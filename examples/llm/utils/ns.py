import os


def get_namespace():
    return os.getenv("DYN_NAMESPACE", "dynamo")
