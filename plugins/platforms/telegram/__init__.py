from .adapter import register as register_platform
from .topic_tool import register_topic_tool


def register(ctx) -> None:
    register_platform(ctx)
    register_topic_tool(ctx)


__all__ = ["register"]
