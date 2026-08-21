from .adapter import register as register_platform
from .tools import register_tools


def register(ctx) -> None:
    """Register the Buzz platform and its opt-in channel tools."""
    register_tools(ctx)
    register_platform(ctx)


__all__ = ["register"]
