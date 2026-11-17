"""Feishu / Lark plugin: registers the ``feishu`` platform adapter and the comment agent's five model tools
through the public PluginContext."""

__all__ = ["register"]


def register(ctx) -> None:
    """Plugin entry point. Discovery imports this package in every process (``provides_tools`` in
    plugin.yaml), so the adapter is imported only here, when the gateway materialises the platform."""
    from .tools import register_tools
    from .adapter import register as register_platform

    register_tools(ctx)
    register_platform(ctx)
