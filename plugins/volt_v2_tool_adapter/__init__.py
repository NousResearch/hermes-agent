"""Volt V2 tool adapter proof plugin."""

from __future__ import annotations

from .adapter import on_post_tool_call, transform_tool_result


__all__ = ["register", "on_post_tool_call", "transform_tool_result"]


def register(ctx) -> None:
    """Register observer/transform hooks only.

    The adapter remains inert unless ``volt_v2.tool_adapter.enabled`` is true
    in config.  No core tool override is registered in v0.
    """
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("transform_tool_result", transform_tool_result)
