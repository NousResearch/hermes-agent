"""Bundled Microsoft 365 Plugin — one coherent, capability-gated integration."""
from __future__ import annotations

from . import tools
from .backend import CAPABILITIES, Microsoft365Settings


def _sdk_available() -> bool:
    try:
        import msgraph  # noqa: F401
        import azure.identity  # noqa: F401
        return True
    except ImportError:
        return False


def register(ctx) -> None:
    """Register preflight and only the capability sections selected by the operator."""
    ctx.register_tool(
        name="microsoft365_preflight", toolset="microsoft365",
        schema={"name": "microsoft365_preflight", "description": "Side-effect-free Microsoft 365 Plugin configuration and permission preflight.", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kwargs: tools.handle_preflight(args, ctx=ctx, **kwargs),
        check_fn=lambda: True, is_async=True, emoji="🧩",
    )
    enabled = ctx.get_config("capabilities", {}) or {}
    settings = Microsoft365Settings.from_mapping({"capabilities": enabled})
    for capability in CAPABILITIES:
        if not settings.enabled(capability):
            continue
        ctx.register_tool(
            name=f"microsoft365_{capability}", toolset="microsoft365",
            schema=tools.schema(capability), handler=tools.capability_handler(capability),
            check_fn=_sdk_available, is_async=True, emoji="📎",
        )
