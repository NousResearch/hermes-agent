"""Bundled Microsoft 365 Plugin — one coherent, capability-gated integration."""
from __future__ import annotations

from . import tools
from .backend import CAPABILITIES


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
    if not isinstance(enabled, dict):
        enabled = {}
    for capability in CAPABILITIES:
        if not bool(enabled.get(capability, False)):
            continue
        ctx.register_tool(
            name=f"microsoft365_{capability}", toolset="microsoft365",
            schema=tools.schema(capability), handler=lambda args, c=capability: tools._run(c, args, ctx),
            check_fn=_sdk_available, is_async=True, emoji="📎",
        )
