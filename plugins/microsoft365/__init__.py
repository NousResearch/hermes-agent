"""Bundled Microsoft 365 Plugin — one coherent, capability-gated integration."""
from __future__ import annotations

from . import tools
from .backend import CAPABILITIES, AUTHENTICATION_MODE, Microsoft365Settings, operation_support, supported_operations, WRITE_OPERATIONS


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
    settings = Microsoft365Settings.from_mapping({
        "tenant_id": ctx.get_config("tenant_id", ""),
        "client_id": ctx.get_config("client_id", ""),
        "client_secret": ctx.get_config("client_secret", ""),
        "user_id": ctx.get_config("user_id", "me"),
        "capabilities": enabled,
    })

    def pre_tool_call(*, tool_name="", args=None, **kwargs):
        prefix = "microsoft365_"
        if not isinstance(tool_name, str) or not tool_name.startswith(prefix):
            return None
        capability = tool_name[len(prefix):]
        if capability not in CAPABILITIES:
            return None
        if not isinstance(args, dict):
            return {"action": "block", "message": f"Microsoft 365 {capability} call rejected: arguments must be an object"}
        action = str(args.get("action") or "").strip().lower()
        status = operation_support(AUTHENTICATION_MODE, capability, action)
        if not status.supported:
            return {"action": "block", "message": f"Microsoft 365 operation unsupported: {capability}.{action}: {status.reason}"}
        if action not in settings.operations(capability):
            return {"action": "block", "message": f"Microsoft 365 operation is disabled: {capability}.{action}"}
        try:
            tools._validate_action(capability, action, args)
        except ValueError as exc:
            return {"action": "block", "message": f"Invalid Microsoft 365 arguments: {exc}"}
        if action in WRITE_OPERATIONS:
            return {"action": "approve", "message": f"Microsoft 365 {action}: external side effect", "rule_key": f"microsoft365.{capability}.{action}"}
        return None

    ctx.register_hook("pre_tool_call", pre_tool_call)
    for capability in CAPABILITIES:
        if not settings.enabled(capability):
            continue
        operations = sorted(supported_operations(settings, capability))
        ctx.register_tool(
            name=f"microsoft365_{capability}", toolset="microsoft365",
            schema=tools.schema(capability, operations), handler=tools.capability_handler(capability),
            check_fn=_sdk_available, is_async=True, emoji="📎",
        )
