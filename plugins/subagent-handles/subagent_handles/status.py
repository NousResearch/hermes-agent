import json
import logging
from typing import Any, Dict, List

from subagent_handles.registry import registry

logger = logging.getLogger(__name__)


def _tool_result(data=None, **kwargs) -> str:
    """Return a JSON result string, using Hermes core helper when importable.

    Falls back to plain json.dumps so the plugin's standalone tests (which
    do not have the hermes-agent core on sys.path) still produce valid JSON
    strings matching the tool-handler contract.
    """
    try:
        from tools.registry import tool_result

        return tool_result(data, **kwargs)
    except ImportError:
        payload = data if data is not None else kwargs
        return json.dumps(payload, ensure_ascii=False)


def _tool_error(message: str, **extra) -> str:
    """Return a JSON error string (Hermes core helper when available)."""
    try:
        from tools.registry import tool_error

        return tool_error(message, **extra)
    except ImportError:
        payload = {"error": message}
        payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def _handle_dict(handle: Any) -> Dict[str, Any]:
    return {
        "subagent_id": handle.subagent_id,
        "session_id": handle.session_id,
        "state": handle.state,
        "role": handle.role,
        "parent_subagent_id": handle.parent_subagent_id,
        "goal": handle.goal,
    }


def handle_subagent_handles(params: Dict[str, Any]) -> str:
    """Resolve handle(s) from the registry — READ-ONLY introspection.

    This plugin does not steer children (the platform's delegation tools
    subagent_send / cancel_subagent are authoritative for live steering);
    it tracks in-flight children as durable handles and reports their state.
    """
    subagent_id = str((params or {}).get("subagent_id") or "").strip()
    if subagent_id:
        handle = registry.resolve(subagent_id)
        if handle is None:
            return _tool_error(
                f"subagent_id={subagent_id!r} not found",
                subagent_id=subagent_id,
            )
        return _tool_result({"subagent_id": subagent_id, "handle": _handle_dict(handle)})

    handles: List[Dict[str, Any]] = []
    for h in registry:
        handles.append(_handle_dict(h))
    return _tool_result({"handles": handles, "count": len(handles)})


SCHEMA = {
    "name": "subagent_handles",
    "description": "List in-flight subagent handles tracked by the subagent-handles registry, "
                   "or resolve one handle by subagent_id (read-only).",
    "parameters": {
        "type": "object",
        "properties": {
            "subagent_id": {
                "type": "string",
                "description": "Optional subagent_id to resolve; omit to list all tracked handles.",
            },
        },
        "required": [],
    },
}


def register_tools(ctx) -> None:
    """Register the read-only introspection tool with the correct PluginContext contract.

    Hermes's PluginContext.register_tool signature is:
        register_hook(name, toolset, schema, handler, check_fn=None, ...)
    Passing the schema as `name` and the handler as `toolset` (as the old code
    did) is silently swallowed and registers nothing.

    NOTE: this plugin deliberately does NOT register subagent_send /
    cancel_subagent — those names belong to the platform delegation toolset
    and register_tool rejects shadowing them without allow_tool_override.
    Live steering stays with the platform tools.
    """
    try:
        ctx.register_tool(
            name="subagent_handles",
            toolset="delegation",
            schema=SCHEMA,
            handler=handle_subagent_handles,
            description="List or resolve subagent handles tracked by the subagent-handles registry (read-only).",
        )
    except Exception as exc:
        logger.debug("register_tool subagent_handles failed: %s", exc)
