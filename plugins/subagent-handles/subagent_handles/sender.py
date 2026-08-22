import json
import logging
from typing import Any, Dict

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


def _resolve_running(subagent_id: str) -> Any:
    handle = registry.resolve(subagent_id)
    if handle is None or handle.state != "running":
        return None
    return handle


def handle_subagent_send(params: Dict[str, Any]) -> str:
    subagent_id = str((params or {}).get("subagent_id") or "").strip()
    text = str((params or {}).get("text") or "").strip()
    if not subagent_id:
        return _tool_error("subagent_id is required")
    if not text:
        return _tool_error("text is required")
    handle = _resolve_running(subagent_id)
    if handle is None:
        return _tool_error(
            f"subagent_id={subagent_id!r} is not running or not found",
            subagent_id=subagent_id,
        )
    logger.debug("subagent_send queued for %s: %r", subagent_id, text[:120])
    return _tool_result({
        "ok": True,
        "subagent_send": {
            "subagent_id": handle.subagent_id,
            "session_id": handle.session_id,
            "state": handle.state,
        },
        "queued": True,
    })


def handle_cancel_subagent(params: Dict[str, Any]) -> str:
    subagent_id = str((params or {}).get("subagent_id") or "").strip()
    if not subagent_id:
        return _tool_error("subagent_id is required")
    handle = registry.resolve(subagent_id)
    if handle is None:
        return _tool_error(
            f"subagent_id={subagent_id!r} not found",
            subagent_id=subagent_id,
        )
    if handle.state != "running":
        return _tool_error(
            f"subagent_id={subagent_id!r} is not running",
            subagent_id=subagent_id,
            state=handle.state,
        )
    updated = registry.set_state(subagent_id, "cancelled")
    if not updated:
        return _tool_error(
            f"subagent_id={subagent_id!r} could not be cancelled",
            subagent_id=subagent_id,
        )
    # Persist the cancelled state so a later session sees it (same store as
    # the hooks use — survive-restart parity with start/stop).
    try:
        from subagent_handles.persister import SessionPersister, default_persist_root

        SessionPersister(default_persist_root()).checkpoint(handle)
    except Exception:
        logger.debug("cancel_subagent checkpoint failed", exc_info=True)
    return _tool_result({
        "ok": True,
        "subagent_id": subagent_id,
        "state": "cancelled",
        "session_id": handle.session_id,
    })


SCHEMA = {
    "subagent_send": {
        "name": "subagent_send",
        "description": "Send steering text to a running subagent by handle.",
        "parameters": {
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string", "description": "Target subagent_id"},
                "text": {"type": "string", "description": "Steer message to deliver"},
            },
            "required": ["subagent_id", "text"],
        },
    },
    "cancel_subagent": {
        "name": "cancel_subagent",
        "description": "Cancel a running subagent by subagent_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string", "description": "subagent_id to cancel"},
            },
            "required": ["subagent_id"],
        },
    },
}


def register_tools(ctx) -> None:
    """Register the two steering tools with the correct PluginContext contract.

    Hermes's PluginContext.register_tool signature is:
        register_hook(name, toolset, schema, handler, check_fn=None, ...)
    Passing the schema as `name` and the handler as `toolset` (as the old code
    did) is silently swallowed and registers nothing.
    """
    try:
        ctx.register_tool(
            name="subagent_send",
            toolset="delegation",
            schema=SCHEMA["subagent_send"],
            handler=handle_subagent_send,
            description="Send steering text to a running subagent by handle.",
        )
    except Exception as exc:
        logger.debug("register_tool subagent_send failed: %s", exc)
    try:
        ctx.register_tool(
            name="cancel_subagent",
            toolset="delegation",
            schema=SCHEMA["cancel_subagent"],
            handler=handle_cancel_subagent,
            description="Cancel a running subagent by subagent_id.",
        )
    except Exception as exc:
        logger.debug("register_tool cancel_subagent failed: %s", exc)
