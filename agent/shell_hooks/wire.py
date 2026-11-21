"""The JSON wire: the stdin payload a hook receives and the stdout directive it may return,
translated into the Hermes shape per event (``_parse_response``)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_BLOCK_MESSAGE = "Blocked by shell hook."

# kwargs promoted to top-level payload keys; everything else lands under ``extra``.
_TOP_LEVEL_PAYLOAD_KEYS = {"tool_name", "args", "session_id", "parent_session_id"}


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _payload_fields(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Common stdin/POST payload fields (shared with outbound webhooks); key order is wire order."""
    try:
        cwd = str(Path.cwd())
    except OSError:
        cwd = ""
    from hermes_cli.profiles import get_active_profile_name
    from agent.delegation_context import owned_kanban_task
    extra = {k: v for k, v in kwargs.items() if k not in _TOP_LEVEL_PAYLOAD_KEYS}
    # Resolve in the firing execution before the descendant environment is scrubbed.
    # This is payload metadata, never permission for the hook to act as the worker.
    extra["kanban_task_id"] = owned_kanban_task()
    return {
        "tool_name": kwargs.get("tool_name"),
        "tool_input": kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
        "session_id": kwargs.get("session_id") or kwargs.get("parent_session_id") or "",
        "cwd": cwd,
        # Resolved at fire time: a multiplexed gateway's hook script must know which profile fired it.
        "profile": get_active_profile_name(),
        "extra": extra,
    }


def _serialize_payload(event: str, kwargs: dict[str, Any]) -> str:
    """Render the stdin JSON payload; unserialisable values are stringified."""
    return json.dumps({"hook_event_name": event, **_payload_fields(kwargs)}, ensure_ascii=False, default=str)


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def _block_message(primary: Any, secondary: Any) -> str:
    """Validated string block message (primary wins), falling back to the default."""
    raw = primary or secondary
    return raw if isinstance(raw, str) and raw else _DEFAULT_BLOCK_MESSAGE

# pre_tool_call dialects in check order — Hermes ``action`` then Claude-Code ``decision`` — as (verb key,
# block-message primary, secondary, modify payload key); both translate to the canonical Hermes shape.
_PRE_TOOL_DIALECTS = (("action", "message", "reason", "args"), ("decision", "reason", "message", "tool_input"))


def _parse_pre_tool_call(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    for verb, primary, secondary, _ in _PRE_TOOL_DIALECTS:
        if data.get(verb) == "block":
            return {"action": "block", "message": _block_message(data.get(primary), data.get(secondary))}
    for verb, _, _, payload in _PRE_TOOL_DIALECTS:
        if data.get(verb) == "modify" and isinstance(data.get(payload), dict):
            return {"action": "modify", "args": data[payload]}
    # Hermes-only escalation to the human-approval gate (#92553). Claude-Code's ``decision:
    # approve`` means auto-ALLOW, so it is deliberately not mapped onto this.
    if data.get("action") == "approve":
        directive: dict[str, Any] = {"action": "approve"}
        for key in ("message", "rule_key"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                directive[key] = value.strip()
        return directive
    return None


def _parse_pre_verify(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    # "continue" (Hermes) / "block" (Claude-Code Stop) both mean keep going; no message is a no-op.
    action = str(data.get("action") or data.get("decision") or "").strip().lower()
    message = data.get("message") or data.get("reason")
    if action in {"continue", "block"} and isinstance(message, str) and message.strip():
        return {"action": "continue", "message": message.strip()}
    return None


def _parse_context(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    context = data.get("context")
    return {"context": context} if isinstance(context, str) and context.strip() else None


def _parse_pre_llm_call(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    # A turn veto before any inference request, in either dialect (as for pre_tool_call); else context.
    for verb, primary, secondary, _ in _PRE_TOOL_DIALECTS:
        if data.get(verb) == "block":
            return {"action": "block", "message": _block_message(data.get(primary), data.get(secondary))}
    return _parse_context(data)

_RESPONSE_PARSERS: dict[str, Callable[[dict[str, Any]], Optional[dict[str, Any]]]] = {
    "pre_tool_call": _parse_pre_tool_call, "pre_verify": _parse_pre_verify, "pre_llm_call": _parse_pre_llm_call,
}


def _parse_response(event: str, stdout: str) -> Optional[dict[str, Any]]:
    """Translate stdout JSON into a Hermes wire-shape dict, or ``None``."""
    stdout = (stdout or "").strip()
    if not stdout:
        return None
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("shell hook stdout was not valid JSON (event=%s): %s", event, stdout[:200])
        return None
    return _RESPONSE_PARSERS.get(event, _parse_context)(data) if isinstance(data, dict) else None
