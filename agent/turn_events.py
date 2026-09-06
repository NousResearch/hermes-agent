"""Small, machine-derived facts about tools used during one agent turn.

The Becky loop auto-close policy deliberately does not inspect assistant prose.
This module records only the tool name, a minimal action argument, and whether
the corresponding result completed successfully.  Raw tool output stays in the
normal transcript and is never copied into the gateway control decision.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.display import _detect_tool_failure

_FAILURE_MARKERS = (
    "cancelled",
    "canceled",
    "did not return a result",
    "error executing tool",
    "timed out",
    "was skipped",
)
_ACTION_ARGUMENTS = frozenset({"action", "operation"})
_SUCCESS_STATUSES = frozenset({
    "added",
    "completed",
    "created",
    "done",
    "inserted",
    "ok",
    "success",
    "succeeded",
})
_NEGATED_SUCCESS_RE = re.compile(
    r"\b(?:not|never|unable to|could not|failed to)\s+(?:be\s+)?"
    r"(?:created|added|inserted)\b",
    re.IGNORECASE,
)
_FAILURE_TEXT_RE = re.compile(
    r"\b(?:cancelled|canceled|error|failed|failure|skipped|timeout|timed\s+out)\b",
    re.IGNORECASE,
)
_POSITIVE_TEXT_RE = re.compile(
    r"(?:\b(?:successfully|succeeded)\b|"
    r"\b(?:task|tasks|event|events)(?:\s+\w+){0,2}\s+"
    r"(?:created|added|inserted)\b|"
    r"\b(?:created|added|inserted)\s+(?:a\s+)?"
    r"(?:task|tasks|event|events)\b)",
    re.IGNORECASE,
)


def _call_id(tool_call: object) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or tool_call.get("call_id") or "").strip()
    return str(
        getattr(tool_call, "id", None)
        or getattr(tool_call, "call_id", None)
        or ""
    ).strip()


def _call_function(tool_call: object) -> object:
    if isinstance(tool_call, dict):
        return tool_call.get("function") or {}
    return getattr(tool_call, "function", None)


def _call_name(tool_call: object) -> str:
    function = _call_function(tool_call)
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    return str(getattr(function, "name", None) or "").strip()


def _call_arguments(tool_call: object) -> dict[str, Any]:
    function = _call_function(tool_call)
    if isinstance(function, dict):
        raw = function.get("arguments")
    else:
        raw = getattr(function, "arguments", None)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _result_data(content: object) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    # MCP results are wrapped for the model with an untrusted-data envelope
    # before they are appended to the conversation. Recover only the body of
    # Hermes's exact wrapper; do not scan arbitrary prose for a success-shaped
    # JSON object supplied by an external result.
    opening_end = content.find(">\n")
    closing_start = content.rfind("\n</untrusted_tool_result>")
    body_start = content.find("\n\n", opening_end + 2)
    if (
        content.startswith("<untrusted_tool_result ")
        and opening_end >= 0
        and body_start >= 0
        and closing_start > body_start
    ):
        body = content[body_start + 2 : closing_start]
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _has_success_receipt(name: str, content: object) -> bool:
    """Return whether a tool result contains affirmative completion evidence."""
    data = _result_data(content)
    if data:
        if data.get("success") is True or data.get("ok") is True:
            return True
        status = data.get("status")
        if isinstance(status, str) and status.casefold() in _SUCCESS_STATUSES:
            return True

        normalized_name = name.casefold().replace("-", "_")
        if normalized_name == "terminal":
            exit_code = data.get("exit_code")
            if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                return exit_code == 0

        for key in ("id", "task_id", "event_id"):
            if data.get(key):
                return True
        for key in ("tasks", "events"):
            values = data.get(key)
            if isinstance(values, list) and any(value for value in values):
                return True
        return False

    if not isinstance(content, str):
        return False
    text = content.strip()
    if not text or _NEGATED_SUCCESS_RE.search(text):
        return False
    return _POSITIVE_TEXT_RE.search(text) is not None


def _tool_result_failed(name: str, content: object) -> bool:
    if content is None:
        return True
    text = _content_text(content)
    data = _result_data(content)
    if data:
        if data.get("success") is False or data.get("ok") is False:
            return True
        if data.get("error"):
            return True
        if str(data.get("status") or "").casefold() in {
            "blocked",
            "cancelled",
            "canceled",
            "error",
            "failed",
            "skipped",
            "timeout",
        }:
            return True
    elif (
        any(marker in text.casefold() for marker in _FAILURE_MARKERS)
        or _FAILURE_TEXT_RE.search(text) is not None
    ):
        return True
    try:
        failed, _ = _detect_tool_failure(name, text)
    except Exception:
        return True
    if failed:
        return True
    # A non-error response is not necessarily proof that the requested side
    # effect landed.  Unknown, empty, and pending responses remain ambiguous.
    return not _has_success_receipt(name, content)


def _safe_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep only arguments needed by the narrow auto-close classifier."""
    normalized = name.casefold().replace("-", "_")
    if normalized != "terminal":
        return {
            key: value[:128]
            for key, value in arguments.items()
            if key in _ACTION_ARGUMENTS and isinstance(value, str)
        }
    command = arguments.get("command")
    result: dict[str, Any] = {
        "background": arguments.get("background") is True,
    }
    if isinstance(command, str):
        result["command"] = command[:4_000]
        if len(command) > 4_000:
            result["command_truncated"] = True
    return result


def record_turn_tool_events(
    agent: object,
    assistant_message: object,
    messages: list[object],
    tool_messages_start: int,
) -> None:
    """Append machine-derived outcomes for the just-executed tool batch.

    The executor appends one ``role=tool`` message per requested call.  A
    missing or malformed result is recorded as unsuccessful so callers fail
    closed and leave the Becky topic open.
    """
    calls = getattr(assistant_message, "tool_calls", None)
    if not isinstance(calls, list):
        calls = list(calls or [])
    tool_messages = [
        message
        for message in messages[max(tool_messages_start, 0) :]
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    recorded = getattr(agent, "_turn_tool_events", None)
    if not isinstance(recorded, list):
        recorded = []
        setattr(agent, "_turn_tool_events", recorded)

    def _append_event(
        tool_call: object,
        result: dict[str, Any] | None,
        *,
        correlated: bool,
    ) -> None:
        requested_name = _call_name(tool_call)
        arguments = _call_arguments(tool_call)
        actual_name = ""
        content: object = None
        if result is not None:
            actual_name = str(
                result.get("name") or result.get("tool_name") or ""
            ).strip()
            content = result.get("content")
        event: dict[str, Any] = {
            "name": actual_name,
            "requested_name": requested_name,
            "success": (
                correlated
                and bool(actual_name)
                and not _tool_result_failed(actual_name, content)
            ),
            "arguments": _safe_arguments(actual_name or requested_name, arguments),
        }
        normalized_requested_name = requested_name.casefold().replace("-", "_")
        if normalized_requested_name in {
            "tool_search",
            "mcp_tool_search",
            "tool_call",
            "mcp_tool_call",
        }:
            event["via_tool_search"] = True
        if actual_name.casefold().replace("-", "_") == "terminal":
            exit_code = _result_data(content).get("exit_code")
            if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                event["exit_code"] = exit_code
        recorded.append(event)

    if not calls:
        return

    call_ids = [_call_id(tool_call) for tool_call in calls]
    result_ids = [str(message.get("tool_call_id") or "").strip() for message in tool_messages]
    ids_are_exact = (
        len(call_ids) == len(set(call_ids))
        and all(call_ids)
        and len(tool_messages) == len(calls)
        and len(result_ids) == len(set(result_ids))
        and all(result_ids)
        and set(call_ids) == set(result_ids)
    )
    if not ids_are_exact:
        # Never guess by position: a missing, duplicated, mismatched, or extra
        # result cannot prove which side effect occurred.  Add a sentinel for
        # batch-shape corruption so a one-call batch can never look singular.
        for tool_call in calls:
            _append_event(tool_call, None, correlated=False)
        recorded.append({
            "name": "",
            "requested_name": "",
            "success": False,
            "arguments": {},
        })
        return

    by_id = {str(message["tool_call_id"]).strip(): message for message in tool_messages}
    for index, tool_call in enumerate(calls):
        call_id = _call_id(tool_call)
        _append_event(tool_call, by_id.get(call_id), correlated=True)


__all__ = ["record_turn_tool_events"]
