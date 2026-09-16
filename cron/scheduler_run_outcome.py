"""Map a finished cron parent turn onto nested delegated-child outcomes.

A cron fire currently treats ``run_conversation`` completing without
``failed=True`` as success. That is the parent turn, not the job: the parent
can narrate a ``delegate_task`` child that returned ``status=failed`` and the
scheduler still writes ``last_status=ok``. This module is the choke point that
turns those structured child results into cron-run failure evidence.

Only ``delegate_task`` payloads are consulted. Terminal exits, quoted JSON in
the parent report, and background dispatch handles are not evidence.
"""
from __future__ import annotations

import json
from typing import Any, Optional

# Same terminal child statuses the parent-facing delegate_task payload uses
# (tools.delegate_tool_progress.SUBAGENT_FAILURE_STATUSES). Kept local so cron
# import does not pull the delegate tool package.
_CHILD_FAILURE_STATUSES = frozenset({"failed", "error", "timeout"})
_DELEGATE_TOOL_NAMES = frozenset({"delegate_task"})
_ERROR_MAX_CHARS = 240


def nested_child_failure(result: Any) -> Optional[str]:
    """Return evidence when a parent result still contains a failed delegated child.

    ``None`` means no structured child failure was found. Malformed tool JSON is
    ignored rather than converted into a false cron failure.
    """
    if not isinstance(result, dict):
        return None
    messages = result.get("messages")
    if not isinstance(messages, list):
        return None

    name_by_call_id = _tool_names_by_call_id(messages)
    failures: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tool_name = name_by_call_id.get(msg.get("tool_call_id") or "") or str(
            msg.get("name") or ""
        ).strip()
        if tool_name not in _DELEGATE_TOOL_NAMES:
            continue
        failures.extend(_failures_in_delegate_payload(_parse_tool_content(msg.get("content"))))
    if not failures:
        return None
    if len(failures) == 1:
        return f"Delegated child failed: {failures[0]}"
    return "Delegated children failed: " + "; ".join(failures)


def _tool_names_by_call_id(messages: list) -> dict[str, str]:
    names: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict) or not call.get("id"):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(fn.get("name") or "").strip()
            if name:
                names[str(call["id"])] = name
    return names


def _parse_tool_content(content: Any) -> Any:
    text = _stringify_tool_content(content).strip()
    if not text:
        return None
    if text[0] in "{[":
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def _stringify_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    if isinstance(content, dict):
        try:
            return json.dumps(content, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def _failures_in_delegate_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    entries = payload.get("results")
    if not isinstance(entries, list):
        # Background dispatch handles and control actions have no results list.
        return []
    found: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status not in _CHILD_FAILURE_STATUSES:
            continue
        found.append(_child_failure_text(entry))
    return found


def _child_failure_text(entry: dict) -> str:
    for key in ("error", "summary"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            if len(text) > _ERROR_MAX_CHARS:
                return text[: _ERROR_MAX_CHARS - 3] + "..."
            return text
    status = str(entry.get("status") or "failed").strip() or "failed"
    return f"child status={status}"
