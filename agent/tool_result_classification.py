"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
import re
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


# Tools whose interrupted/dangling execution is safe to discard because they
# cannot mutate either external state or Hermes session state. Unknown/plugin/
# MCP tools stay effect-capable by default.
NO_EFFECT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "session_search", "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})


def tool_may_have_side_effect(tool_name: str) -> bool:
    return tool_name not in NO_EFFECT_TOOL_NAMES


# Set by a tool that REFUSED a call the harness judged redundant (repeated identical
# read/search). The body still carries ``"error"`` so the model reads it as a stop
# signal, but nothing failed: failure classifiers must not count it, or the cheap
# refusal feeds the streak that fires ``repeated_exact_failure_block``.
GUARDRAIL_REFUSAL_KEY = "guardrail_refusal"


def is_guardrail_refusal(result: Any) -> bool:
    """Return True when ``result`` (JSON string or parsed dict) is a harness refusal."""
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result.strip())
        except Exception:
            return False
    return isinstance(data, dict) and data.get(GUARDRAIL_REFUSAL_KEY) is True


# A quoted "error"/"failed" in a result's head marks it failed UNLESS it is a key whose value is
# empty: success payloads routinely carry ``"error": null`` per item (web_extract) or a
# ``"failed": 0`` count, and a bare substring test flagged every one of them.
_FAILURE_MARKER_RE = re.compile(
    r'"(?:error|failed)"(?!\s*:\s*(?:null|false|0|""|\[\]|\{\})\s*(?:[,}\]]|$))', re.IGNORECASE)


def has_failure_marker(result: str) -> bool:
    """Last-resort text test for a failed tool result: a non-empty ``"error"``/``"failed"``
    STARTING in the first 500 chars, or a body starting with ``Error``. The value is read past
    the cut, so an ``"error": null`` straddling char 500 is not mistaken for a truncated one."""
    head = result[:600]
    return any(m.start() < 500 for m in _FAILURE_MARKER_RE.finditer(head)) or result.startswith("Error")


# ``timeout`` is deliberately absent: process(wait) returns it for a still-running job.
_FAILED_STATUSES = frozenset({"failed", "error"})


def shape_failure_error(data: dict) -> str | None:
    """Error text when a JSON-object result failed by SHAPE rather than by its top-level
    ``error``: a ``status`` of failed/error, or a ``results`` batch where EVERY item carries an
    error (delegate_task with all children interrupted, a connector batch with every call
    refused). None for a clean or partial result. Top-level JSON objects skip the substring
    scan, so this is what keeps those failures counted."""
    if str(data.get("status", "")).lower() in _FAILED_STATUSES:
        return str(data.get("status"))
    items = data.get("results")
    if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
        return None
    errors = [i.get("error") for i in items]
    if not all(errors):
        return None
    first = errors[0]
    return str(first.get("message") or first) if isinstance(first, dict) else str(first)


def structured_tool_result_failed(data: dict[str, Any]) -> bool:
    """Classify the tool envelope, never nested diagnostic findings or text."""
    if any(data.get(key) is False for key in ("success", "ok")):
        return True
    if any(data.get(key) is True for key in ("failed", "isError")):
        return True
    if data.get("status") in ("error", "failed") or data.get("error"):
        return True
    return any(
        isinstance(code, int) and not isinstance(code, bool) and code != 0
        for code in (data.get("exit_code"), data.get("returncode"))
    )


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    try:
        data = json.loads(result.strip())
    except Exception:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False
