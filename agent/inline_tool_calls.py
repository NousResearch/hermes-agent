"""JSON-aware extraction for inline XML-wrapped tool calls."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from agent.transports.types import ToolCall


_START_TAG_RE = re.compile(
    r"<(tool_use|tool_call|tool_calls|function_call|function_calls)\b[^>]*>",
    re.IGNORECASE,
)
_JSON_DECODER = json.JSONDecoder()
_BOUNDARY_CHARS = set(" \t\r\n([{,:;.!?")


@dataclass(frozen=True)
class InlineToolExtraction:
    tool_calls: list[ToolCall]
    content: str | None
    parsed_spans: list[tuple[int, int]]


def extract_inline_tool_calls(text: str | None) -> InlineToolExtraction:
    """Extract valid inline XML tool calls from assistant text.

    This intentionally recognizes only JSON payloads wrapped in known tool-call
    tags. The JSON parser determines the payload boundary, so nested objects and
    strings containing text like ``</tool_use>`` are handled safely.
    """
    if not isinstance(text, str) or "<" not in text:
        return InlineToolExtraction([], text, [])

    tool_calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    lower_text = text.lower()
    pos = 0

    while True:
        match = _START_TAG_RE.search(text, pos)
        if match is None:
            break

        if not _has_safe_boundary(text, match.start()):
            pos = match.end()
            continue

        tag = match.group(1).lower()
        json_start = _skip_ws(text, match.end())
        try:
            payload, json_end = _JSON_DECODER.raw_decode(text, json_start)
        except (json.JSONDecodeError, TypeError, ValueError):
            pos = match.end()
            continue

        close_start = _skip_ws(text, json_end)
        close_tag = f"</{tag}>"
        if not lower_text.startswith(close_tag, close_start):
            pos = match.end()
            continue

        span_end = close_start + len(close_tag)
        raw_payload = text[json_start:json_end]
        recovered = _payload_to_tool_calls(
            payload,
            raw_payload=raw_payload,
            start_offset=match.start(),
            first_sequence=len(tool_calls),
        )
        if recovered:
            tool_calls.extend(recovered)
            spans.append((match.start(), span_end))
            pos = span_end
        else:
            pos = match.end()

    cleaned = _remove_spans(text, spans)
    content = (cleaned if cleaned.strip() else None) if spans else text
    return InlineToolExtraction(
        tool_calls=tool_calls,
        content=content,
        parsed_spans=spans,
    )


def strip_inline_tool_call_blocks(text: str | None) -> str:
    """Remove valid inline XML tool-call blocks without executing them."""
    if not isinstance(text, str) or "<" not in text:
        return text or ""
    extracted = extract_inline_tool_calls(text)
    if not extracted.parsed_spans:
        return text
    return extracted.content or ""


def _has_safe_boundary(text: str, start: int) -> bool:
    if start <= 0:
        return True
    return text[start - 1] in _BOUNDARY_CHARS


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])
    return "".join(parts)


def _payload_to_tool_calls(
    payload: Any,
    *,
    raw_payload: str,
    start_offset: int,
    first_sequence: int,
) -> list[ToolCall]:
    entries = payload if isinstance(payload, list) else [payload]
    tool_calls: list[ToolCall] = []
    for offset, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        tool_call = _entry_to_tool_call(
            entry,
            raw_payload=raw_payload,
            start_offset=start_offset,
            sequence=first_sequence + offset,
        )
        if tool_call is not None:
            tool_calls.append(tool_call)
    return tool_calls


def _entry_to_tool_call(
    entry: dict[str, Any],
    *,
    raw_payload: str,
    start_offset: int,
    sequence: int,
) -> ToolCall | None:
    function = entry.get("function")
    if not isinstance(function, dict):
        function = {}

    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    arguments, ok = _normalize_arguments(entry, function)
    if not ok:
        return None

    call_id = _first_nonempty_string(
        entry.get("id"),
        entry.get("tool_call_id"),
        entry.get("tool_use_id"),
        entry.get("call_id"),
    )
    if call_id is None:
        call_id = _stable_call_id(raw_payload, start_offset, sequence)

    return ToolCall(id=call_id, name=name.strip(), arguments=arguments)


def _normalize_arguments(
    entry: dict[str, Any],
    function: dict[str, Any],
) -> tuple[str, bool]:
    sentinel = object()
    raw_args = entry.get("arguments", sentinel)
    if raw_args is sentinel:
        raw_args = entry.get("input", sentinel)
    if raw_args is sentinel:
        raw_args = function.get("arguments", sentinel)
    if raw_args is sentinel or raw_args is None:
        return "{}", True

    if isinstance(raw_args, dict):
        return json.dumps(raw_args, ensure_ascii=False), True

    if isinstance(raw_args, str):
        stripped = raw_args.strip()
        if not stripped:
            return "{}", True
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            return "", False
        if not isinstance(parsed, dict):
            return "", False
        return json.dumps(parsed, ensure_ascii=False), True

    return "", False


def _first_nonempty_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _stable_call_id(raw_payload: str, start_offset: int, sequence: int) -> str:
    digest = hashlib.sha1(
        f"{raw_payload}\0{start_offset}\0{sequence}".encode("utf-8")
    ).hexdigest()
    return f"call_inline_{digest[:16]}"
