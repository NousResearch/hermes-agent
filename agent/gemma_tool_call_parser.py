"""Recover Gemma / oMLX tool calls emitted as plain text instead of tool_calls."""

from __future__ import annotations

import json
import logging
import re
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# oMLX Gemma4 output parser serializes calls like:
#   <|tool_call>call:web_search{query: "..."}<tool_call|>
_GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|?tool_call\|?>\s*"
    r"(?:call:)?"
    r"(?P<name>[A-Za-z_][\w.-]*)"
    r"\{(?P<args>.*?)\}"
    r"\s*<(?:/tool_call|tool_call\|?)>",
    re.DOTALL | re.IGNORECASE,
)

_GEMMA_TOOL_CALL_MARKER_RE = re.compile(
    r"<\|?tool_call\|?>",
    re.IGNORECASE,
)


def text_contains_gemma_tool_call(text: str) -> bool:
    """Return True when text looks like a Gemma/oMLX serialized tool call."""
    if not isinstance(text, str) or not text.strip():
        return False
    return bool(_GEMMA_TOOL_CALL_RE.search(text) or _GEMMA_TOOL_CALL_MARKER_RE.search(text))


def gemma_args_blob_to_json(args_blob: str) -> str:
    """Convert Gemma pseudo-JSON args ({query: "x"}) to strict JSON."""
    blob = (args_blob or "").strip()
    if not blob:
        return "{}"
    if not blob.startswith("{"):
        blob = "{" + blob + "}"
    try:
        json.loads(blob)
        return blob
    except json.JSONDecodeError:
        pass

    inner = blob[1:-1] if blob.startswith("{") and blob.endswith("}") else blob
    quoted_inner = re.sub(
        r'(?<!["\w])([A-Za-z_][\w.-]*)\s*:',
        r'"\1":',
        inner,
    )
    quoted = "{" + quoted_inner + "}"
    try:
        json.loads(quoted)
        return quoted
    except json.JSONDecodeError:
        logger.debug("Gemma tool args not parseable as JSON: %r", args_blob[:200])
        return "{}"


def extract_gemma_tool_calls_from_text(
    text: str,
    *,
    valid_tool_names: Optional[Iterable[str]] = None,
) -> Tuple[List[Tuple[str, str]], str]:
    """Extract (tool_name, args_json) pairs and return cleaned remaining text."""
    if not isinstance(text, str) or not text.strip():
        return [], text or ""

    valid = set(valid_tool_names) if valid_tool_names is not None else None
    extracted: List[Tuple[str, str]] = []
    consumed: List[Tuple[int, int]] = []

    for match in _GEMMA_TOOL_CALL_RE.finditer(text):
        name = (match.group("name") or "").strip()
        if not name:
            continue
        if valid is not None and name not in valid:
            continue
        args_json = gemma_args_blob_to_json(match.group("args") or "")
        extracted.append((name, args_json))
        consumed.append((match.start(), match.end()))

    if not consumed:
        return [], text.strip()

    consumed.sort()
    merged: List[Tuple[int, int]] = []
    for start, end in consumed:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned


def recover_gemma_text_tool_calls(
    assistant_message,
    *,
    valid_tool_names: Optional[Iterable[str]] = None,
    call_id_prefix: str = "gemma_tc",
) -> bool:
    """Promote Gemma text tool calls on assistant_message.tool_calls.

    Mutates ``assistant_message`` in place. Returns True when at least one
    tool call was recovered.
    """
    existing = getattr(assistant_message, "tool_calls", None) or []
    if existing:
        return False

    content = getattr(assistant_message, "content", None)
    if not isinstance(content, str) or not content.strip():
        return False

    pairs, cleaned = extract_gemma_tool_calls_from_text(
        content, valid_tool_names=valid_tool_names
    )
    if not pairs:
        return False

    from agent.transports.types import ToolCall

    tool_calls = [
        ToolCall(
            id=f"{call_id_prefix}_{index + 1}",
            name=name,
            arguments=args_json,
        )
        for index, (name, args_json) in enumerate(pairs)
    ]
    assistant_message.tool_calls = tool_calls
    assistant_message.content = cleaned or None
    logger.info(
        "Recovered %d Gemma text tool call(s): %s",
        len(tool_calls),
        ", ".join(tc.name for tc in tool_calls),
    )
    return True
