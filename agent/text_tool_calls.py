"""Compatibility parser for models that emit tool calls as assistant text."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from agent.transports.types import ToolCall


_TEXT_TOOL_CALL_BLOCK_RE = re.compile(
    r"<(?P<tag>tool_?calls?)\b[^>]*>\s*(?P<payload>[\s\S]*?)\s*</(?P=tag)>\s*$",
    re.IGNORECASE,
)
_TEXT_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
_MAX_TEXT_TOOL_CALLS = 8


def parse_text_tool_calls(content: Any) -> tuple[str | None, list[ToolCall]] | None:
    """Promote a terminal tool-call block to structured calls.

    Some OpenAI-compatible local models ignore the structured ``tool_calls``
    response field and instead emit a protocol block such as
    ``<TOOLCALL>[{"name": "skill_view", ...}]</TOOLCALL>``. Treat that as a
    tool turn when the block is either the entire assistant message or a
    newline-delimited terminal suffix, and every call has a valid name plus
    object arguments. Any preceding prose is preserved as assistant content.
    Inline examples, fenced examples, malformed JSON, and text after the block
    remain inert.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    match = _TEXT_TOOL_CALL_BLOCK_RE.search(content)
    if not match:
        return None
    prefix = content[: match.start()]
    if prefix.strip():
        # A mixed prose/tool response is only protocol when the opening tag is
        # on its own new line. This rejects inline quoted examples while still
        # accepting the exact shape emitted by qwen during real agent runs.
        if not prefix.rstrip(" \t").endswith(("\n", "\r")):
            return None
        # A closing Markdown fence after the tag already prevents the regex
        # match; reject an opening fence immediately before it as well.
        if prefix.rstrip().endswith("```"):
            return None
        visible_content = prefix.rstrip() or None
    else:
        visible_content = None
    try:
        payload = json.loads(match.group("payload"))
    except (TypeError, ValueError):
        return None
    raw_calls = payload if isinstance(payload, list) else [payload]
    if not raw_calls or len(raw_calls) > _MAX_TEXT_TOOL_CALLS:
        return None

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    parsed: list[ToolCall] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            return None
        function = raw_call.get("function")
        if function is not None and not isinstance(function, dict):
            return None
        source = function if isinstance(function, dict) else raw_call
        name = source.get("name")
        arguments = source.get("arguments", {})
        if not isinstance(name, str) or not _TEXT_TOOL_NAME_RE.fullmatch(name.strip()):
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                return None
        if not isinstance(arguments, dict):
            return None
        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"text_call_{digest}_{index + 1}"
        parsed.append(
            ToolCall(
                id=call_id,
                name=name.strip(),
                arguments=json.dumps(arguments, separators=(",", ":")),
            )
        )
    return visible_content, parsed
