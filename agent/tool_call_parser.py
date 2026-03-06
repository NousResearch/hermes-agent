"""Parse text-based tool calls from model output (e.g. Qwen3.5 <tool_call> XML tags)."""

import json
import re
from dataclasses import dataclass

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict
    raw: str


def parse_tool_calls(content: str) -> list[ParsedToolCall]:
    results: list[ParsedToolCall] = []
    for m in _TOOL_CALL_RE.finditer(content):
        raw = m.group(0)
        body = m.group(1).strip()
        try:
            # Handle single quotes by trying double-quote JSON first, then
            # falling back to replacing single quotes.
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = json.loads(body.replace("'", '"'))
            if not isinstance(data, dict):
                continue
            name = data.get("name") or data.get("function")
            if not name:
                continue
            arguments = data.get("arguments", {})
            if not isinstance(arguments, dict):
                continue
            results.append(ParsedToolCall(name=str(name), arguments=arguments, raw=raw))
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    return results


def strip_tool_calls(content: str) -> str:
    return _TOOL_CALL_RE.sub("", content)


def has_tool_calls(content: str) -> bool:
    return _TOOL_CALL_RE.search(content) is not None


def has_tool_call_start(content: str) -> bool:
    """Return True if content contains a <tool_call> opening tag, even without closing tag."""
    return "<tool_call>" in content


def content_after_tool_calls(content: str) -> str:
    text = strip_tool_calls(content)
    # Collapse multiple blank lines and strip leading/trailing whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
