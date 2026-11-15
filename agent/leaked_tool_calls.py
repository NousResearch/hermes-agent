"""Recover leaked qwen3-XML tool calls from streamed assistant content.

A vLLM server configured with the ``qwen3_xml`` tool-call parser (plus the reasoning
parser) can write the tool-call markup into ``content`` as plain text while returning
zero structured ``tool_calls``: a ``tool_call`` wrapper containing a ``function=NAME``
element with one or more ``parameter=KEY`` bodies, then closing tags. The agent then shows
raw XML as visible text, finds no tool round, and burns the empty-response ladder.

The tag literals are built from parts (never contiguous in source). The contract:
- every complete ``function`` block (wrapper opened, ``function`` opened with a name, at
  least one full ``parameter`` pair, ``function`` and wrapper closed) becomes an
  OpenAI-shaped tool call dict ``{id, type, function: {name, arguments}}``;
- parameter bodies are raw strings; a body that parses as JSON becomes the parsed value,
  otherwise the stripped body stays a string;
- incomplete/dangling markup stays in ``remaining_text`` byte-unchanged — including
  streams that began mid-block (dangling closing tags) and complete wrappers with no
  parameter pair (not function calls);
- text without the markup passes through untouched with an empty list.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("agent.conversation_loop")

# Tag literals built from parts so this module never carries contiguous tool-call markup.
_OPEN_TC = "<" + "tool_call>"
_CLOSE_TC = "</" + "tool_call" + ">"
_FN_OPEN_PREFIX = "<" + "function="
_FN_CLOSE = "</" + "function" + ">"
_P_OPEN_PREFIX = "<" + "parameter="
_P_CLOSE = "</" + "parameter" + ">"

# Complete wrapper: opened, named function, closed, closed. Lazy body so adjacent blocks
# (``...CLOSE_OPEN`` with no separator) each match as one whole.
_BLOCK_RE = re.compile(
    _OPEN_TC + r"\s*<" + r"function=(\w+)\s*>" + r"(.*?)" + r"</" + r"function>" + r"\s*" + _CLOSE_TC,
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<" + r"parameter=(\w+)\s*>" + r"(.*?)" + r"</" + r"parameter>", re.DOTALL)

FUNCTION_MARKER = _FN_OPEN_PREFIX


def _param_value(body: str) -> Any:
    """Raw parameter body; JSON-parsed when it is JSON, plain string otherwise."""
    value = body.strip()
    try:
        return json.loads(value)
    except Exception:
        return value


def parse_leaked_tool_call_blocks(text: str) -> tuple[list[dict[str, Any]], str]:
    """Split leaked qwen3-XML tool calls out of assistant content.

    Returns ``(tool_calls, remaining_text)``; complete blocks are consumed from the text,
    everything incomplete stays byte-unchanged in ``remaining_text``.
    """
    calls: list[dict[str, Any]] = []

    def _consume(match: re.Match) -> str:
        params = _PARAM_RE.findall(match.group(2))
        if not params:
            # Complete wrapper, zero parameter pairs: not a function call. Keep it visible.
            return match.group(0)
        args = {key: _param_value(body) for key, body in params}
        calls.append({
            # Deterministic per-response ids; the tool round uniquifies against session history.
            "id": f"leaked_{len(calls) + 1}",
            "type": "function",
            "function": {"name": match.group(1), "arguments": json.dumps(args)},
        })
        return ""

    remaining = _BLOCK_RE.sub(_consume, text)
    return calls, remaining


def recover_leaked_tool_calls(agent: Any, assistant_message: Any) -> bool:
    """Intake hook: turn a leaked qwen3-XML content round into a structured tool round.

    Fires for provider ``custom`` ONLY, when structured ``tool_calls`` are empty and the
    content carries the ``function=`` marker. On success, ``assistant_message`` gains
    attribute-shaped OpenAI tool calls (the shape every tool-round consumer reads), its
    content is replaced by the remaining text, and True is returned so the caller rebinds
    ``finish_reason``. Any parse miss, empty content, or non-complete block leaves the
    message untouched.
    """
    if getattr(agent, "provider", None) != "custom":
        return False
    content = getattr(assistant_message, "content", None)
    if not isinstance(content, str) or not content:
        return False
    if getattr(assistant_message, "tool_calls", None):
        return False
    if FUNCTION_MARKER not in content:
        return False
    calls, remaining = parse_leaked_tool_call_blocks(content)
    if not calls:
        return False
    assistant_message.tool_calls = [
        SimpleNamespace(
            id=c["id"], type="function",
            function=SimpleNamespace(name=c["function"]["name"], arguments=c["function"]["arguments"]),
        )
        for c in calls
    ]
    assistant_message.content = remaining
    logger.warning(
        "Recovered %d leaked qwen3-XML tool call(s) from streamed content (provider=custom, "
        "model=%s) — treating this response as a tool round.",
        len(calls), getattr(agent, "model", "?"),
    )
    return True
