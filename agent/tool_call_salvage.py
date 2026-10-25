"""Salvage text-form tool calls from reasoning content (IQ3 quantized-model artifact).

Some quantized local models (qwen3.8-flash-next IQ3_XXS on Strata) emit the tool-call
block as plain text inside ``reasoning_content`` — Claude-style
``<function=name>`` / ``<parameter=key>value</parameter>`` markup — while the structured
``tool_calls`` array stays empty and ``finish_reason`` is ``stop``. Without salvage the
turn ends with the planning monologue promoted as the final answer (agent.log:
"Reasoning-only clean stop ... returning the reasoning as the final response") and the
user sees the answer vanish mid-task.

This module parses that markup back into real OpenAI-shaped tool calls so the normal
tool round runs. It is deliberately strict: only calls whose name is in the agent's
valid tool set are salvaged, and a block with no parseable parameters is skipped.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from agent.acp_openai_bridge import build_openai_tool_call

logger = logging.getLogger("agent.conversation_loop")

# <function=NAME> or <function name="NAME"> ... </function>
_FUNCTION_BLOCK_RE = re.compile(
    r"<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>"
    r"(?P<body>.*?)</function>",
    re.DOTALL,
)
_FUNCTION_ATTR_RE = re.compile(
    r'<function\s+name\s*=\s*"([A-Za-z0-9_.:-]+)"\s*>(?P<body>.*?)</function>',
    re.DOTALL,
)
# <parameter=KEY>\nvalue\n</parameter>  (value may span lines)
_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z0-9_.:-]+)\s*>",
)

def _parse_function_body(body: str) -> dict[str, str]:
    """Extract ``{param: value}`` from a function block body. Values keep their raw
    text (multi-line code blocks included); the closing tag ends each value."""
    params: dict[str, str] = {}
    pos = 0
    while True:
        m = _PARAMETER_RE.search(body, pos)
        if m is None:
            break
        key = m.group(1)
        end = body.find("</parameter>", m.end())
        if end == -1:
            break  # malformed tail — stop parsing this block
        value = body[m.end():end]
        # Strip exactly one leading and one trailing newline (the tag convention),
        # preserving indentation inside code values.
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        params[key] = value
        pos = end + len("</parameter>")
    return params

def salvage_tool_calls_from_text(
    text: str, valid_names: Iterable[str] | None,
) -> list[Any]:
    """Parse Claude-style ``<function=...>`` markup out of ``text`` into OpenAI-shaped
    tool calls. Only names present in ``valid_names`` are accepted (None = accept any).
    Returns [] when the text carries no salvageable call."""
    if not isinstance(text, str) or "<function" not in text:
        return []
    allowed = {str(n).strip() for n in valid_names} if valid_names is not None else None
    calls: list[Any] = []
    spans: list[tuple[int, int]] = []
    for pattern in (_FUNCTION_BLOCK_RE, _FUNCTION_ATTR_RE):
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            if allowed is not None and name not in allowed:
                continue
            params = _parse_function_body(m.group("body"))
            if not params:
                continue
            arguments = json.dumps(params, ensure_ascii=False)
            calls.append(build_openai_tool_call(
                call_id=f"salvaged_call_{len(calls) + 1}", name=name, arguments=arguments,
            ))
            spans.append((m.start(), m.end()))
        if calls:
            break
    if calls:
        logger.warning(
            "Salvaged %d text-form tool call(s) from reasoning content (names=%s)",
            len(calls), [c.function.name for c in calls],
        )
    return calls
