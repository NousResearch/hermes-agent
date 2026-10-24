"""Gemma 4 text-based tool-call extraction for stripped-token output.

When Featherless/vLLM serves Gemma 4 without ``--tool-call-parser gemma4``,
special tokens (``<|tool_call>``, ``<tool_call|>``, ``<|"|>``) are stripped and
tool calls arrive as plain text::

    call:get_weather{city:Paris}

This module parses them into ``ToolCall`` objects that the conversation loop
dispatches normally.

Design:
- Schema-aware arg splitting uses tool parameter names from ``agent.tools``
  to split only on ``,<known_param>:`` boundaries, so commas in values are safe.
- All values are strings — no type coercion (avoids mangling "007" → 7).
- Prose guard: skips extraction when ``call:`` blocks are a minority of the
  content (prevents executing quoted examples).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from agent.transports.types import ToolCall, build_tool_call

logger = logging.getLogger(__name__)

_GEMMA4_MODEL_SUBSTRINGS = ("gemma-4", "gemma4", "gemma_4")


def is_gemma4_model(model: str | None) -> bool:
    return any(g in (model or "").lower() for g in _GEMMA4_MODEL_SUBSTRINGS)


def _find_balanced_calls(text: str) -> list[tuple[str, str, int, int]]:
    """Find ``call:name{...}`` with balanced braces.

    Returns ``[(name, args_body, start, end)]`` where *args_body* is the
    content **inside** the outer braces.

    Only matches ``call:`` at the start of text or preceded by whitespace.
    """
    results: list[tuple[str, str, int, int]] = []
    i = 0
    while i < len(text):
        m = re.search(r"(?:^|(?<=\s))call:(\w+)\{", text[i:])
        if not m:
            break
        name = m.group(1)
        brace_start = i + m.end() - 1
        depth = 0
        j = brace_start
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    args_body = text[brace_start + 1 : j]
                    results.append((name, args_body, i + m.start(), j + 1))
                    break
            j += 1
        i = j + 1 if j < len(text) else len(text)
    return results


def _schema_aware_split(args_body: str, param_names: list[str]) -> dict[str, str]:
    """Split on ``,<param>:`` boundaries using known parameter names.

    Single-param tools have no ambiguity: the entire body is the value.
    The last parameter's value runs to the end of the string.
    """
    if not param_names or not args_body.strip():
        return {}

    if len(param_names) == 1:
        key = param_names[0]
        prefix = f"{key}:"
        if args_body.startswith(prefix):
            return {key: args_body[len(prefix) :]}
        if ":" in args_body:
            before_colon = args_body.split(":", 1)[0].strip()
            if re.fullmatch(r"\w+", before_colon):
                return {}
        return {key: args_body}

    result: dict[str, str] = {}
    param_pattern = "|".join(
        re.escape(p) for p in sorted(param_names, key=len, reverse=True)
    )
    segment_re = re.compile(
        r"(?:^|,\s*)(" + param_pattern + r")\s*:(.*?)(?=,\s*(?:" + param_pattern + r")\s*:|$)",
        re.DOTALL,
    )
    for m in segment_re.finditer(args_body):
        result[m.group(1).strip()] = m.group(2).strip()
    return result


def _naive_split(args_body: str) -> dict[str, str]:
    """Fallback: split on commas not inside braces/brackets."""
    result: dict[str, str] = {}
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in args_body:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current))

    for part in parts:
        part = part.strip()
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        result[key.strip()] = value.strip()
    return result


def _get_tool_param_names(
    agent_tools: list[dict[str, Any]] | None, tool_name: str
) -> list[str] | None:
    """Extract parameter names from the agent's tool schema."""
    if not agent_tools:
        return None
    for tool in agent_tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        if fn.get("name") == tool_name:
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            return list(props.keys()) if props else None
    return None


def extract_gemma4_tool_calls(
    content: str,
    agent_tools: list[dict[str, Any]] | None = None,
) -> tuple[list[ToolCall], str]:
    """Extract Gemma 4 text tool calls from stripped-token content.

    *agent_tools*: the agent's tool list (OpenAI format) for schema-aware
    splitting.

    Returns ``(tool_calls, cleaned_content)``.
    """
    if not content or "call:" not in content:
        return [], content or ""

    calls = _find_balanced_calls(content)
    if not calls:
        return [], content

    # Prose guard: only extract when call: blocks dominate the content.
    total_call_chars = sum(end - start for _, _, start, end in calls)
    non_call_text = content
    for _, _, start, end in reversed(calls):
        non_call_text = non_call_text[:start] + non_call_text[end:]
    non_call_text = non_call_text.strip()

    if non_call_text and len(non_call_text) > total_call_chars:
        logger.debug(
            "Skipping Gemma 4 extraction: content is mostly prose "
            "(%d chars non-call vs %d call)",
            len(non_call_text),
            total_call_chars,
        )
        return [], content

    tool_calls: list[ToolCall] = []
    for name, args_body, _start, _end in calls:
        param_names = _get_tool_param_names(agent_tools, name)
        if param_names is not None:
            parsed_args = _schema_aware_split(args_body, param_names)
            if not parsed_args and args_body.strip():
                parsed_args = _naive_split(args_body)
        else:
            parsed_args = _naive_split(args_body)

        tc = build_tool_call(
            id=f"gemma4_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=json.dumps(parsed_args),
        )
        tool_calls.append(tc)

    cleaned = content
    for _, _, start, end in reversed(calls):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = cleaned.strip()

    return tool_calls, cleaned
