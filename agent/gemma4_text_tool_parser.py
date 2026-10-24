"""Gemma 4 text-based tool-call extraction for stripped-token output.

When Featherless/vLLM serves Gemma 4 without ``--tool-call-parser gemma4``,
special tokens (``<|tool_call>``, ``<tool_call|>``, ``<|"|>``) are stripped and
tool calls arrive as plain text::

    call:get_weather{city:Paris}call:get_weather{city:London}

This module parses them into ``ToolCall`` objects that the conversation loop
dispatches normally.

Design (grounded in live Featherless output):
- Generation stops at the tool call, so real calls are the *trailing run* of
  the content: one or more ``call:name{...}`` back to back (parallel calls are
  glued with no separator) with nothing but whitespace after the last ``}``.
  Any ``call:`` followed by more text is prose, not a request.
- Calls inside Markdown examples (fenced blocks, inline code, blockquotes)
  are never dispatched, and an unterminated ``call:name{`` makes the whole
  message ineligible: everything after it is its (truncated) argument text.
- Brace matching tries progressively simpler quote models and takes the first
  that tiles the run exactly to the end of the content, so a mis-read quote
  surfaces as "no calls" rather than a truncated argument.
- Schema-aware arg splitting uses tool parameter names from ``agent.tools``
  to split only on ``,<known_param>:`` boundaries, so commas in values are safe.
- All values are strings — no type coercion (avoids mangling "007" → 7).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from agent.transports.types import ToolCall, build_tool_call

logger = logging.getLogger(__name__)

# Gemma 4 thinking tokens: <|channel>thought\n...<channel|>
_GEMMA4_THINK_OPEN = "<|channel>thought"
_GEMMA4_THINK_RE = re.compile(
    r"<\|channel>thought\s*(.*?)<channel\|>", re.DOTALL,
)

_GEMMA4_MODEL_SUBSTRINGS = ("gemma-4", "gemma4", "gemma_4")

# Not after a word character (``recall:``); after `}`, a backtick or whitespace is fine.
_CALL_START_RE = re.compile(r"(?<!\w)call:(\w+)\{")
# Inside one call's arguments this is the seam of two glued parallel calls.
_GLUED_CALL_RE = re.compile(r"\}\s*call:\w+\{")

# Python string prefixes; a single quote after any other word is an apostrophe.
_STRING_PREFIXES = frozenset({"", "b", "f", "r", "u", "br", "rb", "fr", "rf"})

# Quote models for brace matching, most to least specific:
#   smart: " always delimits, ' only when not an in-word apostrophe (don't)
#   all:   both quote characters delimit (shell: echo prefix'}')
#   none:  quotes are literal
_QUOTE_MODES = ("smart", "all", "none")

# Markdown regions where a ``call:`` is an illustration, never a real call.
# Fences close on a run at least as long as the opener; an unclosed fence runs
# to the end. Inline code pairs equal-length backtick runs and may wrap lines
# but not cross a blank line.
_EXAMPLE_SPAN_RE = re.compile(
    r"^[ \t]*(?P<bt>`{3,})[^`\n]*\n.*?(?:^[ \t]*(?P=bt)`*[ \t]*$|\Z)"
    r"|^[ \t]*(?P<td>~{3,})[^\n]*\n.*?(?:^[ \t]*(?P=td)~*[ \t]*$|\Z)"
    r"|(?<!`)(?P<code>`+)(?!`)(?:(?!\n[ \t]*\n).)*?(?<!`)(?P=code)(?!`)"
    r"|^[ \t]*>[^\n]*$",
    re.DOTALL | re.MULTILINE,
)


def is_gemma4_model(model: str | None) -> bool:
    return any(g in (model or "").lower() for g in _GEMMA4_MODEL_SUBSTRINGS)


class _Quotes:
    """O(1) lookup of the next unescaped quote, so long content stays linear."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._next: dict[str, list[int]] = {}

    def close(self, j: int) -> int | None:
        quote = self._text[j]
        if quote not in self._next:
            self._next[quote] = self._index(quote)
        k = self._next[quote][j]
        return k if k >= 0 else None

    def _index(self, quote: str) -> list[int]:
        text = self._text
        nxt = [-1] * len(text)
        nearest = -1
        for k in range(len(text) - 1, -1, -1):
            nxt[k] = nearest
            if text[k] == quote and not _escaped(text, k):
                nearest = k
        return nxt


def _escaped(text: str, k: int) -> bool:
    backslashes = 0
    while k - backslashes > 0 and text[k - backslashes - 1] == "\\":
        backslashes += 1
    return backslashes % 2 == 1


def _opens_string(text: str, j: int, mode: str) -> bool:
    if mode == "none":
        return False
    if mode == "all" or text[j] == '"':
        return True
    k = j
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_"):
        k -= 1
    return text[k:j].lower() in _STRING_PREFIXES


def _match_brace(text: str, brace_start: int, mode: str, quotes: _Quotes) -> int | None:
    """Index of the ``}`` closing the ``{`` at *brace_start*, or None if unbalanced.

    A quote with no closing partner (``'tis``) is a literal character.
    """
    depth = 0
    j = brace_start
    while j < len(text):
        ch = text[j]
        if ch in "\"'" and _opens_string(text, j, mode):
            end = quotes.close(j)
            if end is not None:
                j = end + 1
                continue
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return None


def _tile(text: str, start: int, mode: str, quotes: _Quotes) -> tuple[list[tuple[str, str, int]], bool]:
    """Parse ``call:name{...}`` back to back from *start* to the end of *text*.

    Returns ``([(name, args_body, start)], covered)``; *covered* is True only
    when the calls reach the end of *text* exactly (whitespace allowed between
    and after them). On failure the list holds the calls visited before it.
    """
    calls: list[tuple[str, str, int]] = []
    pos = start
    while pos < len(text):
        m = _CALL_START_RE.match(text, pos)
        close = _match_brace(text, m.end() - 1, mode, quotes) if m else None
        if close is None:
            return calls, False
        args = text[m.end() : close]
        if _GLUED_CALL_RE.search(args):
            return calls, False
        calls.append((m.group(1), args, pos))
        pos = close + 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return calls, True


def _is_example(text: str, pos: int, spans: list[tuple[int, int]]) -> bool:
    if any(s <= pos < e for s, e in spans):
        return True
    # An unpaired backtick right before the call opens an inline example that
    # never closed; a paired one closing a code span (`x`call:...) is fine.
    return pos > 0 and text[pos - 1] == "`" and not any(e == pos for _, e in spans)


def _find_trailing_calls(text: str) -> list[tuple[str, str, int]]:
    """The trailing run of real calls in *text* (already right-stripped), or []."""
    if not text.endswith("}"):
        return []
    spans = [m.span() for m in _EXAMPLE_SPAN_RE.finditer(text)]
    quotes = _Quotes(text)
    # A chain that failed from one call fails identically from any later call
    # in it, so each start is tiled at most once per quote mode (keeps it linear).
    dead: dict[str, set[int]] = {mode: set() for mode in _QUOTE_MODES}
    args_end = 0
    for m in _CALL_START_RE.finditer(text):
        start = m.start()
        if start < args_end or _is_example(text, start, spans):
            continue
        for mode in _QUOTE_MODES:
            if start in dead[mode]:
                continue
            calls, covered = _tile(text, start, mode, quotes)
            if covered:
                return calls
            dead[mode].update(c[2] for c in calls)
            dead[mode].add(start)
        close = next(
            (c for mode in _QUOTE_MODES if (c := _match_brace(text, m.end() - 1, mode, quotes)) is not None),
            None,
        )
        if close is None:
            # Unterminated call: everything after it is its argument text, so
            # nothing later (even a balanced call) can be a real request.
            return []
        # A call: inside another call's arguments is argument text, not a call.
        args_end = close
    return []


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


def extract_gemma4_thinking(content: str) -> tuple[str, str]:
    """Extract Gemma 4 ``<|channel>thought`` blocks from content.

    Returns ``(thinking_text, cleaned_content)``.  Empty *thinking_text*
    when no blocks are found.

    Handles two cases:
    1. Full tokens present: ``<|channel>thought\\n...\\n<channel|>``
    2. Stripped tokens (Featherless/vLLM): content starts with bare
       ``thought\\n`` — the ``<|channel>`` and ``<channel|>`` delimiters
       are gone, so only the prefix is cleaned (no reasoning/answer
       boundary exists to split on).
    """
    if not content:
        return "", ""

    # Case 1: full tokens present — extract thinking between delimiters.
    if _GEMMA4_THINK_OPEN in content:
        parts: list[str] = []
        cleaned = _GEMMA4_THINK_RE.sub(lambda m: (parts.append(m.group(1).strip()), "")[1], content)
        thinking = "\n\n".join(p for p in parts if p)
        return thinking, cleaned.strip()

    # Case 2: stripped tokens — bare "thought\n" prefix leaked through.
    if content.startswith("thought\n"):
        cleaned = content[len("thought\n"):]
        return "", cleaned
    return "", content


def extract_gemma4_tool_calls(
    content: str,
    agent_tools: list[dict[str, Any]] | None = None,
) -> tuple[list[ToolCall], str]:
    """Extract Gemma 4 text tool calls from stripped-token content.

    *agent_tools*: the agent's tool list (OpenAI format) for schema-aware
    splitting.

    Returns ``(tool_calls, cleaned_content)`` where *cleaned_content* is the
    text before the trailing call run.
    """
    if not content or "call:" not in content:
        return [], content or ""

    text = content.rstrip()
    calls = _find_trailing_calls(text)
    if not calls:
        return [], content

    tool_calls: list[ToolCall] = []
    for name, args_body, _start in calls:
        param_names = _get_tool_param_names(agent_tools, name)
        if param_names is not None:
            parsed_args = _schema_aware_split(args_body, param_names)
            if not parsed_args and args_body.strip():
                parsed_args = _naive_split(args_body)
        else:
            parsed_args = _naive_split(args_body)

        tool_calls.append(build_tool_call(
            id=f"gemma4_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=json.dumps(parsed_args),
        ))

    return tool_calls, text[: calls[0][2]].strip()
