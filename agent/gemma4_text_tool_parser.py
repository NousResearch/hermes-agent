"""Gemma 4 text-based tool-call extraction for stripped-token output.

When Featherless/vLLM serves Gemma 4 without ``--tool-call-parser gemma4``,
special tokens (``<|tool_call>``, ``<tool_call|>``, ``<|"|>``) are stripped and
tool calls arrive as plain text::

    call:get_weather{city:Paris}call:get_weather{city:London}

This module parses them into ``ToolCall`` objects that the conversation loop
dispatches normally. The stripped form is inherently ambiguous (``<|"|>`` was
the only string delimiter), so every rule below fails closed: when a reading
is uncertain, nothing is dispatched.

Design (grounded in live Featherless output):
- Generation stops at the tool call, so real calls are the *trailing run* of
  the content: one or more ``call:name{...}`` back to back (parallel calls are
  glued with no separator) with nothing but whitespace after the last ``}``.
  Any ``call:`` followed by more text is prose, not a request.
- Every call name must be one of the agent's tools, exactly.
- Calls inside Markdown examples (fenced or indented code, blockquotes, or
  right after an unpaired backtick) never dispatch, nor does a call glued to
  such an example. An unterminated ``call:name{`` makes the whole message
  ineligible: everything after it is its argument text.
- Arguments split on ``,<param>:`` only at bracket depth 0 and outside quotes,
  each parameter at most once; values are kept verbatim apart from one
  ``key: value`` separator space and a single wrapping quote pair. Array and
  object parameters are parsed from Gemma's bare ``[a,b]`` / ``{k:v}`` syntax;
  scalars stay strings for the registry's schema coercion.

Known limits (fail closed or out of scope): a genuine argument that contains a
literal ``}call:name{`` is not dispatched; ``\\r``-only line breaks, HTML
``<pre>`` and look-alike backticks are not treated as example markup.
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
# Routes that serve Gemma 4 with special tokens stripped and no tool-call
# parser; providers with native parsing return structured tool_calls instead.
_STRIPPED_TOKEN_PROVIDERS = frozenset({"huggingface", "custom"})

# Not after a word character (``recall:``); after `}`, a backtick or whitespace is fine.
_CALL_START_RE = re.compile(r"(?<!\w)call:(\w+)\{")
# Inside one call's arguments this is the seam of two glued parallel calls.
_GLUED_CALL_RE = re.compile(r"\}\s*call:\w+\{")

# Python string prefixes; a single quote after any other word is an apostrophe.
_STRING_PREFIXES = frozenset({"", "b", "f", "r", "u", "br", "rb", "fr", "rf"})

# Quote models for brace matching, most to least specific:
#   smart: " always delimits, ' only when not an in-word apostrophe (don't)
#   all:   both quote characters delimit (shell: echo prefix'}')
_QUOTE_MODES = ("smart", "all")

# Markdown regions that make a ``call:`` an illustration. Fences close on a run
# at least as long as the opener; an unclosed fence runs to the end.
_EXAMPLE_SPAN_RE = re.compile(
    r"^[^\S\n]*(?P<bt>`{3,})[^`\n]*\n.*?(?:^[^\S\n]*(?P=bt)`*[^\S\n]*$|\Z)"
    r"|^[^\S\n]*(?P<td>~{3,})[^\n]*\n.*?(?:^[^\S\n]*(?P=td)~*[^\S\n]*$|\Z)"
    r"|^[^\S\n]*>[^\n]*$",
    re.DOTALL | re.MULTILINE,
)
# Inline code: equal-length backtick runs, may wrap lines but not cross a blank line.
_INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(?:(?!\n[^\S\n]*\n).)*?(?<!`)\1(?!`)", re.DOTALL)

_ARG_KEY_RE = re.compile(r"\s*(\w+)\s*:")
_WRAPPING_QUOTES_RE = re.compile(r"""\s*(["'])(.*)\1\s*""", re.DOTALL)
_OPENERS, _CLOSERS = "{[(", "}])"


def is_gemma4_model(model: str | None) -> bool:
    return any(g in (model or "").lower() for g in _GEMMA4_MODEL_SUBSTRINGS)


def strips_special_tokens(provider: str | None) -> bool:
    from hermes_cli.providers import normalize_provider

    return normalize_provider((provider or "").split(":", 1)[0]) in _STRIPPED_TOKEN_PROVIDERS


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
    if mode == "all" or text[j] == '"':
        return True
    k = j
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_"):
        k -= 1
    return text[k:j].lower() in _STRING_PREFIXES


def _skip_literal(text: str, j: int, mode: str, quotes: _Quotes) -> int:
    """Index after the escape or quoted string starting at *j*, or *j* if neither."""
    if text[j] == "\\":
        return j + 2
    if text[j] in "\"'" and _opens_string(text, j, mode):
        end = quotes.close(j)
        if end is not None:
            return end + 1
    return j


def _match_brace(text: str, brace_start: int, mode: str, quotes: _Quotes) -> int | None:
    """Index of the ``}`` closing the ``{`` at *brace_start*, or None if unbalanced.

    Escaped characters and quoted strings don't count; a quote with no closing
    partner (``'tis``) is a literal character.
    """
    depth = 0
    j = brace_start
    while j < len(text):
        nxt = _skip_literal(text, j, mode, quotes)
        if nxt != j:
            j = nxt
            continue
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
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


def _is_example(text: str, pos: int, spans: list[tuple[int, int]], code_ends: set[int]) -> bool:
    if any(s <= pos < e for s, e in spans):
        return True
    line_prefix = text[text.rfind("\n", 0, pos) + 1 : pos]
    if line_prefix and not line_prefix.strip() and len(line_prefix.expandtabs(4)) >= 4:
        return True  # indented code block
    # An unpaired backtick just before the call (spaces/emphasis between) opens
    # an inline example that never closed; a backtick closing a paired span
    # (`x`call:...) is the live thinking shape and stays a real call.
    k = pos
    while k > 0 and text[k - 1] in " \t*_":
        k -= 1
    return k > 0 and text[k - 1] == "`" and k not in code_ends


def _find_trailing_calls(text: str) -> list[tuple[str, str, int]]:
    """The trailing run of real calls in *text* (already right-stripped), or []."""
    if not text.endswith("}"):
        return []
    spans = [m.span() for m in _EXAMPLE_SPAN_RE.finditer(text)]
    code_ends = {m.end() for m in _INLINE_CODE_RE.finditer(text)}
    quotes = _Quotes(text)
    # A chain that failed from one call fails identically from any later call
    # in it, so each start is tiled at most once per quote mode (keeps it linear).
    dead: dict[str, set[int]] = {mode: set() for mode in _QUOTE_MODES}
    args_end = 0
    example_end = -1
    for m in _CALL_START_RE.finditer(text):
        start, brace = m.start(), m.end() - 1
        if start < args_end:
            continue  # inside another call's arguments: argument text
        closes = [_match_brace(text, brace, mode, quotes) for mode in _QUOTE_MODES]
        glued_to_example = example_end >= 0 and not text[example_end:start].strip()
        if glued_to_example or _is_example(text, start, spans, code_ends):
            if None in closes:
                return []
            args_end = example_end = max(closes) + 1
            continue
        for mode in _QUOTE_MODES:
            if start in dead[mode]:
                continue
            calls, covered = _tile(text, start, mode, quotes)
            if covered:
                return calls
            dead[mode].update(c[2] for c in calls)
            dead[mode].add(start)
        if all(c is None for c in closes):
            # Unbalanced braces in a final call that ends the message: the
            # closing brace is the end anchor, unless another call: inside
            # could be the real one (an unterminated outer call wrapping it).
            if _CALL_START_RE.search(text, brace + 1) is None:
                return [(m.group(1), text[brace + 1 : -1], start)]
            return []
        if None in closes:
            return []  # quote models disagree on whether it ends: ambiguous
        args_end = max(closes) + 1
    return []


def _split_top_level(body: str, at_boundary) -> list[tuple[int, int]]:
    """Segment *body* at depth-0, outside-quote positions where *at_boundary(i)* is truthy.

    Returns ``[(start, end)]`` ranges; *at_boundary* receives the index of a ``,``.
    """
    quotes = _Quotes(body)
    ranges: list[tuple[int, int]] = []
    seg_start = depth = j = 0
    while j < len(body):
        nxt = _skip_literal(body, j, "smart", quotes)
        if nxt != j:
            j = nxt
            continue
        ch = body[j]
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0 and at_boundary(j):
            ranges.append((seg_start, j))
            seg_start = j + 1
        j += 1
    ranges.append((seg_start, len(body)))
    return ranges


def _unwrap(value: str) -> str:
    """Drop one ``key: value`` separator space and a single wrapping quote pair."""
    if value[:1] == " " and value[1:2] not in ("", " ", "\t", "\n"):
        value = value[1:]
    m = _WRAPPING_QUOTES_RE.fullmatch(value)
    if m and not re.search(rf"(?<!\\){m.group(1)}", m.group(2)):
        return m.group(2)
    return value


def _parse_container(value: str) -> Any:
    """Gemma's bare ``[a,b]`` / ``{k:v}`` (strings unquoted once tokens are stripped)."""
    text = value.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    if len(text) < 2 or (text[0], text[-1]) not in (("[", "]"), ("{", "}")):
        return value
    inner = text[1:-1]
    parts = [inner[s:e] for s, e in _split_top_level(inner, lambda _i: True)]
    if text[0] == "[":
        return [] if not inner.strip() else [_parse_container(p) for p in parts]
    result: dict[str, Any] = {}
    for part in parts:
        key, sep, val = part.partition(":")
        if not sep or not key.strip():
            return value
        result[_unwrap(key.strip())] = _parse_container(val)
    return result


def _typed_value(raw: str, prop: dict[str, Any]) -> Any:
    value = _unwrap(raw)
    types = prop.get("type")
    types = types if isinstance(types, list) else [types]
    if "array" in types or "object" in types:
        return _parse_container(value)
    if {"integer", "number", "boolean"} & set(types):
        return value.strip()
    return value


def _schema_split(body: str, props: dict[str, Any]) -> dict[str, Any] | None:
    """Split on ``,<param>:`` at depth 0 outside quotes, each known param once.

    None when the body doesn't start with a known parameter (caller falls back).
    """
    first = _ARG_KEY_RE.match(body)
    if not first or first.group(1) not in props:
        if len(props) == 1 and not first:
            (key,) = props
            return {key: _typed_value(body, props[key])}
        return None
    seen = {first.group(1)}

    def at_boundary(i: int) -> bool:
        m = _ARG_KEY_RE.match(body, i + 1)
        if m and m.group(1) in props and m.group(1) not in seen:
            seen.add(m.group(1))
            return True
        return False

    result: dict[str, Any] = {}
    for s, e in _split_top_level(body, at_boundary):
        m = _ARG_KEY_RE.match(body, s)
        result[m.group(1)] = _typed_value(body[m.end() : e], props[m.group(1)])
    return result


def _naive_split(body: str) -> dict[str, Any]:
    """No schema: ``key:value`` pairs on depth-0 commas; a segment without a key
    belongs to the previous value."""
    result: dict[str, Any] = {}
    key = None
    for s, e in _split_top_level(body, lambda _i: True):
        m = _ARG_KEY_RE.match(body, s)
        if m and m.end() <= e:
            key = m.group(1)
            result[key] = _unwrap(body[m.end() : e])
        elif key is not None:
            result[key] += "," + body[s:e]
    return result


def _tool_properties(agent_tools: list[dict[str, Any]] | None, tool_name: str) -> dict[str, Any] | None:
    for tool in agent_tools or ():
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        if fn.get("name") == tool_name:
            return (fn.get("parameters") or {}).get("properties") or None
    return None


def _tool_names(agent_tools: list[dict[str, Any]] | None) -> set[str] | None:
    if not agent_tools:
        return None
    return {
        (tool.get("function") or {}).get("name")
        for tool in agent_tools if isinstance(tool, dict)
    }


def extract_gemma4_thinking(content: str) -> tuple[str, str]:
    """Extract Gemma 4 ``<|channel>thought`` blocks from content.

    Returns ``(thinking_text, cleaned_content)``.  Empty *thinking_text*
    when no blocks are found.

    Handles two cases:
    1. Full tokens present: ``<|channel>thought\\n...\\n<channel|>``; an
       unterminated block runs to the end (it is still reasoning).
    2. Stripped tokens (Featherless/vLLM): content starts with bare
       ``thought\\n`` — the ``<|channel>`` and ``<channel|>`` delimiters
       are gone, so only the prefix is cleaned (no reasoning/answer
       boundary exists to split on).
    """
    if not content:
        return "", ""

    if _GEMMA4_THINK_OPEN in content:
        parts: list[str] = []
        cleaned = _GEMMA4_THINK_RE.sub(lambda m: (parts.append(m.group(1).strip()), "")[1], content)
        open_idx = cleaned.find(_GEMMA4_THINK_OPEN)
        if open_idx != -1:
            parts.append(cleaned[open_idx + len(_GEMMA4_THINK_OPEN) :].strip())
            cleaned = cleaned[:open_idx]
        return "\n\n".join(p for p in parts if p), cleaned.strip()

    if content.startswith("thought\n"):
        return "", content[len("thought\n"):]
    return "", content


def extract_gemma4_tool_calls(
    content: str,
    agent_tools: list[dict[str, Any]] | None = None,
) -> tuple[list[ToolCall], str]:
    """Extract Gemma 4 text tool calls from stripped-token content.

    *agent_tools*: the agent's tool list (OpenAI format); when given, every
    call must name one of them and arguments split on its schema.

    Returns ``(tool_calls, cleaned_content)`` where *cleaned_content* is the
    text before the trailing call run.
    """
    if not content or "call:" not in content:
        return [], content or ""

    text = re.sub(r"\r\n?", "\n", content).rstrip()
    calls = _find_trailing_calls(text)
    names = _tool_names(agent_tools)
    if not calls or (names is not None and any(name not in names for name, _, _ in calls)):
        return [], content

    tool_calls: list[ToolCall] = []
    for name, args_body, _start in calls:
        props = _tool_properties(agent_tools, name)
        parsed = _schema_split(args_body, props) if props else None
        tool_calls.append(build_tool_call(
            id=f"gemma4_{uuid.uuid4().hex[:8]}",
            name=name,
            arguments=json.dumps(parsed if parsed is not None else _naive_split(args_body)),
        ))

    return tool_calls, text[: calls[0][2]].strip()


def _reconcile_streamed_text(agent: Any, raw: str, visible: str) -> None:
    """Point the streamed-text record at the visible text after a rewrite, so the
    gateway's exact-match interim dedupe doesn't post the prose a second time."""
    streamed = getattr(agent, "_current_streamed_assistant_text", None)
    if isinstance(streamed, str) and streamed and streamed.split() == raw.split():
        agent._current_streamed_assistant_text = visible


def apply_gemma4_text_protocol(agent: Any, message: Any) -> None:
    """Recover reasoning and tool calls a token-stripping route left in ``message.content``.

    Mutates *message* (content, reasoning, tool_calls, finish_reason). A
    length-truncated reply may end mid-call, so it is never parsed for calls.
    On a tool turn whose stripped thought block opened the reply, the text
    before the call becomes reasoning rather than visible content.
    """
    raw = message.content
    if not raw or not is_gemma4_model(agent.model) or not strips_special_tokens(getattr(agent, "provider", None)):
        return
    thinking, content = extract_gemma4_thinking(raw)
    if thinking and not getattr(message, "reasoning", None):
        message.reasoning = thinking
    if not getattr(message, "tool_calls", None) and message.finish_reason != "length":
        calls, cleaned = extract_gemma4_tool_calls(content, getattr(agent, "tools", None))
        if calls:
            message.tool_calls = calls
            content = cleaned
            # Gemma closes its thought channel right before calling a tool; with the
            # <channel|> stripped the boundary is gone, but on a tool turn the whole
            # preamble after a bare "thought" prefix is that thought block.
            if raw.startswith("thought\n") and cleaned and not getattr(message, "reasoning", None):
                message.reasoning, content = cleaned, ""
            if message.finish_reason == "stop":
                message.finish_reason = "tool_calls"
            logger.info("Gemma 4 text extraction: %d tool call(s) from content", len(calls))
            _reconcile_streamed_text(agent, raw, content)
    if content != raw:
        message.content = content or None
