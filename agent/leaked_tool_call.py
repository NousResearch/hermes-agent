"""Detect a tool call the model wrote as TEXT instead of a structured ``tool_calls`` entry.

Local chat-completions servers (llama.cpp, vLLM) whose tool parser misses the model's call
markup return ``finish_reason="stop"`` with no ``tool_calls`` and the intended call left in
``reasoning``/``content``. Pure, regex-based; it fires only when the markup names a
tool offered on the current request, so prose that merely mentions a tool and examples inside
markdown code never count.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Collection, Iterator, Optional

_NAME = r"[A-Za-z_][\w.\-]*"
_SNIPPET_CHARS = 200

_FENCED_CODE_RE = re.compile(r"(?:```|~~~).*?(?:```|~~~|\Z)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

_HERMES_OPEN_RE = re.compile(r"<tool_call>")
_HERMES_CLOSE = "</tool_call>"
_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"(' + _NAME + r')"')
_XML_FUNCTION_RE = re.compile(r"<function=(" + _NAME + r")>")
_GEMMA_OPEN_RE = re.compile(r"<\|tool_call>\s*call:(" + _NAME + r")")
_GEMMA_COMPLETE_RE = re.compile(r"\s*(\{.*?\})\s*<tool_call\|>", re.DOTALL)
_GEMMA_QUOTED_RE = re.compile(r'<\|"\|>.*?<\|"\|>', re.DOTALL)
_HARMONY_RE = re.compile(r"(?:^|(?<=[\s>|]))to=functions\.(" + _NAME + r")")
_HARMONY_ARGS_RE = re.compile(r"[^\n{]{0,80}?(?:<\|message\|>)?\s*(?=\{)")
_BARE_JSON_RE = re.compile(r'\{\s*"name"\s*:\s*"(' + _NAME + r')"\s*,\s*"(?:arguments|parameters)"\s*:')
# The TAIL of an XML call ending the reply when the reasoning starts mid-call (seen on hosted
# Nemotron 3.5): closers only, at the very end, with no opener anywhere in the text.
_XML_FRAGMENT_TAIL_RE = re.compile(r"(?:\s*</(?:parameter|function|tool_call)>)+\s*\Z")
_XML_OPENER_RE = re.compile(r"<(?:function|parameter|tool_call)\b")

_DECODER = json.JSONDecoder()


@dataclass(frozen=True)
class LeakedToolCall:
    """``fmt``: ``hermes_json`` | ``xml_function`` | ``gemma`` | ``harmony`` | ``bare_json`` |
    ``xml_fragment``. ``complete`` is True only when the markup is closed and its arguments
    parse. ``name`` is ``""`` for ``xml_fragment``: only the call's closing tags survived."""

    name: str
    fmt: str
    snippet: str
    complete: bool

    @property
    def label(self) -> str:
        """Human wording for logs, status lines and the nudge."""
        return f"`{self.name}` tool call" if self.name else "cut-off tool call"


def _blank(match: re.Match) -> str:
    # Same length, so offsets in the masked text index the original.
    return " " * len(match.group(0))


def _mask_code(text: str) -> str:
    return _INLINE_CODE_RE.sub(_blank, _FENCED_CODE_RE.sub(_blank, text))


def _decode_object(text: str, start: int) -> Optional[dict]:
    try:
        obj, _ = _DECODER.raw_decode(text, start)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _arguments_ok(obj: dict) -> bool:
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            return False
    return isinstance(args, dict)


def _hermes_json(text: str) -> Iterator[tuple[int, str, str, bool]]:
    for m in _HERMES_OPEN_RE.finditer(text):
        close = text.find(_HERMES_CLOSE, m.end())
        body = text[m.end():close if close >= 0 else len(text)]
        if "<function=" in body:
            continue  # Qwen3-coder XML inside <tool_call>: the XML scanner owns it.
        stripped = body.lstrip()
        if stripped.startswith("{"):
            obj = _decode_object(text, m.end() + len(body) - len(stripped))
            if obj is not None and isinstance(obj.get("name"), str):
                yield m.start(), obj["name"], "hermes_json", close >= 0 and _arguments_ok(obj)
                continue
        name = _JSON_NAME_RE.search(body)
        if name:
            yield m.start(), name.group(1), "hermes_json", False


def _xml_function(text: str) -> Iterator[tuple[int, str, str, bool]]:
    for m in _XML_FUNCTION_RE.finditer(text):
        close = text.find("</function>", m.end())
        nxt = _XML_FUNCTION_RE.search(text, m.end())
        if nxt and (close < 0 or nxt.start() < close):
            close = -1  # a second call opened before this one closed
        body = text[m.end():close if close >= 0 else len(text)]
        balanced = body.count("<parameter=") == body.count("</parameter>")
        yield m.start(), m.group(1), "xml_function", close >= 0 and balanced


def _gemma(text: str) -> Iterator[tuple[int, str, str, bool]]:
    for m in _GEMMA_OPEN_RE.finditer(text):
        done = _GEMMA_COMPLETE_RE.match(text, m.end())
        complete = False
        if done:
            args = _GEMMA_QUOTED_RE.sub('""', done.group(1))
            complete = args.count("{") == args.count("}") and "<|" not in args
        yield m.start(), m.group(1), "gemma", complete


def _harmony(text: str) -> Iterator[tuple[int, str, str, bool]]:
    for m in _HARMONY_RE.finditer(text):
        lead = _HARMONY_ARGS_RE.match(text, m.end())
        obj = _decode_object(text, lead.end()) if lead else None
        yield m.start(), m.group(1), "harmony", obj is not None


def _bare_json(text: str) -> Iterator[tuple[int, str, str, bool]]:
    for m in _BARE_JSON_RE.finditer(text):
        obj = _decode_object(text, m.start())
        yield m.start(), m.group(1), "bare_json", obj is not None and _arguments_ok(obj)


_SCANNERS: tuple[Callable[[str], Iterator[tuple[int, str, str, bool]]], ...] = (
    _hermes_json, _xml_function, _gemma, _harmony, _bare_json,
)


def _xml_fragment(text: str, masked: str) -> Optional[LeakedToolCall]:
    # No known tool name is required here: closers at the very end of a reply are unambiguous.
    tail = _XML_FRAGMENT_TAIL_RE.search(masked)
    if tail is None or _XML_OPENER_RE.search(masked):
        return None
    return LeakedToolCall(name="", fmt="xml_fragment", snippet=text.rstrip()[-_SNIPPET_CHARS:], complete=False)


def _known(name: str, tool_names: Collection[str]) -> Optional[str]:
    if name in tool_names:
        return name
    short = name.removeprefix("functions.")
    return short if short in tool_names else None


def find_leaked_tool_call(text: Any, tool_names: Collection[str]) -> Optional[LeakedToolCall]:
    """The earliest tool call in ``text`` written as markup naming one of ``tool_names``, or None."""
    if not isinstance(text, str) or not text or not tool_names:
        return None
    masked = _mask_code(text)
    # Every named match must name an offered tool: skip the scanners when none appears.
    named = any(name in text for name in tool_names)
    best: Optional[tuple[int, str, str, bool]] = None
    for scan in _SCANNERS if named else ():
        for start, raw_name, fmt, complete in scan(masked):
            name = _known(raw_name, tool_names)
            if name:  # matches arrive in order: this is the scanner's earliest
                if best is None or start < best[0]:
                    best = (start, name, fmt, complete)
                break
    if best is None:
        return _xml_fragment(text, masked)
    start, name, fmt, complete = best
    return LeakedToolCall(name=name, fmt=fmt, snippet=text[start:start + _SNIPPET_CHARS], complete=complete)


def leaked_tool_call_nudge(call: LeakedToolCall, where: str) -> str:
    """Corrective user turn: the call was NOT executed; emit it structurally or answer."""
    if not call.name:
        what = (f"ended with the closing tags of a tool call written as text in your {where}; "
                "the call was cut off")
    else:
        partial = "" if call.complete else " (the call text was also incomplete)"
        what = f"contained a {call.label} written as text in your {where}{partial}"
    return (
        f"[System: Your previous reply {what}. It was NOT executed: tools only run when you emit "
        "a structured tool call. If you still need this action, issue it now as a proper tool "
        "call, not as text. If no action is needed, give your final answer instead.]"
    )
