"""Detect a tool call the model wrote as TEXT instead of a structured ``tool_calls`` entry.

Local chat-completions servers (llama.cpp, vLLM) whose tool parser misses the model's call
markup return ``finish_reason="stop"`` with no ``tool_calls`` and the intended call left in
``reasoning``/``content``. Pure, regex-based; it fires only when the markup names a
tool offered on the current request, so prose that merely mentions a tool and examples inside
markdown code never count.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
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
# Nemotron 3.5): closers only, at the very end, with no opener anywhere in the text. A suffix test,
# not a regex search: ``\s*`` backtracks quadratically over the blank runs ``_mask_code`` leaves.
_XML_FRAGMENT_CLOSERS = ("</parameter>", "</function>", "</tool_call>")
_XML_OPENER_RE = re.compile(r"<(?:function|parameter|tool_call)\b")
# Closing tokens that may follow a call and still belong to it.
_XML_WRAPPER_CLOSE_RE = re.compile(r"(?:\s*</tool_call>)?")
_HARMONY_CALL_END_RE = re.compile(r"(?:\s*<\|call\|>)?")
# Call markup a reply may stop partway through, after its last complete call: the cut-off start of
# one more call (or a stray closer) is still the tail, not prose.
_CALL_TOKENS = ("<tool_call>", "<|tool_call>", "<function=", "<parameter=", "to=functions.",
                "</tool_call>", "</function>", "</parameter>", "<tool_call|>", "<|call|>")
_WRAPPER_OPENER_RE = re.compile(r"<\|?tool_call>\s*")
_NAMED_OPENER_RE = re.compile(r"(?:<function=|<parameter=|to=functions\.)[\w.\-]*")
_GEMMA_CALL_PREFIX_RE = re.compile(r"c(?:a(?:l(?:l(?::[\w.\-]*)?)?)?)?")
# Where a named call starts, in any format: one linear pass finds the last of them. Groups 1-3 sit
# at the call's own start (XML, gemma, harmony); group 4 is a JSON "name" key inside the call.
_CALL_MARKER_RE = re.compile(
    r"<function=(" + _NAME + r")>|<\|tool_call>\s*call:(" + _NAME + r")|to=functions\.(" + _NAME + r")"
    r'|"name"\s*:\s*"(' + _NAME + r')"'
)
_JSON_MARKER = 4
# Known-tool markers tried from the end when the last one is not itself a call (a "name" key inside
# another call's arguments). A small constant: each try rescans only the text after its marker.
_TRAILING_MARKERS = 4

_DECODER = json.JSONDecoder()

# (start, name, fmt, complete, end): ``end`` is just past the call, or the text's end if unclosed.
_Match = tuple[int, str, str, bool, int]


@dataclass(frozen=True)
class LeakedToolCall:
    """``fmt``: ``hermes_json`` | ``xml_function`` | ``gemma`` | ``harmony`` | ``bare_json`` |
    ``xml_fragment``. ``complete`` is True only when the markup is closed and its arguments
    parse. ``name`` is ``""`` for ``xml_fragment``: only the call's closing tags survived.
    ``end`` indexes the scanned text just past the call."""

    name: str
    fmt: str
    snippet: str
    complete: bool
    end: int

    @property
    def label(self) -> str:
        """Human wording for logs, status lines and the nudge."""
        return f"`{self.name}` tool call" if self.name else "cut-off tool call"


def _blank(match: re.Match) -> str:
    # Same length, so offsets in the masked text index the original.
    return " " * len(match.group(0))


def _mask_code(text: str) -> str:
    return _INLINE_CODE_RE.sub(_blank, _FENCED_CODE_RE.sub(_blank, text))


def _decode_object(text: str, start: int) -> tuple[Optional[dict], int]:
    """The JSON object at ``start`` and the offset past it, or ``(None, len(text))``."""
    try:
        obj, end = _DECODER.raw_decode(text, start)
    except (ValueError, RecursionError):  # deeply nested JSON exhausts the C decoder
        return None, len(text)
    return (obj, end) if isinstance(obj, dict) else (None, len(text))


def _arguments_ok(obj: dict) -> bool:
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except (ValueError, RecursionError):
            return False
    return isinstance(args, dict)


# Scanners take ``known`` (the offered tool a raw name resolves to, or None) and check it before any
# lookahead past the opener, so text full of calls to tools that are not offered stays linear.
_Known = Callable[[str], Optional[str]]


def _hermes_name(head: str) -> Optional[str]:
    # Decode within ``head`` only: a failed decode costs O(offset) (JSONDecodeError counts lines).
    stripped = head.lstrip()
    if stripped.startswith("{"):
        obj, _ = _decode_object(head, len(head) - len(stripped))
        if obj is not None and isinstance(obj.get("name"), str):
            return obj["name"]
    name = _JSON_NAME_RE.search(head)
    return name.group(1) if name else None


def _hermes_json(text: str, known: _Known) -> Iterator[_Match]:
    next_close = -2  # memo: the first closer at or after the last search start (-1: none left)
    for m in _HERMES_OPEN_RE.finditer(text):
        if next_close == -2 or 0 <= next_close < m.end():
            next_close = text.find(_HERMES_CLOSE, m.end())
        close = next_close
        # Name the call from its text up to the next opener first: an unclosed call's body runs on
        # to a later closer, and reading it for every call to an unoffered tool is quadratic.
        nxt = _HERMES_OPEN_RE.search(text, m.end())
        stop = min(p for p in (close, nxt.start() if nxt else -1, len(text)) if p >= 0)
        head = text[m.end():stop]
        if "<function=" in head:
            continue  # Qwen3-coder XML inside <tool_call>: the XML scanner owns it.
        if not known(_hermes_name(head) or ""):
            continue
        body = text[m.end():close if close >= 0 else len(text)]
        end = close + len(_HERMES_CLOSE) if close >= 0 else len(text)
        if "<function=" in body:
            continue
        stripped = body.lstrip()
        if stripped.startswith("{"):
            obj, obj_end = _decode_object(text, m.end() + len(body) - len(stripped))
            if obj is not None and isinstance(obj.get("name"), str):
                if known(obj["name"]):
                    yield (m.start(), known(obj["name"]), "hermes_json", close >= 0 and _arguments_ok(obj),
                           end if close >= 0 else obj_end)
                continue
        name = _JSON_NAME_RE.search(body)
        if name and known(name.group(1)):
            yield m.start(), known(name.group(1)), "hermes_json", False, end


def _xml_function(text: str, known: _Known) -> Iterator[_Match]:
    for m in _XML_FUNCTION_RE.finditer(text):
        name = known(m.group(1))
        if not name:
            continue
        nxt = _XML_FUNCTION_RE.search(text, m.end())
        # A closer after the next opener doesn't close this call: bound the search there (linear).
        close = text.find("</function>", m.end(), nxt.start() if nxt else len(text))
        body = text[m.end():close if close >= 0 else len(text)]
        balanced = body.count("<parameter=") == body.count("</parameter>")
        if close >= 0:
            end = _XML_WRAPPER_CLOSE_RE.match(text, close + len("</function>")).end()
        else:
            end = nxt.start() if nxt else len(text)
        yield m.start(), name, "xml_function", close >= 0 and balanced, end


def _gemma(text: str, known: _Known) -> Iterator[_Match]:
    for m in _GEMMA_OPEN_RE.finditer(text):
        name = known(m.group(1))
        if not name:
            continue
        done = _GEMMA_COMPLETE_RE.match(text, m.end())
        complete = False
        if done:
            args = _GEMMA_QUOTED_RE.sub('""', done.group(1))
            complete = args.count("{") == args.count("}") and "<|" not in args
        yield m.start(), name, "gemma", complete, done.end() if done else len(text)


def _harmony(text: str, known: _Known) -> Iterator[_Match]:
    for m in _HARMONY_RE.finditer(text):
        name = known(m.group(1))
        if not name:
            continue
        lead = _HARMONY_ARGS_RE.match(text, m.end())
        obj, end = _decode_object(text, lead.end()) if lead else (None, len(text))
        if obj is not None:
            end = _HARMONY_CALL_END_RE.match(text, end).end()
        yield m.start(), name, "harmony", obj is not None, end


def _bare_json(text: str, known: _Known) -> Iterator[_Match]:
    for m in _BARE_JSON_RE.finditer(text):
        name = known(m.group(1))
        if not name:
            continue
        obj, end = _decode_object(text, m.start())
        yield m.start(), name, "bare_json", obj is not None and _arguments_ok(obj), end


_SCANNERS: tuple[Callable[[str, _Known], Iterator[_Match]], ...] = (
    _hermes_json, _xml_function, _gemma, _harmony, _bare_json,
)


def _xml_fragment(text: str, masked: str) -> Optional[LeakedToolCall]:
    # No known tool name is required here: closers at the very end of a reply are unambiguous.
    # Cut at the reply's own last character: a trailing code block masks to blanks, not to its end.
    if not masked[:len(text.rstrip())].endswith(_XML_FRAGMENT_CLOSERS) or _XML_OPENER_RE.search(masked):
        return None
    return LeakedToolCall(name="", fmt="xml_fragment", snippet=text.rstrip()[-_SNIPPET_CHARS:], complete=False,
                          end=len(text))


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
    best: Optional[_Match] = None

    def known(raw_name: str) -> Optional[str]:
        return _known(raw_name, tool_names)

    for scan in _SCANNERS if named else ():
        # Scanners yield offered tools only, in order: the first is the scanner's earliest.
        match = next(scan(masked, known), None)
        if match is not None and (best is None or match[0] < best[0]):
            best = match
    if best is None:
        return _xml_fragment(text, masked)
    start, name, fmt, complete, end = best
    return LeakedToolCall(name=name, fmt=fmt, snippet=text[start:start + _SNIPPET_CHARS], complete=complete,
                          end=end)


def find_trailing_leaked_tool_call(text: Any, tool_names: Collection[str]) -> Optional[LeakedToolCall]:
    """The leaked call ``text`` ends on (only whitespace after it), or None. Calls drafted mid-text
    and then set aside are stepped past; an unclosed call runs to the end, so it always counts. The
    cut-off start of one more call after the last one (``<tool``, ``<function=na``) also counts as
    the tail. Cost: one pass to find the last call's start, then single-call detection from there."""
    if not isinstance(text, str) or not text or not tool_names:
        return None
    masked = _mask_code(text)
    if not any(name in text for name in tool_names):
        return _xml_fragment(text, masked)  # a fragment ends the text by construction
    markers: deque[re.Match] = deque(maxlen=_TRAILING_MARKERS)
    for m in _CALL_MARKER_RE.finditer(masked):
        if _known(m.group(m.lastindex), tool_names):
            markers.append(m)
    tried: set[int] = set()
    for m in reversed(markers):
        start = _call_start(masked, m)
        if start in tried:
            continue
        tried.add(start)
        leak = find_leaked_tool_call(text[start:], tool_names)
        if leak is not None:
            end = start + leak.end
            return replace(leak, end=end) if _is_call_tail(text[end:]) else None
    return None if markers else _xml_fragment(text, masked)


def _call_start(masked: str, marker: re.Match) -> int:
    """Where the call holding ``marker`` starts: the marker itself, or for a JSON "name" key the open
    ``<tool_call>`` wrapper around it, else the nearest ``{`` before it."""
    pos = marker.start()
    if marker.lastindex != _JSON_MARKER:
        return pos
    wrapper = masked.rfind("<tool_call>", 0, pos)
    if wrapper > masked.rfind(_HERMES_CLOSE, 0, pos):
        return wrapper
    brace = masked.rfind("{", 0, pos)
    return brace if brace >= 0 else pos


def _is_call_tail(rest: str) -> bool:
    """True when ``rest`` (the text after a call) is blank or only the cut-off start of another call."""
    rest = rest.strip()
    wrapper = _WRAPPER_OPENER_RE.match(rest)
    if wrapper:
        rest = rest[wrapper.end():]
    if not rest or _NAMED_OPENER_RE.fullmatch(rest):
        return True
    if wrapper and wrapper.group(0).startswith("<|") and _GEMMA_CALL_PREFIX_RE.fullmatch(rest):
        return True  # gemma: "<|tool_call>call:na"
    # A bare "t"/"to" is too likely to be prose; harmony's opener counts from "to=".
    return any(token.startswith(rest) for token in _CALL_TOKENS) and (rest[0] == "<" or len(rest) >= 3)


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
