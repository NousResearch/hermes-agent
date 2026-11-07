"""Split one ``MEDIA:`` tag that lists several references into one tag per reference.

Models routinely serialize several attachments behind a single keyword instead of writing
one tag per file:

* ``MEDIA:"/ws/first image.png" "/ws/second image.png"`` (space-separated quoted)
* ``MEDIA:"/ws/a b.png", "/ws/c d.png"`` (comma-separated quoted)
* ``MEDIA:["/ws/a.png", "/ws/b.png"]`` (a serialized JSON array)
* ``MEDIA:/ws/a.png, /ws/b.png`` (comma-separated bare absolute paths)

The anchored tag regexes in ``gateway/platforms/base.py`` read a quoted payload as ONE path, so
every shape above delivered only the first file and leaked the rest as literal text. Rewriting
the list into ``MEDIA:<ref> MEDIA:<ref> …`` up front lets the existing single-tag pipeline
(extraction, display strip, stream cleanup) handle each reference with its normal validation,
so a member that does not validate stays visible instead of being welded into a neighbour.

Only an unambiguous list is rewritten: the payload must open with two or more explicitly quoted
references, or two or more tokens that each start at a filesystem root (``/``, ``~/``, ``X:\\``),
followed by whitespace, sentence punctuation or the end of the line. A spaced bare path
(``MEDIA:/data/map data.kmz``), a lone quoted name holding an apostrophe
(``MEDIA:'/tmp/team's.png'``) and prose after a single reference (``MEDIA:"/a.png", the first
one``) all fail that test and are left untouched. Ported from openclaw/openclaw#150958,
#163448, #163627.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

_QUOTES = "`\"'"
_KEYWORD_RE = re.compile(r"MEDIA:", re.IGNORECASE)
_ROOTED_TOKEN_RE = re.compile(r"(?:~/|/|[A-Za-z]:[/\\])[^\s,]+")
_JSON_ARRAY_RE = re.compile(r"\[([^\n\]]*)\]")
_SEPARATOR_RE = re.compile(r"[^\S\n]*,?[^\S\n]*")
_LIST_TERMINATORS = ".,;:!?)"
# A quote followed by one of these closes a reference (``MEDIA:"a", "b".``); the comma is handled
# separately because it only closes when another quoted reference follows it.
_SENTENCE_PUNCT = ".;:!?)"


def _after_separator(payload: str, index: int) -> int:
    """Index past the optional ``[ws][,][ws]`` separator at ``index`` (the pattern matches empty)."""
    sep = _SEPARATOR_RE.match(payload, index)
    return sep.end() if sep else index


def _quoted_reference_end(payload: str, start: int) -> int:
    """Index of the quote closing the reference opened at ``start``, else -1. A quote closes only
    when whitespace, the end of the payload, ``]`` or ``,`` + optional whitespace + another quote
    follows it; an earlier same-kind quote (``it's``) is part of the value. Linear: each
    whitespace run is walked at most once."""
    quote = payload[start]
    for index in range(start + 1, len(payload)):
        if payload[index] != quote:
            continue
        after = index + 1
        if after >= len(payload) or payload[after].isspace() or payload[after] in _SENTENCE_PUNCT:
            return index
        if payload[after] == ",":
            after += 1
            while after < len(payload) and payload[after].isspace():
                after += 1
            if after < len(payload) and payload[after] in _QUOTES:
                return index
    return -1


def _read_quoted_list(payload: str) -> Tuple[List[str], int]:
    """Leading run of quoted references (commas between them allowed) → ``(tokens, end)``;
    ``end`` is where the run stops (the first unquoted token, or the payload end)."""
    tokens: List[str] = []
    index = 0
    while index < len(payload):
        if tokens:
            index = _after_separator(payload, index)
            if index >= len(payload):
                break
        if payload[index] not in _QUOTES:
            break
        end = _quoted_reference_end(payload, index)
        if end == -1:
            break
        tokens.append(payload[index:end + 1])
        index = end + 1
    return tokens, index


def _read_bare_list(payload: str) -> Tuple[List[str], int]:
    """Leading run of whitespace/comma-separated tokens that each start at a filesystem root."""
    tokens: List[str] = []
    index = 0
    while True:
        probe = _after_separator(payload, index) if tokens else index
        token = _ROOTED_TOKEN_RE.match(payload, probe)
        if not token:
            break
        tokens.append(token.group(0))
        index = token.end()
    return tokens, index


def _clean_break(body: str, end: int) -> bool:
    """True when the list ends at the payload end, whitespace or sentence punctuation (a
    reference running straight into other text is not a list)."""
    return end >= len(body) or body[end].isspace() or body[end] in _LIST_TERMINATORS


def _split_payload(payload: str) -> Optional[Tuple[List[str], str]]:
    """``(tokens, trailing)`` when ``payload`` opens with a list of references, else None.
    ``trailing`` is whatever follows the list (sentence punctuation, prose, whitespace)."""
    body = payload.lstrip()
    array = _JSON_ARRAY_RE.match(body)
    if array:
        inner = array.group(1)
        tokens, end = _read_quoted_list(inner)
        if tokens and end >= len(inner.rstrip()) and _clean_break(body, array.end()):
            return tokens, body[array.end():]
        return None
    tokens, end = _read_quoted_list(body)
    if not tokens:
        tokens, end = _read_bare_list(body)
    if len(tokens) < 2 or not _clean_break(body, end):
        return None
    return tokens, body[end:]


def expand_media_list_tags(text: str, mask: Callable[[str], str]) -> str:
    """``text`` with every list-shaped ``MEDIA:`` payload rewritten as one tag per reference.
    ``mask`` is the offset-preserving protected-span masker (code, blockquotes): keywords are
    located on the masked copy so examples inside code are never rewritten, while the payload
    is read from the original so a backtick-quoted member is not blanked by the inline-code
    mask. Returns ``text`` unchanged (same object) when no list is present."""
    if not _KEYWORD_RE.search(text):
        return text
    masked = mask(text)
    out: List[str] = []
    cursor = 0
    for match in _KEYWORD_RE.finditer(masked):
        start = match.start()
        if start < cursor:
            continue  # inside the previous keyword's rewritten payload
        line_end = text.find("\n", match.end())
        line_end = len(text) if line_end == -1 else line_end
        nxt = _KEYWORD_RE.search(masked, match.end(), line_end)
        payload_end = nxt.start() if nxt else line_end
        split = _split_payload(text[match.end():payload_end])
        if not split:
            continue
        tokens, trailing = split
        out.append(text[cursor:start])
        out.append(" ".join(f"MEDIA:{token}" for token in tokens) + trailing)
        cursor = payload_end
    if not out:
        return text
    out.append(text[cursor:])
    return "".join(out)
