"""Telegram long-message splitting that keeps MarkdownV2 inline entities intact across chunks.

``BasePlatformAdapter.truncate_message`` picks split points by newline/space, so a cut can land
inside ``*bold*``, ``_italic_``, ``__underline__``, ``~strike~`` or ``||spoiler||``. Each half then
carries an unbalanced delimiter: Telegram refuses the formatted chunk (``can't parse entities``) and
the plain-text fallback strips the formatting, or, when raw chunks are formatted one by one, the
unpaired marker is escaped and shows as a literal ``**``. Here an entity still open at a chunk's end
is closed there and reopened at the start of the next chunk, so both halves render styled.
"""

import re
from typing import Callable, Optional

_INDICATOR_RE = re.compile(r" \((\d+)/(\d+)\)$")
_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(r'(?m)^``` (?P<indicator>(?:\\)?\(\d+/\d+(?:\\)?\))$')
_MDV2_LINK_RE = re.compile(r"\[(?:\\.|`(?:\\.|[^`\\])*`|[^\]\\`])*\]\((?:\\.|[^)\\])*\)")
_QUOTE_MARKER_RE = re.compile(r"(?:\*\*)?>+")
_MDV2_MARKS = ("||", "__", "*", "_", "~")
_RAW_CODE_RE = re.compile(r"```[\s\S]*?(?:```|$)|`[^`\n]*`")
_RAW_MARK_RE = re.compile(r"\*\*|~~|\|\||\*|_")
# Room for the closers a carried entity adds (``||__*_~`` = 7 at most) plus a moved trailing backslash.
ENTITY_CARRY_RESERVE = 20


def separate_chunk_indicator_from_fence(text: str) -> str:
    """Move a ``(N/M)`` chunk marker that ``truncate_message()`` appended to a synthesized closing
    fence onto its own line — Telegram rejects ````` \\(1/2\\)`` as a fence."""
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(r'```\n\g<indicator>', text)


def _code_span_end(text: str, i: int) -> Optional[int]:
    end = i + 1
    while end < len(text) and text[end] != "`":
        end += 2 if text[end] == "\\" else 1
    return end + 1 if end < len(text) else None


def _mdv2_skip(text: str, i: int) -> int:
    """Index past an escape, ``pre``/``code`` span or link starting at ``text[i]`` (entities inside
    them are not inline-style delimiters), else ``i``."""
    if text[i] == "\\":
        return i + 2
    if text.startswith("```", i):
        end = text.find("```", i + 3)
        return len(text) if end < 0 else end + 3
    if text[i] == "`":
        return _code_span_end(text, i) or len(text)
    if text[i] == "[":
        m = _MDV2_LINK_RE.match(text, i)
        return m.end() if m else i
    return i


def open_mdv2_entities(text: str) -> list[str]:
    """Inline-style delimiters still open at the end of MarkdownV2 *text*, outermost first.
    Mirrors the Bot API parser: a marker closes the innermost entity of its kind, else opens one;
    a line-final ``||`` on an expandable quote (``**>``) ends the quote, not a spoiler."""
    i, n, stack, expandable = 0, len(text), [], False
    while i < n:
        skipped = _mdv2_skip(text, i)
        if skipped != i:
            i = skipped
            continue
        if (i == 0 or text[i - 1] == "\n") and (q := _QUOTE_MARKER_RE.match(text, i)):
            expandable, i = expandable or text[i] == "*", q.end()
            continue
        if expandable and text.startswith("||", i) and text[i + 2:i + 3] in ("", "\n") and stack[-1:] != ["||"]:
            expandable, i = False, i + 2
            continue
        mark = next((mk for mk in _MDV2_MARKS if text.startswith(mk, i)), None)
        if mark is None:
            i += 1
            continue
        if stack and stack[-1] == mark:
            stack.pop()
        else:
            stack.append(mark)
        i += len(mark)
    return stack


def _carry(chunks: list[str], open_marks: Callable[[str], list[str]], indicator: str,
           closes_reopened: Optional[Callable[[str, str], str]] = None) -> list[str]:
    """Close each chunk's still-open entities before its ``(N/M)`` marker and reopen them at the
    start of the next chunk; *indicator* is the marker template (``" \\(%s/%s\\)"`` when escaped).
    ``closes_reopened(marks, rest)`` (raw markdown only) keeps just the marks the rest of the
    message really closes, so a literal marker with no partner is never turned into styling. Formatted
    MarkdownV2 (no ``closes_reopened``) also moves a backslash cut from the char it escapes."""
    out, reopen, carry_text = [], "", ""
    for index, chunk in enumerate(chunks):
        m = _INDICATOR_RE.search(chunk)
        body = reopen + carry_text + (chunk[:m.start()] if m else chunk)
        carry_text = reopen = ""
        last = index == len(chunks) - 1
        trailing = len(body) - len(body.rstrip("\\"))
        if trailing % 2 and not last and closes_reopened is None:  # a cut between "\\" and its char
            body, carry_text = body[:-1], "\\"
        if not last:
            marks = open_marks(body)
            if closes_reopened is not None:
                marks = closes_reopened(marks, "".join(_INDICATOR_RE.sub("", rest) for rest in chunks[index + 1:]))
            reopen = "".join(marks)
            body += "".join(reversed(marks))
        out.append(body + (indicator % m.groups() if m else ""))
    return out


def split_formatted_mdv2(formatted: str, max_length: int, truncate, len_fn) -> list[str]:
    """``TelegramAdapter.send``'s split of already-formatted MarkdownV2 with ``truncate`` (the
    adapter's ``truncate_message``): entities are carried across cuts and the ``(N/M)`` markers are
    MarkdownV2-escaped. Only a split that carries an entity re-cuts with the carry reserve, so plain
    text splits exactly where ``truncate_message`` alone would."""
    chunks = truncate(formatted, max_length, len_fn=len_fn)
    if len(chunks) <= 1:
        return chunks
    escaped = " \\(%s/%s\\)"
    carried = _carry(chunks, open_mdv2_entities, escaped)
    if any(len(c) != len(_INDICATOR_RE.sub(lambda m: escaped % m.groups(), r)) for c, r in zip(carried, chunks)):
        carried = _carry(truncate(formatted, max_length - ENTITY_CARRY_RESERVE, len_fn=len_fn), open_mdv2_entities, escaped)
    return [separate_chunk_indicator_from_fence(c) for c in carried]


def _raw_emphasis_tokens(text: str):
    """``(mark, opens, closes)`` for each raw-markdown emphasis delimiter outside code, skipping
    backslash-escaped characters (``\\*args`` is literal). A single ``*``/``_`` delimits only per
    CommonMark flanking: never before a space (so not a ``* `` bullet); an intraword ``_`` is literal."""
    masked = _RAW_CODE_RE.sub(lambda c: " " * len(c.group()), text)
    for m in _RAW_MARK_RE.finditer(masked):
        slashes = len(masked[:m.start()]) - len(masked[:m.start()].rstrip("\\"))
        if slashes % 2:
            continue
        mark, before, after = m.group(), masked[m.start() - 1:m.start()], masked[m.end():m.end() + 1]
        if len(mark) == 1:
            yield (mark, bool(after) and not after.isspace() and (mark == "*" or not before.isalnum()),
                   bool(before) and not before.isspace() and (mark == "*" or not after.isalnum()))
        else:
            yield mark, True, True


def _open_raw_emphasis(text: str) -> list[str]:
    """Raw-markdown emphasis still open at the end of *text*, outermost first."""
    stack: list[str] = []
    for mark, opens, closes in _raw_emphasis_tokens(text):
        if stack and stack[-1] == mark and closes:
            stack.pop()
        elif opens:
            stack.append(mark)
    return stack


def _raw_marks_closed_in(marks: list[str], body: str) -> list[str]:
    """The *marks* that *body* (the rest of the message) closes once they are reopened before it,
    outermost first; a mark nothing closes was a literal character, not an entity cut by the split."""
    stack = [(mark, True) for mark in marks]
    closed: set[int] = set()
    for mark, opens, closes in _raw_emphasis_tokens(body):
        if stack and stack[-1][0] == mark and closes:
            if stack[-1][1]:
                closed.add(len(stack) - 1)
            stack.pop()
        elif opens:
            stack.append((mark, False))
    return [mark for i, mark in enumerate(marks) if i in closed]


def split_raw_carrying_emphasis(content: str, max_length: int, truncate, len_fn) -> tuple[list[str], list[str]]:
    """``(raw_chunks, chunks)`` for RAW markdown formatted one chunk at a time (the overflow-edit
    path): ``**bold**``, ``*italic*``, ``~~strike~~`` and ``||spoiler||`` cut across chunks are closed and reopened
    in ``chunks`` so ``format_message`` pairs them instead of escaping a lone marker into literal
    text; ``raw_chunks`` are the uncarried cuts (what the visible prefix is built from)."""
    raw = truncate(content, max_length, len_fn=len_fn)
    if len(raw) <= 1:
        return raw, raw
    carried = _carry(raw, _open_raw_emphasis, " (%s/%s)", _raw_marks_closed_in)
    if carried != raw:
        raw = truncate(content, max_length - ENTITY_CARRY_RESERVE, len_fn=len_fn)
        carried = _carry(raw, _open_raw_emphasis, " (%s/%s)", _raw_marks_closed_in)
    return raw, carried
