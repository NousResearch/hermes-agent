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
_RAW_MARK_RE = re.compile(r"\*+|~~|\|\||\n")
_MDV2_SPECIAL_RE = re.compile(r"[_*\[\]()~`>#+\-=|{}.!\\]")
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


def _carry(chunks: list[str], marks_at: Callable[[int, str], list[str]], indicator: str, raw: bool = False) -> list[str]:
    """Close each chunk's still-open entities before its ``(N/M)`` marker and reopen them at the
    start of the next chunk; *indicator* is the marker template (``" \\(%s/%s\\)"`` when escaped).
    ``marks_at(index, body)`` names the entities to carry out of chunk *index*, outermost first.
    Formatted MarkdownV2 (not *raw*) also moves a backslash cut from the char it escapes."""
    out, reopen, carry_text = [], "", ""
    for index, chunk in enumerate(chunks):
        m = _INDICATOR_RE.search(chunk)
        body = reopen + carry_text + (chunk[:m.start()] if m else chunk)
        carry_text = reopen = ""
        last = index == len(chunks) - 1
        trailing = len(body) - len(body.rstrip("\\"))
        if trailing % 2 and not last and not raw:  # a cut between "\\" and its char
            body, carry_text = body[:-1], "\\"
        if not last:
            marks = marks_at(index, body)
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
    def open_at(_index: int, body: str) -> list[str]:
        return open_mdv2_entities(body)

    carried = _carry(chunks, open_at, escaped)
    if any(len(c) != len(_INDICATOR_RE.sub(lambda m: escaped % m.groups(), r)) for c, r in zip(carried, chunks)):
        carried = _carry(truncate(formatted, max_length - ENTITY_CARRY_RESERVE, len_fn=len_fn), open_at, escaped)
    return [separate_chunk_indicator_from_fence(c) for c in carried]


def _raw_emphasis_tokens(text: str):
    """``(mark, opens, closes)`` for each raw-markdown emphasis delimiter outside code, skipping
    backslash-escaped characters (``\\*args`` is literal), plus ``("\\n", False, False)`` per line
    break (``"\\n\\n"`` when a blank line follows). A ``*`` run delimits only as ``format_message``
    pairs it: an opener before a non-space, a closer after one (so a ``* `` bullet is literal).
    ``_`` is never a delimiter: ``format_message`` shows it as written."""
    masked = _RAW_CODE_RE.sub(lambda c: re.sub(r"[^\n]", " ", c.group()), text)
    for m in _RAW_MARK_RE.finditer(masked):
        mark = m.group()
        if mark == "\n":
            yield ("\n\n" if masked.startswith("\n", m.end()) else "\n"), False, False
            continue
        slash = start = m.start()
        while slash and masked[slash - 1] == "\\":
            slash -= 1
        if (start - slash) % 2:
            if len(mark) == 1 or mark[0] != "*":
                continue
            start, mark = start + 1, mark[1:]  # only the escaped first star is literal
        if mark in ("~~", "||"):
            yield mark, True, True
            continue
        before, after = masked[start - 1:start] if start else "", masked[m.end():m.end() + 1]
        yield mark, bool(after) and not after.isspace(), bool(before) and not before.isspace()


def _cut_whitespace(content: str, pieces: list[str]) -> list[str]:
    """The whitespace ``truncate_message`` dropped at each cut between *pieces* (its chunks without
    the ``(N/M)`` marker), read back from *content*; ``"\\n"`` where a piece cannot be located."""
    gaps, pos = [], 0
    for piece in pieces[:-1]:
        head = piece.split("\n", 1)[1] if piece.startswith("```") and "\n" in piece else piece
        tail = [p[:-4] for p in (piece, head) if p.endswith("\n```")]
        core = next((c for c in (piece, head, *tail) if content.startswith(c, pos)), None)
        if core is None:
            gaps.append("\n")
            continue
        end = pos = pos + len(core)
        while pos < len(content) and content[pos].isspace():
            pos += 1
        gaps.append(content[end:pos])
    return gaps


def _raw_carry_plan(pieces: list[str], gaps: list[str]) -> list[list[str]]:
    """For each cut: the raw emphasis marks open there that the rest of the reply really closes,
    outermost first. One pass pairs every delimiter of the whole reply (a ``*``, ``~~`` or ``||``
    still open at a line break, or a ``**`` at a blank line, can never pair, as in
    ``format_message``), a second replays only the pairs that close, so no cut rescans the rest.
    A mark nothing closes is a literal character, not an entity cut by the split."""
    ops: list[tuple[str, int]] = []
    stack: list[tuple[str, int, int, int]] = []  # (mark, entry id, line, paragraph)
    closed: set[int] = set()
    line = para = 0
    for index, piece in enumerate(pieces):
        tokens = list(_raw_emphasis_tokens(piece))
        if index < len(pieces) - 1:
            gap = gaps[index]
            if "\n" in gap:
                tokens.append(("\n\n" if "\n\n" in gap else "\n", False, False))
            tokens.append(("", False, False))  # the cut
        for mark, opens, closes in tokens:
            if mark in ("\n", "\n\n"):
                line, para = line + 1, para + (mark == "\n\n")
                continue
            if not mark:
                ops.append(("cut", 0))
                continue
            while stack and (stack[-1][3] != para if stack[-1][0] == "**" else stack[-1][2] != line):
                stack.pop()  # an entity open at a line or paragraph break never closes
            left = mark
            if closes:  # a star run closes as many open star entities as it has stars (``***`` = both)
                while left and stack and stack[-1][0] in (left, left[:len(stack[-1][0])]):
                    closed.add(stack[-1][1])
                    left = left[len(stack[-1][0]):]
                    ops.append(("pop", stack.pop()[1]))
            if left == mark and opens:
                for part in ([left] if left[0] != "*" else ["**"] * (len(left) // 2) + ["*"] * (len(left) % 2)):
                    stack.append((part, len(ops), line, para))
                    ops.append((part, len(ops)))
    plan: list[list[str]] = []
    open_marks: list[str] = []
    for op, entry in ops:
        if op == "cut":
            plan.append(list(open_marks))
        elif op == "pop":
            open_marks.pop()
        elif entry in closed:
            open_marks.append(op)
    return plan


def split_raw_carrying_emphasis(content: str, max_length: int, truncate, len_fn,
                                formatted_len: Optional[Callable[[str], int]] = None) -> tuple[list[str], list[str]]:
    """``(raw_chunks, chunks)`` for RAW markdown formatted one chunk at a time (the overflow-edit
    path): ``**bold**``, ``*italic*``, ``~~strike~~`` and ``||spoiler||`` cut across chunks are closed and reopened
    in ``chunks`` so ``format_message`` pairs them instead of escaping a lone marker into literal
    text; ``raw_chunks`` are the uncarried cuts (what the visible prefix is built from).
    ``formatted_len`` measures a chunk once formatted: escaping can grow a raw chunk past
    *max_length*, so such a split is re-cut with the overshoot taken off the budget."""
    def carry(chunks: list[str]) -> list[str]:
        if len(chunks) <= 1:
            return chunks
        pieces = [_INDICATOR_RE.sub("", chunk) for chunk in chunks]
        plan = _raw_carry_plan(pieces, _cut_whitespace(content, pieces))
        return _carry(chunks, lambda index, _body: plan[index], " (%s/%s)", raw=True)

    def overshoot(chunks: list[str]) -> int:
        if formatted_len is None:
            return 0
        # Escaping adds at most one unit per special character, so most chunks fit without formatting.
        return max((formatted_len(c) - max_length for c in chunks
                    if len_fn(c) + len(_MDV2_SPECIAL_RE.findall(c)) > max_length), default=0)

    raw = truncate(content, max_length, len_fn=len_fn)
    carried = carry(raw)
    budget = max_length - ENTITY_CARRY_RESERVE if carried != raw else max_length
    for _ in range(16):  # a cut snaps back to whitespace, so a smaller budget may not move it at once
        if budget != max_length:
            raw = truncate(content, budget, len_fn=len_fn)
            carried = carry(raw)
        over = overshoot(carried)
        if over <= 0:
            break
        budget -= over + ENTITY_CARRY_RESERVE
    return raw, carried
