"""``BasePlatformAdapter.truncate_message``: split text into chunks that each fit one post.

A platform posts every chunk as one API call against a hard cap, so ``max_length`` bounds the
whole chunk: a reopened fence, its closing fence and the ``(i/n)`` suffix included.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from gateway.platforms.helpers import fence_state_after

FENCE_CLOSE = "\n```"
# Room for " (XX/XX)"; widened and re-split for a count that needs more.
INDICATOR_RESERVE = 10
# Longest HTML entity body a hard split must not cut (``&amp;``, ``&#128512;``).
_ENTITY_MAX = 9
# A ``<`` that opens a span: a Slack link / mention (``<https:``, ``<@U1>``, ``<#C1>``,
# ``<!here>``) or an HTML tag. A comparison such as ``x < 5`` opens nothing.
_SPAN_OPEN = re.compile(r"<[@#!/A-Za-z]")


def _indicator(index: int, count: int) -> str:
    return f" ({index}/{count})"


def _fits_whole(remaining: str, start: int, closer: str, cp_limit: int) -> bool:
    """The span opening at ``start`` closes, on its line, within one chunk's ``cp_limit``."""
    end = remaining.find(closer, start + 1, start + cp_limit)
    return end != -1 and "\n" not in remaining[start:end]


def _unbroken_split(remaining: str, split_at: int, hard_cut: bool, cp_limit: int) -> int:
    """``split_at``, or an earlier offset so the chunk does not end inside an inline code span, an
    HTML entity (Slack mrkdwn and Telegram HTML escape ``& < >``) or an open ``<...>`` span (a
    Slack link or mention, an HTML tag): its tail would render broken. A span that fits a chunk on
    its own moves whole to the next one however early it opens; a longer span keeps its tail
    only when a break above ``floor`` (a quarter of the budget) precedes it, else it is split
    inside. ``hard_cut`` is a cut between two non-space characters; an entity holds no
    whitespace, so a natural break can never be inside one."""
    candidate = remaining[:split_at]
    floor = cp_limit // 4
    backtick_count = candidate.count("`") - candidate.count("\\`")
    if backtick_count % 2 == 1:
        last_bt = candidate.rfind("`")
        while last_bt > 0 and candidate[last_bt - 1] == "\\":
            last_bt = candidate.rfind("`", 0, last_bt)
        if last_bt > 0:
            safe = max(candidate.rfind(" ", 0, last_bt), candidate.rfind("\n", 0, last_bt))
            if safe > floor:
                split_at, hard_cut = safe, False  # now a natural break
            elif _fits_whole(remaining, last_bt, "`", cp_limit):
                split_at, hard_cut = last_bt, False  # a span boundary is a natural break too
    amp = candidate.rfind("&", 0, split_at)
    entity_head = candidate[amp + 1:split_at].lstrip("#")
    # An empty head is a cut right after the ``&``: the next chunk would open on ``amp;``.
    if (hard_cut and amp > floor and amp >= split_at - _ENTITY_MAX
            and (not entity_head or entity_head.isalnum())):
        split_at = amp
    lt = candidate.rfind("<", 0, split_at)
    if (lt > 0 and _SPAN_OPEN.match(candidate, lt) and candidate.rfind(">", 0, split_at) < lt
            and "\n" not in candidate[lt:split_at]
            and (lt > floor or _fits_whole(remaining, lt, ">", cp_limit))):
        split_at = lt
    return split_at


def _split_point(remaining: str, cp_limit: int) -> int:
    """Natural break (newline, then space) inside the first ``cp_limit`` codepoints."""
    region = remaining[:cp_limit]
    split_at = region.rfind("\n")
    if split_at < cp_limit // 2:
        split_at = region.rfind(" ")
    if split_at < 1:
        # Floor at one codepoint: a zero cp_limit (max_length 0/1, or a surrogate pair wider than
        # the utf16 budget) would never shrink ``remaining``; overshooting beats a hang.
        split_at = max(1, cp_limit)
    hard_cut = split_at < len(remaining) and not remaining[split_at].isspace()
    return max(1, _unbroken_split(remaining, split_at, hard_cut, cp_limit))


def _split(content: str, max_length: int, _len: Callable[[str], int], reserve: int) -> list[str]:
    from gateway.platforms.base import _custom_unit_to_cp

    chunks: list[str] = []
    remaining = content
    carry_lang: Optional[str] = None  # language tag ("" ok) when previous chunk ended mid-fence
    while remaining:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
        # Body budget after prefix/fence/indicator; floored so a tiny max_length can't stall.
        headroom = max_length - reserve - _len(prefix) - _len(FENCE_CLOSE)
        if headroom < 1:
            headroom = max(1, max_length // 2)
        # Remainder fits in one final chunk, with the close of a reopened fence still open.
        if _len(prefix) + _len(remaining) <= max_length - reserve:
            close = FENCE_CLOSE if (
                carry_lang is not None and fence_state_after(remaining, True, carry_lang)[0]) else ""
            if _len(prefix) + _len(remaining) + _len(close) <= max_length - reserve:
                chunks.append(prefix + remaining + close)
                break
        # A custom _len budget maps to a codepoint offset.
        cp_limit = _custom_unit_to_cp(remaining, headroom, _len) if _len is not len else headroom
        split_at = _split_point(remaining, cp_limit)
        chunk_body = remaining[:split_at]
        remaining = remaining[split_at:].lstrip()
        # Walk only chunk_body (not the prepended prefix) for the fence state.
        in_code, lang = fence_state_after(chunk_body, carry_lang is not None, carry_lang or "")
        carry_lang = lang if in_code else None
        # Close the orphaned fence so the chunk stands alone.
        chunks.append(prefix + chunk_body + (FENCE_CLOSE if in_code else ""))
    return chunks


def split_message(content: str, max_length: int = 4096,
                  len_fn: Optional[Callable[[str], int]] = None) -> list[str]:
    """Split a long message into chunks preserving code blocks: a split inside a fence closes it
    at the chunk end and reopens it (same language tag) in the next; multi-chunk output gets
    ``(1/3)`` indicators. ``len_fn`` overrides ``len`` (``utf16_len`` for Telegram)."""
    _len = len_fn or len
    if _len(content) <= max_length:
        return [content]
    reserve = INDICATOR_RESERVE
    chunks = _split(content, max_length, _len, reserve)
    while len(chunks) > 1 and _len(_indicator(len(chunks), len(chunks))) > reserve:
        reserve = _len(_indicator(len(chunks), len(chunks)))
        chunks = _split(content, max_length, _len, reserve)
    if len(chunks) > 1:
        chunks = [chunk + _indicator(i + 1, len(chunks)) for i, chunk in enumerate(chunks)]
    return chunks
