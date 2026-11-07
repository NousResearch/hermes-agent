"""Markdown-aware message chunking behind ``BasePlatformAdapter.truncate_message``."""

from typing import Callable, Optional

from gateway.platforms.helpers import fence_state_after, table_header_to_repeat, table_rollback_point


def _open_link_bracket(text: str) -> int:
    """Index of the ``[`` opening a Markdown link still unclosed at *text* end, else -1.

    Escaped brackets are plain text: platform formatters (Telegram MarkdownV2) escape
    literal ``[]()`` in prose, so an unescaped ``[...](…)`` run really is a link.
    """
    i, open_at, in_dest = 0, -1, False
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2  # skip the escaped char (or an escaped backslash)
            continue
        if in_dest:
            if ch == ")":
                in_dest = False
                open_at = -1
        elif ch == "[":
            if open_at < 0:
                open_at = i
        elif ch == "]":
            if open_at >= 0 and text.startswith("(", i + 1):
                in_dest = True
                i += 1
            else:
                open_at = -1  # lone "]" without "(" — not a link start
        i += 1
    return open_at


def chunk_markdown_message(content: str, max_length: int = 4096,
                           len_fn: Optional[Callable[[str], int]] = None) -> list[str]:
    """Split *content* into chunks of at most *max_length* ``len_fn`` units (``utf16_len`` for
    Telegram). A split inside a fence closes it and reopens it (same language tag) in the next
    chunk; splits retreat out of inline code spans and Markdown links; a pipe table moves whole
    to the next chunk when it fits, else splits between rows and repeats its header. Multi-chunk
    output gets ``(1/3)`` indicators."""
    from gateway.platforms.base import _custom_unit_to_cp  # base imports this module lazily
    _len = len_fn or len
    if _len(content) <= max_length:
        return [content]
    INDICATOR_RESERVE = 10   # room for " (XX/XX)"
    FENCE_CLOSE = "\n```"
    chunks: list[str] = []
    remaining = content
    carry_lang: Optional[str] = None  # language tag ("" ok) when previous chunk ended mid-fence
    carry_header: Optional[str] = None  # table header + separator when it ended mid-table
    while remaining:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
        if carry_header:
            prefix = f"{carry_header}\n"
        # Body budget after prefix/fence/indicator; floored so a tiny max_length can't stall.
        headroom = max_length - INDICATOR_RESERVE - _len(prefix) - _len(FENCE_CLOSE)
        if headroom < 1:
            headroom = max(1, max_length // 2)
        # Remainder fits in one final chunk; close a reopened fence if still open.
        if _len(prefix) + _len(remaining) <= max_length - INDICATOR_RESERVE:
            final_chunk = prefix + remaining
            if carry_lang is not None and fence_state_after(remaining, True, carry_lang)[0]:
                final_chunk += FENCE_CLOSE
            chunks.append(final_chunk)
            break
        # Natural split (newline, then space); a custom _len budget maps to a codepoint offset.
        _cp_limit = (
            _custom_unit_to_cp(remaining, headroom, _len) if _len is not len else headroom)
        region = remaining[:_cp_limit]
        split_at = region.rfind("\n")
        if split_at < _cp_limit // 2:
            split_at = region.rfind(" ")
        if split_at < 1:
            # Floor at one codepoint: a zero _cp_limit (max_length 0/1, or a surrogate pair
            # wider than the utf16 budget) would never shrink ``remaining``; overshooting beats
            # a hang.
            split_at = max(1, _cp_limit)
        # Don't split inside an inline code span or an open Markdown link:
        # an unpaired backtick breaks MarkdownV2, and a cut between "[" and
        # its closing ")" strands a half link in each chunk. One guard's
        # retreat can re-expose the other's hazard in the shorter candidate
        # (retreating past the "[" may also cross a backtick), so the guards
        # settle split_at to a fixed point; it strictly decreases per move,
        # which guarantees termination.
        while True:
            candidate = remaining[:split_at]
            moved = False
            backtick_count = candidate.count("`") - candidate.count("\\`")
            if backtick_count % 2 == 1:
                last_bt = candidate.rfind("`")
                while last_bt > 0 and candidate[last_bt - 1] == "\\":
                    last_bt = candidate.rfind("`", 0, last_bt)
                if last_bt > 0:
                    safe_split = max(
                        candidate.rfind(" ", 0, last_bt), candidate.rfind("\n", 0, last_bt))
                    if safe_split > _cp_limit // 4:
                        split_at = safe_split
                        moved = True
            link_lb = _open_link_bracket(candidate)
            if link_lb > 0:
                safe_split = max(
                    candidate.rfind(" ", 0, link_lb), candidate.rfind("\n", 0, link_lb))
                if safe_split > _cp_limit // 4:
                    split_at = safe_split
                    moved = True
            if not moved:
                break
        # Pipe tables: move a split that lands mid-table to before the header when the
        # whole table fits a fresh chunk; a larger table splits between rows and the next
        # chunk repeats its header (``carry_header``). Fenced pipes are not tables.
        if not fence_state_after(remaining[:split_at], carry_lang is not None, carry_lang or "")[0]:
            split_at = table_rollback_point(
                remaining, split_at, lambda t: _len(t) <= max_length - INDICATOR_RESERVE)
        chunk_body = remaining[:split_at]
        remaining = remaining[split_at:].lstrip()
        full_chunk = prefix + chunk_body
        # Walk only chunk_body (not the prepended prefix) for the fence state.
        in_code, lang = fence_state_after(chunk_body, carry_lang is not None, carry_lang or "")
        carry_lang = lang if in_code else None
        carry_header = None if in_code else table_header_to_repeat(full_chunk, remaining, max_length // 4, _len)
        # Close the orphaned fence so the chunk stands alone.
        chunks.append(full_chunk + FENCE_CLOSE if in_code else full_chunk)
    if len(chunks) > 1:
        chunks = [f"{chunk} ({i + 1}/{len(chunks)})" for i, chunk in enumerate(chunks)]
    return chunks
