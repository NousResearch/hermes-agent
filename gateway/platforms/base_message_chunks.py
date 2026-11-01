"""Canonical message splitting with optional raw source receipt offsets."""

from typing import Callable, Optional

from gateway.platforms.helpers import fence_state_after


def truncate_message_with_offsets(content: str, max_length: int = 4096,
                                  len_fn: Optional[Callable[[str], int]] = None, *,
                                  chunk_end_offsets: Optional[list[int]] = None) -> list[str]:
    """Split a long message into chunks preserving code blocks: a split inside a fence closes it
    at the chunk end and reopens it (same language tag) in the next; multi-chunk output gets
    ``(1/3)`` indicators. ``len_fn`` overrides ``len`` (``utf16_len`` for Telegram).
    Optional offsets identify consumed source characters before fence/counter decoration."""
    from gateway.platforms.base import _custom_unit_to_cp

    _len = len_fn or len
    if _len(content) <= max_length:
        if chunk_end_offsets is not None:
            chunk_end_offsets.append(len(content))
        return [content]
    INDICATOR_RESERVE = 10   # room for " (XX/XX)"
    FENCE_CLOSE = "\n```"
    chunks: list[str] = []
    remaining = content
    carry_lang: Optional[str] = None  # language tag ("" ok) when previous chunk ended mid-fence
    while remaining:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
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
            if chunk_end_offsets is not None:
                chunk_end_offsets.append(len(content))
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
        # Don't split inside an inline code span: an unpaired backtick breaks MarkdownV2.
        candidate = remaining[:split_at]
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
        chunk_body = remaining[:split_at]
        remaining = remaining[split_at:].lstrip()
        if chunk_end_offsets is not None:
            chunk_end_offsets.append(len(content) - len(remaining))
        full_chunk = prefix + chunk_body
        # Walk only chunk_body (not the prepended prefix) for the fence state.
        in_code, lang = fence_state_after(chunk_body, carry_lang is not None, carry_lang or "")
        carry_lang = lang if in_code else None
        # Close the orphaned fence so the chunk stands alone.
        chunks.append(full_chunk + FENCE_CLOSE if in_code else full_chunk)
    if len(chunks) > 1:
        chunks = [f"{chunk} ({i + 1}/{len(chunks)})" for i, chunk in enumerate(chunks)]
    return chunks
