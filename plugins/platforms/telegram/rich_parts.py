"""Split an oversized rich-eligible Telegram message into ``sendRichMessage``-sized parts.

``sendRichMessage`` caps one message at ``RICH_MESSAGE_MAX_CHARS`` (32,768). Above that cap the
adapter used to hand the WHOLE message to the legacy MarkdownV2 chunker, which cannot render the
constructs that made the content rich-eligible in the first place — ``<details>`` blocks, pipe
tables and task lists arrived as literal text, chopped at 4,096 units, with delivery still reported
successful (#141272).

This module keeps the rich path for oversized content: the message is split on block boundaries
(blank lines) and each part goes out as its own ``sendRichMessage``. A ``<details>`` element, a
fenced code block and a table are atomic — a part never ends inside one. A single block larger than
the cap is the one case no boundary choice can solve, so it splits on line boundaries (a line wider
than the cap is cut by characters and logged); a split fence is closed and reopened with its
language tag, and a split table repeats its header, so the continuation parts still render.

Lives beside the adapter (not in it) because ``plugins/platforms/telegram/adapter.py`` is over the
repo's file cap: the facade imports :func:`send_rich_parts` and stays line-neutral.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)

# A part is joined from blocks with a blank line, exactly the separator a Markdown block boundary
# uses — so re-joining the parts reproduces the original message byte-for-byte.
_BLOCK_SEPARATOR = "\n\n"

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_DETAILS_OPEN_RE = re.compile(r"<details\b", re.IGNORECASE)
_DETAILS_CLOSE_RE = re.compile(r"</details\s*>", re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"^\s*\|")


def _starts_details(line: str) -> bool:
    return line.lstrip().lower().startswith("<details")


def _fence_marker(line: str) -> Optional[str]:
    """The fence run (`` ``` `` / ``~~~``) opening *line*, or ``None``."""
    match = _FENCE_RE.match(line)
    return match.group(1) if match else None


def _split_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """*lines* as ``(kind, block)`` pairs: ``fence``, ``details``, ``table`` or ``paragraph``."""
    blocks: list[tuple[str, list[str]]] = []
    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        marker = _fence_marker(line)
        if marker is not None:
            block = [line]
            index += 1
            while index < total:
                block.append(lines[index])
                closed = lines[index].strip().startswith(marker)
                index += 1
                if closed:
                    break
            blocks.append(("fence", block))
            continue
        if _starts_details(line):
            # Nested <details> stay in one block: depth rather than the first close tag.
            depth = 0
            block = []
            while index < total:
                current = lines[index]
                block.append(current)
                depth += len(_DETAILS_OPEN_RE.findall(current)) - len(_DETAILS_CLOSE_RE.findall(current))
                index += 1
                if depth <= 0:
                    break
            blocks.append(("details", block))
            continue
        if _TABLE_ROW_RE.match(line):
            block = []
            while index < total and _TABLE_ROW_RE.match(lines[index]):
                block.append(lines[index])
                index += 1
            blocks.append(("table", block))
            continue
        block = [line]
        index += 1
        while index < total and lines[index].strip() and not (
            _fence_marker(lines[index]) is not None
            or _starts_details(lines[index])
            or _TABLE_ROW_RE.match(lines[index])
        ):
            block.append(lines[index])
            index += 1
        blocks.append(("paragraph", block))
    return blocks


def _greedy_lines(lines: list[str], budget: int) -> list[list[str]]:
    """Group *lines* into runs whose joined length is at most *budget* (an over-wide line is cut)."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        if len(line) > budget:
            if current:
                groups.append(current)
            current, current_len = [], 0
            for start in range(0, len(line), max(1, budget)):
                groups.append([line[start:start + max(1, budget)]])
            logger.warning("Telegram rich part: a %d-char line exceeds the %d-char cap; cut mid-line",
                           len(line), budget)
            continue
        if current and current_len + 1 + len(line) > budget:
            groups.append(current)
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + (1 if current_len else 0)
    if current:
        groups.append(current)
    return groups


def _fit_fence(block: list[str], limit: int) -> list[str]:
    """Split a fenced block, closing each part's fence and reopening it with the language tag."""
    marker = _fence_marker(block[0]) or "```"
    opener = block[0].rstrip()
    has_closer = len(block) > 1 and block[-1].strip().startswith(marker)
    body = block[1:-1] if has_closer else block[1:]
    closer = f"\n{marker}" if has_closer else ""
    budget = limit - len(opener) - len(closer) - 2
    if budget < 1:
        return ["\n".join(block[:1] + block[1:])[:limit]]
    return [opener + "\n" + "\n".join(group) + closer for group in _greedy_lines(body, budget)]


def _fit_table(block: list[str], limit: int) -> list[str]:
    """Split a table, repeating its header rows so every continuation part still renders."""
    header = block[:2] if len(block) > 2 else block[:1]
    header_len = len("\n".join(header)) + 1
    budget = limit - header_len
    rows = block[len(header):]
    if budget < 1 or not rows:
        return ["\n".join(group) for group in _greedy_lines(block, limit)]
    return ["\n".join(header + group) for group in _greedy_lines(rows, budget)]


def _fit_block(kind: str, block: list[str], limit: int) -> list[str]:
    """*block* as texts of at most *limit* chars, keeping its own structure where possible."""
    text = "\n".join(block)
    if len(text) <= limit:
        return [text]
    if kind == "fence":
        return _fit_fence(block, limit)
    if kind == "table":
        return _fit_table(block, limit)
    return ["\n".join(group) for group in _greedy_lines(block, limit)]


def split_rich_parts(content: str, limit: int) -> list[str]:
    """*content* as consecutive parts of at most *limit* chars, split on block boundaries.

    Parts re-join into *content* with a blank line whenever no single block exceeds *limit* itself;
    a block that does is split (fence closed/reopened, table header repeated), so its continuation
    parts carry the few extra characters that keeps them renderable. Content already within *limit*
    comes back as the single original string.
    """
    if limit <= 0 or len(content) <= limit:
        return [content]
    pieces = [
        piece
        for kind, block in _split_blocks(content.splitlines())
        for piece in _fit_block(kind, block, limit)
    ]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        added = len(piece) + (len(_BLOCK_SEPARATOR) if current else 0)
        if current and current_len + added > limit:
            parts.append(_BLOCK_SEPARATOR.join(current))
            current, current_len = [], 0
            added = len(piece)
        current.append(piece)
        current_len += added
    if current:
        parts.append(_BLOCK_SEPARATOR.join(current))
    return parts


async def send_rich_parts(
    adapter: Any, chat_id: str, content: str, reply_to: Optional[str], metadata: Optional[dict[str, Any]] = None,
) -> Optional[SendResult]:
    """Deliver *content* through ``sendRichMessage``, one message or several parts when oversized.

    Returns the ``SendResult`` the caller hands back, or ``None`` when the legacy MarkdownV2 chunker
    should own the message: not rich-shaped, one part only, or the first part found the endpoint
    unavailable (nothing was delivered yet). Every part passes the same rich shape checks as a
    single message — only the 32,768-char cap moves from the whole message to each part.
    """
    if adapter._should_attempt_rich(content, metadata=metadata):
        return await adapter._try_send_rich(chat_id, content, reply_to, metadata)
    # Only SIZE may be the reason this message missed the rich path: ordinary long prose keeps the
    # legacy MarkdownV2 chunker (rich sends stay reserved for the constructs it cannot render), and
    # an ``expect_edits`` metadata still means "this is a preview, not a new message".
    if (
        (metadata or {}).get("expect_edits")
        or not adapter._needs_rich_rendering(content)
        or not adapter._rich_delivery_enabled()
        or getattr(adapter, "_rich_send_disabled", False)
    ):
        return None
    parts = split_rich_parts(content, adapter.RICH_MESSAGE_MAX_CHARS)
    cap = adapter.RICH_MESSAGE_MAX_CHARS
    if len(parts) < 2 or not all(
        len(part) <= cap and adapter._rich_content_ok(part, check_size=False) for part in parts
    ):
        return None
    sent: list[SendResult] = []
    for index, part in enumerate(parts):
        # Only the first part anchors the reply; later parts continue the same reply in order.
        result = await adapter._try_send_rich(chat_id, part, reply_to if index == 0 else None, metadata)
        if result is None:
            if index == 0:
                return None  # nothing delivered — the legacy path may own the whole message
            # A mid-message capability loss: a legacy resend would duplicate the parts already in
            # the chat, so the partial delivery is reported instead (#141272).
            return SendResult(
                success=False, error="rich_parts_unavailable",
                message_id=sent[-1].message_id,
                continuation_message_ids=tuple(
                    str(r.message_id) for r in sent if r.message_id is not None),
                retryable=False)
        if not result.success:
            return result
        sent.append(result)
    logger.info("Telegram: delivered an oversized rich message in %d parts (%d chars)", len(parts), len(content))
    return sent[-1]
