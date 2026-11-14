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
from typing import Any, Optional, Sequence

from gateway.platforms.base import SendResult
from gateway.platforms.helpers import TABLE_SEPARATOR_RE
from plugins.platforms.telegram.rich_markdown import closes_fence, fence_opener

logger = logging.getLogger(__name__)

# A part is joined from blocks with a blank line, exactly the separator a Markdown block boundary
# uses — so re-joining the parts reproduces the original message byte-for-byte.
_BLOCK_SEPARATOR = "\n\n"

_DETAILS_OPEN_RE = re.compile(r"<details\b", re.IGNORECASE)
_DETAILS_CLOSE_RE = re.compile(r"</details\s*>", re.IGNORECASE)
_TABLE_ROW_RE = re.compile(r"^\s*\|")


def _starts_table(lines: list[str], index: int) -> bool:
    """A pipe row, or a header row without outer pipes whose next line is the separator."""
    return bool(_TABLE_ROW_RE.match(lines[index])) or (
        "|" in lines[index] and index + 1 < len(lines) and bool(TABLE_SEPARATOR_RE.match(lines[index + 1])))


def _starts_details(line: str) -> bool:
    return line.lstrip().lower().startswith("<details")


def _fence_marker(line: str) -> Optional[str]:
    """The full fence run opening *line* (a ```` ```` ```` block keeps all four), or ``None``."""
    return fence_opener(line)


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
                closed = closes_fence(lines[index], marker)
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
        if _starts_table(lines, index):
            block, piped = [], bool(_TABLE_ROW_RE.match(line))
            while index < total and lines[index].strip() and (
                    _TABLE_ROW_RE.match(lines[index]) or (not piped and "|" in lines[index])):
                block.append(lines[index])
                index += 1
            blocks.append(("table", block))
            continue
        block = [line]
        index += 1
        while index < total and lines[index].strip() and not (
            _fence_marker(lines[index]) is not None
            or _starts_details(lines[index])
            or _starts_table(lines, index)
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
    has_closer = len(block) > 1 and closes_fence(block[-1], marker)
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


_SUMMARY_END_RE = re.compile(r"</summary\s*>", re.IGNORECASE)


def _fit_details(block: list[str], limit: int) -> Optional[list[str]]:
    """Split a ``<details>`` element: every part closes it and the next reopens it with its
    ``<summary>``, so no part shows a bare tag as text. Its body is split recursively (fences,
    tables and nested ``<details>`` inside keep their own rules). ``None`` when the element has no
    recognizable ``<summary>`` head or closing tag, or the head alone leaves no room."""
    text = "\n".join(block)
    head_end = _SUMMARY_END_RE.search(text)
    tail = text.rstrip()
    if head_end is None or not tail.lower().endswith("</details>"):
        return None
    head, body = text[:head_end.end()], text[head_end.end():len(tail) - len("</details>")].strip("\n")
    if _DETAILS_OPEN_RE.search(head[1:]):
        return None  # a nested element inside the head: no safe reopen point
    budget = limit - len(head) - len("\n\n\n\n</details>")
    if budget < 64 or not body.strip():
        return None
    return [f"{head}\n\n{part}\n\n</details>" for part in split_rich_parts(body, budget)]


def _fit_block(kind: str, block: list[str], limit: int) -> list[str]:
    """*block* as texts of at most *limit* chars, keeping its own structure where possible."""
    text = "\n".join(block)
    if len(text) <= limit:
        return [text]
    if kind == "fence":
        return _fit_fence(block, limit)
    if kind == "table":
        return _fit_table(block, limit)
    if kind == "details":
        fitted = _fit_details(block, limit)
        if fitted is not None:
            return fitted
    return ["\n".join(group) for group in _greedy_lines(block, limit)]


def _details_balanced(text: str) -> bool:
    return len(_DETAILS_OPEN_RE.findall(text)) == len(_DETAILS_CLOSE_RE.findall(text))


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


def rich_part_count(adapter: Any, content: str, metadata: Optional[dict[str, Any]] = None) -> Optional[int]:
    """Rich messages :func:`send_rich_parts` would post for *content*, or ``None`` when the legacy
    MarkdownV2 path owns it (same gates, nothing sent)."""
    if adapter._should_attempt_rich(content, metadata=metadata):
        return 1
    if ((metadata or {}).get("expect_edits") or not adapter._needs_rich_rendering(content)
            or not adapter._rich_delivery_enabled() or getattr(adapter, "_rich_send_disabled", False)):
        return None
    parts = _fitting_parts(adapter, content)
    return len(parts) if parts and all(adapter._rich_content_ok(part, check_size=False) for part in parts) else None


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
    parts = _fitting_parts(adapter, content)
    if parts is None or not all(adapter._rich_content_ok(part, check_size=False) for part in parts):
        return None
    sent: list[SendResult] = []
    result = await _deliver_parts(adapter, chat_id, parts, sent, reply_to, metadata, len(parts), whole=content)
    if result is not None and not result.success and sent:
        return _with_delivered_prefix(result, content, parts, sent)
    return result


async def _deliver_parts(
    adapter: Any, chat_id: str, parts: list[str], sent: list[SendResult], reply_to: Optional[str],
    metadata: Optional[dict[str, Any]], total: int, whole: Optional[str] = None,
) -> Optional[SendResult]:
    """Send *parts* as rich messages after the ones already in *sent* (appending each that lands).
    ``None`` only when nothing is in the chat yet and the first part found rich unavailable."""
    for index, part in enumerate(parts):
        # Only the first part of the reply anchors it; later parts continue the same reply in order.
        result = await adapter._try_send_rich(chat_id, part, None if sent else reply_to, metadata)
        if result is None and not sent:
            return None  # nothing delivered — the legacy path may own the whole message
        if result is None:
            # A later part was refused for good (or rich went away): its text never reached the chat,
            # so the rest goes out as MarkdownV2 after the parts already shown, never re-sending them.
            return await _send_legacy_tail(adapter, chat_id, parts[index:], sent, metadata, whole=whole)
        if not result.success:
            raw = result.raw_response if isinstance(result.raw_response, dict) else {}
            if raw.get("delivery_uncertain") and index + 1 < len(parts):
                # The part in flight may show; the parts after it never left, so the stream
                # consumer sends them (``unattempted_chunks``) before counting the reply shown.
                result.raw_response = {**raw, "unattempted_chunks": tuple(
                    _legacy_chunks(adapter, _BLOCK_SEPARATOR.join(parts[index + 1:])))}
            if not sent:
                return result  # nothing landed: the caller's normal retry still owns the message
            return _partial_failure(result, sent, total, rest=parts[index:])
        sent.append(result)
    logger.info("Telegram: delivered an oversized rich message in %d parts", total)
    return SendResult(
        success=True, message_id=sent[0].message_id,
        raw_response={"message_ids": [r.message_id for r in sent if r.message_id is not None]})


async def resume_rich_parts(adapter: Any, chat_id: str, result: SendResult,
                            metadata: Optional[dict[str, Any]]) -> SendResult:
    """``_resume_partial_send`` for a rich-part send stopped by a failure that proved the part
    never reached the chat (``undelivered_rich_parts``): send that part and the ones after it as
    rich messages, after the parts already shown, under the same per-chat lock and flood window."""
    from plugins.platforms.telegram.adapter import _flood_cap_result
    raw = result.raw_response
    rest = list(raw["undelivered_rich_parts"])
    sent = [SendResult(success=True, message_id=str(i)) for i in raw.get("delivered_message_ids") or ()]
    total = len(sent) + len(rest)
    async with adapter._chat_send_lock(chat_id):
        cooldown = adapter._send_flood_cooldown_remaining(chat_id)
        if cooldown is not None:
            return _partial_failure(_flood_cap_result(cooldown), sent, total, rest=rest)
        return await _deliver_parts(adapter, chat_id, rest, sent, None, metadata, total)


async def edit_rich_parts(
    adapter: Any, chat_id: str, message_id: str, content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[SendResult]:
    """Finalize an edit-based stream using rich messages even when its answer is oversized.

    The first rich part replaces the existing preview; subsequent parts are fresh replies
    chained to the previous message. Return the last id and continuation ids exactly as
    ``_edit_overflow_split`` does, so the stream consumer retargets later edits correctly.
    """
    if adapter._rich_eligible(content):
        return await adapter._try_edit_rich(chat_id, message_id, content, metadata=metadata)
    if (
        not adapter._needs_rich_rendering(content)
        or not adapter._rich_delivery_enabled()
        or getattr(adapter, "_rich_send_disabled", False)
    ):
        return None
    parts = _fitting_parts(adapter, content)
    if not parts or not all(adapter._rich_content_ok(part, check_size=False) for part in parts):
        return None

    first = await adapter._try_edit_rich(chat_id, message_id, parts[0], metadata=metadata)
    if first is None or not first.success:
        return first  # nothing new was sent: the legacy edit may still own the final

    sent = [SendResult(success=True, message_id=str(message_id))]
    for part in parts[1:]:
        previous_id = sent[-1].message_id
        outcome = await adapter._try_send_rich(chat_id, part, previous_id, metadata)
        if outcome is None:
            tail = await _send_legacy_tail(adapter, chat_id, parts[len(sent):], sent, metadata)
            if tail.success:
                ids = [str(i) for i in tail.raw_response["message_ids"]]
                return SendResult(success=True, message_id=ids[-1], continuation_message_ids=tuple(ids[1:]),
                                  raw_response={"message_ids": ids})
            # The stream fallback resends past ``delivered_prefix``: exact only while no MarkdownV2 chunk landed.
            prefix = _BLOCK_SEPARATOR.join(parts[:len(sent)])
            raw = tail.raw_response if isinstance(tail.raw_response, dict) else {}
            if raw.get("delivered_chunks") == len(sent) and content.startswith(prefix):
                raw["delivered_prefix"] = prefix
            return tail
        if not outcome.success or outcome.message_id is None:
            if outcome is not None and outcome.success:
                outcome = SendResult(success=False, error="rich_part_missing_message_id", retryable=False)
            failure = _partial_failure(outcome, sent, len(parts))
            # This is an edit, so the stream consumer needs the actually delivered
            # prefix for a tail-only fallback. It must not resend the edited head, nor a part
            # whose receipt timed out after Telegram may have accepted it (``delivery_uncertain``):
            # that part counts as shown and only the parts after it are sent again.
            in_flight = bool(failure.raw_response.get("delivery_uncertain"))
            prefix = _BLOCK_SEPARATOR.join(parts[:len(sent) + in_flight])
            if content.startswith(prefix):
                failure.raw_response["delivered_prefix"] = prefix
            return failure
        sent.append(outcome)

    return SendResult(
        success=True,
        message_id=sent[-1].message_id,
        continuation_message_ids=tuple(str(r.message_id) for r in sent[1:]),
        raw_response={"message_ids": [str(r.message_id) for r in sent]},
    )


def _fitting_parts(adapter: Any, content: str) -> Optional[list[str]]:
    """Parts whose SENT markdown fits the cap. Normalization grows a part a little (hard breaks,
    paragraph spacer rows, ``\\$`` escapes), so a split whose parts overflow is redone with a
    smaller budget, including raw text under the cap whose sent markdown is over it (one part at
    the full cap). ``None`` when even the smallest budget overflows, or one part already fits."""
    cap = adapter.RICH_MESSAGE_MAX_CHARS
    for budget in (cap, cap * 7 // 8, cap * 3 // 4):
        parts = split_rich_parts(content, budget)
        if all(adapter._content_fits_rich_limits(part) for part in parts):
            break
    else:
        return None
    if len(parts) <= 1:
        return None
    # A part cut inside a <details> that could not be closed and reopened would show the bare tag
    # as text: the legacy chunker owns that message instead.
    if _details_balanced(content) and not all(_details_balanced(part) for part in parts):
        return None
    return parts


def _legacy_chunks(adapter: Any, text: str) -> list[str]:
    """*text* as the MarkdownV2 chunks ``send()``'s legacy path sends."""
    from gateway.platforms.base import utf16_len
    from plugins.platforms.telegram.adapter import _separate_chunk_indicator_from_fence
    chunks = adapter.truncate_message(adapter.format_message(text), adapter.MAX_MESSAGE_LENGTH, len_fn=utf16_len)
    if len(chunks) > 1:
        chunks = [_separate_chunk_indicator_from_fence(re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk))
                  for chunk in chunks]
    return chunks


async def _send_legacy_tail(
    adapter: Any, chat_id: str, rest: list[str], sent: list[SendResult], metadata: Optional[dict[str, Any]],
    whole: Optional[str] = None,
) -> SendResult:
    """Deliver the rich parts *rest* (none of them in the chat) as MarkdownV2 chunks after the
    parts in *sent*, through the adapter's partial-send resume: same lock, flood window, retries
    and ``partial_overflow`` contract, so a failure there still names only the undelivered tail."""
    logger.warning("Telegram: a rich part was refused after %d part(s) landed; sending the rest as MarkdownV2",
                   len(sent))
    delivered = tuple(str(r.message_id) for r in sent if r.message_id is not None)
    chunks = tuple(_legacy_chunks(adapter, _BLOCK_SEPARATOR.join(rest)))
    if whole is not None:  # recount: the landed parts plus this tail may pass the reply's message cap
        from gateway.platforms.helpers import hand_off_split
        if as_file := await hand_off_split(adapter, chat_id, whole, len(chunks), delivered, metadata=metadata):
            return as_file
    pending = SendResult(success=False, raw_response={
        "undelivered_chunks": chunks,
        "delivered_message_ids": delivered})
    result = await adapter._resume_partial_send(chat_id, pending, reply_to=None, metadata=metadata)
    if result is None:  # nothing to resume over (no bot): report the honest partial
        return _partial_failure(None, sent, len(sent) + len(rest))
    return result


def _with_delivered_prefix(result: SendResult, content: str, parts: list[str], sent: list[SendResult]) -> SendResult:
    """Name the raw text the landed rich parts show (``delivered_prefix``), so the stream consumer
    resends only what follows it: exact only while no MarkdownV2 chunk landed after them and the
    parts are a raw prefix of *content* (a reopened ``<details>`` part is not)."""
    raw = result.raw_response if isinstance(result.raw_response, dict) else {}
    prefix = _BLOCK_SEPARATOR.join(parts[:len(sent)])
    if sent and raw.get("partial_overflow") and raw.get("delivered_chunks") == len(sent) and content.startswith(prefix):
        raw["delivered_prefix"] = prefix
    return result


def _partial_failure(result: Optional[SendResult], sent: list[SendResult], total: int,
                     rest: Sequence[str] = ()) -> SendResult:
    """A failure after earlier parts landed, under the ``partial_overflow`` contract
    (``BasePlatformAdapter._is_partial_delivery``): ``_send_with_retry`` then neither re-sends the
    whole message nor falls back to plain text, either of which would duplicate the visible parts.
    When the failure proves the part in flight never left (``retryable`` and not
    ``delivery_uncertain``), it and the parts after it (*rest*) are attached as
    ``undelivered_rich_parts``, so ``_send_with_retry`` resumes exactly them."""
    if result is None:  # no adapter to resume the tail through
        result = SendResult(success=False, error="rich_parts_unavailable", retryable=False)
    ids = [str(r.message_id) for r in sent if r.message_id is not None]
    raw = dict(result.raw_response) if isinstance(result.raw_response, dict) else {}
    raw.update({
        "partial_overflow": True, "delivered_chunks": len(sent), "total_chunks": total,
        "last_message_id": ids[-1] if ids else None, "continuation_message_ids": tuple(ids[1:])})
    if rest and result.retryable and not raw.get("delivery_uncertain"):
        raw["undelivered_rich_parts"], raw["delivered_message_ids"] = tuple(rest), tuple(ids)
    result.raw_response = raw
    result.message_id = ids[-1] if ids else result.message_id
    result.continuation_message_ids = tuple(ids[1:])
    return result
