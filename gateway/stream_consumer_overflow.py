"""Lossless overflow delivery for message-buffered streams."""

import time

from gateway.platforms.base import _custom_unit_to_cp
from gateway.stream_consumer_fences import ensure_closed_code_fences


def overflow_head_end(consumer):
    """Prefer a late newline, otherwise cut within the platform's length budget."""
    budget = _custom_unit_to_cp(consumer._accumulated, consumer._safe_limit, consumer._len_fn)
    split_at = consumer._accumulated.rfind("\n", 0, budget)
    return budget if split_at < budget // 2 else split_at


async def split_buffered_first_send(consumer, tick):
    """Commit one head at a time; retain only unacknowledged text on retries."""
    reply_to = consumer._initial_reply_to_id
    while consumer._overflows() and consumer._should_edit(tick):
        split_at = overflow_head_end(consumer)
        chunk = ensure_closed_code_fences(consumer._accumulated[:split_at])
        new_id = await consumer._send_new_chunk(chunk, reply_to, seal_draft=True)
        if new_id is None or new_id == reply_to:
            # The gateway must not mistake a partial delivery for the completed answer.
            return tick.got_done
        reply_to = new_id
        consumer._accumulated = consumer._accumulated[split_at:]
        consumer._turn_split_delivery = True
        consumer._message_id = consumer._message_created_ts = None
        consumer._last_sent_text = ""
        consumer._initial_text_visible = False
        consumer._last_edit_time = time.monotonic()
        if consumer._use_draft_streaming and not consumer._stream_is_message():
            consumer._bump_draft_id()
    if tick.got_done:
        consumer._final_response_sent = (
            not consumer._accumulated
            or await consumer._send_or_edit(consumer._accumulated, finalize=True))
        if consumer._final_response_sent:
            consumer._mark_final_delivered(record=consumer._accumulated)
        return True
    return False
