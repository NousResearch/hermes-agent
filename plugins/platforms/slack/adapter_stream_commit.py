"""Commit Slack draft messages in place, at reply or explicit overflow boundaries."""

import logging

from gateway.platforms.base import SendResult

logger = logging.getLogger(__name__)


async def try_finalize_stream(adapter, chat_id, content, metadata=None):
    """Unrelated sends leave the draft alone; a matching sealed segment owns its draft."""
    md = metadata or {}
    segment_id = md.get("_finalize_draft_id")
    segment = segment_id is not None
    if not segment and (md.get("_interim_send") or md.get("expect_edits")):
        return None
    key = adapter._stream_key(chat_id, md)
    stream = adapter._active_streams.get(key) if key else None
    if stream is None:
        return None
    if segment and segment_id != stream.get("draft_id"):
        return SendResult(success=False, error="draft segment identity mismatch")
    text = adapter._strip_stream_cursor(content)
    sent = stream.get("sent", "")
    kind, delta = adapter._stream_relation(sent, text)
    replace = segment and (kind == "unrelated" or stream.get("_segment_commit_pending", False))
    if kind == "unrelated" and not segment:
        # Only a restyled turn final may rewrite an unrelated-looking draft.
        if not (md.get("notify") and text.strip() and sent.strip()
                and adapter._mrkdwn_insensitive(text) == adapter._mrkdwn_insensitive(sent)):
            return None
        replace = True
    if not replace and kind == "extends" and len(delta) > adapter.MAX_MESSAGE_LENGTH:
        adapter._active_streams.pop(key, None)
        await adapter._seal_stream(key, stream)
        return None
    return await _commit_stream(adapter, key, stream, text, metadata,
                                delta=delta, replace=replace, segment=segment)


async def _commit_stream(adapter, key, stream, text, metadata, *, delta="", replace=False,
                         segment=False):
    """A sealed overflow head is not the turn final and must never fall back to a duplicate."""
    chat_id, ts = key[1], stream["ts"]
    adapter._active_streams.pop(key, None)
    # stopStream appends: a rewrite (or ambiguous prior append) must use chat.update.
    append = "" if replace else delta
    sealed = await adapter._seal_stream(key, stream, delta=append)
    if not sealed and not (append and append.strip()):
        sealed = await adapter._seal_stream(key, stream)
    if sealed and not replace and not adapter._maybe_blocks(text):
        committed = True
    else:
        shown = text if replace else text[stream.get("base", 0):]
        edited = await adapter.edit_message(chat_id, ts, shown, finalize=True, metadata=metadata)
        committed = edited.success or (sealed and not replace)
        if not sealed:
            logger.warning(
                "[Slack] chat.stopStream failed for %s/%s; in-place commit %s",
                chat_id, ts, "succeeded" if committed else "failed")
    if not committed:
        if segment:
            # Retry by replacement, never by appending a delta whose delivery is ambiguous.
            stream["_segment_commit_pending"] = True
            adapter._active_streams[key] = stream
            return SendResult(success=False, message_id=ts, error="draft segment commit failed")
        return None
    if not segment:
        await adapter.stop_typing(chat_id, metadata)
    return SendResult(success=True, message_id=ts)
