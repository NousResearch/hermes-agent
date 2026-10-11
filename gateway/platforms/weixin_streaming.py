"""Append-only block delivery over the gateway's callback and boundary rails."""

from __future__ import annotations

import asyncio
import copy
import concurrent.futures
import logging
import time

from gateway.response_filters import is_partial_silence_marker, is_intentional_silence_response
from gateway.stream_consumer import GatewayStreamConsumer, _APPROVAL_BOUNDARY
from gateway.platforms.weixin_markdown import StreamingMarkdownFilter, filter_markdown

logger = logging.getLogger(__name__)


class WeixinBlockConsumer(GatewayStreamConsumer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        settings = self.adapter.config.extra.get("block_streaming") or {}
        self._min_chars = max(1, int(settings.get("min_chars", 200)))
        self._idle_seconds = max(0.05, float(settings.get("idle_ms", 3000)) / 1000)
        self._last_delta_at = time.monotonic()
        self._parser = StreamingMarkdownFilter()
        self._acked_parser = copy.deepcopy(self._parser)
        self._fed_raw = self._acked_raw = ""
        self._pending_display = ""
        self._stream_failed = False

    def _append_accumulated(self, text):
        super()._append_accumulated(text)
        if text:
            self._last_delta_at = time.monotonic()

    def _display_payload(self, text):
        return filter_markdown(self._clean_for_display(text or "")).strip()

    def _visible_prefix(self):
        return self._display_payload(self._acked_raw)

    def acknowledged_block_prefix(self):
        return self._acked_raw

    def close_for_approval_prompt(self, placeholder=None, reason="Approval", reopen=False):
        try:
            future = asyncio.get_running_loop().create_future()
        except RuntimeError:
            future = concurrent.futures.Future()
        self._queue.put((_APPROVAL_BOUNDARY, future, None))
        return future

    def _adopt_final_text(self, final_raw):
        if not (self._accumulated or self._acked_raw):
            return
        cleaned = self._clean_for_display(final_raw)
        if cleaned.startswith(self._acked_raw):
            # Rebuild only the unacknowledged tail; never replay sealed blocks.
            self._accumulated = final_raw
            self._stream_ledger = final_raw
            self._fed_raw = self._acked_raw
            self._parser = copy.deepcopy(self._acked_parser)
            self._pending_display = ""
            self._stream_failed = False
        else:
            # An append-only transport cannot correct a rewritten prefix by editing it.
            # Leave final delivery to the gateway rather than declaring a false match.
            self._stream_failed = True

    async def run(self):
        try:
            while self._run_still_current():
                tick = self._drain_queue()
                if tick.got_done:
                    self._flush_think_buffer()
                boundary = tick.got_done or tick.got_segment_break or tick.approval_boundary is not None
                if boundary or tick.commentary_text is not None or self._block_due():
                    await self._flush_block(final=boundary or tick.commentary_text is not None, turn_final=tick.got_done)
                if tick.approval_boundary is not None:
                    future, cancelled = tick.approval_boundary
                    if not future.done():
                        future.set_result(not self._stream_failed)
                    self._reset_block_segment()
                if tick.got_done:
                    await self._notify_before_finalize()
                    if not self._stream_failed and self._acked_raw:
                        self._mark_final_delivered(record=self._accumulated)
                    return
                if tick.commentary_text is not None:
                    self._reset_block_segment()
                    await self._send_commentary(tick.commentary_text)
                    self._reset_block_segment()
                if tick.got_segment_break:
                    self._reset_block_segment()
                self._signal_flush(tick.flush_event)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Weixin block stream failed")
        finally:
            self._wake_flush_waiters()

    def _block_due(self):
        if self.cfg.buffer_only or not self.stream_deltas_enabled or self._stream_failed:
            return False
        unseen = self._clean_for_display(self._accumulated)[len(self._fed_raw):]
        return bool(unseen) and (len(unseen) >= self._min_chars
                                or time.monotonic() - self._last_delta_at >= self._idle_seconds)

    async def _flush_block(self, *, final, turn_final=False):
        if self._stream_failed:
            return
        raw = self._clean_for_display(self._accumulated)
        if is_partial_silence_marker(raw) or is_intentional_silence_response(raw):
            return
        if not final:
            # A partial directive must not leak before the anchored display filter can see it.
            last_line = raw.rsplit("\n", 1)[-1].lstrip()
            if last_line.startswith(("MEDIA:", "[[")) or "MEDIA:".startswith(last_line):
                raw = raw[:len(raw) - len(raw.rsplit("\n", 1)[-1])]
        parser = copy.deepcopy(self._parser)
        display = self._pending_display + parser.feed(raw[len(self._fed_raw):])
        if final:
            display += parser.flush()
        if not final and (parser.fence or parser.inline or parser.buf):
            # Keep an open code block / inline construct together for correct rendering.
            self._parser, self._fed_raw, self._pending_display = parser, raw, display
            return
        if display.strip():
            metadata = self._metadata_for_send(final=turn_final) or {}
            if not turn_final:
                metadata["_interim_send"] = True
            result = await self.adapter.send(
                self.chat_id, display, reply_to=self._initial_reply_to_id if not self._already_sent else None,
                metadata=metadata)
            if not result.success:
                self._stream_failed = True
                return
            self._message_id = result.message_id
            self._already_sent = True
            self._turn_split_delivery = True
            self._notify_new_message()
        self._parser, self._fed_raw, self._acked_raw = parser, raw, raw
        self._acked_parser = copy.deepcopy(parser)
        self._pending_display = ""
        self._last_sent_text = self._display_payload(raw)

    def _reset_block_segment(self):
        if self._stream_failed:
            return
        self._reset_segment_state()
        self._parser = StreamingMarkdownFilter()
        self._acked_parser = copy.deepcopy(self._parser)
        self._fed_raw = self._acked_raw = self._pending_display = ""


def unsent_block_tail(text, prefix):
    """Trim acknowledged text only at egress, preserving the full stored transcript and media."""
    from gateway.platforms.base import BasePlatformAdapter
    if not prefix or not isinstance(text, str):
        return text
    cleaned = BasePlatformAdapter.strip_media_directives_for_display(text)
    if not cleaned.startswith(prefix):
        return text
    media, _ = BasePlatformAdapter.extract_media(text)
    tail = cleaned[len(prefix):]
    tags = [f'MEDIA:"{path}"' for path, _ in media]
    if any(voice for _, voice in media):
        tags.insert(0, "[[audio_as_voice]]")
    return "\n".join([tail, *tags]) if tags else tail
