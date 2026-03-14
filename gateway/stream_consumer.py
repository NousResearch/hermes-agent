# ============================================================
# Gateway Streaming Consumer
# ============================================================
# Bridges the sync stream_delta_callback from AIAgent to the
# async Telegram adapter streaming methods (draft / edit).
# Follows the same queue-based pattern as tool_progress_callback.
# ============================================================

import asyncio
import queue
import time
import logging
import random

logger = logging.getLogger(__name__)


def _cfg(cfg, key, default):
    """Read a field from either a StreamingConfig dataclass or a plain dict."""
    return getattr(cfg, key) if hasattr(cfg, key) else cfg.get(key, default)


class GatewayStreamConsumer:
    """Manages streaming token delivery from agent thread to Telegram.

    Usage inside _run_agent:
        consumer = GatewayStreamConsumer(adapter, chat_id, streaming_cfg, metadata, loop)
        # Pass consumer.on_delta as stream_delta_callback to AIAgent
        # Start consumer.run() as asyncio task
        # On completion, check consumer.already_sent
    """

    def __init__(self, adapter, chat_id, streaming_cfg, metadata, loop):
        self.adapter = adapter
        self.chat_id = chat_id
        self.metadata = metadata
        self.loop = loop

        # Config — accepts both StreamingConfig dataclass and plain dict
        self.enabled          = _cfg(streaming_cfg, "enabled",          True)
        self.transport        = _cfg(streaming_cfg, "transport",        "edit")   # auto|draft|edit
        self.buffer_threshold = _cfg(streaming_cfg, "buffer_threshold", 20)
        self.edit_interval    = _cfg(streaming_cfg, "edit_interval",    0.15)
        self.cursor           = _cfg(streaming_cfg, "cursor",           " \u2589")

        # State
        self._queue          = queue.Queue()
        self._buffer         = ""
        self._sent_text      = ""
        self._msg_id         = None
        self._draft_id       = None
        self._draft_ok       = None   # None = untested, True = working, False = failed
        self._draft_msg_id   = None   # Telegram message_id of the active draft
        self._first_delta    = False
        self._already_sent   = False
        self._last_edit_time = 0.0
        self._draft_interval = 0.1    # min seconds between sendMessageDraft calls
        self._last_draft_time = 0.0

    @property
    def already_sent(self):
        return self._already_sent

    def on_delta(self, text):
        """Sync callback — called from agent thread per token."""
        if not text or not self.enabled:
            return
        if not self._first_delta:
            logger.debug("[stream] first delta received (%d chars)", len(text))
            self._first_delta = True
        self._queue.put(text)

    def finish(self):
        """Signal that streaming is complete."""
        self._queue.put(None)  # sentinel

    async def _try_draft(self, text):
        """Attempt draft transport. Returns True if the update was delivered.

        sendMessageDraft is PRIVATE CHAT ONLY (Telegram Bot API by design).
        Groups/supergroups/channels return Textdraft_peer_invalid permanently.
        Skip the round-trip for negative chat_ids — go straight to edit-mode.
        """
        if not hasattr(self.adapter, "send_draft"):
            return False
        if not getattr(self.adapter, "supports_draft_streaming", False):
            return False

        try:
            if int(self.chat_id) < 0:
                return False  # group/channel → edit-mode
        except (ValueError, TypeError):
            pass

        now = time.monotonic()
        if now - self._last_draft_time < self._draft_interval:
            return self._draft_ok is True  # rate-limited, skip this tick

        if self._draft_id is None:
            self._draft_id = random.randrange(1, 2**31)

        ok = await self.adapter.send_draft(
            self.chat_id, self._draft_id, text, metadata=self.metadata)
        if ok:
            self._last_draft_time = time.monotonic()
            # Cache message_id from first successful call so finalize_draft can
            # edit the existing draft in-place rather than sending a new message.
            if self._draft_msg_id is None and isinstance(ok, str):
                self._draft_msg_id = ok
        return bool(ok)

    async def _send_or_edit(self, text, *, with_cursor=True):
        """Send first message or throttle-edit an existing one (edit transport)."""
        display = text + self.cursor if with_cursor else text
        now = time.monotonic()

        if self._msg_id is not None:
            if with_cursor and (now - self._last_edit_time) < self.edit_interval:
                return  # skip — will catch up on next tick
            result = await self.adapter.edit_message_raw(
                self.chat_id, self._msg_id, display)
            if result.success:
                self._last_edit_time = now
        else:
            result = await self.adapter.send_raw(
                self.chat_id, display, metadata=self.metadata)
            if result.success and result.message_id:
                self._msg_id = result.message_id
                self._last_edit_time = now

    async def _finalize_draft(self, final_text):
        """Commit the draft stream as a permanent Telegram message."""
        logger.info("[stream] finalizing via DRAFT chat=%s draft_id=%s (%d chars)",
                    self.chat_id, self._draft_id, len(final_text))
        if hasattr(self.adapter, "finalize_draft"):
            result = await self.adapter.finalize_draft(
                self.chat_id, final_text,
                metadata=self.metadata,
                draft_message_id=self._draft_msg_id,
                draft_id=self._draft_id,
            )
            if result.success:
                self._already_sent = True
                return True
        return False

    async def _finalize_edit(self, final_text):
        """Apply final MarkdownV2 formatting edit on the streaming placeholder.

        If the edit fails, deletes the placeholder BEFORE sending a new
        formatted message so the user never sees two messages in the chat.
        """
        logger.info("[stream] finalizing via EDIT-MODE chat=%s msg_id=%s (%d chars)",
                    self.chat_id, self._msg_id, len(final_text))
        if self._msg_id:
            result = await self.adapter.edit_message(
                self.chat_id, self._msg_id, final_text)
            if result.success:
                self._already_sent = True
                return True
            # Edit failed — delete placeholder before fallback so only one
            # message appears in chat (Bug 2 fix).
            logger.warning("[stream] final edit failed (%s), deleting placeholder "
                           "before fallback send", result.error)
            try:
                await self.adapter.delete_message(self.chat_id, self._msg_id)
            except Exception as del_err:
                logger.warning("[stream] could not delete placeholder: %s", del_err)
            self._msg_id = None

        # Fallback: send a new formatted message
        result = await self.adapter.send(
            chat_id=self.chat_id,
            content=final_text,
            metadata=self.metadata,
        )
        if result.success:
            self._already_sent = True
            return True
        return False

    async def run(self):
        """Main async consumer loop. Run as an asyncio task alongside the agent."""
        logger.info("[stream] consumer started enabled=%s transport=%s",
                    self.enabled, self.transport)
        if not self.enabled:
            return

        use_draft = self.transport in ("auto", "draft")
        use_edit  = self.transport in ("auto", "edit")

        try:
            _chat_type = "private-DM" if int(self.chat_id) > 0 else "group/channel"
        except (ValueError, TypeError):
            _chat_type = "unknown"
        logger.info("[stream] ▶ START chat=%s type=%s transport=%s threshold=%d",
                    self.chat_id, _chat_type, self.transport, self.buffer_threshold)

        try:
            while True:
                # Drain all available tokens from the agent thread
                got_sentinel = False
                while True:
                    try:
                        item = self._queue.get_nowait()
                        if item is None:
                            got_sentinel = True
                            break
                        self._buffer += item
                    except queue.Empty:
                        break

                # Flush when enough new chars have accumulated or stream is done
                new_chars = len(self._buffer) - len(self._sent_text)
                if self._buffer and (new_chars >= self.buffer_threshold or got_sentinel):
                    display_text = self._buffer

                    if not got_sentinel:
                        # Intermediate update
                        if use_draft and self._draft_ok is not False:
                            ok = await self._try_draft(display_text + self.cursor)
                            if ok:
                                if self._draft_ok is None:
                                    logger.info("[stream] ✅ DRAFT confirmed chat=%s",
                                                self.chat_id)
                                self._draft_ok = True
                            else:
                                if self._draft_ok is None:
                                    logger.info("[stream] draft failed, switching to "
                                                "EDIT-MODE chat=%s", self.chat_id)
                                self._draft_ok = False
                                if use_edit:
                                    await self._send_or_edit(display_text)
                        elif use_edit:
                            await self._send_or_edit(display_text)

                        self._sent_text = display_text
                    else:
                        break  # sentinel received — proceed to finalization

                if got_sentinel:
                    break

                await asyncio.sleep(0.03)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("[stream] consumer error: %s", e)

        # ── Finalization ──────────────────────────────────────────────────────
        final_text = self._buffer
        if not final_text:
            logger.info("[stream] ◀ END chat=%s (empty buffer, nothing to send)",
                        self.chat_id)
            return

        if self._draft_ok:
            ok = await self._finalize_draft(final_text)
            if not ok and use_edit:
                await self._finalize_edit(final_text)
        elif self._msg_id:
            await self._finalize_edit(final_text)
        else:
            # Short/instant response: buffer filled before any intermediate update.
            # Still route through the configured transport before plain send.
            logger.info("[stream] short response — finalizing directly chat=%s",
                        self.chat_id)
            if use_draft:
                ok = await self._try_draft(final_text)
                if ok:
                    if await self._finalize_draft(final_text):
                        _transport = "DRAFT"
                        logger.info("[stream] ◀ END chat=%s transport=%s already_sent=%s",
                                    self.chat_id, _transport, self._already_sent)
                        return
            if use_edit:
                await self._finalize_edit(final_text)

        _transport = "DRAFT" if self._draft_ok else ("EDIT" if self._msg_id is not None else "SEND")
        logger.info("[stream] ◀ END chat=%s transport=%s already_sent=%s",
                    self.chat_id, _transport, self._already_sent)

    async def run_with_timeout(self, timeout=300):
        """Run with a safety timeout to prevent silent hangs on adapter failures."""
        try:
            await asyncio.wait_for(self.run(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[stream] consumer timed out after %ds chat=%s",
                           timeout, self.chat_id)
