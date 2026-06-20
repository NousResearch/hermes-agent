"""Streaming card controller for Feishu CardKit.

Manages the lifecycle of a single streaming card:
  idle → creating → streaming → completed

Mirrors the openclaw-lark ``StreamingCardController`` (TypeScript) which
uses a state machine and a FlushController for throttled API updates.

The controller is owned by the Feishu adapter on a per-chat basis.  The
adapter calls:
  - ``start(chat_id, reply_to)``      → create card, send to chat
  - ``stream_chunk(text)``             → throttled content update
  - ``finalize()``                     → final content push, close streaming

Throttle: minimum 200ms between API calls.  Chunks arriving faster are
buffered and flushed as a single batch on the next timer tick.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
import traceback
from typing import Any, Optional

from gateway.platforms.feishu_cardkit import (
    STREAMING_ELEMENT_ID,
    build_streaming_card_json,
    cardkit_available,
    create_card_entity,
    send_card_by_id,
    set_card_streaming_mode,
    stream_card_content,
)

logger = logging.getLogger("gateway.platforms.feishu_streaming_card")

# Minimum interval between CardKit API calls (seconds).
_THROTTLE_INTERVAL = 0.2


class _Phase(enum.Enum):
    IDLE = "idle"
    CREATING = "creating"
    STREAMING = "streaming"
    COMPLETED = "completed"


class StreamingCardController:
    """Stateful controller for one streaming card in one chat.

    Thread-safety: all public methods are ``async`` and expected to be
    called from the same asyncio event loop (the adapter's loop).  No
    explicit locking is needed beyond the single-threaded event loop.
    """

    def __init__(self, client: Any, *, adapter_name: str = "Feishu") -> None:
        self._client = client
        self._adapter_name = adapter_name
        self._phase = _Phase.IDLE

        # Card identity
        self._card_id: Optional[str] = None
        self._message_id: Optional[str] = None
        self._chat_id: Optional[str] = None

        # Sequence counter — monotonically increasing, starting at 0.
        self._sequence: int = 0

        # Accumulated text for the streaming element.
        self._accumulated_text: str = ""

        # Throttle state
        self._last_flush_time: float = 0.0
        self._dirty: bool = False
        self._flush_task: Optional[asyncio.Task] = None

        # Auto-finalize timer:  closes the streaming card after a period of
        # inactivity so the blinking cursor does not stay forever.
        self._idle_finalize_task: Optional[asyncio.Task] = None

        # Error tracking — if cardkit operations fail, we degrade gracefully
        # rather than blocking the entire response pipeline.
        self._failed: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        return self._phase.value

    @property
    def card_id(self) -> Optional[str]:
        return self._card_id

    @property
    def message_id(self) -> Optional[str]:
        return self._message_id

    @property
    def is_streaming(self) -> bool:
        return self._phase == _Phase.STREAMING

    @property
    def is_completed(self) -> bool:
        return self._phase == _Phase.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self._failed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        chat_id: str,
        *,
        reply_to: Optional[str] = None,
        initial_content: str = "",
        title: str = "Hermes",
    ) -> Optional[str]:
        """Create a streaming card, deliver it to the chat, and enter
        the streaming phase.

        Returns the ``message_id`` on success, or ``None`` on failure
        (caller should fall back to normal message sending).
        """
        if self._phase != _Phase.IDLE:
            logger.warning(
                "[%s] StreamingCardController.start() called in phase=%s",
                self._adapter_name, self._phase.value,
            )
            return None

        if not cardkit_available():
            logger.debug("[%s] CardKit SDK not available", self._adapter_name)
            self._failed = True
            return None

        self._phase = _Phase.CREATING
        self._chat_id = chat_id
        self._accumulated_text = initial_content

        try:
            # Step 1: Build card JSON and create the card entity
            card_json = build_streaming_card_json(
                title=title,
                initial_content=initial_content,
            )
            card_id = await create_card_entity(self._client, card_json)
            if not card_id:
                self._failed = True
                self._phase = _Phase.IDLE
                return None
            self._card_id = card_id

            # Streaming mode is already enabled via "streaming_mode": true
            # in the card JSON config — no separate settings() call needed.
            self._sequence = 1

            # Step 2: Send the card to the chat
            message_id = await send_card_by_id(
                self._client,
                receive_id=chat_id,
                card_id=card_id,
                reply_to_message_id=reply_to,
            )
            if not message_id:
                # Card created but couldn't deliver — mark failed so
                # the adapter falls back.
                self._failed = True
                self._phase = _Phase.IDLE
                return None

            self._message_id = message_id
            self._phase = _Phase.STREAMING
            self._last_flush_time = time.monotonic()

            logger.info(
                "[%s] Streaming card created: card_id=%s msg=%s",
                self._adapter_name, card_id, message_id,
            )
            return message_id

        except Exception as exc:
            logger.warning(
                "[%s] StreamingCardController.start() error: %s\n%s",
                self._adapter_name, exc, traceback.format_exc(),
            )
            self._failed = True
            self._phase = _Phase.IDLE
            return None

    async def stream_chunk(self, text: str, *, finalize: bool = False) -> bool:
        """Accumulate text and schedule a throttled content update.

        If ``finalize`` is True, the full accumulated text is pushed
        immediately and the card is transitioned to completed state
        (streaming mode off).
        """
        if self._failed:
            return False

        if finalize:
            return await self._do_finalize(text)

        if self._phase not in (_Phase.STREAMING,):
            # Silently ignore chunks outside streaming phase — the caller
            # might still be sending after a finalize.
            return False

        self._accumulated_text = text
        self._dirty = True
        logger.info(
            "[%s] stream_chunk: len=%d dirty=%s",
            self._adapter_name, len(text), self._dirty,
        )

        # Reset idle-finalize timer on every new chunk
        self._reset_idle_finalize()

        # Throttle check
        now = time.monotonic()
        elapsed = now - self._last_flush_time
        if elapsed >= _THROTTLE_INTERVAL:
            # Enough time since last flush — flush immediately
            return await self._flush()

        # Schedule a delayed flush if one isn't already pending
        if self._flush_task is None or self._flush_task.done():
            remaining = _THROTTLE_INTERVAL - elapsed
            self._flush_task = asyncio.ensure_future(self._delayed_flush(remaining))

        return True  # buffered

    async def finalize(self, final_text: Optional[str] = None) -> bool:
        """Finalize the streaming card: push last content and disable
        streaming mode.
        """
        text = final_text if final_text is not None else self._accumulated_text
        return await self._do_finalize(text)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _delayed_flush(self, delay: float) -> None:
        """Wait ``delay`` seconds then flush if dirty."""
        await asyncio.sleep(delay)
        if self._dirty and self._phase == _Phase.STREAMING:
            await self._flush()

    # ------------------------------------------------------------------
    # Idle auto-finalize
    # ------------------------------------------------------------------

    _IDLE_FINALIZE_SECONDS = 10  # close card after 10s of no updates

    def _reset_idle_finalize(self) -> None:
        """Cancel any pending idle-finalize and reschedule."""
        if self._idle_finalize_task and not self._idle_finalize_task.done():
            self._idle_finalize_task.cancel()
        self._idle_finalize_task = asyncio.ensure_future(
            self._auto_finalize(self._IDLE_FINALIZE_SECONDS)
        )

    async def _auto_finalize(self, delay: float) -> None:
        """Finalize the card after ``delay`` seconds of silence."""
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if self._phase == _Phase.STREAMING and not self._failed:
            logger.info(
                "[%s] Auto-finalizing streaming card after %.0fs idle: card_id=%s",
                self._adapter_name, delay, self._card_id,
            )
            try:
                await self._do_finalize(self._accumulated_text)
            except Exception as exc:
                logger.warning("[%s] Auto-finalize error: %s", self._adapter_name, exc)

    async def _flush(self) -> bool:
        """Push the current accumulated text to the card element."""
        if not self._card_id or not self._dirty:
            return True

        self._dirty = False
        self._last_flush_time = time.monotonic()
        seq = self._sequence
        self._sequence += 1
        text = self._accumulated_text
        logger.info(
            "[%s] _flush: seq=%d len=%d text[-50:]=%s",
            self._adapter_name, seq, len(text), text[-50:] if len(text) > 50 else text,
        )

        ok = await stream_card_content(
            self._client,
            card_id=self._card_id,
            element_id=STREAMING_ELEMENT_ID,
            content=self._accumulated_text,
            sequence=seq,
        )
        if not ok:
            logger.debug(
                "[%s] Content flush failed seq=%d", self._adapter_name, seq,
            )
        return ok

    async def _do_finalize(self, text: str) -> bool:
        """Push final content and close streaming mode."""
        if self._phase == _Phase.COMPLETED:
            return True

        if self._failed or not self._card_id:
            self._phase = _Phase.COMPLETED
            return False

        self._accumulated_text = text
        self._dirty = False

        logger.info(
            "[%s] _do_finalize: seq=%d len=%d text[-50:]=%s",
            self._adapter_name, self._sequence, len(text),
            text[-50:] if len(text) > 50 else text,
        )

        # Cancel any pending delayed flush
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        try:
            # Push final content
            seq = self._sequence
            self._sequence += 1
            await stream_card_content(
                self._client,
                card_id=self._card_id,
                element_id=STREAMING_ELEMENT_ID,
                content=text,
                sequence=seq,
            )

            # Disable streaming mode
            seq = self._sequence
            self._sequence += 1
            await set_card_streaming_mode(
                self._client,
                card_id=self._card_id,
                streaming=False,
                sequence=seq,
            )

            self._phase = _Phase.COMPLETED
            logger.info(
                "[%s] Streaming card finalized: card_id=%s",
                self._adapter_name, self._card_id,
            )
            return True

        except Exception as exc:
            logger.warning(
                "[%s] StreamingCard finalize error: %s", self._adapter_name, exc,
            )
            self._phase = _Phase.COMPLETED
            self._failed = True
            return False
