"""Flood-aware native media uploads for the Telegram adapter (#64137, #125857).

A flood-refused ``sendDocument``/``sendPhoto``/``sendVideo``/``sendVoice`` used to fall straight to
the base adapter's text notice ("Couldn't deliver the file attachment") milliseconds after Telegram
said "retry in N seconds": the file was dropped and the notice itself fired into the open penalty.
Now a short wait (≤ the inline cap) is slept and the upload retried once, as ``send()`` does; a
longer one arms the chat's send cooldown and schedules ONE native re-upload after the window. The
failure notice is posted only if that re-upload fails too, by which time it can be delivered. The
caller gets the canonical ``flood_control:<s>`` result flagged ``failure_notice_sent`` (the notice is
owned by the deferred upload), so the base dispatcher does not post a second one into the penalty.
"""

import asyncio
import logging
import os
import re
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")

_FLOOD_TEXT_RE = re.compile(r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)

# How far the current upload has already gone through flood recovery: 0 fresh, 1 inline retry
# done, 2 running as the deferred re-upload. Context-local, so a nested media call (send_voice →
# send_document) inherits it and recovery can never loop.
_FLOOD_STAGE: ContextVar[int] = ContextVar("telegram_media_flood_stage", default=0)

Redo = Callable[[], Awaitable[SendResult]]


def flood_wait_seconds(error: Exception) -> Optional[float]:
    """Seconds Telegram asked this call to wait (``RetryAfter``), or ``None`` for any other error."""
    retry_after: Any = getattr(error, "retry_after", None)
    if retry_after is not None:
        if hasattr(retry_after, "total_seconds"):  # newer PTB hands out a timedelta
            retry_after = retry_after.total_seconds()
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            return None
    match = _FLOOD_TEXT_RE.search(str(error))
    return float(match.group(1)) if match else None


class TelegramMediaFloodMixin:
    """Native local-file upload shell + flood recovery shared by every media sender."""

    async def _send_local_file(
        self, label: str, path: str, chat_id, reply_to, metadata, media_key: str, build_kwargs, on_error,
        redo: Optional[Redo] = None,
    ) -> SendResult:
        """Existence check, open, send with routing. ``build_kwargs(f)`` supplies the media kwargs;
        ``await on_error(exc)`` handles a failure flood recovery did not resolve. ``redo`` repeats
        the whole public send: a caller whose upload or thumbnail is a temp file it deletes on return
        must pass one, so a deferred re-upload rebuilds it instead of reading a removed path."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        if not await asyncio.to_thread(os.path.exists, path):
            return SendResult(success=False, error=self._missing_media_path_error(label, path))
        if redo is None:
            def redo() -> Awaitable[SendResult]:
                return self._send_local_file(label, path, chat_id, reply_to, metadata, media_key, build_kwargs, on_error)

        what = f"{media_key} {os.path.basename(path)}"
        deferred = self._media_flood_precheck(chat_id, redo, what)
        if deferred is not None:
            return deferred
        try:
            with await asyncio.to_thread(open, path, "rb") as f:
                msg = await self._send_media(
                    getattr(self._bot, f"send_{media_key}"), chat_id, reply_to, metadata, media_key,
                    reset_media=lambda: f.seek(0), **build_kwargs(f))
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:  # health: allow BLE001 -- upload boundary: flood recovery or the caller's on_error (which logs) owns every failure
            recovered = await self._media_flood_recover(e, chat_id, redo, what)
            return recovered if recovered is not None else await on_error(e)

    def _media_flood_precheck(self, chat_id, redo: Redo, what: str) -> Optional[SendResult]:
        """Inside a known flood window, defer the upload without spending a request on it."""
        from plugins.platforms.telegram.adapter import _flood_cap_result
        cooldown = self._send_flood_cooldown_remaining(chat_id)
        if cooldown is None or _FLOOD_STAGE.get() >= 2:
            return None
        return self._defer_media_upload(redo, cooldown, _flood_cap_result(cooldown), what)

    async def _media_flood_recover(self, error: Exception, chat_id, redo: Redo, what: str) -> Optional[SendResult]:
        """Recover a media send refused by flood control. ``None``: not a flood, or recovery is
        exhausted (the caller then reports the failure; a deferred re-upload first sleeps out the
        new window so that report can land)."""
        from plugins.platforms.telegram.adapter import _FLOOD_INLINE_WAIT_CAP_SECS
        wait = flood_wait_seconds(error)
        if wait is None:
            return None
        stage = _FLOOD_STAGE.get()
        if stage == 0 and wait <= _FLOOD_INLINE_WAIT_CAP_SECS:
            logger.warning("[%s] Telegram flood control on %s, retrying in %.1fs", self.name, what, wait)
            await asyncio.sleep(wait)
            token = _FLOOD_STAGE.set(1)
            try:
                return await redo()
            finally:
                _FLOOD_STAGE.reset(token)
        refused = self._record_send_flood_cooldown(chat_id, wait)
        if stage >= 2:
            await asyncio.sleep(wait)
            return None
        return self._defer_media_upload(redo, wait, refused, what)

    def _defer_media_upload(self, redo: Redo, wait: float, refused: SendResult, what: str) -> SendResult:
        """Schedule ``redo()`` after the flood window (tracked, so shutdown cancels it); returns
        ``refused`` flagged so the dispatcher leaves the failure notice to the deferred upload."""
        logger.warning("[%s] Telegram flood control on %s; re-uploading natively in %.0fs", self.name, what, wait)

        async def _later() -> None:
            await asyncio.sleep(wait)
            _FLOOD_STAGE.set(2)  # this task's own context
            try:
                result = await redo()
            except Exception:
                logger.exception("[%s] Deferred upload of %s failed", self.name, what)
                return
            (logger.info if result.success else logger.error)(
                "[%s] Deferred upload of %s: %s", self.name, what, "delivered" if result.success else result.error)

        task = asyncio.get_running_loop().create_task(_later())
        tracked = getattr(self, "_background_tasks", None)
        if isinstance(tracked, set):
            tracked.add(task)
            task.add_done_callback(tracked.discard)
        raw = dict(refused.raw_response) if isinstance(refused.raw_response, dict) else {}
        refused.raw_response = {**raw, "failure_notice_sent": True, "deferred_upload": True}
        return refused
