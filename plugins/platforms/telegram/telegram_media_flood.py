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
import contextlib
import logging
import os
import re
import shutil
import tempfile
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")

_FLOOD_TEXT_RE = re.compile(r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)", re.IGNORECASE)

# How far the current upload has already gone through flood recovery: 0 fresh, 1 inline retry
# done, 2 running as the deferred re-upload. Context-local, so a nested media call (send_voice →
# send_document) inherits it and recovery can never loop.
_FLOOD_STAGE: ContextVar[int] = ContextVar("telegram_media_flood_stage", default=0)

Redo = Callable[[Optional[str]], Awaitable[SendResult]]

# How many times a deferred re-upload may wait out a flood window that a sibling upload re-armed
# before its request, so a window kept open by other traffic cannot park it forever.
_MAX_WINDOW_WAITS = 3


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


def _keep_private_copy(path: str) -> Optional[str]:
    """Copy ``path`` into a private temp dir for a deferred upload: callers (auto-TTS, the image
    document fallback) delete their file as soon as the send returns. Blocking: use ``to_thread``."""
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="hermes-tg-flood-")
        kept = os.path.join(tmp_dir, os.path.basename(path))
        shutil.copy2(path, kept)
        return kept
    except OSError as exc:
        logger.warning("Could not keep a copy of %s for the deferred upload: %s", os.path.basename(path), exc)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


async def _owned_private_copy(source: str) -> Optional[str]:
    """``_keep_private_copy`` on a worker thread that cancellation cannot orphan: cancelling the
    awaiting send does not stop the thread, so the copy it still finishes is removed when it lands."""
    copying = asyncio.ensure_future(asyncio.to_thread(_keep_private_copy, source))
    try:
        return await asyncio.shield(copying)
    except asyncio.CancelledError:
        copying.add_done_callback(_discard_finished_copy)
        raise


def _discard_finished_copy(copying: "asyncio.Future[Optional[str]]") -> None:
    if not copying.cancelled() and copying.exception() is None and (kept := copying.result()):
        shutil.rmtree(os.path.dirname(kept), ignore_errors=True)


class _KnownFloodWindow(Exception):
    """A media request refused locally because the chat's flood window is still open (no API call)."""

    def __init__(self, wait: float) -> None:
        super().__init__(f"Telegram flood window open; retry in {wait:.1f} seconds")
        self.retry_after = wait


class TelegramMediaFloodMixin:
    """Native local-file upload shell + flood recovery shared by every media sender."""

    async def _send_local_file(
        self, label: str, path: str, chat_id, reply_to, metadata, media_key: str, build_kwargs, on_error,
        redo: Optional[Redo] = None, source: Optional[str] = None,
    ) -> SendResult:
        """Existence check, open, send with routing. ``build_kwargs(f)`` supplies the media kwargs;
        ``await on_error(exc)`` handles a failure flood recovery did not resolve. ``redo(path)`` repeats
        the whole public send from ``source`` (or a private copy of it): a caller whose upload or
        thumbnail is a temp file it deletes on return must pass both, so a deferred re-upload rebuilds
        it instead of reading a removed path."""
        if not self._bot:
            return SendResult(success=False, error="Not connected")
        if not await asyncio.to_thread(os.path.exists, path):
            return SendResult(success=False, error=self._missing_media_path_error(label, path))
        if redo is None:
            def redo(p: Optional[str]) -> Awaitable[SendResult]:
                return self._send_local_file(label, p or path, chat_id, reply_to, metadata, media_key, build_kwargs, on_error)
            source = path

        what = f"{media_key} {os.path.basename(path)}"
        try:
            with await asyncio.to_thread(open, path, "rb") as f:
                msg = await self._send_media(
                    getattr(self._bot, f"send_{media_key}"), chat_id, reply_to, metadata, media_key,
                    reset_media=lambda: f.seek(0), **build_kwargs(f))
            return SendResult(success=True, message_id=str(msg.message_id))
        except Exception as e:  # health: allow BLE001 -- upload boundary: flood recovery or the caller's on_error (which logs) owns every failure
            recovered = await self._media_flood_recover(e, chat_id, redo, what, source, metadata)
            return recovered if recovered is not None else await on_error(e)

    async def _media_flood_recover(
        self, error: Exception, chat_id, redo: Redo, what: str, source: Optional[str] = None, metadata=None,
    ) -> Optional[SendResult]:
        """Recover a media send refused by flood control. ``None``: not a flood, or recovery is
        exhausted (the caller then reports the failure; a deferred re-upload first sleeps out the
        new window so that report can land)."""
        from plugins.platforms.telegram.adapter import _FLOOD_INLINE_WAIT_CAP_SECS, _flood_cap_result
        wait = flood_wait_seconds(error)
        if wait is None:
            return None
        stage = _FLOOD_STAGE.get()
        if isinstance(error, _KnownFloodWindow):  # refused locally: the window is already armed
            return await self._defer_media_upload(redo, source, wait, _flood_cap_result(wait), what, (chat_id, metadata))
        if stage == 0 and wait <= _FLOOD_INLINE_WAIT_CAP_SECS:
            logger.warning("[%s] Telegram flood control on %s, retrying in %.1fs", self.name, what, wait)
            if wait > 0:  # siblings queued on the chat lock must not fire into the window this one waits out
                self._record_send_flood_cooldown(chat_id, wait)
            await asyncio.sleep(wait)
            token = _FLOOD_STAGE.set(1)
            try:
                return await redo(source)
            finally:
                _FLOOD_STAGE.reset(token)
        refused = self._record_send_flood_cooldown(chat_id, wait)
        if stage >= 2:
            await asyncio.sleep(wait)
            return None
        return await self._defer_media_upload(redo, source, wait, refused, what, (chat_id, metadata))

    @contextlib.asynccontextmanager
    async def _media_send_slot(self, chat_id):
        """The chat's send lock for one native media request, gated on the chat's flood window INSIDE
        the lock, right before the request: a check made before the lock races a sibling refused while
        this one queued. A fresh upload inside a window raises ``_KnownFloodWindow`` without a request,
        so its caller's flood recovery defers it. A retry waits the window out with the lock released,
        without spending its remaining attempt: the inline one only up to the inline cap (past it, it
        is deferred too), the deferred re-upload at most ``_MAX_WINDOW_WAITS`` times."""
        from plugins.platforms.telegram.adapter import _FLOOD_INLINE_WAIT_CAP_SECS
        stage = _FLOOD_STAGE.get()
        waits = 0
        while True:
            async with self._chat_send_lock(chat_id):
                wait = self._send_flood_cooldown_remaining(chat_id)
                if wait is None or (stage >= 2 and waits >= _MAX_WINDOW_WAITS):
                    yield
                    return
                if stage == 0 or (stage == 1 and (wait > _FLOOD_INLINE_WAIT_CAP_SECS or waits >= _MAX_WINDOW_WAITS)):
                    raise _KnownFloodWindow(wait)
            waits += 1
            await asyncio.sleep(wait)

    async def _defer_media_upload(
        self, redo: Redo, source: Optional[str], wait: float, refused: SendResult, what: str, target: tuple,
    ) -> SendResult:
        """Schedule ``redo()`` after the flood window (tracked, so shutdown cancels it) on a private copy
        of ``source``; returns ``refused`` flagged so the dispatcher leaves the failure notice to it. The
        task owns that notice on every path: ``redo``'s own failure handling posts it, and where ``redo``
        cannot run (the copy failed and the caller deleted its file, or it raised) the task posts it to
        ``target`` (``(chat_id, metadata)``) itself."""
        logger.warning("[%s] Telegram flood control on %s; re-uploading natively in %.0fs", self.name, what, wait)
        kept = await _owned_private_copy(source) if source else None

        async def _later() -> None:
            await asyncio.sleep(wait)
            _FLOOD_STAGE.set(2)  # this task's own context
            upload = kept or source
            try:
                result = await redo(upload)
            except Exception:
                logger.exception("[%s] Deferred upload of %s failed", self.name, what)
                result = None
            if result is not None and result.success:
                logger.info("[%s] Deferred upload of %s: delivered", self.name, what)
                return
            # redo's own failure handling posts the notice once it reached the upload; a file that is
            # gone (no private copy, and the caller deleted its own) or a raise never got that far.
            if result is None or (upload and not await asyncio.to_thread(os.path.exists, upload)):
                chat_id, metadata = target
                logger.error("[%s] Deferred upload of %s lost: %s", self.name, what,
                             "file gone" if result is not None else "raised")
                if (still := self._send_flood_cooldown_remaining(chat_id)) is not None:
                    await asyncio.sleep(still)  # the window can outlast ``wait`` (1 s floor): land the notice
                await self._notify_media_delivery_failure(
                    chat_id, source or "", is_voice=what.startswith("voice "), metadata=metadata)
            else:
                logger.error("[%s] Deferred upload of %s: %s", self.name, what, result.error)

        task = asyncio.get_running_loop().create_task(_later())
        if kept:  # a done callback also runs when shutdown cancels the task before it started
            task.add_done_callback(lambda _t: shutil.rmtree(os.path.dirname(kept), ignore_errors=True))
        tracked = getattr(self, "_background_tasks", None)
        if isinstance(tracked, set):
            tracked.add(task)
            task.add_done_callback(tracked.discard)
        raw = dict(refused.raw_response) if isinstance(refused.raw_response, dict) else {}
        refused.raw_response = {**raw, "failure_notice_sent": True, "deferred_upload": True}
        return refused
