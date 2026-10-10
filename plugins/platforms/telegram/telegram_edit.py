"""Telegram message edits outside the streaming-preview body of ``edit_message``: caller-formatted content
(``edit_message(parse_mode=...)``) and the flood-wait and failure settlement it shares with that body."""

import asyncio
import logging
from typing import Optional

from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")

# Transient network errors must not permanently disable progress-message editing.
_TRANSIENT_EDIT_ERROR_MARKERS = (
    "connecterror", "connect error", "connection error", "networkerror", "network error", "timed out", "readtimeout",
    "writetimeout", "server disconnected", "temporarily unavailable", "temporary failure", "httpx")


class TelegramEditMixin:
    """Caller-formatted edits and the shared edit-error settlement for ``TelegramAdapter``."""

    async def _edit_caller_formatted(self, chat_id: str, message_id: str, content: str, parse_mode: str) -> SendResult:
        """Edit with ``content`` sent verbatim in ``parse_mode``: no MarkdownV2 conversion, no rich upgrade, no
        plain-text fallback (rejected markup comes back as a failed SendResult so the caller degrades on its own
        terms) and no split or truncation (markup does not count toward the cap, so an overflow is the API's
        verdict). Not a preview the next edit supersedes: a busy shared slot (#116312) delays it like a send."""
        slot_remaining = self._chat_outbound_slot_remaining(chat_id)
        if slot_remaining > 0:
            logger.debug(
                "[%s] pacing %s edit for chat %s (shared send+edit budget: slot in %.1fs)",
                self.name, parse_mode, chat_id, slot_remaining)
            await asyncio.sleep(slot_remaining)
        self._hold_chat_outbound_slot(chat_id)
        # The formatted text replaces whatever preview the message showed: no stale saturation state.
        self._last_overflow_preview.pop((str(chat_id), str(message_id)), None)
        try:
            await self._edit_text(chat_id, message_id, content, parse_mode)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:  # health: allow BLE001 -- every edit error becomes a SendResult, logged redacted below
            return await self._settle_caller_formatted_edit_error(e, chat_id, message_id, content, parse_mode)

    async def _settle_caller_formatted_edit_error(
        self, e: Exception, chat_id: str, message_id: str, content: str, parse_mode: str,
    ) -> SendResult:
        """``_edit_caller_formatted``'s failure path: like the preview path's, minus split and truncation."""
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        err_str = str(e).lower()
        if "not modified" in err_str:
            return SendResult(success=True, message_id=message_id)
        if "message_too_long" in err_str or "too long" in err_str:
            safe_error = _redact_telegram_error_text(e)
            logger.warning(
                "[%s] %s edit of message %s too long, not split (caller-formatted): %s",
                self.name, parse_mode, message_id, safe_error)
            return SendResult(success=False, error=safe_error)
        flooded = await self._retry_edit_after_flood(e, err_str, chat_id, message_id, content, parse_mode)
        if flooded is not None:
            return flooded
        return self._edit_failure_result(e, err_str, message_id)

    async def _retry_edit_after_flood(
        self, e: Exception, err_str: str, chat_id: str, message_id: str, content: str, parse_mode: Optional[str] = None,
    ) -> Optional[SendResult]:
        """Flood control: short waits retry inline (same text, same ``parse_mode``); long waits fail immediately so
        streaming falls back to a normal final send instead of a clipped partial. None: ``e`` is not a flood refusal."""
        from plugins.platforms.telegram.adapter import (
            _FLOOD_INLINE_WAIT_CAP_SECS, _flood_cap_result, _redact_telegram_error_text)

        retry_after = getattr(e, "retry_after", None)
        if retry_after is None and "retry after" not in err_str:
            return None
        wait = retry_after if retry_after else 1.0
        if wait > _FLOOD_INLINE_WAIT_CAP_SECS:
            # Log AFTER the cap check: "waiting 33.0s" followed by no wait misled an investigation.
            logger.warning(
                "[%s] Telegram flood control, refusing edit (retry_after %.1fs > %.0fs inline cap)",
                self.name, wait, _FLOOD_INLINE_WAIT_CAP_SECS)
            return _flood_cap_result(wait)
        logger.warning("[%s] Telegram flood control, waiting %.1fs", self.name, wait)
        await asyncio.sleep(wait)
        try:
            await self._edit_text(chat_id, message_id, content, parse_mode)
            return SendResult(success=True, message_id=message_id)
        except Exception as retry_err:  # health: allow BLE001 -- moved verbatim from edit_message; a failed retry is a SendResult
            safe_retry_error = _redact_telegram_error_text(retry_err)
            logger.error("[%s] Edit retry failed after flood wait: %s", self.name, safe_retry_error)
            retry_wait = getattr(retry_err, "retry_after", None)
            if retry_wait is not None or "retry after" in str(retry_err).lower():
                # Still flooded after the inline wait, and typically for much longer than the
                # first refusal asked for. Fail closed canonically so the ledger arms its
                # timer on this delay rather than storing the platform's raw wording, which
                # it would read as an ordinary failure and never redeliver.
                return _flood_cap_result(
                    float(retry_wait) if retry_wait is not None else wait)
            return SendResult(success=False, error=safe_retry_error)

    def _edit_failure_result(self, e: Exception, err_str: str, message_id: str) -> SendResult:
        """The terminal failed SendResult of an edit: retryable for transient network errors, else final."""
        from plugins.platforms.telegram.adapter import _redact_telegram_error_text

        safe_error = _redact_telegram_error_text(e)
        if any(m in err_str for m in _TRANSIENT_EDIT_ERROR_MARKERS):
            logger.warning("[%s] Transient network error editing message %s (will retry): %s", self.name, message_id, safe_error)
            return SendResult(success=False, error=safe_error, retryable=True)
        logger.error("[%s] Failed to edit Telegram message %s: %s", self.name, message_id, safe_error)
        return SendResult(success=False, error=safe_error)
