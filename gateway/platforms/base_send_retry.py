"""The shared send retry / plain-text-fallback path for platform adapters.

Split out of ``base.py``: that facade is already over its file-size target and may only shrink
(``scripts/code_health``), and new behaviour belongs in a topical sibling. ``BasePlatformAdapter``
inherits :class:`SendRetryMixin`, so every ``self.`` hook an adapter already overrides — ``send``,
``_send_retry_is_final``, ``_send_plain_fallback``, ``_resume_partial_send``, ``_is_partial_delivery``
and the error classifiers — still resolves through the instance.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any, Optional

from agent.i18n import t
from hermes_cli.observability.shared_metrics_gateway import records_delivery

if TYPE_CHECKING:  # annotation-only; importing it at runtime would cycle through base.py
    from gateway.platforms.base import SendResult

# Same logger name as ``base.py``: the send path is one subsystem, and existing ``caplog``-based
# assertions filter on it by name.
logger = logging.getLogger("gateway.platforms.base")

#: Cap on a server-requested ``retry_after`` this coroutine will sleep inline (see the loop below).
SEND_RETRY_INLINE_WAIT_CAP_SECS = 60.0


class SendRetryMixin:
    @records_delivery
    async def _send_with_retry(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Any = None,
        max_retries: int = 2, base_delay: float = 2.0) -> SendResult:
        """Send with exponential-backoff retry on transient network errors; permanent
        failures fall back to a plain-text send, exhausted retries notify the user."""
        async def _send(text: str) -> SendResult:
            return await self.send(chat_id=chat_id, content=text, reply_to=reply_to, metadata=metadata)

        async def _send_again(previous: SendResult) -> Optional[SendResult]:
            """Retry: the whole payload normally; only the undelivered remainder after a partial split
            delivery (``raw_response["partial_overflow"]``). ``None`` when the adapter cannot resume — the
            caller then keeps the partial failure rather than re-sending the already-visible head."""
            if not self._is_partial_delivery(previous):
                return await _send(content)
            return await self._resume_partial_send(chat_id, previous, reply_to=reply_to, metadata=metadata)

        result = await _send(content)
        if result.success or self._send_retry_is_final(result):
            return result
        error_str = result.error or ""
        # A rate-limited / flood-capped send is transient: it should back off
        # (honoring the server's retry_after when present) rather than fall
        # through to the plain-text fallback, which re-enters the ban and can
        # truncate content.  Gate on the platform-neutral classifier as well so
        # platforms that surface a rate limit without a retry_after field
        # (e.g. Weixin raising a bare RuntimeError) get the same treatment.
        is_rate_limited = self._is_rate_limited_error(error_str)
        is_network = (
            result.retryable
            or is_rate_limited
            or result.retry_after is not None
            or self._is_retryable_error(error_str)
        )
        # Timeouts: not safe to retry (may have delivered) and not a formatting error.
        if not is_network and self._is_timeout_error(error_str):
            return result
        if is_network:
            # A server-requested retry_after (Telegram FloodWait) overrides backoff, once per send.
            server_retry_after = result.retry_after
            for attempt in range(1, max_retries + 1):
                backoff = server_retry_after
                if backoff is None:
                    backoff = base_delay * (2 ** (attempt - 1))
                elif backoff > SEND_RETRY_INLINE_WAIT_CAP_SECS:
                    # Never hold this coroutine open for a long server penalty: a 97-minute
                    # FloodWait slept verbatim once froze inbound on every platform (#91969).
                    # Return the typed failure; the delivery ledger redelivers after the cooldown.
                    logger.error(
                        "[%s] Server asked to retry after %.0fs (> %.0fs inline cap); returning "
                        "typed failure for redelivery instead of sleeping: %s",
                        self.name, backoff, SEND_RETRY_INLINE_WAIT_CAP_SECS, error_str,
                    )
                    return result
                delay = backoff + random.uniform(0, 1)
                server_retry_after = None
                logger.warning("[%s] Send failed (attempt %d/%d, retrying in %.1fs): %s", self.name,
                               attempt, max_retries, delay, error_str)
                await asyncio.sleep(delay)
                resumed = await _send_again(result)
                if resumed is None:
                    logger.warning(
                        "[%s] Split send partly delivered and the remainder cannot be resumed safely; "
                        "not re-sending the whole payload (would duplicate the visible head): %s", self.name, error_str)
                    return result
                result = resumed
                if result.success:
                    logger.info("[%s] Send succeeded on retry %d", self.name, attempt)
                    return result
                error_str = result.error or ""
                if self._send_retry_is_final(result):
                    return result
                if result.retry_after is not None:
                    server_retry_after = result.retry_after
                # The failure kind can change between attempts (a transient error may
                # later surface as a flood/rate-limit, or a rate-limited send may give
                # way to a permanent formatting error). Reclassify from the refreshed
                # error_str on every attempt so the break/continue decision below
                # reflects the current attempt, not a stale first-send classification.
                is_rate_limited = self._is_rate_limited_error(error_str)
                # A retry that ends in a read/write timeout is the same "may have delivered" case the
                # first attempt already refuses to retry or reformat: falling through to the plain-text
                # fallback would re-send the whole payload and the user sees the message twice (#147536).
                # Rate limits keep their own path — the delivery ledger owns that redelivery.
                if not is_rate_limited and self._is_timeout_error(error_str):
                    logger.warning(
                        "[%s] Retry %d/%d ended in a timeout — not re-sending as plain text "
                        "(the payload may already be delivered): %s",
                        self.name, attempt, max_retries, error_str,
                    )
                    return result
                if not (
                    result.retryable
                    or is_rate_limited
                    or result.retry_after is not None
                    or self._is_retryable_error(error_str)
                ):
                    break  # error switched to non-transient — fall through to plain-text fallback
            else:
                # All retries exhausted (loop completed without break) — notify user.
                # If the final failure is a rate-limit / still carries a server
                # retry_after, do NOT send the delivery-failure notice now: the notice
                # send would land inside the same flood penalty and re-enter the ban
                # (a fourth send at t=378 in an [0, 189, 378, 378] sequence). Return
                # the typed failure so the delivery ledger owns redelivery after the
                # cooldown instead — no extra sleep or request needed.
                if self._is_rate_limited_error(error_str) or result.retry_after is not None:
                    logger.error(
                        "[%s] Rate-limited send exhausted retries; returning typed failure "
                        "for redelivery (no notice sent inside active flood penalty): %s",
                        self.name, error_str,
                    )
                    return result
                logger.error("[%s] Failed to deliver response after %d retries: %s", self.name, max_retries, error_str)
                # Not a diagnostic: the requested result itself was lost and this is its only signal.
                notice = t("gateway.notify.delivery_failed_retry")
                try:
                    await _send(notice)
                except Exception as notify_err:
                    logger.debug("[%s] Could not send delivery-failure notice: %s", self.name, notify_err, exc_info=True)
                return result
        # Non-network / post-retry formatting failure: try plain text as fallback. A
        # rate-limited error never reaches here: it classifies as network above and the
        # loop only breaks on a non-transient, non-rate-limited error.
        if self._is_partial_delivery(result):
            # Part of a split payload is already on screen; a plain-text re-send of the whole would duplicate it.
            logger.warning("[%s] Send failed after partial delivery: %s — not re-sending as plain text", self.name, error_str)
            return result
        logger.warning("[%s] Send failed: %s — trying plain-text fallback", self.name, error_str)
        fallback_result = await self._send_plain_fallback(chat_id, content, reply_to=reply_to, metadata=metadata)
        if not fallback_result.success:
            logger.error("[%s] Fallback send also failed: %s", self.name, fallback_result.error)
        return fallback_result
