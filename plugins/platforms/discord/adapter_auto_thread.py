"""Automatic thread creation and notices for surfaced Discord rate limits.

Leave discord.py's shared REST retry policy untouched. In particular, an outer
cancellation deadline can strand its global-429 gate and block unrelated sends.
"""

import asyncio
import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)


class _AutoThreadRateLimited(Exception):
    """A surfaced rate limit, with a delivery-independent retry estimate."""

    def __init__(self, retry_after: float | None) -> None:
        self.retry_at = None if retry_after is None else math.ceil(time.time() + retry_after)
        self.retry_until = None if retry_after is None else time.monotonic() + retry_after
        super().__init__(f"Discord rate-limited thread creation; estimated retry at {self.retry_at}")

    def notice(self) -> str:
        instruction = "Try again later"
        if self.retry_at is not None:
            instruction = (
                f"Next attempt estimate: <t:{self.retry_at}:R> (<t:{self.retry_at}:F>)"
            )
        return (
            "⚠️ Discord is temporarily rate-limiting Hermes from creating another "
            f"thread. Hermes could not confirm a new thread for this request. {instruction}. "
            "Check for an existing thread before retrying, or continue in one."
        )


async def _delete_failed_seed(seed: Any) -> None:
    if seed is None:
        return
    try:
        await seed.delete()
    except Exception:
        logger.debug("Failed to remove auto-thread fallback seed after rate limit", exc_info=True)


class DiscordAutoThreadMixin:
    async def _auto_create_thread_or_notify(self, message: Any) -> Any | None:
        from .adapter import t

        now = time.monotonic()
        # Local repeat suppression only: discord.py still owns route/global coordination.
        self._auto_thread_cooldowns = {
            channel: entry for channel, entry in getattr(self, "_auto_thread_cooldowns", {}).items()
            if entry[0] > now
        }
        generic_notice = "Hermes could not confirm a new thread. Check for an existing thread before retrying."
        cached = self._auto_thread_cooldowns.get(message.channel.id)
        if cached:
            thread, notice = None, cached[1]
        else:
            try:
                thread = await self._auto_create_thread(message)
            except _AutoThreadRateLimited as error:
                thread, notice = None, error.notice()
                # Another in-flight attempt may have learned the deadline while we waited.
                cached = self._auto_thread_cooldowns.get(message.channel.id)
                if cached and cached[0] > time.monotonic():
                    notice = cached[1]
                elif error.retry_until is not None:
                    self._auto_thread_cooldowns[message.channel.id] = (error.retry_until, notice)
            else:
                notice = t("platform.discord.thread.auto_create_failed")
                generic_notice = t("platform.discord.thread.auto_create_failed_generic")
        if thread:
            return thread
        # Never dispatch in the shared parent when its thread routing target failed (#20243).
        try:
            await message.channel.send(self.warning_text(
                notice, generic_notice,
            ))
        except Exception:
            logger.warning("[%s] Failed to notify user of auto-thread failure", self.name, exc_info=True)
        return None

    async def _auto_create_thread(self, message: Any) -> Any | None:
        """Create a thread, stopping on surfaced 429s without another creation attempt.

        Non-rate-limit failures retain the existing seed fallback and short retry.
        SDK-internal waits and retries are not bounded or cancelled by this helper.
        """
        from .adapter import t

        thread_name = self._derive_auto_thread_name(message.content or "")
        display_name = getattr(getattr(message, "author", None), "display_name", None) or "unknown user"
        reason = f"Auto-threaded from mention by {display_name}"
        last_direct_error: Exception | None = None
        last_fallback_error: Exception | None = None
        for attempt in range(2):
            try:
                thread = await message.create_thread(name=thread_name, auto_archive_duration=1440)
                return self._stamp_auto_thread_name(thread, thread_name)
            except Exception as direct_error:
                if self._is_discord_rate_limit(direct_error):
                    raise _AutoThreadRateLimited(self._extract_discord_retry_after(direct_error)) from direct_error
                logger.debug("[%s] Direct auto-thread creation failed", self.name, exc_info=True)
                last_direct_error = direct_error
                seed_msg = None
                try:
                    seed_msg = await message.channel.send(t("platform.discord.thread.seed_message", name=thread_name))
                    thread = await seed_msg.create_thread(name=thread_name, auto_archive_duration=1440, reason=reason)
                    return self._stamp_auto_thread_name(thread, thread_name)
                except Exception as fallback_error:
                    if self._is_discord_rate_limit(fallback_error):
                        # Capture before cleanup, which can itself wait inside the SDK.
                        rate_limit = _AutoThreadRateLimited(self._extract_discord_retry_after(fallback_error))
                        await _delete_failed_seed(seed_msg)
                        raise rate_limit from fallback_error
                    logger.debug("[%s] Fallback auto-thread creation failed", self.name, exc_info=True)
                    last_fallback_error = fallback_error
                    if attempt == 0:
                        await asyncio.sleep(0.75)
                        continue
        logger.warning(
            "[%s] Auto-thread creation failed after retry. Direct error: %s. Fallback error: %s",
            self.name, last_direct_error, last_fallback_error,
        )
        return None
