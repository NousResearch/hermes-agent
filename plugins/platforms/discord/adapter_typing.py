"""Discord "typing…" indicator lifecycle.

One persistent loop per chat posts ``POST /channels/{id}/typing`` every
``_TYPING_POST_INTERVAL_SECONDS`` because a single POST only lasts ~10 s. The loop must die at
turn end: a leaked loop keeps a channel showing "typing…" indefinitely after the reply landed,
which users read as the bot still working.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from discord.http import Route

logger = logging.getLogger("plugins.platforms.discord.adapter")

# Refresh cadence: one POST keeps the indicator alive ~10 s.
_TYPING_POST_INTERVAL_SECONDS = 12
# Hard ceiling for one loop. The loop is normally cancelled by stop_typing() at turn end; this
# bounds the blast radius when a stop is missed at all (a channel that shows "typing…" forever
# is the symptom that surfaced this: a cancelled loop's unconditional map pop orphaned its
# replacement, so nothing could reach the orphan any more).
_TYPING_LOOP_MAX_SECONDS = 3600


class DiscordTypingMixin:
    """``send_typing`` / ``stop_typing`` for the Discord adapter."""

    # Provided by the adapter: the tracked loop per chat, and a per-chat generation counter
    # that stop_typing() bumps so a loop whose map entry was lost still exits.
    _client: Any
    _typing_tasks: dict
    _typing_epoch: dict

    def _extract_discord_retry_after(self, exc: BaseException) -> Optional[float]:  # pragma: no cover
        raise NotImplementedError

    async def send_typing(self, chat_id: str, metadata: Optional[Any] = None) -> None:
        """Start a persistent typing loop (POST typing every 12s; indicator lasts ~10s).
        TYPING_START is unreliable for bots in DMs; 429 sleeps ``retry_after``; CancelledError ends it."""
        if getattr(self, "_typing_tasks", None) is None:
            self._typing_tasks = {}
        if getattr(self, "_typing_epoch", None) is None:
            self._typing_epoch = {}
        if not self._client:
            return
        if chat_id in self._typing_tasks:
            return

        async def _typing_loop() -> None:
            started = asyncio.get_running_loop().time()
            epoch = self._typing_epoch.get(chat_id, 0)
            try:
                while True:
                    if self._typing_epoch.get(chat_id, 0) != epoch:
                        return  # a stop_typing() superseded this loop
                    try:
                        route = Route(
                            "POST", "/channels/{channel_id}/typing", channel_id=chat_id,
                        )
                        await self._client.http.request(route)
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        retry_after = self._extract_discord_retry_after(e)
                        if retry_after is not None:
                            logger.warning(
                                "Typing indicator rate-limited for %s; retrying in %.1fs",
                                chat_id, retry_after,
                            )
                        else:
                            logger.debug("Discord typing indicator failed for %s: %s", chat_id, e)
                            return
                        await asyncio.sleep(retry_after)
                        continue
                    await asyncio.sleep(_TYPING_POST_INTERVAL_SECONDS)
                    if asyncio.get_running_loop().time() - started > _TYPING_LOOP_MAX_SECONDS:
                        logger.warning(
                            "Discord typing loop for %s ran %ss without a stop; ending it "
                            "(a turn-end stop_typing() was missed)",
                            chat_id, _TYPING_LOOP_MAX_SECONDS,
                        )
                        return
            except asyncio.CancelledError:
                pass
            finally:
                # Drop ONLY our own entry. An unconditional pop let a late-firing cancelled loop
                # delete its replacement's map entry, orphaning that task: it kept posting every
                # 12s after the turn ended and no stop_typing could reach it any more.
                if self._typing_tasks.get(chat_id) is asyncio.current_task():
                    self._typing_tasks.pop(chat_id, None)

        self._typing_tasks[chat_id] = asyncio.create_task(_typing_loop())

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the persistent typing indicator for a channel."""
        # Supersede every loop for this chat first: even a loop whose map entry was lost keeps
        # a private epoch and exits on its next tick (≤ one interval), so "typing…" cannot
        # outlive the turn.
        if getattr(self, "_typing_epoch", None) is None:
            self._typing_epoch = {}
        if getattr(self, "_typing_tasks", None) is None:
            self._typing_tasks = {}
        self._typing_epoch[chat_id] = self._typing_epoch.get(chat_id, 0) + 1
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
