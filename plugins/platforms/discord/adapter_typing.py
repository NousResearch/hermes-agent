"""Native Discord typing transport and lifecycle."""

import asyncio
import logging

logger = logging.getLogger(__name__)


class DiscordTypingMixin:
    _TYPING_REQUEST_TIMEOUT = 4.0

    async def _request_typing(self, http, route):
        from aiohttp import ClientTimeout

        # Bound socket I/O, not discord.py's rate-limit waits. Cancelling request()
        # during a global 429 sleep strands its shared gate (no finally upstream).
        request = asyncio.create_task(http.request(
            route, timeout=ClientTimeout(total=self._TYPING_REQUEST_TIMEOUT)))
        try:
            await asyncio.shield(request)
        except asyncio.CancelledError:
            async def drain():
                gate = getattr(http, "_global_over", None)
                while not request.done() and gate is not None and not gate.is_set():
                    await gate.wait()
                # No await between observing the open gate and cancellation:
                # request cannot enter another global backoff in that interval.
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

            cleanup = asyncio.create_task(drain())
            # Repeated stop/shutdown cancellation must not interrupt gate recovery
            # or leave an HTTP request orphaned after its typing owner exits.
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise

    def _start_typing_refresh(self, event, interrupt_event, metadata):
        if self._disconnecting or not self._client:
            return None
        task = super()._start_typing_refresh(event, interrupt_event, metadata)
        if task is not None:
            # These are typing owners, not a second worker registry. A new turn
            # replaces only its own session's refresh handle.
            owners = self._typing_owners.setdefault(event.source.chat_id, {})
            owners[self._event_session_key(event)] = (task, interrupt_event)
        return task

    def _typing_active(self, chat_id):
        if self._disconnecting or not self._client:
            return False
        from tools.async_delegation import has_live_for_session
        from tools.process_registry import process_registry

        owners = self._typing_owners.get(chat_id, {})
        active = False
        for key, (refresh, interrupted) in list(owners.items()):
            # Registry queries use the canonical, profile-namespaced session key,
            # not the transport task's captured context (one channel may serve
            # several profiles). Neither query performs I/O.
            live = not interrupted.is_set() and (
                (not refresh.done() and not refresh.cancelling())
                or has_live_for_session(session_key=key)
                or process_registry.has_completion_work_for_session(key))
            if live:
                active = True
            else:
                owners.pop(key, None)
        return active

    async def interrupt_session_activity(self, session_key: str, chat_id: str, metadata=None) -> None:
        owner = self._typing_owners.get(chat_id, {}).get(session_key)
        if owner is not None:
            refresh, interrupted = owner
            interrupted.set()
            refresh.cancel()
        await super().interrupt_session_activity(session_key, chat_id, metadata)

    async def cancel_background_tasks(self) -> None:
        # Stop owners before base cleanup: its finally blocks call stop_typing,
        # which must no longer preserve detached workers during shutdown.
        for owners in self._typing_owners.values():
            for refresh, interrupted in owners.values():
                interrupted.set()
                refresh.cancel()
        try:
            await super().cancel_background_tasks()
        finally:
            self._typing_owners.clear()
            for chat_id in list(self._typing_tasks):
                await self.stop_typing(chat_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Start a persistent typing loop (POST typing every 5s; indicator lasts ~10s).
        TYPING_START is unreliable for bots in DMs; 429 sleeps ``retry_after``; CancelledError ends it."""
        from .adapter import discord  # optional dependency, resolved by the facade

        if self._disconnecting or not self._client:
            return
        if chat_id in self._typing_tasks:
            return

        managed = bool(self._typing_owners.get(chat_id))

        async def _typing_loop() -> None:
            try:
                while self._client and not self._disconnecting:
                    if managed and not self._typing_active(chat_id):
                        return
                    if chat_id in self._typing_paused:
                        await asyncio.sleep(5)
                        continue
                    try:
                        route = discord.http.Route(
                            "POST", "/channels/{channel_id}/typing", channel_id=chat_id,
                        )
                        await self._request_typing(self._client.http, route)
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
                            if not managed:
                                return
                            # Detached workers have no base refresh left to restart
                            # a failed transport. Retry while their owner is live.
                            retry_after = 5
                        # Respect the full backoff without retaining a finished
                        # owner's task for a potentially very long Retry-After.
                        while retry_after > 0:
                            if managed and not self._typing_active(chat_id):
                                return
                            delay = min(5, retry_after)
                            await asyncio.sleep(delay)
                            retry_after -= delay
                        continue
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                pass
            finally:
                if self._typing_tasks.get(chat_id) is asyncio.current_task():
                    self._typing_tasks.pop(chat_id, None)
                    if self._typing_active(chat_id):
                        # A successor may finish its foreground turn while this
                        # transport drains, leaving only its worker as an owner.
                        await self.send_typing(chat_id)
                    else:
                        self._typing_owners.pop(chat_id, None)
        self._typing_tasks[chat_id] = asyncio.create_task(_typing_loop())

    async def stop_typing(self, chat_id: str) -> None:
        """Stop the persistent typing indicator for a channel."""
        if self._typing_active(chat_id):
            return
        self._typing_owners.pop(chat_id, None)
        # Keep the task registered until its HTTP cleanup finishes so a concurrent
        # refresh cannot replace it while a global rate-limit gate is recovering.
        task = self._typing_tasks.get(chat_id)
        if task:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            finally:
                # A task cancelled before its first step never runs its finally.
                if task.done() and self._typing_tasks.get(chat_id) is task:
                    self._typing_tasks.pop(chat_id, None)
                    if self._typing_active(chat_id):
                        await self.send_typing(chat_id)
                    else:
                        self._typing_owners.pop(chat_id, None)
