"""Discord gateway ↔ realtime-voice supervisor bridge.

Supplies the Discord :class:`~agent.voice_supervisor.TurnRunner` for
:class:`agent.voice_supervisor.VoiceSupervisorController`. Consults are
normal agent turns (not the STT pipeline): no fuzzy transcript dedup, no
``MessageType.VOICE``. Busy/queue state comes from the adapter session guards.

Turns are attributed to the last *transcribed* speaker (falling back to the
last PCM contributor). No speaker → ``submit`` returns False. Gateway ops
are serialized on the event loop so interrupt-then-submit stays ordered.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class DiscordVoiceTurnRunner:
    """TurnRunner backed by the gateway's Discord message pipeline."""

    def __init__(self, runner: Any, adapter: Any, guild_id: int, loop: asyncio.AbstractEventLoop):
        self._runner = runner
        self._adapter = adapter
        self._guild_id = guild_id
        self._loop = loop
        self._active_user_id: int = 0
        self._op_tail: Optional[Future] = None

    def _speaker_user_id(self) -> int:
        transcribed = getattr(self._adapter, "_voice_realtime_last_transcribed", {}).get(
            self._guild_id
        )
        if transcribed:
            try:
                uid = int(transcribed)
                if uid:
                    return uid
            except (TypeError, ValueError):
                pass
        last = getattr(self._adapter, "_voice_realtime_last_speaker", {}).get(
            self._guild_id
        ) or 0
        try:
            return int(last)
        except (TypeError, ValueError):
            return 0

    def _turn_user_id(self) -> int:
        return self._active_user_id or self._speaker_user_id()

    def _source(self):
        user_id = self._turn_user_id()
        if not user_id:
            return None
        return self._runner._voice_channel_source(self._adapter, self._guild_id, user_id)

    def _session_key(self) -> Optional[str]:
        source = self._source()
        if source is None:
            return None
        try:
            return self._runner._session_key_for_source(source)
        except Exception:
            logger.debug("voice turn runner: session key resolution failed", exc_info=True)
            return None

    def _enqueue(self, factory: Callable[[], Awaitable[None]]) -> None:
        """Run *factory* on the gateway loop after any prior op finishes."""
        prev = self._op_tail

        async def _linked() -> None:
            if prev is not None:
                try:
                    await asyncio.wrap_future(prev, loop=self._loop)
                except Exception:
                    pass
            await factory()

        self._op_tail = asyncio.run_coroutine_threadsafe(_linked(), self._loop)

    def submit(self, task: str) -> bool:
        user_id = self._speaker_user_id()
        if not user_id:
            logger.warning(
                "voice consult dropped: no speaker context for guild %d", self._guild_id
            )
            return False
        self._active_user_id = user_id

        async def _submit() -> None:
            await self._runner._handle_voice_channel_input(
                self._guild_id, user_id, task, consult=True, adapter=self._adapter
            )

        self._enqueue(_submit)
        return True

    def interrupt(self) -> None:
        source = self._source()
        session_key = self._session_key() if source is not None else None
        if not session_key:
            return
        chat_id = str(source.chat_id)  # the bound text channel: consults share its session

        async def _interrupt() -> None:
            await self._adapter.interrupt_session_activity(session_key, chat_id)

        self._enqueue(_interrupt)

    def is_busy(self) -> bool:
        session_key = self._session_key()
        active = getattr(self._adapter, "_active_sessions", {})
        return bool(session_key) and session_key in active

    def is_queue_empty(self) -> bool:
        session_key = self._session_key()
        pending = getattr(self._adapter, "_pending_messages", {})
        # Truthiness, not key membership: a leftover empty/None entry under
        # the key would otherwise report "not empty" forever and permanently
        # block stale-consult recovery for the guild.
        return not session_key or not pending.get(session_key)


__all__ = ["DiscordVoiceTurnRunner"]
