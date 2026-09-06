"""Maps MatrixRTC transcripts onto the room's gateway session.

The Matrix analogue of Discord's ``_voice_text_channels`` / ``_voice_sources`` pair
(``gateway/run_voice.py``). Discord needs two ids because a voice channel and the text
channel its transcripts land in are different objects; a Matrix call lives *in* the room
it is about, so the pair collapses to one map: ``room_id -> SessionSource``. A spoken
turn therefore arrives in exactly the session the room's typed messages already use —
same thread, same profile, same channel prompt — instead of a synthetic sibling.

Nothing here joins a call. Binding happens when something else joins one, which is the
``/voice join`` wiring's job.
"""

from __future__ import annotations

import copy
import logging
from contextlib import suppress
from typing import Optional

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


def split_identity(identity: str) -> tuple[str, str]:
    """LiveKit's ``{matrix_user_id}:{device_id}`` -> ``("@user:server", "DEVICE")``.

    The gateway session is keyed on the user, not the device, so the device half is
    dropped everywhere but the logs. Splitting on the *last* colon is safe because a
    Matrix user id always contains one of its own: a bare ``@user:server`` with no device
    suffix comes back whole, since the remainder would not itself contain a colon.
    """
    user_id, sep, device_id = identity.rpartition(":")
    if sep and user_id.startswith("@") and ":" in user_id:
        return user_id, device_id
    return identity, ""


class MatrixRTCSessions:
    """Room id -> the ``SessionSource`` that call audio in that room speaks into."""

    def __init__(self, adapter):
        self._adapter = adapter
        self._sources: dict[str, SessionSource] = {}

    # --- binding ---

    def bind(self, room_id: str, source: SessionSource) -> None:
        """Bind a joined call to *source*, the room's own session source."""
        self._sources[room_id] = source
        logger.info("MatrixRTC: call in %s bound to %s", room_id, source.description)

    def unbind(self, room_id: str) -> None:
        """Drop the bind on leave. Later audio for the room is discarded, not guessed at."""
        if self._sources.pop(room_id, None) is not None:
            logger.info("MatrixRTC: call in %s unbound", room_id)

    def source_for(self, room_id: str, user_id: str,
                   user_name: Optional[str] = None) -> Optional[SessionSource]:
        """The bound source with *this speaker* stamped on it, or None when unbound.

        ``copy.copy`` rather than ``dataclasses.replace``: the transport-adapter weakref
        that authorization delegation reads is set after construction, so rebuilding
        through ``__init__`` would silently drop it.
        """
        bound = self._sources.get(room_id)
        if bound is None:
            return None
        source = copy.copy(bound)
        source.user_id = user_id
        source.user_name = user_name or user_id
        return source

    # --- authorization ---

    def is_authorized(self, room_id: str, identity: str) -> bool:
        """Allowlist verdict for one speaker — cheap enough to run *before* STT.

        Applies the two gates the text path applies, so a participant who may not type in
        the room may not talk into it either, and their audio never reaches Whisper.
        """
        user_id, _device = split_identity(identity)
        source = self.source_for(room_id, user_id)
        if source is None:
            logger.debug("MatrixRTC: no session bound for %s, dropping audio", room_id)
            return False
        # MATRIX_ALLOWED_ROOMS, with DMs exempt — the same shape as _resolve_message_context,
        # so a project-scoped allowlist does not silence the operator's own DM call.
        allowed_rooms = getattr(self._adapter, "_allowed_rooms", None)
        if allowed_rooms and source.chat_type != "dm" and room_id not in allowed_rooms:
            logger.debug("MatrixRTC: %s not in MATRIX_ALLOWED_ROOMS, dropping audio", room_id)
            return False
        return self._user_allowed(source)

    def _user_allowed(self, source: SessionSource) -> bool:
        """The gateway's full allowlist policy, which resolves MATRIX_ALLOWED_USERS through
        the platform registry. Without a runner (adapter driven standalone) fall back to the
        adapter's own check — still a real allowlist, never an open door."""
        runner = getattr(self._adapter, "gateway_runner", None)
        if (check := getattr(runner, "_is_user_authorized", None)) is not None:
            return bool(check(source))
        return bool(self._adapter._is_authorized_user(source.user_id))

    # --- barge-in ---

    async def barge_in(self, room_id: str, identity: str) -> None:
        """*identity* talked over the bot: stop the reply and the turn generating it.

        Everything this needs already exists one level up. Setting the session's interrupt
        guard is what the runner's monitor loop turns into ``agent.interrupt()`` plus
        ``StreamingTTSConsumer.abort("barge-in")``, and that abort comes straight back here as
        ``abort_streaming_tts`` -> ``publisher.clear()``. Stopping only the audio would leave
        the model still generating a reply nobody will hear.

        Idempotent (the guard is an already-set ``Event`` the second time) and a no-op when no
        turn is running. Unauthorized speakers cannot interrupt: the same allowlist that keeps
        their words out of the session keeps them from cancelling someone else's turn.
        """
        user_id, _device = split_identity(identity)
        source = self.source_for(room_id, user_id)
        if source is None or not self.is_authorized(room_id, identity):
            return
        adapter = self._adapter
        # The same key derivation the spoken turn itself uses — ``_event_session_key`` reads
        # nothing but ``event.source``, and a key built any other way would not find the guard.
        session_key = adapter._event_session_key(MessageEvent(text="", source=source))
        logger.info("MatrixRTC: barge-in from %s in %s", user_id, room_id)
        await adapter.interrupt_session_activity(session_key, room_id)

    # --- dispatch ---

    def _is_duplicate(self, room_id: str, user_id: str, transcript: str) -> bool:
        """Reuse the runner's suppressor. Its ``(guild_id, user_id)`` parameters are only
        ever used as an opaque dict key, so Matrix ids go in unchanged — one utterance
        emitted twice a few seconds apart would otherwise queue a second run."""
        runner = getattr(self._adapter, "gateway_runner", None)
        check = getattr(runner, "_is_duplicate_voice_transcript", None)
        return bool(check(room_id, user_id, transcript)) if check is not None else False

    async def on_transcript(self, room_id: str, identity: str, transcript: str) -> None:
        """Receiver callback: one finished utterance -> one ``MessageEvent(VOICE)``.

        Bind with ``functools.partial(sessions.on_transcript, room_id)`` to match
        ``MatrixRTCReceiver``'s ``on_transcript(identity, transcript)`` signature.
        """
        user_id, _device = split_identity(identity)
        # Runs even when the receiver already asked: the pre-STT predicate is optional and
        # this is the boundary that reaches the agent.
        if not self.is_authorized(room_id, identity):
            logger.info("MatrixRTC: dropping voice input from %s in %s", user_id, room_id)
            return
        if self._is_duplicate(room_id, user_id, transcript):
            logger.info("MatrixRTC: suppressing duplicate transcript for %s in %s: %s",
                        user_id, room_id, transcript[:100])
            return
        display_name = user_id
        with suppress(Exception):  # room member cache; a miss is not worth losing the turn
            display_name = await self._adapter._get_display_name(room_id, user_id)
        source = self.source_for(room_id, user_id, display_name)
        if source is None:  # unbound between the check and here
            return
        await self._echo_transcript(source, transcript)
        # Top-level user fields mirror source.* because downstream prompt code reads them,
        # exactly as the adapter's own _build_inbound_event does.
        await self._adapter.handle_message(MessageEvent(
            text=transcript, source=source, message_type=MessageType.VOICE,
            user_id=user_id, user_name=display_name))

    async def _echo_transcript(self, source: SessionSource, transcript: str) -> None:
        """Post what we heard back into the room when ``stt_echo_transcripts`` is on.

        The runner's own helper, which needs nothing but ``adapter.send`` — so a spoken turn
        gets the same 🎙️ line a Telegram voice note gets, and STT quality is checkable from
        the room instead of the logs. No runner (adapter driven standalone): nothing is echoed.
        """
        runner = getattr(self._adapter, "gateway_runner", None)
        echo = getattr(runner, "_echo_stt_transcripts", None)
        if echo is not None and runner._should_echo_stt_transcripts():
            await echo(self._adapter, source, [transcript])
