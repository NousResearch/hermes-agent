"""Maps MatrixRTC transcripts onto the room's gateway session.

The Matrix analogue of Discord's ``_voice_text_channels`` / ``_voice_sources`` pair
(``gateway/run_voice.py``). Discord needs two ids because a voice channel and the text
channel its transcripts land in are different objects; a Matrix call lives *in* the room
it is about, so the pair collapses to one map from room id to ``SessionSource``. A spoken
turn therefore arrives in the session that the room's typed messages already use, with
the same thread, profile and channel prompt, and not in a synthetic sibling session.

Nothing here joins a call. Binding happens when something else joins one, which is the
``/voice join`` wiring's job.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_identity import canonical_identity, clear_identity, identity_of, replace_source

logger = logging.getLogger(__name__)

# How long the audio path reuses a speaker verdict. A change to the room's call state or
# to the binding discards verdicts at once. The expiry covers changes that happen without a
# state event, such as a membership reaching its expiry time or a change to the allowlist.
VERDICT_TTL_SECONDS = 5.0


@dataclass(frozen=True)
class _CallBinding:
    source: SessionSource
    fields: tuple
    client: object
    api: object
    account: tuple
    home: Path
    physical_home: Path
    physical_authorization_home: Optional[Path]


def _source_fields(source: SessionSource) -> tuple:
    return (source.platform, source.chat_id, source.chat_type, source.user_id,
            source.thread_id, source.profile, getattr(source, "_identity", None),
            getattr(source, "_transport_adapter_ref", None))


def _is_call_participant(events, user_id: str, device_id: str) -> bool:
    """Whether room state shows *user_id* joined to the room with a live call membership.

    A LiveKit identity from ``/sfu/get`` has a user id that the authorisation service
    verified and a device id that the client only claimed. The claimed device must
    therefore match one of that user's own live memberships.
    """
    from .membership import live_call_memberships
    events = list(events)
    if not any(event.get("type") == "m.room.member" and event.get("state_key") == user_id
               and (event.get("content") or {}).get("membership") == "join" for event in events):
        return False
    return any(membership.user_id == user_id and (not device_id or membership.device_id == device_id)
               for membership in live_call_memberships(events))


def split_identity(identity: str) -> tuple[str, str]:
    """Split LiveKit's ``{matrix_user_id}:{device_id}`` into ``("@user:server", "DEVICE")``.

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
    """For each room, the ``SessionSource`` that call audio in that room speaks into."""

    def __init__(self, adapter, clock: Callable[[], float] = time.monotonic):
        self._adapter = adapter
        self._clock = clock
        self._sources: dict[str, SessionSource] = {}
        self._bindings: dict[str, _CallBinding] = {}
        self._verdicts: dict[str, dict[tuple[str, str], tuple[bool, float]]] = {}

    # --- binding ---

    def bind(self, room_id: str, source: SessionSource) -> None:
        """Bind a joined call to *source*, the room's own session source."""
        self._sources[room_id] = source
        from hermes_constants import get_hermes_home
        adapter = self._adapter
        client = getattr(adapter, "_client", None)
        runner = getattr(adapter, "gateway_runner", None)
        resolve = getattr(runner, "_resolve_profile_home_for_source", None)
        home = Path(resolve(source) if resolve else get_hermes_home())
        identity = identity_of(source)
        authorization_home = identity.authorization_home.resolve() if identity is not None else None
        self._bindings[room_id] = _CallBinding(
            source, _source_fields(source), client, getattr(client, "api", None),
            self._account(), home, home.resolve(), authorization_home)
        self.invalidate(room_id)
        logger.info("MatrixRTC: call in %s bound to %s", room_id, source.description)

    def unbind(self, room_id: str) -> None:
        """Drop the bind on leave. Later audio for the room is discarded, not guessed at."""
        self._bindings.pop(room_id, None)
        self.invalidate(room_id)
        if self._sources.pop(room_id, None) is not None:
            logger.info("MatrixRTC: call in %s unbound", room_id)

    def _speaker_source(self, bound: SessionSource, user_id: Optional[str],
                        user_name: Optional[str] = None) -> Optional[SessionSource]:
        source = replace_source(bound, user_id=user_id, user_name=user_name or user_id,
                                role_authorized=False)
        runner = getattr(self._adapter, "gateway_runner", None)
        if runner is not None and callable(getattr(runner, "_canonicalize", None)):
            registered, _profile = runner._owning_profile(self._adapter, source.platform)
            if not registered:
                return None
            clear_identity(source)
            if canonical_identity(source, runner=runner, adapter=self._adapter) is None:
                return None
        return source

    def source_for(self, room_id: str, user_id: str,
                   user_name: Optional[str] = None) -> Optional[SessionSource]:
        """Return the current speaker source only within the call's original profile."""
        if not self.current(room_id):
            return None
        binding = self._bindings[room_id]
        source = self._speaker_source(binding.source, user_id, user_name)
        if source is None or not self._same_home(source, binding):
            return None
        return source

    def _same_home(self, source: SessionSource, binding: _CallBinding) -> bool:
        runner = getattr(self._adapter, "gateway_runner", None)
        resolve = getattr(runner, "_resolve_profile_home_for_source", None)
        home = Path(resolve(source)) if resolve is not None else binding.home
        identity = identity_of(source)
        authorization_home = identity.authorization_home.resolve() if identity is not None else None
        return (home == binding.home and home.resolve() == binding.physical_home
                and authorization_home == binding.physical_authorization_home)

    # --- authorization ---

    def _account(self) -> tuple:
        return tuple(getattr(self._adapter, key, None) for key in
                     ("_homeserver", "_user_id", "_access_token", "_device_id"))

    def binding_for(self, room_id: str):
        return self._bindings.get(room_id)

    def scope_for(self, room_id: str):
        binding = self._bindings.get(room_id)
        if binding is None or not self.current(room_id):
            return nullcontext()
        from gateway.run import _profile_runtime_scope
        return _profile_runtime_scope(binding.home)

    def current(self, room_id: str, binding=None) -> bool:
        bound = self._bindings.get(room_id)
        if bound is None or (binding is not None and binding is not bound):
            return False
        adapter = self._adapter
        if getattr(adapter, "_closing", False) or self._account() != bound.account:
            return False
        if self._sources.get(room_id) is not bound.source or _source_fields(bound.source) != bound.fields:
            return False
        client = getattr(adapter, "_client", None)
        if client is not bound.client or getattr(client, "api", None) is not bound.api:
            return False
        if hasattr(adapter, "_client") and client is None:
            return False
        if hasattr(adapter, "_joined_rooms") and room_id not in adapter._joined_rooms:
            return False
        if room_id in getattr(adapter, "_rtc_call_state", {}) and not adapter._rtc_call_state[room_id]:
            return False
        runner = getattr(adapter, "gateway_runner", None)
        delivery = getattr(runner, "_delivery_adapter_for", None)
        if delivery is not None and delivery(bound.source) is not adapter:
            return False
        source = self._speaker_source(bound.source, bound.source.user_id)
        return source is not None and self._same_home(source, bound)

    def is_authorized(self, room_id: str, identity: str) -> bool:
        """Allowlist verdict for one LiveKit participant."""
        user_id, device = split_identity(identity)
        return self.is_user_authorized(room_id, user_id, device)

    def audio_allowed(self, room_id: str, identity: str) -> bool:
        """``is_authorized`` for one audio frame, reusing a recent verdict for the participant."""
        user_id, device = split_identity(identity)
        return self.user_audio_allowed(room_id, user_id, device)

    def user_audio_allowed(self, room_id: str, user_id: str, device: str = "") -> bool:
        """``is_user_authorized`` for one audio frame or chunk, reusing a recent verdict.

        The full verdict scans the room's state and runs the gateway's allowlist, which
        is too much work to repeat for every 10 ms frame.
        """
        verdicts = self._verdicts.setdefault(room_id, {})
        now = self._clock()
        cached = verdicts.get((user_id, device))
        if cached is not None and cached[1] > now:
            return cached[0]
        verdict = self.is_user_authorized(room_id, user_id, device)
        verdicts[(user_id, device)] = (verdict, now + VERDICT_TTL_SECONDS)
        return verdict

    def invalidate(self, room_id: str) -> None:
        """Discard the room's reused verdicts after its call state or binding changes."""
        self._verdicts.pop(room_id, None)

    def is_user_authorized(self, room_id: str, user_id: str, device: str = "") -> bool:
        """Allowlist verdict for one Matrix user, and for one device when *device* is set.

        Applies the two gates that the text path applies, so a participant who may not
        type in the room may not talk into it either, and their audio never reaches STT.
        The user must also be joined to the room with a live call membership.
        """
        if not self.current(room_id):
            return False
        source = self.source_for(room_id, user_id)
        if source is None:
            logger.debug("MatrixRTC: no session bound for %s, dropping audio", room_id)
            return False
        state = getattr(self._adapter, "_rtc_call_state", {}).get(room_id)
        if not state or not _is_call_participant(state.values(), user_id, device):
            return False
        # MATRIX_ALLOWED_ROOMS, with DMs exempt, in the same shape as _resolve_message_context,
        # so a project-scoped allowlist does not silence the operator's own DM call.
        allowed_rooms = getattr(self._adapter, "_allowed_room_ids", None)
        if allowed_rooms is None:
            allowed_rooms = getattr(self._adapter, "_allowed_rooms", None)
        if allowed_rooms and source.chat_type != "dm" and room_id not in allowed_rooms:
            logger.debug("MatrixRTC: %s not in MATRIX_ALLOWED_ROOMS, dropping audio", room_id)
            return False
        bound = self._sources[room_id]
        requester = self._speaker_source(bound, bound.user_id)
        return requester is not None and self._user_allowed(requester) and self._user_allowed(source)

    def _user_allowed(self, source: SessionSource) -> bool:
        """The gateway's full allowlist policy, which resolves MATRIX_ALLOWED_USERS through
        the platform registry. Without a runner (adapter driven standalone) fall back to the
        adapter's own check, which is still a real allowlist."""
        runner = getattr(self._adapter, "gateway_runner", None)
        check = getattr(runner, "_is_user_authorized_for_source", None)
        if check is None:
            check = getattr(runner, "_is_user_authorized", None)
        if check is not None:
            return bool(check(source))
        return bool(self._adapter._is_authorized_user(source.user_id))

    # --- barge-in ---

    async def barge_in(self, room_id: str, identity: str) -> None:
        """*identity* talked over the bot: stop the reply and the turn generating it.

        Everything this needs already exists one level up. Setting the session's interrupt
        guard is what the runner's monitor loop turns into ``agent.interrupt()`` plus
        ``StreamingTTSConsumer.abort("barge-in")``. That abort comes back here as
        ``abort_streaming_tts``, which calls ``publisher.clear()``. Stopping only the audio
        would leave the model generating a reply that nobody will hear.

        Idempotent (the guard is an already-set ``Event`` the second time) and a no-op when no
        turn is running. Unauthorized speakers cannot interrupt: the same allowlist that keeps
        their words out of the session keeps them from cancelling someone else's turn.
        """
        user_id, _device = split_identity(identity)
        source = self.source_for(room_id, user_id)
        if source is None or not self.is_authorized(room_id, identity):
            return
        adapter = self._adapter
        # The same key derivation that the spoken turn uses: ``_event_session_key`` reads
        # nothing but ``event.source``, and a key built any other way would not find the guard.
        session_key = adapter._event_session_key(MessageEvent(text="", source=source))
        logger.info("MatrixRTC: barge-in from %s in %s", user_id, room_id)
        await adapter.interrupt_session_activity(session_key, room_id)

    # --- dispatch ---

    def _is_duplicate(self, room_id: str, user_id: str, transcript: str) -> bool:
        """Reuse the runner's suppressor. Its ``(guild_id, user_id)`` parameters are only
        ever used as an opaque dict key, so Matrix ids go in unchanged. One utterance
        emitted twice a few seconds apart would otherwise queue a second run."""
        runner = getattr(self._adapter, "gateway_runner", None)
        check = getattr(runner, "_is_duplicate_voice_transcript", None)
        return bool(check(room_id, user_id, transcript)) if check is not None else False

    async def on_transcript(self, room_id: str, identity: str, transcript: str) -> None:
        binding = self.binding_for(room_id)
        with self.scope_for(room_id):
            await self._dispatch_transcript(room_id, identity, transcript, binding)

    async def _dispatch_transcript(self, room_id: str, identity: str, transcript: str, binding) -> None:
        """Receiver callback: turn one finished utterance into one ``MessageEvent(VOICE)``.

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
        if not self.current(room_id, binding) or not self.is_authorized(room_id, identity):
            return
        source = self.source_for(room_id, user_id, display_name)
        if source is None:  # unbound between the check and here
            return
        await self._echo_transcript(source, transcript)
        if not self.current(room_id, binding) or not self.is_authorized(room_id, identity):
            return
        # Top-level user fields mirror source.* because downstream prompt code reads them,
        # exactly as the adapter's own _build_inbound_event does.
        await self._adapter.handle_message(MessageEvent(
            text=transcript, source=source, message_type=MessageType.VOICE,
            user_id=user_id, user_name=display_name))

    async def _echo_transcript(self, source: SessionSource, transcript: str) -> None:
        """Post what we heard back into the room when ``stt_echo_transcripts`` is on.

        The runner's own helper, which needs only ``adapter.send``, so a spoken turn
        gets the same 🎙️ line a Telegram voice note gets, and STT quality is checkable from
        the room instead of the logs. No runner (adapter driven standalone): nothing is echoed.
        """
        runner = getattr(self._adapter, "gateway_runner", None)
        echo = getattr(runner, "_echo_stt_transcripts", None)
        if echo is not None and runner._should_echo_stt_transcripts():
            await echo(self._adapter, source, [transcript])
