"""Joining a MatrixRTC call — the ``/voice join`` surface.

``gateway/run_voice.py`` drives live voice through five adapter methods Discord defines:
``get_user_voice_channel`` / ``join_voice_channel`` / ``leave_voice_channel`` /
``is_in_voice_channel`` / ``get_voice_channel_info``. This mixin answers them with Matrix
semantics, and behind them wires Phases 1-3 together: the JWT exchange (``focus``), the
LiveKit room and STT (``receiver``), the outbound track (``outbound.start_rtc_audio``), and
the room's own gateway session (``session``).

One Discord shape does not survive translation. Discord keys a live call on the *guild*,
because a voice channel and the text channel its transcripts land in are different objects;
a Matrix call lives in the room it is about, so the key is the room id — a ``str``, never a
guild. ``voice_scope = "chat"`` is how the gateway is told that, and it is the only
Matrix-specific thing the gateway has to know.

Membership state is read defensively on purpose. Two event types and two content shapes are
in the wild depending on which client started the call, so ``live_call_members`` accepts all
of them and reads "no expiry stated" as live: refusing to join a call that is plainly
running, because a field we guessed at is missing, is the worse failure.

Writing it is the opposite: exactly one shape, copied off a live Element Desktop 1.12.27
call. matrix-js-sdk silently drops a membership missing any of ``application`` / ``call_id``
/ ``device_id`` / ``focus_active`` / ``foci_preferred``, and a dropped membership is a bot
nobody can see in the call — which is what the SFU-only join looked like in production.
Media and signalling are independent here (``/sfu/get`` checks neither room membership nor
room existence), so the state event is the only thing that puts the bot in the widget, and a
homeserver refusing it is never a reason to hang up on a call that is already audible.
"""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from gateway.platforms.base import _lazy_attr

from .focus import fetch_livekit_credentials
from .receiver import MatrixRTCReceiver
from .session import MatrixRTCSessions, split_identity

logger = logging.getLogger(__name__)

# MSC4143's type and the MSC3401 name Element shipped first. Which one a room carries
# depends on the client that started the call, so both count.
RTC_MEMBER_TYPES = frozenset({"m.rtc.member", "org.matrix.msc3401.call.member"})

# What we *write*. Element Desktop 1.12.27 publishes the MSC3401 name and nothing else, and
# a client that reads both would list the bot twice if we sent both — so we send what the
# client in the room actually sends, and keep reading both.
CALL_MEMBER_TYPE = "org.matrix.msc3401.call.member"

# Element's own value. There is no ``created_ts``: matrix-js-sdk falls back to the event's
# ``origin_server_ts``, which the homeserver stamps and we cannot lie about.
MEMBERSHIP_EXPIRY_MS = 14_400_000  # 4 hours


@dataclass
class MatrixCall:
    """What ``get_user_voice_channel`` hands back to ``/voice join``.

    The gateway reads ``.name`` for its confirmation message and passes the object straight
    back into ``join_voice_channel``; Discord's channel object is just as opaque to it.
    """

    room_id: str
    name: str


def membership_user_id(state_key: str) -> str:
    """The Matrix user id inside an RTC membership state key.

    The suffix count is not fixed: ``@u:hs`` (MSC3401 as first shipped), the per-device
    ``@u:hs_DEVICE`` / ``_@u:hs_DEVICE``, and Element's ``_@u:hs_DEVICE_m.call``, which
    appends the application too. So we cut at the *front* of the suffixes rather than
    counting them off the end — the user id stops at the first underscore after the
    server name, which cannot itself contain one (the grammar admits only a hostname,
    an IP literal and a port). A localpart may, so the scan starts at the colon and
    ``@my_bot:hs`` survives whole.
    """
    key = state_key[1:] if state_key.startswith("_") else state_key
    colon = key.find(":")
    if not key.startswith("@") or colon < 0:
        return key
    cut = key.find("_", colon)
    return key[:cut] if cut > 0 else key


def _membership_live(entry: dict, now_ms: float) -> bool:
    """One membership dict: has it not expired yet? Absent expiry reads as live."""
    expires_ts = entry.get("expires_ts")
    if isinstance(expires_ts, (int, float)) and not isinstance(expires_ts, bool):
        return expires_ts > now_ms
    expires, created = entry.get("expires"), entry.get("created_ts")
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (expires, created)):
        return created + expires > now_ms
    return True


def live_call_members(state_events, now_ms: Optional[float] = None) -> set[str]:
    """The user ids with a live RTC membership in a room's raw state events.

    Leaving a call is published as an *empty content* state event rather than a redaction,
    so empty content is the "not in the call" signal — and the usual state of a room whose
    call has ended.
    """
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    live: set[str] = set()
    for event in state_events or []:
        if not isinstance(event, dict) or event.get("type") not in RTC_MEMBER_TYPES:
            continue
        content = event.get("content")
        if not isinstance(content, dict) or not content:
            continue
        memberships = content.get("memberships")
        entries = memberships if isinstance(memberships, list) else [content]
        if not any(isinstance(e, dict) and _membership_live(e, now_ms) for e in entries):
            continue
        if user_id := membership_user_id(str(event.get("state_key") or "")):
            live.add(user_id)
    return live


def call_membership_content(room_id: str, device_id: str, service_url: str) -> dict:
    """Our own RTC membership, in the shape a live Element call publishes it.

    Every key is load-bearing: matrix-js-sdk validates the five of them and ignores a
    membership that is missing one, so a guessed-at schema shows up as a bot that joined
    the SFU and appears in nobody's call UI. ``foci_preferred`` advertises the JWT service
    URL — the one clients POST ``/sfu/get`` to, not the SFU websocket we connected on.
    """
    return {
        "application": "m.call",
        "call_id": "",  # the room's own call
        "device_id": device_id,
        "expires": MEMBERSHIP_EXPIRY_MS,
        "focus_active": {"type": "livekit", "focus_selection": "oldest_membership"},
        "foci_preferred": [{"type": "livekit", "livekit_alias": room_id,
                            "livekit_service_url": service_url}],
        "scope": "m.room",
    }


class MatrixRTCVoiceMixin:
    """Joining half of a MatrixRTC call. Mixed into ``MatrixAdapter``."""

    # The call lives in the room, so the gateway keys it on chat_id and never asks for a
    # guild id it could not produce. See ``GatewayVoiceMixin._voice_scope_id``.
    voice_scope = "chat"

    # --- registries (getattr-guarded: object.__new__ test instances skip __init__) ---

    @property
    def rtc_sessions(self) -> MatrixRTCSessions:
        """Room id -> the gateway session call audio in that room speaks into."""
        return _lazy_attr(self, "_rtc_sessions", lambda: MatrixRTCSessions(self))

    @property
    def rtc_receivers(self) -> Dict[str, MatrixRTCReceiver]:
        """Room id -> the receiver listening to that room's call."""
        return _lazy_attr(self, "_rtc_receivers", dict)

    # --- gateway duck-types ---

    def bind_voice_session(self, room_id: str, source) -> None:
        """Point the room's call at the session its typed messages already use.

        Called *before* the join: audio can arrive with the first frame, and an unbound
        room drops it.
        """
        self.rtc_sessions.bind(room_id, source)

    async def get_user_voice_channel(self, room_id: str, user_id: str) -> Optional[MatrixCall]:
        """The room's call when *user_id* is in it — the Matrix reading of Discord's
        "which voice channel is this user sitting in"."""
        if user_id not in live_call_members(await self._fetch_room_state(room_id)):
            logger.debug("MatrixRTC: %s has no live call membership in %s", user_id, room_id)
            return None
        return MatrixCall(room_id=room_id, name=await self._rtc_room_name(room_id))

    async def join_voice_channel(self, channel) -> bool:
        """Hear the call (``receiver``) and speak into it (``publisher``).

        Idempotent per room: joining twice reuses the connection rather than opening a
        second one. Whatever the JWT exchange or the SFU raised propagates — the gateway
        turns it into the user-visible failure message.
        """
        room_id = getattr(channel, "room_id", None) or str(channel)
        if room_id in self.rtc_receivers:
            return True
        try:
            sfu_url, jwt, focus_url = await fetch_livekit_credentials(
                self._homeserver, self._user_id, self._access_token, room_id,
                self._rtc_device_id(), session=self._rtc_http_session())
            receiver = MatrixRTCReceiver(
                on_transcript=functools.partial(self.rtc_sessions.on_transcript, room_id),
                is_authorized=functools.partial(self.rtc_sessions.is_authorized, room_id),
                # The two halves of the duplex have to know about each other for exactly one
                # reason: what the bot says comes back to its own ears. While the publisher is
                # playing, the receiver drops what it hears instead of transcribing the reply
                # as if the user had said it — and speech loud enough to survive that gate is
                # the user cutting the reply off.
                is_speaking=functools.partial(self.is_speaking_in, room_id),
                on_barge_in=functools.partial(self.rtc_sessions.barge_in, room_id))
            await receiver.connect(sfu_url, jwt)
        except Exception:
            self.rtc_sessions.unbind(room_id)  # the gateway bound it before calling us
            raise
        self.rtc_receivers[room_id] = receiver
        try:
            await self.start_rtc_audio(room_id, receiver.room)
        except Exception as exc:
            # Half a call beats no call: we still hear the user, and play_tts falls back to
            # sending the reply as a voice message.
            logger.warning("MatrixRTC: joined %s without an outbound track: %s", room_id, exc)
        await self._publish_call_membership(
            room_id, call_membership_content(room_id, self._rtc_device_id(), focus_url))
        return True

    async def leave_voice_channel(self, room_id: str) -> None:
        """Stop speaking, stop listening, unbind, and take ourselves out of the call UI.

        The order is load-bearing: ``close()`` flushes the utterance still sitting in the
        segmenter, and that transcript needs the bind to reach a session. Clearing the
        membership goes last for the same reason it goes last on join — the state event
        describes what the media plane is already doing.

        Every step is unconditional, and the membership PUT especially so: the three
        registries live in this process and the membership lives in the room, so after a
        restart this runs with nothing to close and the state event is the only thing left
        to clear. That is the case the ghost participant comes from, not the happy path.
        """
        await self.stop_rtc_audio(room_id)
        if (receiver := self.rtc_receivers.pop(room_id, None)) is not None:
            await receiver.close()
        self.rtc_sessions.unbind(room_id)
        await self._publish_call_membership(room_id, {})

    def get_voice_channel_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """``/voice status``: who else is on the call, or None when we are not in one.

        Sync like Discord's, so it reports the SFU's own participant list instead of
        re-reading room state. Speaking flags are Phase 5's; ``/voice status`` already
        treats that key as optional.
        """
        room = getattr(self.rtc_receivers.get(room_id), "room", None)
        if room is None:
            return None
        members = []
        for identity, participant in (getattr(room, "remote_participants", None) or {}).items():
            user_id, _device = split_identity(str(identity))
            members.append({"user_id": user_id, "is_bot": False,
                            "display_name": getattr(participant, "name", "") or user_id})
        return {"channel_name": self._rtc_cached_room_name(room_id),
                "member_count": len(members), "members": members}

    # --- internals ---

    def _rtc_device_id(self) -> str:
        """The device the access token actually belongs to. The LiveKit identity is
        ``{user_id}:{device_id}``, and the client's resolved device is the one the
        homeserver recognises when the configured value is stale."""
        client = getattr(self, "_client", None)
        return str(getattr(client, "device_id", "") or getattr(self, "_device_id", "") or "")

    def _rtc_http_session(self):
        """The adapter's own aiohttp session — proxy and TLS already configured — or None
        to let ``focus`` open a short-lived one."""
        return getattr(getattr(getattr(self, "_client", None), "api", None), "session", None)

    async def _fetch_room_state(self, room_id: str) -> List[dict]:
        """The room's raw state events.

        Raw on purpose: ``m.rtc.member`` is not an event type mautrix models, so the typed
        client would drop the very event we came for.
        """
        api = getattr(getattr(self, "_client", None), "api", None)
        if api is None:
            return []
        try:
            from mautrix.api import Method
            state = await api.request(
                Method.GET, f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/state")
        except Exception as exc:
            logger.debug("MatrixRTC: could not read state of %s: %s", room_id, exc)
            return []
        return state if isinstance(state, list) else []

    async def _publish_call_membership(self, room_id: str, content: dict) -> None:
        """PUT our RTC membership for *room_id*; empty *content* is how leaving is said.

        Raw ``api.request`` for the same reason the state read is raw: mautrix models
        neither event type, so the typed client cannot send this one. Never raises — the
        call is already up (or already down) by the time we get here, and a homeserver that
        refuses the state event costs a widget entry, not the conversation.
        """
        api = getattr(getattr(self, "_client", None), "api", None)
        if api is None:
            return
        state_key = f"_{self._user_id}_{self._rtc_device_id()}_m.call"
        try:
            from mautrix.api import Method
            await api.request(
                Method.PUT,
                f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}"
                f"/state/{CALL_MEMBER_TYPE}/{quote(state_key, safe='')}",
                content=content)
        except Exception as exc:
            logger.warning("MatrixRTC: could not %s call membership in %s: %s",
                           "clear" if not content else "publish", room_id, exc)

    async def _rtc_room_name(self, room_id: str) -> str:
        """The room's display name for the join confirmation; the id if it has none."""
        resolve = getattr(self, "_resolve_room_identity", None)
        if resolve is not None:
            try:
                return (await resolve(room_id)).display_name or room_id
            except Exception:
                pass
        return room_id

    def _rtc_cached_room_name(self, room_id: str) -> str:
        """Same name from the identity cache only — ``get_voice_channel_info`` is sync."""
        identity = (getattr(self, "_room_identities", None) or {}).get(room_id)
        return getattr(identity, "display_name", None) or room_id
