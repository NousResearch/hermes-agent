"""Joining a MatrixRTC call: the ``/voice join`` surface.

``gateway/run_voice.py`` drives live voice through five adapter methods from the
Discord adapter: ``get_user_voice_channel``, ``join_voice_channel``,
``leave_voice_channel``, ``is_in_voice_channel`` and ``get_voice_channel_info``. This
mixin answers them with Matrix semantics. Behind them it connects the credential
exchange (``focus``), the LiveKit room and STT (``receiver``), the outbound track
(``outbound.start_rtc_audio``), the call membership (``membership``) and the room's own
gateway session (``session``).

Discord keys a live call on the guild, because a voice channel and the text channel for
its transcripts are different objects. A Matrix call belongs to its room, so the key is
the room id. ``voice_scope = "chat"`` tells the gateway that.

A join succeeds only when the bot is both connected to the SFU and listed in the room's
call membership state. The authorisation service does not check room membership for a
``/sfu/get`` token, so the media connection alone would leave the bot audible but
missing from every client's participant list. If any step fails or the join is
cancelled, the media connection closes and the membership is cleared before the error
reaches the gateway.

Hermes connects only to the SFU behind its own homeserver's MatrixRTC service. A join is
refused when the requester publishes through another deployment's service, because the
bot would not hear them.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

from gateway.platforms.base import _lazy_attr

from .focus import MatrixRTCError, fetch_livekit_credentials
from .membership import (
    CALL_MEMBER_TYPE, CallMembershipLease, call_membership_content, call_membership_state_key,
    leave_delay_ms, live_call_members, live_call_memberships)
from .receiver import MatrixRTCReceiver
from .session import MatrixRTCSessions, split_identity

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 45

# Authorisation reads call memberships and room memberships from room state.
_TRACKED_STATE_TYPES = frozenset({CALL_MEMBER_TYPE, "m.room.member"})


@dataclass
class MatrixCall:
    """What ``get_user_voice_channel`` hands back to ``/voice join``.

    The gateway reads ``.name`` for its confirmation message and passes the object back
    into ``join_voice_channel``, exactly as it does with Discord's channel object.
    """

    room_id: str
    name: str


class MatrixRTCVoiceMixin:
    """Joining half of a MatrixRTC call. Mixed into ``MatrixAdapter``."""

    voice_scope = "chat"

    # Set by the gateway at ``/voice join``, and called with the room id when the bot
    # leaves a call without ``/voice leave``.
    _on_voice_disconnect: Optional[Callable[[str], None]] = None

    # --- registries (getattr-guarded: object.__new__ test instances skip __init__) ---

    @property
    def rtc_sessions(self) -> MatrixRTCSessions:
        """For each room, the gateway session that call audio speaks into."""
        return _lazy_attr(self, "_rtc_sessions", lambda: MatrixRTCSessions(self))

    @property
    def rtc_receivers(self) -> Dict[str, MatrixRTCReceiver]:
        """For each room, the receiver listening to its call."""
        return _lazy_attr(self, "_rtc_receivers", dict)

    @property
    def _rtc_call_bindings(self) -> Dict[str, Any]:
        """For each room, the session binding that owns the live call."""
        return _lazy_attr(self, "_rtc_call_binding_map", dict)

    @property
    def _rtc_foci(self) -> Dict[str, str]:
        """For each room with a live call, the MatrixRTC service URL that the bot uses."""
        return _lazy_attr(self, "_rtc_focus_map", dict)

    # --- gateway duck-types ---

    def bind_voice_session(self, room_id: str, source) -> None:
        """Point the room's call at the session that its typed messages already use.

        Called before the join, because audio can arrive with the first frame and an
        unbound room drops it.
        """
        self.rtc_sessions.bind(room_id, source)

    async def get_user_voice_channel(self, room_id: str, user_id: str) -> Optional[MatrixCall]:
        """The room's call when *user_id* has a live membership of it."""
        state = await asyncio.wait_for(self._fetch_room_state(room_id), REQUEST_TIMEOUT)
        self._remember_call_state(room_id, state)
        if user_id not in live_call_members(state):
            logger.debug("MatrixRTC: %s has no live call membership in %s", user_id, room_id)
            return None
        return MatrixCall(room_id=room_id, name=await self._rtc_room_name(room_id))

    async def join_voice_channel(self, channel) -> bool:
        """Hear the call (``receiver``), speak into it (``publisher``) and join its membership.

        Joining a room that already has a live call keeps the connection and moves the
        call to the session that the gateway has just bound. A join that arrives while an
        earlier one is still connecting does the same to that join, and both return its
        result. Errors from the credential exchange, the SFU or the homeserver propagate,
        and the gateway turns them into the failure message.
        """
        room_id = getattr(channel, "room_id", None) or str(channel)
        binding = self.rtc_sessions.binding_for(room_id)
        tasks = _lazy_attr(self, "_rtc_join_tasks", dict)
        if room_id in tasks and not tasks[room_id].done():
            self._rtc_call_bindings[room_id] = binding
            return await asyncio.shield(tasks[room_id])
        if room_id in self.rtc_receivers:
            self._check_call_owner(room_id, binding, self._rtc_foci.get(room_id))
            self._rtc_call_bindings[room_id] = binding
            return True
        task = asyncio.create_task(self._join_call(room_id))
        tasks[room_id] = task
        try:
            return await task
        finally:
            if tasks.get(room_id) is task:
                tasks.pop(room_id, None)

    async def _join_call(self, room_id: str) -> bool:
        binding = self.rtc_sessions.binding_for(room_id)
        self._rtc_call_bindings[room_id] = binding
        receiver = None
        membership_sent = False
        try:
            with self.rtc_sessions.scope_for(room_id):
                state = await asyncio.wait_for(self._fetch_room_state(room_id), REQUEST_TIMEOUT)
                self._remember_call_state(room_id, state)
                if any(isinstance(event, dict) and event.get("type") == "m.room.encryption"
                       for event in state):
                    raise MatrixRTCError(
                        "calls in encrypted rooms are not supported, because Hermes does not "
                        "implement MatrixRTC media encryption")
                self._check_call(room_id)
                sfu_url, jwt, focus_url = await asyncio.wait_for(fetch_livekit_credentials(
                    self._homeserver, self._user_id, self._access_token, room_id,
                    self._rtc_device_id(), session=self._rtc_http_session()), REQUEST_TIMEOUT)
                self._check_call(room_id, focus_url)
            receiver = MatrixRTCReceiver(
                on_transcript=functools.partial(self.rtc_sessions.on_transcript, room_id),
                is_authorized=functools.partial(self.rtc_sessions.audio_allowed, room_id),
                # The room plays the bot's own reply back to it. While the publisher is
                # speaking, the receiver drops that audio, and speech loud enough to pass
                # the gate is a barge-in.
                is_speaking=functools.partial(self.is_speaking_in, room_id),
                on_barge_in=functools.partial(self.rtc_sessions.barge_in, room_id))
            with self.rtc_sessions.scope_for(room_id):
                await asyncio.wait_for(receiver.connect(sfu_url, jwt), REQUEST_TIMEOUT)
                self._check_call(room_id, focus_url)
                self._rtc_foci[room_id] = focus_url
                self.rtc_receivers[room_id] = receiver
                try:
                    await self.start_rtc_audio(room_id, receiver.room)
                except Exception as exc:
                    logger.warning("MatrixRTC: joined %s without an outbound track: %s", room_id, exc)
                self._check_call(room_id, focus_url)
                lease = self._membership_lease(room_id, binding, focus_url)
                _lazy_attr(self, "_rtc_leases", dict)[room_id] = lease
                membership_sent = True
                await lease.join()
                self._check_call(room_id, focus_url)
                return True
        except BaseException:
            with self.rtc_sessions.scope_for(room_id):
                self.rtc_sessions.unbind(room_id)
                await self._close_call(room_id, receiver, binding, clear_membership=membership_sent)
            raise

    def _membership_lease(self, room_id: str, binding, focus_url: str) -> CallMembershipLease:
        user_id = binding.account[1]
        device_id = str(getattr(binding.client, "device_id", "") or "")

        def content(created_ts, expires):
            return call_membership_content(user_id, room_id, device_id, focus_url,
                                           created_ts=created_ts, expires=expires)

        return CallMembershipLease(
            room_id, call_membership_state_key(user_id, device_id),
            request=binding.api.request,
            publish=functools.partial(self._publish_call_membership, room_id, binding=binding),
            content=content,
            check=functools.partial(self._check_call, room_id),
            on_lost=functools.partial(self._leave_on_own, room_id),
            delay_ms=leave_delay_ms())

    def _check_call(self, room_id: str, focus_url: Optional[str] = None) -> None:
        """``_check_call_owner`` for the binding that owns the room's call now."""
        self._check_call_owner(room_id, self._rtc_call_bindings.get(room_id), focus_url)

    def _check_call_owner(self, room_id: str, binding, focus_url: Optional[str] = None) -> None:
        """Raise unless *binding* still owns the room's call and its requester may use it.

        The requester must still pass the gateway's policy and still have a live
        membership of the call, so the bot leaves after the requester hangs up. With
        *focus_url*, the requester must also publish through that MatrixRTC service.
        """
        if binding is None or not self.rtc_sessions.current(room_id, binding):
            raise RuntimeError("MatrixRTC receiving session is no longer available")
        if not self.rtc_sessions.is_user_authorized(room_id, binding.source.user_id):
            raise RuntimeError("MatrixRTC requester is no longer authorised")
        if focus_url is None:
            return
        requester = binding.source.user_id
        memberships = live_call_memberships(getattr(self, "_rtc_call_state", {}).get(room_id, {}).values())
        services = {membership.publishing_service_url(memberships)
                    for membership in memberships if membership.user_id == requester}
        if focus_url.rstrip("/") in services:
            return
        where = ", ".join(sorted(service for service in services if service)) or "an unknown transport"
        raise MatrixRTCError(
            f"{requester} publishes call media through {where}, but Hermes can only use its "
            f"homeserver's MatrixRTC service at {focus_url}. Calls must stay within one deployment.")

    async def _leave_on_own(self, room_id: str) -> None:
        """Leave a call without ``/voice leave``, and tell the gateway that the call ended."""
        await self.leave_voice_channel(room_id)
        if self._on_voice_disconnect is not None:
            self._on_voice_disconnect(room_id)

    async def leave_voice_channel(self, room_id: str) -> None:
        """Stop speaking, stop listening, unbind, and clear our call membership.

        ``receiver.close()`` flushes the utterance still in the segmenter, and that
        transcript needs the binding to reach a session, so unbinding comes last.

        Every step runs whether or not this process joined the call. The receivers
        live in this process, but the membership lives in the room, so after a restart
        clearing the state event is the only work left.
        """
        task = getattr(self, "_rtc_join_tasks", {}).get(room_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        with self.rtc_sessions.scope_for(room_id):
            await self._close_call(room_id)
        self.rtc_sessions.unbind(room_id)

    async def _close_call(self, room_id: str, receiver=None, binding=None, *,
                          clear_membership: bool = True) -> None:
        binding = (binding or self._rtc_call_bindings.get(room_id)
                   or self.rtc_sessions.binding_for(room_id))
        self._rtc_call_bindings.pop(room_id, None)
        self._rtc_foci.pop(room_id, None)
        lease = getattr(self, "_rtc_leases", {}).pop(room_id, None)
        try:
            if lease is not None:
                await lease.close()
            await self.stop_rtc_audio(room_id)
        finally:
            receiver = self.rtc_receivers.pop(room_id, None) or receiver
            try:
                if receiver is not None:
                    await receiver.close()
            finally:
                if clear_membership:
                    try:
                        await self._publish_call_membership(room_id, {}, binding=binding)
                    except Exception:
                        logger.warning("MatrixRTC: could not clear call membership in %s", room_id,
                                       exc_info=True)

    async def close_rtc_calls(self) -> None:
        """Leave every call on disconnect, including joins still in progress."""
        joining = {room for room, task in getattr(self, "_rtc_join_tasks", {}).items() if not task.done()}
        live = set(self.rtc_receivers) - joining
        await asyncio.gather(*(self._leave_on_own(room) for room in live),
                             *(self.leave_voice_channel(room) for room in joining),
                             return_exceptions=True)

    def _remember_call_state(self, room_id: str, events: list) -> None:
        states = _lazy_attr(self, "_rtc_call_state", dict)
        states[room_id] = {(event.get("type"), event.get("state_key")): event
                           for event in events if isinstance(event, dict)
                           and event.get("type") in _TRACKED_STATE_TYPES}
        self.rtc_sessions.invalidate(room_id)

    def update_rtc_call_state(self, sync_data: dict) -> None:
        """Apply call and room membership changes from a sync to rooms with a known call.

        A change in the room of a live call rechecks the call's requester at once, so the
        bot leaves soon after the requester hangs up or leaves the room.
        """
        states = getattr(self, "_rtc_call_state", {})
        rooms = sync_data.get("rooms", {})
        changed = set()
        for room_id in rooms.get("leave", {}):
            if room_id in states:
                states[room_id] = {}
                changed.add(room_id)
        for room_id, room in rooms.get("join", {}).items():
            if room_id not in states:
                continue
            for section in ("state", "timeline"):
                for event in room.get(section, {}).get("events", []):
                    if event.get("type") in _TRACKED_STATE_TYPES and "state_key" in event:
                        states[room_id][event.get("type"), event.get("state_key")] = event
                        changed.add(room_id)
        leases = getattr(self, "_rtc_leases", {})
        for room_id in changed:
            self.rtc_sessions.invalidate(room_id)
            if room_id in leases:
                with self.rtc_sessions.scope_for(room_id):
                    leases[room_id].recheck()

    def get_voice_channel_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """``/voice status``: who else is on the call, or None when we are not in one.

        Synchronous like Discord's, so it reports the SFU's participant list and does
        not read room state.
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
        """The device that owns the access token. The LiveKit identity is
        ``{user_id}:{device_id}``, and the homeserver recognises the client's resolved
        device even when the configured value is stale."""
        client = getattr(self, "_client", None)
        return str(getattr(client, "device_id", "") or getattr(self, "_device_id", "") or "")

    def _rtc_http_session(self):
        """The adapter's own aiohttp session, with proxy and TLS already configured, or
        None to let ``focus`` open a short-lived one."""
        return getattr(getattr(getattr(self, "_client", None), "api", None), "session", None)

    async def _fetch_room_state(self, room_id: str) -> List[dict]:
        """The room's raw state events.

        Raw because mautrix does not model ``org.matrix.msc3401.call.member``, so the
        typed client would drop the events that this module needs.
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

    async def _publish_call_membership(self, room_id: str, content: dict, *, binding=None) -> None:
        """PUT our call membership for *room_id*. Empty *content* means that we have left.

        Uses the raw ``api.request`` because mautrix does not model the event type.
        Raises when the homeserver refuses the event, so a join never reports success
        for a bot that clients cannot see.
        """
        api = binding.api if binding is not None else getattr(getattr(self, "_client", None), "api", None)
        if api is None:
            raise RuntimeError("MatrixRTC Matrix client is disconnected")
        user_id = binding.account[1] if binding is not None else self._user_id
        device_id = (str(getattr(binding.client, "device_id", "") or "") if binding is not None
                     else self._rtc_device_id())
        state_key = call_membership_state_key(user_id, device_id)
        from mautrix.api import Method
        await asyncio.wait_for(api.request(
                Method.PUT,
                f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}"
                f"/state/{CALL_MEMBER_TYPE}/{quote(state_key, safe='')}",
                content=content), REQUEST_TIMEOUT)

    async def _rtc_room_name(self, room_id: str) -> str:
        """The room's display name for the join confirmation, or the id if it has none."""
        resolve = getattr(self, "_resolve_room_identity", None)
        if resolve is not None:
            try:
                return (await resolve(room_id)).display_name or room_id
            except Exception:
                pass
        return room_id

    def _rtc_cached_room_name(self, room_id: str) -> str:
        """The same name from the identity cache, because ``get_voice_channel_info`` is
        synchronous."""
        identity = (getattr(self, "_room_identities", None) or {}).get(room_id)
        return getattr(identity, "display_name", None) or room_id
