"""MatrixRTC call membership: reading the room's call and keeping our own place in it.

Hermes uses the session form of MatrixRTC membership, which matrix-js-sdk still reads
and writes for room calls: one ``org.matrix.msc3401.call.member`` state event per
device, with empty content once that device has left. The newer ``m.rtc.member``
form travels in sticky timeline events (MSC4354) rather than room state, and Hermes
does not read it.

A membership is live until ``created_ts + expires``. As in matrix-js-sdk,
``created_ts`` defaults to the event's ``origin_server_ts`` and ``expires`` defaults
to four hours. A client that crashes cannot clear its own membership, so before
joining it also schedules a delayed leave event (MSC4140) and keeps restarting it
while the call is up. The homeserver sends the leave when the restarts stop.

Each membership also says where its media is published. With ``focus_selection:
"multi_sfu"``, which current Element Call writes, a member publishes on the first
transport in its own ``foci_preferred``. With ``"oldest_membership"``, it publishes on
the transport of the call's oldest membership.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional
from urllib.parse import quote

from .segmenter import _positive_float, _rtc_config

logger = logging.getLogger(__name__)

CALL_MEMBER_TYPE = "org.matrix.msc3401.call.member"

# matrix-js-sdk's DEFAULT_EXPIRE_DURATION, used when a membership states no expiry.
DEFAULT_EXPIRY_MS = 4 * 60 * 60 * 1000

# matrix-js-sdk's default delayed leave timeout. ``matrix.rtc.leave_delay_seconds``
# overrides it, within the homeserver's ``max_event_delay_duration``.
DEFAULT_LEAVE_DELAY_MS = 8_000

# How often the lease checks the call's requester between room state changes. The periodic
# check covers changes that happen without a state event, such as a membership reaching its
# expiry time or a change to the allowlist.
REQUESTER_CHECK_MS = 30_000

REQUEST_TIMEOUT = 15


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def membership_user_id(state_key: str) -> str:
    """The Matrix user id inside an RTC membership state key.

    The suffix count is not fixed: ``@u:hs`` (MSC3401 as first shipped), the per-device
    ``@u:hs_DEVICE`` / ``_@u:hs_DEVICE``, and Element's ``_@u:hs_DEVICE_m.call``, which
    appends the application too. The user id stops at the first underscore after the
    server name, because a server name cannot contain one. A localpart may, so the scan
    starts at the colon and ``@my_bot:hs`` survives whole.
    """
    key = state_key[1:] if state_key.startswith("_") else state_key
    colon = key.find(":")
    if not key.startswith("@") or colon < 0:
        return key
    cut = key.find("_", colon)
    return key[:cut] if cut > 0 else key


def _is_room_call_session(content: Any) -> bool:
    """Whether *content* is a session membership of the room's own call.

    The shape checks are the ones in matrix-js-sdk's ``checkSessionsMembershipData``.
    Element ignores a membership that fails them, so it does not count here either.
    """
    if not isinstance(content, dict):
        return False
    focus = content.get("focus_active")
    foci = content.get("foci_preferred", [])
    device = content.get("device_id")
    return (content.get("application") == "m.call"
            and content.get("call_id") == ""
            and isinstance(device, str) and bool(device)
            and isinstance(focus, dict) and isinstance(focus.get("type"), str)
            and isinstance(foci, list)
            and all(isinstance(f, dict) and isinstance(f.get("type"), str) for f in foci)
            and ("created_ts" not in content or _is_number(content["created_ts"])))


def _service_url(transport: Any) -> Optional[str]:
    """The MatrixRTC authorisation service URL of a LiveKit transport, without a trailing slash."""
    if not isinstance(transport, dict) or transport.get("type") != "livekit":
        return None
    url = transport.get("livekit_service_url")
    return url.rstrip("/") if isinstance(url, str) and url else None


@dataclass(frozen=True)
class CallMembership:
    """One device's membership of a room's call, as read from room state.

    *service_url* comes from the first entry of ``foci_preferred`` and is None when that
    entry is not a LiveKit transport.
    """

    user_id: str
    device_id: str
    created_ms: float
    expires_at_ms: float
    focus_selection: Optional[str]
    service_url: Optional[str]

    @classmethod
    def from_event(cls, event: Any) -> Optional["CallMembership"]:
        """Parse one raw state event. Anything that is not a valid membership is None."""
        if not isinstance(event, dict) or event.get("type") != CALL_MEMBER_TYPE:
            return None
        content = event.get("content")
        if not _is_room_call_session(content):
            return None
        sender = event.get("sender")
        if not isinstance(sender, str) or membership_user_id(str(event.get("state_key") or "")) != sender:
            return None
        created = content.get("created_ts", event.get("origin_server_ts"))
        if not _is_number(created):
            return None
        expires = content.get("expires")
        if not _is_number(expires):
            expires = DEFAULT_EXPIRY_MS
        foci = content.get("foci_preferred") or []
        return cls(sender, content["device_id"], created, created + expires,
                   content["focus_active"].get("focus_selection"),
                   _service_url(foci[0]) if foci else None)

    def is_live(self, now_ms: float) -> bool:
        return self.expires_at_ms > now_ms

    def publishing_service_url(self, memberships: Iterable["CallMembership"]) -> Optional[str]:
        """The service URL of the transport that this member publishes on, or None if unknown.

        This is matrix-js-sdk's ``CallMembership.getTransport``. *memberships* are the
        call's live memberships, which decide the oldest one.
        """
        if self.focus_selection == "multi_sfu":
            return self.service_url
        if self.focus_selection != "oldest_membership":
            return None
        oldest = min(memberships, key=lambda membership: membership.created_ms, default=self)
        if oldest.focus_selection not in ("multi_sfu", "oldest_membership"):
            return None
        return oldest.service_url


def live_call_memberships(state_events: Iterable[Any],
                          now_ms: Optional[float] = None) -> list[CallMembership]:
    """Every unexpired call membership in a room's raw state events."""
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    parsed = (CallMembership.from_event(event) for event in state_events or [])
    return [membership for membership in parsed
            if membership is not None and membership.is_live(now_ms)]


def live_call_members(state_events: Iterable[Any], now_ms: Optional[float] = None) -> set[str]:
    """The user ids with at least one live device in the room's call."""
    return {membership.user_id for membership in live_call_memberships(state_events, now_ms)}


def call_membership_state_key(user_id: str, device_id: str) -> str:
    """The state key matrix-js-sdk uses for a room call outside MSC3757 room versions."""
    return f"_{user_id}_{device_id}_m.call"


def call_membership_content(user_id: str, room_id: str, device_id: str, service_url: str, *,
                            created_ts: Optional[int] = None,
                            expires: int = DEFAULT_EXPIRY_MS) -> dict:
    """Our own membership, in the shape matrix-js-sdk's ``makeMyMembership`` writes.

    ``multi_sfu`` tells clients that the bot publishes on the first transport in
    ``foci_preferred``, whatever transport the call's oldest member uses. That transport
    lists the authorisation service URL that clients call for a LiveKit token, not the
    SFU websocket URL. ``membershipID`` is the LiveKit identity that the service assigns
    to a session membership. A renewal keeps ``created_ts`` from the first join and
    extends ``expires``.
    """
    content = {
        "application": "m.call",
        "call_id": "",
        "scope": "m.room",
        "device_id": device_id,
        "membershipID": f"{user_id}:{device_id}",
        "expires": expires,
        "focus_active": {"type": "livekit", "focus_selection": "multi_sfu"},
        "foci_preferred": [{"type": "livekit", "livekit_alias": room_id,
                            "livekit_service_url": service_url}],
    }
    if created_ts is not None:
        content["created_ts"] = created_ts
    return content


def leave_delay_ms() -> int:
    """The delayed leave timeout from ``matrix.rtc.leave_delay_seconds``."""
    seconds = _positive_float(_rtc_config().get("leave_delay_seconds"), DEFAULT_LEAVE_DELAY_MS / 1000)
    return int(seconds * 1000)


class CallMembershipLease:
    """Keeps one device's call membership valid until the call is closed.

    *request* is the bound client's raw ``api.request`` and *publish* writes membership
    content. *content* builds that content from ``(created_ts, expires)``. *check*
    raises when the call's receiving session is no longer valid, and *on_lost* leaves
    the call. The lease runs *check* at every renewal, at least every
    ``REQUESTER_CHECK_MS``, and whenever ``recheck`` reports a change to the room's call
    state. *sleep* and *wall_ms* are injectable so that tests can drive the renewal
    schedule without waiting.
    """

    def __init__(self, room_id: str, state_key: str, *,
                 request: Callable[..., Awaitable[Any]],
                 publish: Callable[[dict], Awaitable[None]],
                 content: Callable[[Optional[int], int], dict],
                 check: Callable[[], None],
                 on_lost: Callable[[], Awaitable[None]],
                 delay_ms: int,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 wall_ms: Callable[[], float] = lambda: time.time() * 1000):
        self.room_id = room_id
        self.state_key = state_key
        self._request = request
        self._publish = publish
        self._content = content
        self._check = check
        self._on_lost = on_lost
        self.delay_ms = delay_ms
        self._sleep = sleep
        self._wall_ms = wall_ms
        self.delay_id: Optional[str] = None
        self._created_ts = 0
        self._task: Optional[asyncio.Task] = None
        self._lost = False

    async def _call(self, method, path: str, **kwargs):
        return await asyncio.wait_for(self._request(method, path, **kwargs), REQUEST_TIMEOUT)

    def _delayed_event_path(self, action: str) -> str:
        return (f"/_matrix/client/unstable/org.matrix.msc4140/delayed_events/"
                f"{quote(str(self.delay_id), safe='')}/{action}")

    async def join(self) -> None:
        """Schedule the delayed leave, then publish the membership.

        The delayed leave goes first, as in matrix-js-sdk's MembershipManager, so a
        crash between the two requests leaves no membership behind.
        """
        from mautrix.api import Method

        versions = await self._call(Method.GET, "/_matrix/client/versions")
        if (versions.get("unstable_features") or {}).get("org.matrix.msc4140"):
            path = (f"/_matrix/client/v3/rooms/{quote(self.room_id, safe='')}"
                    f"/state/{CALL_MEMBER_TYPE}/{quote(self.state_key, safe='')}")
            response = await self._call(Method.PUT, path, content={},
                                        query_params={"org.matrix.msc4140.delay": self.delay_ms})
            self.delay_id = response["delay_id"]
        else:
            logger.warning("MatrixRTC: the homeserver does not offer delayed events, so a crash "
                           "leaves the call membership in %s until it expires", self.room_id)
        self._created_ts = int(self._wall_ms())
        await self._publish(self._content(None, DEFAULT_EXPIRY_MS))
        self._task = asyncio.create_task(self._renew())

    def recheck(self) -> None:
        """Run *check* now, because the room's call state has changed, and leave if it fails."""
        if self._lost or self._task is None or self._task.done():
            return
        try:
            self._check()
        except Exception as exc:
            logger.info("MatrixRTC: leaving the call in %s: %s", self.room_id, exc)
            self._lost = True
            self._task.cancel()
            self._task = asyncio.create_task(self._lose())

    async def _lose(self) -> None:
        self._lost = True
        try:
            await self._on_lost()
        except Exception:
            logger.warning("MatrixRTC: could not leave the call in %s", self.room_id, exc_info=True)

    async def _renew(self) -> None:
        from mautrix.api import Method

        renew_ms = self.delay_ms / 2 if self.delay_id else DEFAULT_EXPIRY_MS / 2
        interval_ms = min(renew_ms, REQUESTER_CHECK_MS)
        periods = 1
        try:
            while True:
                await self._sleep(interval_ms / 1000)
                self._check()
                if self.delay_id:
                    await self._call(Method.POST, self._delayed_event_path("restart"), content={})
                if self._wall_ms() - self._created_ts >= DEFAULT_EXPIRY_MS * periods / 2:
                    periods += 1
                    await self._publish(self._content(self._created_ts, DEFAULT_EXPIRY_MS * periods))
                self._check()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("MatrixRTC: call membership renewal failed in %s", self.room_id, exc_info=True)
            await self._lose()

    async def close(self) -> None:
        """Stop renewing and send the delayed leave now, if one is scheduled."""
        from mautrix.api import Method

        task, self._task = self._task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self.delay_id is None:
            return
        try:
            await self._call(Method.POST, self._delayed_event_path("send"), content={})
        except Exception:
            logger.warning("MatrixRTC: delayed leave could not be sent in %s", self.room_id, exc_info=True)
