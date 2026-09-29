"""Room state builders shared by the MatrixRTC tests."""

from __future__ import annotations

import time

from plugins.platforms.matrix.rtc.join import CALL_MEMBER_TYPE

FOCUS_URL = "https://call.hs.tld/livekit/jwt"
SESSION = {"application": "m.call", "call_id": "", "device_id": "DEVICEAAA", "scope": "m.room",
           "focus_active": {"type": "livekit", "focus_selection": "oldest_membership"},
           "foci_preferred": [{"type": "livekit", "livekit_service_url": FOCUS_URL}]}


def call_member_event(user: str, device: str = "DEVICEAAA", *, event_type: str = CALL_MEMBER_TYPE,
                      sender: str | None = None, age_ms: float = 1_000, now_ms: float | None = None,
                      **content) -> dict:
    """An ``org.matrix.msc3401.call.member`` state event as ``/state`` returns it.

    A content value of None removes that key from the session membership.
    """
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    body = {**SESSION, "device_id": device, **content}
    return {"type": event_type, "state_key": f"_{user}_{device}_m.call", "sender": sender or user,
            "origin_server_ts": now_ms - age_ms,
            "content": {key: value for key, value in body.items() if value is not None}}


def room_member_event(user: str, membership: str = "join") -> dict:
    return {"type": "m.room.member", "state_key": user, "sender": user,
            "content": {"membership": membership}}


def call_state(*participants: tuple[str, str]) -> list[dict]:
    """Room state for users who have joined the room and have a live call membership."""
    events = []
    for user, device in participants:
        events += [call_member_event(user, device), room_member_event(user)]
    return events


def remembered(events: list[dict]) -> dict:
    """*events* keyed by type and state key, as ``_remember_call_state`` stores them."""
    return {(event["type"], event["state_key"]): event for event in events}


def sync(room_id: str, *events: dict) -> dict:
    """A sync response whose timeline for *room_id* contains *events*."""
    return {"rooms": {"join": {room_id: {"timeline": {"events": list(events)}}}}}
