"""Park unmentioned MSC3245 voice messages until the sender's bare @mention claims them.

Element X sends a mention typed while recording as a SEPARATE ``m.text`` event after the
voice event (which carries an empty ``m.mentions``). Under ``require_mention`` the voice is
parked instead of forgotten; a bare mention from the same sender in the same room within
the window claims it. Unmentioned voices are never downloaded or transcribed while parked.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional, Tuple

CLAIM_WINDOW_SECONDS = 120.0
# How long a bare mention waits for a voice from the same /sync batch that is still being gated.
SETTLE_TIMEOUT_SECONDS = 5.0

# (voice event_id, content, relates_to)
ParkedVoice = Tuple[str, dict, dict]


def has_voice_marker(content: dict) -> bool:
    """The single MSC3245 voice-message check (shared with the adapter's media classifier)."""
    return content.get("org.matrix.msc3245.voice") is not None


def is_voice_event(content: dict) -> bool:
    return content.get("msgtype") == "m.audio" and has_voice_marker(content)


class ParkedVoices:
    def __init__(self) -> None:
        # (room_id, sender) -> (parked_at, parked voice)
        self._parked: Dict[Tuple[str, str], Tuple[float, ParkedVoice]] = {}
        # (room_id, sender) -> set once that sender's voice has been gated (parked or not).
        # mautrix runs one /sync batch's events as concurrent tasks, so the voice may still be
        # awaiting room identity when its bare mention is handled.
        self._inflight: Dict[Tuple[str, str], asyncio.Event] = {}

    def _prune(self) -> None:
        cutoff = time.monotonic() - CLAIM_WINDOW_SECONDS
        self._parked = {k: v for k, v in self._parked.items() if v[0] >= cutoff}

    def pending(self, room_id: str, sender: str) -> bool:
        """Cheap pre-check: a voice is parked (maybe expired; ``claim`` prunes) or still being gated."""
        key = (room_id, sender)
        return key in self._parked or key in self._inflight

    def begin(self, room_id: str, sender: str) -> asyncio.Event:
        """Mark a voice as being gated. Call before the first await; always pair with ``release``."""
        gate = self._inflight[(room_id, sender)] = asyncio.Event()
        return gate

    def release(self, room_id: str, sender: str, gate: asyncio.Event) -> None:
        gate.set()
        if self._inflight.get((room_id, sender)) is gate:
            del self._inflight[(room_id, sender)]

    async def settle(self, room_id: str, sender: str) -> None:
        """Wait (bounded) for a concurrently gated voice from this sender to be parked or dropped."""
        gate = self._inflight.get((room_id, sender))
        if gate is not None:
            try:
                await asyncio.wait_for(gate.wait(), SETTLE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                pass

    def park(self, room_id: str, sender: str, event_id: str, content: dict, relates_to: dict) -> None:
        self._prune()
        self._parked[(room_id, sender)] = (time.monotonic(), (event_id, content, relates_to))

    def claim(self, room_id: str, sender: str) -> Optional[ParkedVoice]:
        """Pop the sender's parked voice for this room (the caller re-dispatches it with
        ``mention_claimed=True`` so it passes the mention gate without re-parking)."""
        self._prune()
        entry = self._parked.pop((room_id, sender), None)
        return entry[1] if entry else None
