"""Park unmentioned MSC3245 voice messages until the sender's bare @mention claims them.

Element X sends a mention typed while recording as a SEPARATE ``m.text`` event after the
voice event (which carries an empty ``m.mentions``). Under ``require_mention`` the voice is
parked instead of forgotten; a bare mention from the same sender in the same room within
the window claims it. Unmentioned voices are never downloaded or transcribed while parked.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from typing import Dict, List, Optional, Tuple

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


class VoiceGate:
    """One voice being gated; ``seq`` orders voices from the same sender by arrival."""

    __slots__ = ("seq", "done")

    def __init__(self, seq: int) -> None:
        self.seq = seq
        self.done = asyncio.Event()


class ParkedVoices:
    def __init__(self) -> None:
        # (room_id, sender) -> (parked_at, seq, parked voice)
        self._parked: Dict[Tuple[str, str], Tuple[float, int, ParkedVoice]] = {}
        # (room_id, sender) -> every voice of that sender still being gated. mautrix runs one
        # /sync batch's events as concurrent tasks, so a voice may still be awaiting room
        # identity when its bare mention is handled -- and a sender can have several in flight.
        self._inflight: Dict[Tuple[str, str], List[VoiceGate]] = {}
        # (room_id, sender) -> seq of the newest voice parked while gates were in flight, so an
        # older voice finishing late never lands over (or after the claim of) a newer one.
        self._newest: Dict[Tuple[str, str], int] = {}
        self._seq = itertools.count()

    def _prune(self) -> None:
        cutoff = time.monotonic() - CLAIM_WINDOW_SECONDS
        self._parked = {k: v for k, v in self._parked.items() if v[0] >= cutoff}

    def pending(self, room_id: str, sender: str) -> bool:
        """Cheap pre-check: a voice is parked (maybe expired; ``claim`` prunes) or still being gated."""
        key = (room_id, sender)
        return key in self._parked or key in self._inflight

    def begin(self, room_id: str, sender: str) -> VoiceGate:
        """Mark a parkable voice as being gated. Call before the first await; always pair with
        ``release`` (idempotent, so it may run early and again in a ``finally``)."""
        gate = VoiceGate(next(self._seq))
        self._inflight.setdefault((room_id, sender), []).append(gate)
        return gate

    def release(self, room_id: str, sender: str, gate: VoiceGate) -> None:
        gate.done.set()
        key = (room_id, sender)
        gates = self._inflight.get(key)
        if gates and gate in gates:
            gates.remove(gate)
            if not gates:  # no older voice can park any more
                del self._inflight[key]
                self._newest.pop(key, None)

    async def settle(self, room_id: str, sender: str) -> None:
        """Wait (bounded) for every concurrently gated voice from this sender to park or drop."""
        gates = self._inflight.get((room_id, sender))
        if gates:
            waits = [asyncio.ensure_future(g.done.wait()) for g in gates]
            try:
                await asyncio.wait(waits, timeout=SETTLE_TIMEOUT_SECONDS)
            finally:
                for w in waits:
                    w.cancel()

    def park(self, room_id: str, sender: str, gate: VoiceGate, event_id: str, content: dict,
             relates_to: dict) -> None:
        key = (room_id, sender)
        if gate.seq < self._newest.get(key, -1):
            return  # a newer voice from this sender already parked (and maybe was claimed)
        self._prune()
        self._parked[key] = (time.monotonic(), gate.seq, (event_id, content, relates_to))
        self._newest[key] = gate.seq

    def claim(self, room_id: str, sender: str) -> Optional[ParkedVoice]:
        """Pop the sender's parked voice for this room (the caller re-dispatches it with
        ``mention_claimed=True`` so it passes the mention gate without re-parking)."""
        self._prune()
        entry = self._parked.pop((room_id, sender), None)
        return entry[2] if entry else None
