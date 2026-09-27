"""Park unmentioned MSC3245 voice messages until the sender's bare @mention claims them.

Element X sends a mention typed while recording as a SEPARATE ``m.text`` event after the
voice event (which carries an empty ``m.mentions``). Under ``require_mention`` the voice is
parked instead of forgotten; a bare mention from the same sender in the same room within
the window claims it. Unmentioned voices are never downloaded or transcribed while parked.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Set, Tuple

CLAIM_WINDOW_SECONDS = 120.0


def is_voice_event(content: dict) -> bool:
    return content.get("msgtype") == "m.audio" and "org.matrix.msc3245.voice" in content


class ParkedVoices:
    def __init__(self, window: float = CLAIM_WINDOW_SECONDS) -> None:
        self._window = window
        # (room_id, sender) -> (parked_at, event_id, content, relates_to)
        self._parked: Dict[Tuple[str, str], Tuple[float, str, dict, dict]] = {}
        self._claimed: Set[str] = set()

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._window
        self._parked = {k: v for k, v in self._parked.items() if v[0] >= cutoff}

    def park(self, room_id: str, sender: str, event_id: str, content: dict, relates_to: dict) -> None:
        self._prune()
        self._parked[(room_id, sender)] = (time.monotonic(), event_id, content, relates_to)

    def claim(self, room_id: str, sender: str) -> Optional[Tuple[str, dict, dict]]:
        """Pop the sender's parked voice for this room; the claimed event id then passes the
        mention gate exactly once (see ``consume_claim``) instead of being re-parked."""
        self._prune()
        entry = self._parked.pop((room_id, sender), None)
        if entry is None:
            return None
        _ts, event_id, content, relates_to = entry
        self._claimed.add(event_id)
        return event_id, content, relates_to

    def consume_claim(self, event_id: str) -> bool:
        if event_id in self._claimed:
            self._claimed.discard(event_id)
            return True
        return False
