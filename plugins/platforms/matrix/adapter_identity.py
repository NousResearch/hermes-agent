"""Own-account identity and event replay tracking for the Matrix adapter — split out of ``adapter.py``
for its size cap.

Two related answers to "who sent this, and did we already see it?": own-account classification (the
self-chat gate plus the echo-loop guards of #15763) and the bounded event-ID tracker that drops
replays of our own sends.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Bounded tracker: keep the newest N event IDs, evict the oldest (the WhatsApp bridge's outbound_id
# pattern).
DEDUPE_MAXLEN = 1000


class MatrixIdentity:
    """Own-account classification + the event-ID tracker, composed by ``MatrixAdapter``.

    ``own_user_id`` is a callable rather than a value: the adapter resolves its MXID from ``whoami``
    *after* construction, and every predicate must see the resolved identity. ``dm_room_check`` is the
    adapter's room classifier, so self-chat scoping follows the same DM rule as every other Matrix path.
    """

    def __init__(
        self,
        own_user_id: Callable[[], str],
        dm_room_check: Callable[[str], Awaitable[bool]],
    ) -> None:
        self._own_user_id = own_user_id
        self._dm_room_check = dm_room_check
        self._sent: deque[str] = deque(maxlen=DEDUPE_MAXLEN)
        self._sent_set: set[str] = set()

    @staticmethod
    def is_system_or_bridge_sender(sender: str) -> bool:
        """True for appservice/bridge/system identities (``@_telegram_123:server``) or malformed IDs.
        Never offer these a pairing code: an approved bridge would relay every outbound message
        back as an "authorized user message" (echo loop).

        We treat these as system identities for pairing purposes: they should never be offered a pairing
        code, because an operator approving the code would hand the bridge itself permanent authorization —
        and every outbound message relayed by the bridge would then loop back into the agent as an
        "authorized user message", which is the root of issue #15763.
        """
        localpart = (sender or "").strip().lstrip("@").partition(":")[0]
        return not localpart or localpart.startswith("_")

    def is_duplicate_event(self, event_id) -> bool:
        """Return True if this event was already processed. Tracks the ID otherwise."""
        if not event_id:
            return False
        if event_id in self._sent_set:
            return True
        self.remember_event(event_id)
        return False

    def remember_event(self, event_id) -> None:
        """Track an event ID we SENT (self-chat echo suppression): when the sync replays our own
        send back (sender == the bot account), ``is_duplicate_event`` drops it as already seen.
        Same bounded tracker as inbound dedup — the WhatsApp bridge's outbound_id pattern."""
        if not event_id or event_id in self._sent_set:
            return
        if len(self._sent) == self._sent.maxlen:
            self._sent_set.discard(self._sent[0])
        self._sent.append(event_id)
        self._sent_set.add(event_id)

    def is_self_sender(self, sender: str) -> bool:
        """True if *sender* is the bot itself (case-insensitive: homeservers vary localpart case). With
        no resolved user_id we can't prove a sender is NOT us, so return True — dropping our own
        events beats an echo loop ("hall of mirrors").

        Matrix user IDs are byte-compared after trimming whitespace and lowercasing — some homeservers
        normalize the localpart case differently at different API surfaces, and the reply-loop tail of the
        "hall of mirrors" bug (#15763) has been observed with the bot's own account bypassing a
        case-sensitive equality check.
        """
        own = (self._own_user_id() or "").strip().lower()
        return not own or sender.strip().lower() == own

    def is_confirmed_self_sender(self, sender: str) -> bool:
        """True only when we KNOW the sender is the bot account (resolved user_id, compared like
        ``is_self_sender``). Never True on an unresolved identity: ``is_self_sender``'s fail-closed
        heuristic ("can't prove it isn't us") must not be read as "the owner is typing" — that would
        turn every DM from every other user into owner input (#15763)."""
        own = (self._own_user_id() or "").strip().lower()
        return bool(own) and sender.strip().lower() == own

    async def is_self_chat_room(self, room_id: str) -> bool:
        """Self-chat triggers only in DM-classified rooms (incl. a 1-member notes-to-self room) —
        never in group rooms. MATRIX_HOME_ROOM is deliberately NOT an override: it is a delivery
        target that /sethome can write from any chat, groups included."""
        try:
            return await self._dm_room_check(room_id)
        except Exception:
            # Fails closed (never treats an undecidable room as self-chat) but keeps the traceback:
            # a silent drop here would be undiagnosable.
            logger.warning("Matrix: self-chat room classification failed for %s", room_id, exc_info=True)
            return False
