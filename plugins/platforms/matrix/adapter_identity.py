"""Self/dedupe identity for the Matrix adapter — moved out of ``adapter.py`` for its size cap.

Two related answers to "who sent this, and did we already see it?": own-account detection
(the self-chat gate plus the echo-loop guards of #15763) and the bounded event-ID tracker
that drops replays of our own sends. Kept as a mixin (the platform siblings' pattern) so
every ``self._x(...)`` call site in ``adapter.py`` and in the tests keeps resolving.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MatrixIdentityMixin:
    """Own-account classification + the event-ID tracker. Uses only ``self._user_id``,
    ``self._processed_events``/``_processed_events_set`` and ``self._is_dm_room``."""

    def _is_duplicate_event(self, event_id) -> bool:
        """Return True if this event was already processed. Tracks the ID otherwise."""
        if not event_id:
            return False
        if event_id in self._processed_events_set:
            return True
        self._remember_event(event_id)
        return False

    def _remember_event(self, event_id) -> None:
        """Track an event ID we SENT (self-chat echo suppression): when the sync replays our own
        send back (sender == the bot account), ``_is_duplicate_event`` drops it as already seen.
        Same bounded tracker as inbound dedup — the WhatsApp bridge's outbound_id pattern."""
        if not event_id or event_id in self._processed_events_set:
            return
        if len(self._processed_events) == self._processed_events.maxlen:
            self._processed_events_set.discard(self._processed_events[0])
        self._processed_events.append(event_id)
        self._processed_events_set.add(event_id)

    def _is_self_sender(self, sender: str) -> bool:
        """True if *sender* is the bot itself (case-insensitive: homeservers vary localpart case). With
        no resolved user_id we can't prove a sender is NOT us, so return True — dropping our own
        events beats an echo loop ("hall of mirrors").

        Matrix user IDs are byte-compared after trimming whitespace and lowercasing — some homeservers
        normalize the localpart case differently at different API surfaces, and the reply-loop tail of the
        "hall of mirrors" bug (#15763) has been observed with the bot's own account bypassing a
        case-sensitive equality check.
        """
        own = (self._user_id or "").strip().lower()
        return not own or sender.strip().lower() == own

    def _is_confirmed_self_sender(self, sender: str) -> bool:
        """True only when we KNOW the sender is the bot account (resolved ``_user_id``, compared
        like ``_is_self_sender``). Never True on an unresolved identity: ``_is_self_sender``'s
        fail-closed heuristic ("can't prove it isn't us") must not be read as "the owner is
        typing" — that would turn every DM from every other user into owner input (#15763)."""
        own = (self._user_id or "").strip().lower()
        return bool(own) and sender.strip().lower() == own

    async def _is_self_chat_room(self, room_id: str) -> bool:
        """Self-chat triggers only in DM-classified rooms (incl. a 1-member notes-to-self room) —
        never in group rooms. MATRIX_HOME_ROOM is deliberately NOT an override: it is a delivery
        target that /sethome can write from any chat, groups included."""
        try:
            return await self._is_dm_room(room_id)
        except Exception:
            # Fails closed (never treats an undecidable room as self-chat) but keeps the traceback:
            # a silent drop here would be undiagnosable.
            logger.warning("Matrix: self-chat room classification failed for %s", room_id, exc_info=True)
            return False

    @staticmethod
    def _is_system_or_bridge_sender(sender: str) -> bool:
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
