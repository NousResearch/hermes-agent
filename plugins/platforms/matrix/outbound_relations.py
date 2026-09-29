"""Track the Matrix events that decide the relations of outbound messages.

Threads are keyed by their root event, and a room's main timeline by an empty
thread ID.
"""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from plugins.platforms.matrix.relations import MatrixRelation

_RECENT_EVENTS_PER_THREAD = 32


def _recent_events() -> deque[str]:
    return deque(maxlen=_RECENT_EVENTS_PER_THREAD)


@dataclass
class _ThreadEvents:
    latest: str
    answered: deque[str] = field(default_factory=_recent_events)
    sent: deque[str] = field(default_factory=_recent_events)


class ThreadFallbackTracker:
    def __init__(self, max_threads: int = 500) -> None:
        self._max_threads = max_threads
        self._threads: OrderedDict[tuple[str, str], _ThreadEvents] = OrderedDict()

    def remember(self, room_id: str, thread_id: str, event_id: str) -> None:
        self._update(room_id, thread_id, event_id, own=False, answered=None)

    def remember_sent(self, room_id: str, content: dict[str, Any], event_id: str, *, notice: bool = False) -> None:
        relation = MatrixRelation.from_content(content.get("m.relates_to"))
        if relation.is_edit:
            return

        answered = None if notice else relation.reply_target
        self._update(room_id, relation.thread_root or "", event_id, own=True, answered=answered)

    def latest(self, room_id: str, thread_id: str) -> str | None:
        events = self._threads.get((room_id, thread_id))
        return events.latest if events else None

    def is_continuation(
        self, room_id: str, thread_id: str, reply_to: str, *, allow_repeat: bool = False, notice: bool = False,
    ) -> bool:
        """Whether a send anchored on ``reply_to`` continues output already in the thread or room.

        The gateway anchors every message of a response on the triggering event, and
        the stream consumer anchors each new chunk on the previous one, so only the
        first of those messages is a genuine reply. A reply to the thread root is
        never genuine because the thread relation already refers to the root.
        ``allow_repeat`` lets an event that already has a genuine reply receive another.

        Other responses and inbound posts can arrive between two messages of one
        response, so the check covers several recent answered and sent events, not
        only the latest of each.

        A notice, such as the confirmation of a reaction prompt, is a deliberate reply
        to ``reply_to`` and never continues a response, even when Hermes sent the
        target. Only the root check applies to it.
        """
        if reply_to == thread_id:
            return True

        if notice:
            return False

        events = self._threads.get((room_id, thread_id))
        if events is None:
            return False

        if not allow_repeat and reply_to in events.answered:
            return True

        return reply_to in events.sent

    def _update(
        self, room_id: str, thread_id: str, event_id: str, *, own: bool, answered: str | None,
    ) -> None:
        if not room_id or not event_id:
            return

        key = (room_id, thread_id)
        events = self._threads.setdefault(key, _ThreadEvents(latest=event_id))
        events.latest = event_id
        if own:
            events.sent.append(event_id)
        if answered:
            events.answered.append(answered)

        self._threads.move_to_end(key)
        if len(self._threads) > self._max_threads:
            self._threads.popitem(last=False)
