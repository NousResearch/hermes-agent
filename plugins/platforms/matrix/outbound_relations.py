"""Track the latest Matrix event available for each outbound thread fallback."""

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.thread_context import NON_CONVERSATIONAL_KEY

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

    def remember_sent(
        self, room_id: str, content: dict[str, Any], event_id: str
    ) -> None:
        if content.get(NON_CONVERSATIONAL_KEY) is True:
            return

        relation = MatrixRelation.from_content(content.get("m.relates_to"))
        if relation.thread_root:
            self._update(
                room_id,
                relation.thread_root,
                event_id,
                own=True,
                answered=relation.reply_target,
            )

    def latest(self, room_id: str, thread_id: str) -> str | None:
        events = self._threads.get((room_id, thread_id))
        return events.latest if events else None

    def is_continuation(self, room_id: str, thread_id: str, reply_to: str) -> bool:
        """Whether a send anchored on ``reply_to`` continues output already in the thread.

        The gateway anchors every message of a response on the triggering event, and
        the stream consumer anchors each new chunk on the previous one, so only the
        first of those messages is a genuine reply. A reply to the thread root is
        never genuine because the thread relation already refers to the root.

        Other responses and inbound posts can arrive between two messages of one
        response, so the check covers several recent answered and sent events, not
        only the latest of each.
        """
        if reply_to == thread_id:
            return True

        events = self._threads.get((room_id, thread_id))
        if events is None:
            return False

        return reply_to in events.answered or reply_to in events.sent

    def _update(
        self,
        room_id: str,
        thread_id: str,
        event_id: str,
        *,
        own: bool,
        answered: str | None,
    ) -> None:
        if not room_id or not thread_id or not event_id:
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
