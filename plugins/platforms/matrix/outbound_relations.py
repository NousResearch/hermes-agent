"""Track the latest Matrix event available for each outbound thread fallback."""

from collections import OrderedDict
from typing import Any

from plugins.platforms.matrix.relations import MatrixRelation


class ThreadFallbackTracker:
    def __init__(self, max_threads: int = 500) -> None:
        self._max_threads = max_threads
        self._events: OrderedDict[tuple[str, str], str] = OrderedDict()

    def remember(self, room_id: str, thread_id: str, event_id: str) -> None:
        if not room_id or not thread_id or not event_id:
            return
        key = (room_id, thread_id)
        self._events[key] = event_id
        self._events.move_to_end(key)
        if len(self._events) > self._max_threads:
            self._events.popitem(last=False)

    def remember_sent(self, room_id: str, content: dict[str, Any], event_id: str) -> None:
        thread_id = MatrixRelation.from_content(content.get("m.relates_to")).thread_root
        if thread_id:
            self.remember(room_id, thread_id, event_id)

    def latest(self, room_id: str, thread_id: str) -> str | None:
        return self._events.get((room_id, thread_id))
