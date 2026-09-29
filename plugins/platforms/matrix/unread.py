"""Bot-account notification observations and explicit scoped read receipts."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.client_events import raw_event
from plugins.platforms.matrix.read_context import _decrypted_event, _read_access
from plugins.platforms.matrix.relations import MatrixRelation
from plugins.platforms.matrix.reply_context import _content_dict

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        POST = "POST"

        def __str__(self) -> str:
            return self.value


SYNC_FILTER = '{"room":{"timeline":{"unread_thread_notifications":true}}}'


@dataclass(frozen=True)
class NotificationCounts:
    notification_count: int | None = None
    highlight_count: int | None = None

    def update(self, raw: dict[str, Any]) -> NotificationCounts:
        def count(key: str, previous: int | None) -> int | None:
            value = raw.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else previous

        return NotificationCounts(
            count("notification_count", self.notification_count),
            count("highlight_count", self.highlight_count),
        )


@dataclass
class RoomUnreadObservation:
    main: NotificationCounts = field(default_factory=NotificationCounts)
    threads: OrderedDict[str, NotificationCounts] = field(default_factory=OrderedDict)
    threads_complete: bool = False
    marked_unread: bool | None = None
    generation: int = 0
    pending_receipts: set[str] = field(default_factory=set)


class MatrixUnreadState:
    def __init__(
        self, *, max_rooms: int = 256, max_threads: int = 128,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rooms: OrderedDict[str, RoomUnreadObservation] = OrderedDict()
        self._owner: Any = None
        self._generation = 0
        self._last_sync: float | None = None
        self._max_rooms = max_rooms
        self._max_threads = max_threads
        self._clock = clock

    def reset(self) -> None:
        self._rooms.clear()
        self._owner = None
        self._generation = 0
        self._last_sync = None

    def observe(self, client: Any, data: dict[str, Any], *, initial: bool = False) -> None:
        if initial or self._owner is not client:
            self.reset()
        self._owner = client
        self._generation += 1
        self._last_sync = self._clock()
        rooms = data.get("rooms", {})
        for room_id in rooms.get("leave", {}):
            self._rooms.pop(room_id, None)
        for room_id, delta in rooms.get("join", {}).items():
            if not isinstance(delta, dict):
                continue
            observation = self._rooms.pop(room_id, RoomUnreadObservation())
            observation.generation = self._generation
            raw_counts = delta.get("unread_notifications")
            if isinstance(raw_counts, dict):
                observation.main = observation.main.update(raw_counts)
                if all(isinstance(raw_counts.get(key), int) and not isinstance(raw_counts[key], bool) and raw_counts[key] >= 0 for key in ("notification_count", "highlight_count")):
                    observation.pending_receipts.discard("main")
            # When the server includes a room, it recalculates every thread and omits threads that
            # have no notifications. A thread missing from this map therefore has zero counts.
            raw_threads = delta.get("unread_thread_notifications")
            threads: OrderedDict[str, NotificationCounts] = OrderedDict()
            observation.threads_complete = True
            for thread_id, counts in (raw_threads.items() if isinstance(raw_threads, dict) else ()):
                if not isinstance(thread_id, str) or not thread_id.startswith("$") or not isinstance(counts, dict):
                    continue
                threads[thread_id] = observation.threads.get(thread_id, NotificationCounts()).update(counts)
                if all(isinstance(counts.get(key), int) and not isinstance(counts[key], bool) and counts[key] >= 0 for key in ("notification_count", "highlight_count")):
                    observation.pending_receipts.discard(thread_id)
                if len(threads) > self._max_threads:
                    threads.popitem(last=False)
                    observation.threads_complete = False
            observation.pending_receipts.intersection_update({"main", *threads})
            observation.threads = threads
            account_data = delta.get("account_data", {}).get("events", [])
            if initial and observation.marked_unread is None:
                observation.marked_unread = False
            for event in account_data:
                if event.get("type") != "m.marked_unread":
                    continue
                unread = event.get("content", {}).get("unread")
                if isinstance(unread, bool):
                    observation.marked_unread = unread
            self._rooms[room_id] = observation
            while len(self._rooms) > self._max_rooms:
                self._rooms.popitem(last=False)

    def read(self, client: Any, room_id: str, thread_id: str) -> dict[str, Any]:
        observation = self._rooms.get(room_id) if self._owner is client else None
        counts = NotificationCounts()
        if observation is not None and thread_id == "main":
            counts = observation.main
        elif observation is not None:
            absent = NotificationCounts(0, 0) if observation.threads_complete else counts
            counts = observation.threads.get(thread_id, absent)
        age = max(0.0, self._clock() - self._last_sync) if self._last_sync is not None and self._owner is client else None
        status = "unavailable"
        if counts.notification_count is not None or counts.highlight_count is not None:
            status = "stale" if age is not None and age > 90 else "observed"
        if observation is not None and thread_id in observation.pending_receipts:
            status = "await_sync"
        return {
            "notification_count": counts.notification_count,
            "highlight_count": counts.highlight_count,
            "marked_unread": observation.marked_unread if observation is not None else None,
            "status": status,
            "observation_generation": observation.generation if observation is not None else None,
            "last_sync_age_seconds": round(age, 3) if age is not None else None,
        }

    def receipt_sent(self, client: Any, room_id: str, thread_id: str) -> None:
        observation = self._rooms.get(room_id) if self._owner is client else None
        if observation is None:
            return
        scopes = {"main", *observation.threads} if thread_id == "room" else {
            scope for scope in (thread_id,) if scope == "main" or scope in observation.threads
        }
        observation.pending_receipts.update(scopes)


async def read_matrix_unread(
    adapter: Any, room_id: str, thread_id: str, *, requester: str,
) -> dict[str, Any]:
    client, _chat_type, error = await _read_access(adapter, room_id, requester)
    if error is not None:
        return error
    if adapter._closing:
        return {"error": "Matrix client is disconnected"}
    if thread_id != "main" and not thread_id.startswith("$"):
        return {"error": "thread_id must be main or a thread root event ID"}
    return {
        "room_id": room_id, "account_user_id": adapter._user_id,
        "count_basis": "bot_account_push_rules", "thread_id": thread_id,
        **adapter._unread.read(client, room_id, thread_id),
    }


def _operation_error(operation: str, exc: Exception) -> dict[str, Any]:
    result = {"operation": operation, "error": str(getattr(exc, "errcode", None) or type(exc).__name__)}
    retry_after = getattr(exc, "retry_after_ms", None)
    if isinstance(retry_after, int):
        result["retry_after_ms"] = retry_after
    return result


@dataclass(frozen=True)
class ReadTarget:
    room_id: str
    event_id: str
    thread_id: str
    visibility: str
    requester: str


async def _validate_target(adapter: Any, client: Any, target: ReadTarget, chat_type: str | None) -> dict[str, Any] | None:
    try:
        membership = await asyncio.wait_for(
            client.get_state_event(target.room_id, "m.room.member", target.requester), timeout=10,
        )
        content = raw_event(membership)
        content = content.get("content", content)
        if content.get("membership") != "join":
            return {"error": "Matrix requester is not a joined room member"}
        raw = raw_event(await asyncio.wait_for(
            client.get_event(target.room_id, target.event_id), timeout=10,
        ))
        if raw.get("room_id") != target.room_id or raw.get("event_id") != target.event_id:
            return {"error": "Matrix event does not belong to the selected room"}
        event, error = await _decrypted_event(client, raw)
    except Exception as exc:
        return {"error": "Matrix read target could not be verified", "errors": [_operation_error("target", exc)]}
    if error is not None:
        return error
    content = _content_dict(event)
    if not content.get("msgtype"):
        return {"error": "Matrix event has no visible message"}
    sender = str(raw.get("sender") or "")
    if sender != adapter._user_id and adapter._is_sender_authorized(
        sender, chat_type=chat_type, chat_id=target.room_id,
    ) is not True:
        return {"error": "Matrix event sender is not authorized for this room"}
    relates_to = content.get("m.relates_to")
    relation = MatrixRelation.from_content(relates_to)
    if target.thread_id != "room" and (
        relation.is_edit or (relates_to is not None and not isinstance(relates_to, dict))
        or (isinstance(relates_to, dict) and relates_to.get("rel_type") == "m.thread" and not relation.thread_root)
    ):
        return {"error": "Matrix event has no verifiable receipt scope"}
    root = relation.thread_root
    if target.thread_id == "main" and root is not None:
        return {"error": "Matrix event is in a thread, not the main timeline"}
    if target.thread_id.startswith("$") and root != target.thread_id and target.event_id != target.thread_id:
        return {"error": "Matrix event does not belong to the selected thread"}
    if target.event_id == target.thread_id and root is not None:
        return {"error": "Matrix thread root is itself a reply in another thread"}
    return None


async def mark_matrix_read(adapter: Any, target: ReadTarget) -> dict[str, Any]:
    if target.visibility not in {"public", "private"}:
        return {"error": "visibility must be public or private"}
    if target.thread_id not in {"main", "room"} and not target.thread_id.startswith("$"):
        return {"error": "thread_id must be main, room, or a thread root event ID"}
    if not target.event_id.startswith("$"):
        return {"error": "event_id is required"}
    client, chat_type, error = await _read_access(adapter, target.room_id, target.requester)
    if error is not None:
        return error
    error = await _validate_target(adapter, client, target, chat_type)
    if error is not None:
        return error
    current_client, _chat_type, error = await _read_access(adapter, target.room_id, target.requester)
    if error is not None:
        return error
    if current_client is not client or adapter._closing:
        return {"error": "Matrix transport changed before the receipt was sent"}

    result: dict[str, Any] = {
        "room_id": target.room_id, "account_user_id": adapter._user_id,
        "event_id": target.event_id, "thread_id": target.thread_id, "visibility": target.visibility,
        "receipt_sent": False, "marked_unread_reset": False,
        "fully_read_marker_changed": False, "counts": "await_sync", "errors": [],
    }
    receipt_type = "m.read.private" if target.visibility == "private" else "m.read"
    path = f"/_matrix/client/v3/rooms/{quote(target.room_id, safe='')}/receipt/{receipt_type}/{quote(target.event_id, safe='')}"
    body = {} if target.thread_id == "room" else {"thread_id": target.thread_id}
    try:
        await asyncio.wait_for(client.api.request(Method.POST, path, body, retry_count=0), timeout=20)
        result["receipt_sent"] = True
        adapter._unread.receipt_sent(client, target.room_id, target.thread_id)
    except Exception as exc:
        if not getattr(exc, "errcode", None):
            result["receipt_sent"] = None
        result["counts"] = "unchanged" if result["receipt_sent"] is False else "unknown"
        result["errors"].append(_operation_error("receipt", exc))
        return result
    if target.thread_id != "room":
        return result
    current_client, _chat_type, error = await _read_access(adapter, target.room_id, target.requester)
    if error is not None or current_client is not client or adapter._closing:
        result["errors"].append({"operation": "marked_unread", "error": "Matrix access changed after the receipt was sent"})
        return result
    try:
        await asyncio.wait_for(
            client.set_account_data("m.marked_unread", {"unread": False}, room_id=target.room_id), timeout=20,
        )
        result["marked_unread_reset"] = True
    except Exception as exc:
        if not getattr(exc, "errcode", None):
            result["marked_unread_reset"] = None
        result["errors"].append(_operation_error("marked_unread", exc))
    return result


class MatrixUnreadMixin:
    async def read_matrix_unread(
        self, room_id: str, thread_id: str, *, requester: str,
    ) -> dict[str, Any]:
        return await read_matrix_unread(self, room_id, thread_id, requester=requester)


    async def mark_matrix_read(
        self, room_id: str, event_id: str, thread_id: str, visibility: str, *, requester: str,
    ) -> dict[str, Any]:
        return await mark_matrix_read(self, ReadTarget(room_id, event_id, thread_id, visibility, requester))
