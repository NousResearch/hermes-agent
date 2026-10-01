"""Bot-account notification observations and explicit scoped read receipts."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

from plugins.platforms.matrix.client_events import Method, UndecryptableEvent, decrypt_history_event, raw_event
from plugins.platforms.matrix.effective_event import event_content
from plugins.platforms.matrix.read_context import _read_access
from plugins.platforms.matrix.relations import MatrixRelation
from tools.matrix_tool_runtime import MatrixOwner


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
    adapter: Any, room_id: str, thread_id: object, *, requester: str,
) -> dict[str, Any]:
    if not isinstance(thread_id, str) or (thread_id != "main" and not thread_id.startswith("$")):
        return {"error": "thread_id must be main or a thread root event ID"}
    client, _chat_type, error = await _read_access(adapter, room_id, requester)
    if error is not None:
        return error
    if adapter._closing:
        return {"error": "Matrix client is disconnected"}
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

    @classmethod
    def parse(
        cls, room_id: str, event_id: object, thread_id: object, visibility: object, requester: str,
    ) -> ReadTarget | dict[str, Any]:
        if not isinstance(visibility, str) or visibility not in {"public", "private"}:
            return {"error": "visibility must be public or private"}
        if not isinstance(thread_id, str) or (thread_id not in {"main", "room"} and not thread_id.startswith("$")):
            return {"error": "thread_id must be main, room, or a thread root event ID"}
        if not isinstance(event_id, str) or not event_id.startswith("$"):
            return {"error": "event_id is required"}
        return cls(room_id, event_id, thread_id, visibility, requester)


async def _validate_target(adapter: Any, client: Any, target: ReadTarget, chat_type: str | None) -> str | dict[str, Any]:
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
        event = await decrypt_history_event(client, raw)
    except UndecryptableEvent as exc:
        return {"event_id": target.event_id, "error": str(exc)}
    except Exception as exc:
        return {"error": "Matrix read target could not be verified", "errors": [_operation_error("target", exc)]}
    content = event_content(event)
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
    return sender


@dataclass
class ReadProgress:
    result: dict[str, Any] | None = None
    operation: Literal["receipt", "marked_unread"] = "receipt"
    dispatched: bool = False

    def interrupted_result(self, error: str) -> dict[str, Any]:
        if self.result is None:
            return {"error": error}
        result = {**self.result, "errors": [*self.result["errors"], {"operation": self.operation, "error": error}]}
        if self.dispatched:
            result["receipt_sent" if self.operation == "receipt" else "marked_unread_reset"] = None
            if self.operation == "receipt":
                result["counts"] = "unknown"
        return result


async def _write_access(
    adapter: Any, client: Any, target: ReadTarget, owner: MatrixOwner,
    interrupted: Callable[[], bool], sender: str,
) -> dict[str, Any] | None:
    current_client, chat_type, error = await _read_access(adapter, target.room_id, target.requester)
    if error is not None:
        return error
    if current_client is not client or adapter._closing:
        return {"error": "Matrix transport changed before the receipt was sent"}
    try:
        membership = await asyncio.wait_for(
            client.get_state_event(target.room_id, "m.room.member", target.requester), timeout=10,
        )
    except Exception as exc:
        return {"error": "Matrix membership could not be verified", "errors": [_operation_error("membership", exc)]}
    if interrupted():
        return {"error": "Matrix read acknowledgement interrupted"}
    if not owner.matches():
        return {"error": "Matrix session or client ownership changed"}
    if (adapter._closing or target.room_id not in adapter._joined_rooms
            or not adapter._is_allowed_matrix_room(target.room_id, chat_type)
            or adapter._is_sender_authorized(target.requester, chat_type=chat_type, chat_id=target.room_id) is not True):
        return {"error": "Matrix access changed before the receipt was sent"}
    if sender != adapter._user_id and adapter._is_sender_authorized(
        sender, chat_type=chat_type, chat_id=target.room_id,
    ) is not True:
        return {"error": "Matrix event sender is not authorized for this room"}
    content = raw_event(membership)
    if content.get("content", content).get("membership") != "join":
        return {"error": "Matrix requester is not a joined room member"}
    return None


async def mark_matrix_read(
    adapter: Any, room_id: str, event_id: object, thread_id: object, visibility: object, *, requester: str,
    interrupt_check: Callable[[], bool] = lambda: False,
    before_write: Callable[[], None] = lambda: None,
    progress: ReadProgress | None = None,
) -> dict[str, Any]:
    progress = progress or ReadProgress()
    owner = MatrixOwner.capture()
    target = ReadTarget.parse(room_id, event_id, thread_id, visibility, requester)
    if isinstance(target, dict):
        return target
    client, chat_type, error = await _read_access(adapter, target.room_id, target.requester)
    if error is not None:
        return error
    sender = await _validate_target(adapter, client, target, chat_type)
    if isinstance(sender, dict):
        return sender
    error = await _write_access(adapter, client, target, owner, interrupt_check, sender)
    if error is not None:
        return error

    result: dict[str, Any] = {
        "room_id": target.room_id, "account_user_id": adapter._user_id,
        "event_id": target.event_id, "thread_id": target.thread_id, "visibility": target.visibility,
        "receipt_sent": False, "marked_unread_reset": False,
        "fully_read_marker_changed": False, "counts": "await_sync", "errors": [],
    }
    progress.result = result
    receipt_type = "m.read.private" if target.visibility == "private" else "m.read"
    path = f"/_matrix/client/v3/rooms/{quote(target.room_id, safe='')}/receipt/{receipt_type}/{quote(target.event_id, safe='')}"
    body = {} if target.thread_id == "room" else {"thread_id": target.thread_id}
    try:
        before_write()
        progress.dispatched = True
        await asyncio.wait_for(client.api.request(Method.POST, path, body, retry_count=0), timeout=20)
        result["receipt_sent"] = True
        progress.dispatched = False
        progress.operation = "marked_unread"
        adapter._unread.receipt_sent(client, target.room_id, target.thread_id)
    except Exception as exc:
        if not getattr(exc, "errcode", None):
            result["receipt_sent"] = None
        result["counts"] = "unchanged" if result["receipt_sent"] is False else "unknown"
        result["errors"].append(_operation_error("receipt", exc))
        return result
    if target.thread_id != "room":
        return result
    error = await _write_access(adapter, client, target, owner, interrupt_check, sender)
    if error is not None:
        reason = error["error"] if error["error"] == "Matrix read acknowledgement interrupted" else "Matrix access changed after the receipt was sent"
        result["errors"].append({"operation": "marked_unread", "error": reason})
        return result
    try:
        before_write()
        progress.dispatched = True
        await asyncio.wait_for(
            client.set_account_data("m.marked_unread", {"unread": False}, room_id=target.room_id), timeout=20,
        )
        result["marked_unread_reset"] = True
        progress.dispatched = False
    except Exception as exc:
        if not getattr(exc, "errcode", None):
            result["marked_unread_reset"] = None
        result["errors"].append(_operation_error("marked_unread", exc))
    return result


class MatrixUnreadMixin:
    async def read_matrix_unread(
        self, room_id: str, thread_id: object, *, requester: str,
    ) -> dict[str, Any]:
        return await read_matrix_unread(self, room_id, thread_id, requester=requester)

    async def mark_matrix_read(
        self, room_id: str, event_id: object, thread_id: object, visibility: object, *, requester: str,
        interrupt_check: Callable[[], bool] = lambda: False,
        before_write: Callable[[], None] = lambda: None,
        progress: ReadProgress | None = None,
    ) -> dict[str, Any]:
        return await mark_matrix_read(
            self, room_id, event_id, thread_id, visibility, requester=requester,
            interrupt_check=interrupt_check, before_write=before_write, progress=progress,
        )
