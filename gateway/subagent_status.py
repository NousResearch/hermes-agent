"""Privacy-empty presentation state for detached subagents."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Awaitable, Callable

from gateway.platforms.base import BasePlatformAdapter, SendResult

from tools.delegation_status import (
    DetachedStatusEvent as SubagentStatusEvent,
    DetachedStatusPhase as SubagentStatusPhase,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SubagentStatusRow:
    ordinal: int
    phase: SubagentStatusPhase
    started_at: float
    finished_at: float | None = None


@dataclass(frozen=True, slots=True)
class SubagentStatusSnapshot:
    rows: tuple[SubagentStatusRow, ...]


@dataclass(frozen=True, slots=True)
class SubagentStatusBatch:
    batch_id: int
    start_ordinal: int
    task_count: int


_PHASE_GLYPHS = {
    SubagentStatusPhase.WAITING: "🚀",
    SubagentStatusPhase.THINKING: "💭",
    SubagentStatusPhase.DONE: "✅",
    SubagentStatusPhase.FAILED: "❌",
}
_TERMINAL_PHASES = frozenset(
    {SubagentStatusPhase.DONE, SubagentStatusPhase.FAILED}
)


class SubagentStatusBoard:
    """Loop-owned projection of accepted detached batches."""

    def __init__(self) -> None:
        self._rows: list[SubagentStatusRow] = []
        self._open_batches: set[int] = set()
        self._sealed = False
        self._next_batch_id = 1

    def admit_batch(self, *, task_count: int, at: float) -> SubagentStatusBatch:
        if self._sealed:
            raise RuntimeError("status board is sealed")
        batch = SubagentStatusBatch(
            batch_id=self._next_batch_id,
            start_ordinal=len(self._rows) + 1,
            task_count=task_count,
        )
        self._next_batch_id += 1
        self._open_batches.add(batch.batch_id)
        self._rows.extend(
            SubagentStatusRow(
                ordinal=batch.start_ordinal + index,
                phase=SubagentStatusPhase.WAITING,
                started_at=at,
            )
            for index in range(task_count)
        )
        return batch

    def finalize_batch(
        self,
        batch: SubagentStatusBatch,
        *,
        outcomes: tuple[SubagentStatusPhase, ...],
        at: float,
    ) -> None:
        if batch.batch_id not in self._open_batches:
            return
        if len(outcomes) != batch.task_count:
            raise ValueError("one outcome is required for every admitted task")
        for index, outcome in enumerate(outcomes):
            if outcome not in _TERMINAL_PHASES:
                raise ValueError("batch outcomes must be terminal")
            row_index = batch.start_ordinal - 1 + index
            row = self._rows[row_index]
            if row.phase in _TERMINAL_PHASES:
                continue
            self._rows[row_index] = replace(
                row, phase=outcome, finished_at=at
            )
        self._open_batches.remove(batch.batch_id)

    def observe(
        self, batch: SubagentStatusBatch, event: SubagentStatusEvent
    ) -> None:
        if batch.batch_id not in self._open_batches or event.depth != 1:
            return
        if not 0 <= event.task_index < batch.task_count:
            return
        row_index = batch.start_ordinal - 1 + event.task_index
        row = self._rows[row_index]
        if row.phase in _TERMINAL_PHASES:
            return
        finished_at = event.at if event.phase in _TERMINAL_PHASES else None
        self._rows[row_index] = replace(
            row, phase=event.phase, finished_at=finished_at
        )

    def seal(self) -> None:
        self._sealed = True

    def fail_open_and_seal(self, *, at: float) -> None:
        self._rows = [
            row
            if row.phase in _TERMINAL_PHASES
            else replace(
                row,
                phase=SubagentStatusPhase.FAILED,
                finished_at=at,
            )
            for row in self._rows
        ]
        self._open_batches.clear()
        self._sealed = True

    @property
    def is_terminal(self) -> bool:
        return self._sealed and not self._open_batches

    def snapshot(self) -> SubagentStatusSnapshot:
        return SubagentStatusSnapshot(rows=tuple(self._rows))


class _SubagentStatusSink:
    def __init__(self, owner: "SubagentStatusOwner", batch: SubagentStatusBatch):
        self._owner = owner
        self._batch = batch

    def observe(self, event: SubagentStatusEvent) -> None:
        self._owner._loop.call_soon_threadsafe(
            self._owner._observe, self._batch, event
        )

    def finalize(self, outcomes: tuple[SubagentStatusPhase, ...]) -> None:
        self._owner._loop.call_soon_threadsafe(
            self._owner._finalize, self._batch, outcomes
        )


class SubagentStatusOwner:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        on_change: Callable[[SubagentStatusSnapshot, bool], Any],
        clock: Callable[[], float] = time.monotonic,
        admission_timeout: float = 5.0,
        on_first_admission: Callable[["SubagentStatusOwner"], bool] | None = None,
    ) -> None:
        self._loop = loop
        self._on_change = on_change
        self._clock = clock
        self._admission_timeout = admission_timeout
        self._on_first_admission = on_first_admission
        self._board = SubagentStatusBoard()

    def admit_batch(self, task_count: int) -> _SubagentStatusSink:
        response: Future[_SubagentStatusSink] = Future()

        def admit_on_loop() -> None:
            if not response.set_running_or_notify_cancel():
                return
            try:
                sink = self._admit_batch_on_loop(task_count)
            except BaseException as exc:
                response.set_exception(exc)
            else:
                response.set_result(sink)

        self._loop.call_soon_threadsafe(admit_on_loop)
        try:
            return response.result(timeout=self._admission_timeout)
        except FutureTimeoutError:
            if response.cancel():
                raise
            return response.result()

    def _admit_batch_on_loop(self, task_count: int) -> _SubagentStatusSink:
        if self._on_first_admission is not None:
            if not self._on_first_admission(self):
                raise RuntimeError("detached status registry is closed")
        batch = self._board.admit_batch(task_count=task_count, at=self._clock())
        self._notify()
        return _SubagentStatusSink(self, batch)

    def _observe(
        self, batch: SubagentStatusBatch, event: SubagentStatusEvent
    ) -> None:
        self._board.observe(batch, event)
        self._notify()

    def _finalize(
        self,
        batch: SubagentStatusBatch,
        outcomes: tuple[SubagentStatusPhase, ...],
    ) -> None:
        self._board.finalize_batch(batch, outcomes=outcomes, at=self._clock())
        self._notify()

    def seal(self) -> None:
        self._board.seal()
        self._notify()

    def request_seal(self) -> None:
        self._loop.call_soon_threadsafe(self.seal)

    def fail_open_and_seal(self) -> None:
        self._board.fail_open_and_seal(at=self._clock())
        self._notify()

    def snapshot(self) -> SubagentStatusSnapshot:
        return self._board.snapshot()

    def _notify(self) -> None:
        snapshot = self._board.snapshot()
        if not snapshot.rows:
            return
        try:
            self._on_change(snapshot, self._board.is_terminal)
        except Exception:
            logger.warning("Detached status change callback failed", exc_info=True)


async def _wait_for_wake(wake: asyncio.Event, delay: float) -> bool:
    try:
        await asyncio.wait_for(wake.wait(), timeout=delay)
        return True
    except asyncio.TimeoutError:
        return False


class SubagentStatusRegistry:
    def __init__(self) -> None:
        self._entries: dict[
            TelegramSubagentStatusPublisher, SubagentStatusOwner
        ] = {}
        self._empty = asyncio.Event()
        self._empty.set()
        self._closed = False
        self._cleanup_tasks: set[asyncio.Task] = set()

    @property
    def active_count(self) -> int:
        return len(self._entries)

    @property
    def cleanup_task_count(self) -> int:
        return len(self._cleanup_tasks)

    def register(
        self,
        owner: SubagentStatusOwner,
        publisher: "TelegramSubagentStatusPublisher",
    ) -> bool:
        if self._closed:
            return False
        if publisher in self._entries:
            return True
        self._entries[publisher] = owner
        self._empty.clear()
        cleanup_task = asyncio.create_task(self._remove_when_closed(publisher))
        self._cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._cleanup_tasks.discard)
        return True

    def close(self) -> None:
        self._closed = True

    async def _remove_when_closed(
        self, publisher: "TelegramSubagentStatusPublisher"
    ) -> None:
        await publisher.wait_closed()
        self._entries.pop(publisher, None)
        if not self._entries:
            self._empty.set()

    async def wait_empty(self) -> None:
        await self._empty.wait()

    async def shutdown(self, *, timeout: float) -> None:
        self.close()
        entries = list(self._entries.items())
        for _publisher, owner in entries:
            owner.seal()

        terminal_budget = min(1.0, max(0.0, timeout * 0.2))
        grace_budget = max(0.0, timeout - terminal_budget)
        try:
            await asyncio.wait_for(self.wait_empty(), timeout=grace_budget)
            return
        except asyncio.TimeoutError:
            pass

        for _publisher, owner in entries:
            owner.fail_open_and_seal()
        try:
            await asyncio.wait_for(self.wait_empty(), timeout=terminal_budget)
        except asyncio.TimeoutError:
            await asyncio.gather(
                *(publisher.cancel() for publisher, _owner in entries),
                return_exceptions=True,
            )
            await self.wait_empty()


class _PublicationResult(Enum):
    SUCCESS = "success"
    RETRY = "retry"
    ABANDON = "abandon"


class TelegramSubagentStatusPublisher:
    def __init__(
        self,
        *,
        send: Callable[[str], Awaitable[SendResult]],
        edit: Callable[[Any, str], Awaitable[SendResult]],
        clock: Callable[[], float] = time.monotonic,
        interval: float = 5.0,
        wait: Callable[[asyncio.Event, float], Awaitable[bool]] = _wait_for_wake,
    ) -> None:
        self._send = send
        self._edit = edit
        self._clock = clock
        self._interval = interval
        self._wait = wait
        self._latest: SubagentStatusSnapshot | None = None
        self._message_id: Any = None
        self._terminal_requested = False
        self._terminal_edit_attempts = 0
        self._shutdown_requested = False
        self._abandoned = False
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._closed = asyncio.Event()

    def notify(
        self, snapshot: SubagentStatusSnapshot, is_terminal: bool
    ) -> None:
        if self._shutdown_requested or self._abandoned or self._closed.is_set():
            return
        self._latest = snapshot
        if is_terminal:
            self._terminal_requested = True
            self._wake.set()
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def shutdown(self) -> None:
        self._shutdown_requested = True
        self._wake.set()
        if self._task is None:
            self._closed.set()
        await self.wait_closed()

    async def cancel(self) -> None:
        self._shutdown_requested = True
        task = self._task
        if task is None:
            self._closed.set()
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _run(self) -> None:
        try:
            while self._latest is not None:
                if self._shutdown_requested:
                    return
                snapshot = self._latest
                terminal = self._terminal_requested
                terminal_edit = terminal and self._message_id is not None
                if terminal_edit:
                    self._terminal_edit_attempts += 1
                result = await self._publish(snapshot)
                if result is _PublicationResult.ABANDON:
                    self._abandoned = True
                    return
                if self._shutdown_requested:
                    return
                if terminal and result is _PublicationResult.SUCCESS:
                    return
                if (
                    terminal_edit
                    and result is _PublicationResult.RETRY
                    and self._terminal_edit_attempts >= 2
                ):
                    return
                await self._wait(self._wake, self._interval)
                self._wake.clear()
                if self._shutdown_requested:
                    return
        finally:
            self._closed.set()

    async def _publish(
        self, snapshot: SubagentStatusSnapshot
    ) -> _PublicationResult:
        text = render_subagent_status(snapshot, now=self._clock())
        try:
            if self._message_id is None:
                result = await self._send(text)
                if not result.success or result.message_id is None:
                    return _PublicationResult.ABANDON
                self._message_id = result.message_id
                return _PublicationResult.SUCCESS
            result = await self._edit(self._message_id, text)
            if result.success:
                return _PublicationResult.SUCCESS
            if result.retryable and result.retry_after is None:
                return _PublicationResult.RETRY
            return _PublicationResult.ABANDON
        except Exception:
            logger.warning("Detached Telegram status publication failed", exc_info=True)
            return _PublicationResult.ABANDON


def create_telegram_subagent_status(
    *,
    source: Any,
    adapter: Any,
    route_metadata: dict | None,
    loop: asyncio.AbstractEventLoop,
    clock: Callable[[], float] = time.monotonic,
    registry: SubagentStatusRegistry | None = None,
) -> tuple[SubagentStatusOwner, TelegramSubagentStatusPublisher] | None:
    platform = getattr(getattr(source, "platform", None), "value", None)
    edit_implementation = getattr(type(adapter), "edit_message", None)
    if (
        platform != "telegram"
        or not callable(getattr(adapter, "edit_message", None))
        or edit_implementation is BasePlatformAdapter.edit_message
    ):
        return None

    descriptor_for_chat = getattr(adapter, "_descriptor_for_chat", None)
    if callable(descriptor_for_chat):
        try:
            descriptor = descriptor_for_chat(str(source.chat_id))
            supports_op = getattr(descriptor, "supports_op", None)
            if (
                str(getattr(descriptor, "platform", "")).lower()
                != "telegram"
                or not bool(getattr(descriptor, "supports_edit", False))
                or not callable(supports_op)
                or not supports_op("edit")
            ):
                return None
        except Exception:
            return None

    chat_id = source.chat_id
    metadata = copy.deepcopy(route_metadata or {})
    send_metadata = {**metadata, "_interim_send": True}
    finalize = bool(getattr(adapter, "REQUIRES_EDIT_FINALIZE", False))

    async def send(text: str) -> SendResult:
        return await adapter.send(chat_id, text, metadata=send_metadata)

    async def edit(message_id: Any, text: str) -> SendResult:
        return await adapter.edit_message(
            chat_id, message_id, text, finalize=finalize
        )

    publisher = TelegramSubagentStatusPublisher(
        send=send, edit=edit, clock=clock
    )
    on_first_admission = None
    if registry is not None:
        on_first_admission = lambda owner: registry.register(owner, publisher)
    owner = SubagentStatusOwner(
        loop=loop,
        on_change=publisher.notify,
        clock=clock,
        on_first_admission=on_first_admission,
    )
    return owner, publisher


def _format_elapsed(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    minutes, remaining = divmod(whole_seconds, 60)
    if minutes:
        return f"{minutes}m{remaining:02d}s"
    return f"{remaining}s"


def render_subagent_status(snapshot: SubagentStatusSnapshot, *, now: float) -> str:
    """Render one compact, plaintext status board."""
    rows = snapshot.rows
    done = sum(row.phase in _TERMINAL_PHASES for row in rows)
    lines = [f"🔀 Subagents · {done}/{len(rows)} done"]
    for index, row in enumerate(rows):
        branch = "└" if index == len(rows) - 1 else "├"
        finished_at = row.finished_at if row.finished_at is not None else now
        elapsed = _format_elapsed(finished_at - row.started_at)
        lines.append(
            f"{branch} #{row.ordinal} {_PHASE_GLYPHS[row.phase]} "
            f"{row.phase.value} · {elapsed}"
        )
    return "\n".join(lines)
