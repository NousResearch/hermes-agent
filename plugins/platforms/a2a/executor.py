"""Official A2A AgentExecutor bridge into Hermes request dispatch."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue_v2 import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types.a2a_pb2 import (
    ROLE_USER,
    TASK_STATE_AUTH_REQUIRED,
    TASK_STATE_INPUT_REQUIRED,
    TASK_STATE_SUBMITTED,
    Part,
    Task,
    TaskStatus,
)

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


@dataclass
class _ContextLockEntry:
    lock: asyncio.Lock
    references: int = 0


class RefCountedContextLocks:
    def __init__(self):
        self._guard = asyncio.Lock()
        self._entries: dict[str, _ContextLockEntry] = {}

    @property
    def size(self) -> int:
        return len(self._entries)

    async def acquire(self, key: str) -> _ContextLockEntry:
        async with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _ContextLockEntry(asyncio.Lock())
                self._entries[key] = entry
            entry.references += 1
        try:
            await entry.lock.acquire()
            return entry
        except BaseException:
            async with self._guard:
                current = self._entries.get(key)
                if current is entry:
                    entry.references -= 1
                    if entry.references == 0:
                        self._entries.pop(key, None)
            raise

    async def release(self, key: str, entry: _ContextLockEntry) -> None:
        entry.lock.release()
        async with self._guard:
            current = self._entries.get(key)
            if current is not entry:
                return
            entry.references -= 1
            if entry.references == 0:
                self._entries.pop(key, None)


@dataclass
class _RunRecord:
    source: SessionSource
    context_key: str
    updater: TaskUpdater
    task: asyncio.Task | None = None
    context_lock: _ContextLockEntry | None = None
    cancel_requested: bool = False
    cancel_emitted: bool = False
    cancel_signal: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_task: asyncio.Task | None = None
    terminal: str | None = None


class HermesA2AExecutor(AgentExecutor):
    """Emit task-only A2A lifecycle events around one Hermes dispatch."""

    def __init__(
        self,
        adapter: Any,
        *,
        active_profile: str,
        cancel_wait_seconds: float = 5.0,
    ):
        self.adapter = adapter
        self.active_profile = active_profile
        self.cancel_wait_seconds = cancel_wait_seconds
        self._context_locks = RefCountedContextLocks()
        self._runs: dict[str, _RunRecord] = {}
        self._runs_guard = asyncio.Lock()
        self._owned_cleanup_tasks: set[asyncio.Task] = set()

    @staticmethod
    def _chat_id(principal: str, context_id: str) -> str:
        digest = hashlib.sha256(f"a2a\0{principal}\0{context_id}".encode()).hexdigest()
        return f"a2a_{digest[:40]}"

    def _identity(self, context: RequestContext) -> tuple[str, str, str]:
        user = context.call_context.user
        if not user.is_authenticated or not user.user_name:
            raise ValueError("authenticated A2A identity required")
        if not context.task_id or not context.context_id:
            raise ValueError("A2A task and context identifiers are required")
        return user.user_name, context.task_id, context.context_id

    @staticmethod
    def _text(context: RequestContext) -> str:
        message = context.message
        if message is None or message.role != ROLE_USER or not message.parts:
            raise ValueError("A2A request requires user text")
        texts = []
        for part in message.parts:
            if (
                part.WhichOneof("content") != "text"
                or not part.text.strip()
                or part.metadata.fields
                or part.filename
                or part.media_type
            ):
                raise ValueError("A2A accepts nonempty text parts only")
            texts.append(part.text.strip())
        text = "\n".join(texts).strip()
        if not text or text.lstrip().startswith("/"):
            raise ValueError("A2A slash commands are not allowed")
        return text

    def _source(self, principal: str, task_id: str, context_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform("a2a"),
            chat_id=self._chat_id(principal, context_id),
            chat_type="dm",
            user_id=principal,
            user_name=principal,
            message_id=task_id,
            profile=self.active_profile,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        principal, task_id, context_id = self._identity(context)
        source = self._source(principal, task_id, context_id)
        updater = TaskUpdater(event_queue, task_id, context_id)
        current_task = context.current_task
        if current_task is not None and current_task.status.state not in {
            TASK_STATE_INPUT_REQUIRED,
            TASK_STATE_AUTH_REQUIRED,
        }:
            raise RuntimeError("A2A task is not awaiting continuation input")
        record = _RunRecord(source=source, context_key=source.chat_id, updater=updater)
        async with self._runs_guard:
            if task_id in self._runs:
                raise RuntimeError("A2A task is already executing")
            self._runs[task_id] = record
        acquire_task: asyncio.Task | None = None
        cancel_wait: asyncio.Task | None = None
        try:
            if current_task is None:
                await event_queue.enqueue_event(
                    Task(
                        id=task_id,
                        context_id=context_id,
                        status=TaskStatus(state=TASK_STATE_SUBMITTED),
                    )
                )
            await updater.start_work()
            text = self._text(context)
            if record.cancel_requested:
                await self._cancel_record(record)
                return
            acquire_task = asyncio.create_task(
                self._context_locks.acquire(record.context_key)
            )
            cancel_wait = asyncio.create_task(record.cancel_signal.wait())
            done, _pending = await asyncio.wait(
                {acquire_task, cancel_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if cancel_wait in done:
                if acquire_task.done() and not acquire_task.cancelled():
                    acquired = acquire_task.result()
                    await self._context_locks.release(record.context_key, acquired)
                else:
                    acquire_task.cancel()
                    await asyncio.gather(acquire_task, return_exceptions=True)
                acquire_task = None
                cancel_wait = None
                await self._cancel_record(record)
                return
            cancel_wait.cancel()
            await asyncio.gather(cancel_wait, return_exceptions=True)
            cancel_wait = None
            record.context_lock = acquire_task.result()
            acquire_task = None
            if record.cancel_requested:
                await self._cancel_record(record)
                return
            event = MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=task_id,
                metadata={},
            )
            gateway_task = asyncio.create_task(self.adapter.dispatch_request(event))
            # Establish strong ownership in the same synchronous turn as task
            # creation. Acquiring _runs_guard below is a cancellation point;
            # leaving the assignment after it could orphan the dispatch when
            # that guard is contended.
            record.task = gateway_task
            async with self._runs_guard:
                cancel_requested = record.cancel_requested
            if cancel_requested:
                await self._cancel_record(record)
                return
            result = await asyncio.shield(gateway_task)
            output = str(result or "").strip()
            if not output:
                raise RuntimeError("Hermes produced no response")
            await self._complete_record(record, output)
        except asyncio.CancelledError:
            # The producer owns bounded cleanup. Never wait indefinitely for a
            # separate CancelTask request that may itself have timed out.
            await self._cancel_record(record)
            raise
        except Exception:
            await self._fail_record(record)
        finally:
            if cancel_wait is not None:
                cancel_wait.cancel()
                await asyncio.gather(cancel_wait, return_exceptions=True)
            if acquire_task is not None:
                if not acquire_task.done():
                    acquire_task.cancel()
                    await asyncio.gather(acquire_task, return_exceptions=True)
                if (
                    acquire_task.done()
                    and not acquire_task.cancelled()
                    and acquire_task.exception() is None
                    and record.context_lock is None
                ):
                    acquired = acquire_task.result()
                    await self._context_locks.release(record.context_key, acquired)
            gateway_task = record.task
            if gateway_task is not None and not gateway_task.done():
                # A dispatch is allowed to defer cancellation while it unwinds.
                # Keep the authoritative run and context lock until that work
                # genuinely exits; otherwise another request could overlap the
                # old Hermes session after CancelTask has already returned.
                if record.cleanup_task is None:
                    cleanup = asyncio.create_task(
                        self._finish_deferred_record(task_id, record, gateway_task),
                        name=f"a2a-dispatch-reaper-{task_id}",
                    )
                    record.cleanup_task = cleanup
                    self._own_cleanup_task(cleanup)
            else:
                if gateway_task is not None:
                    self._consume_task(gateway_task)
                await self._release_record(task_id, record)

    @staticmethod
    def _remaining(deadline: float | None, fallback: float) -> float:
        if deadline is None:
            return fallback
        return max(0.001, deadline - asyncio.get_running_loop().time())

    async def _complete_record(self, record: _RunRecord, output: str) -> None:
        async with record.lifecycle_lock:
            if record.terminal is not None or record.cancel_requested:
                return
            # Artifact and terminal completion are one serialized lifecycle
            # transaction relative to cancel/failure.
            await record.updater.add_artifact(
                parts=[Part(text=output)], last_chunk=True
            )
            if record.cancel_requested:
                return
            await record.updater.complete()
            record.terminal = "completed"

    async def _fail_record(self, record: _RunRecord) -> None:
        async with record.lifecycle_lock:
            if record.terminal is not None or record.cancel_requested:
                return
            await record.updater.failed()
            record.terminal = "failed"

    @staticmethod
    def _consume_task(task: asyncio.Task) -> None:
        if not task.done() or task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            pass

    def _own_cleanup_task(self, task: asyncio.Task) -> None:
        if task in self._owned_cleanup_tasks:
            return
        self._owned_cleanup_tasks.add(task)

        def reap(done: asyncio.Task) -> None:
            self._owned_cleanup_tasks.discard(done)
            self._consume_task(done)

        task.add_done_callback(reap)

    async def _release_record(self, task_id: str, record: _RunRecord) -> None:
        if record.context_lock is not None:
            await self._context_locks.release(record.context_key, record.context_lock)
            record.context_lock = None
        async with self._runs_guard:
            if self._runs.get(task_id) is record:
                self._runs.pop(task_id, None)
        record.settled.set()

    async def _finish_deferred_record(
        self, task_id: str, record: _RunRecord, gateway_task: asyncio.Task
    ) -> None:
        await asyncio.gather(gateway_task, return_exceptions=True)
        await self._release_record(task_id, record)

    async def _observe_without_waiting(
        self, task: asyncio.Task, *, cancel_pending: bool = False
    ) -> bool:
        if task.done():
            self._consume_task(task)
            return True
        # Ownership must precede both task.cancel() and the first yield. The
        # observer itself may be canceled at either point, while the child is
        # allowed to suppress cancellation and continue running.
        self._own_cleanup_task(task)
        if cancel_pending:
            task.cancel()
        # Yield once so ordinary cancellation can finish, but never await a
        # child which is free to suppress CancelledError.
        await asyncio.sleep(0)
        if task.done():
            self._consume_task(task)
            return True
        return False

    async def _cancel_record(
        self, record: _RunRecord, *, deadline: float | None = None
    ) -> None:
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + self.cancel_wait_seconds
        async with record.cancel_lock:
            async with record.lifecycle_lock:
                if record.terminal is not None or record.cancel_emitted:
                    return
                record.cancel_requested = True
                record.cancel_signal.set()
                interrupt_task = asyncio.create_task(
                    self.adapter.request_session_interrupt(record.source),
                    name="a2a-session-interrupt",
                )
                self._own_cleanup_task(interrupt_task)
                task = record.task
                if task is not None and task is not asyncio.current_task() and not task.done():
                    task.cancel()
                cancel_update = asyncio.create_task(
                    record.updater.cancel(), name="a2a-cancel-event"
                )
                self._own_cleanup_task(cancel_update)
                try:
                    done, _pending = await asyncio.wait(
                        {cancel_update},
                        timeout=self._remaining(deadline, self.cancel_wait_seconds),
                    )
                    if done:
                        self._consume_task(cancel_update)
                    else:
                        cancel_update.cancel()
                        await self._observe_without_waiting(cancel_update)
                finally:
                    record.terminal = "canceled"
                    record.cancel_emitted = True
                    await self._observe_without_waiting(
                        interrupt_task, cancel_pending=True
                    )
                    if task is not None and task is not asyncio.current_task():
                        # The run record (or its producer/reaper) remains the
                        # strong owner of a resistant dispatch.
                        if task.done():
                            self._consume_task(task)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        _principal, task_id, context_id = self._identity(context)
        del event_queue, context_id
        async with self._runs_guard:
            record = self._runs.get(task_id)
        if record is None:
            return
        await self._cancel_record(record)

    async def shutdown(self) -> None:
        async with self._runs_guard:
            records = list(self._runs.values())
        if records:
            deadline = asyncio.get_running_loop().time() + self.cancel_wait_seconds
            await asyncio.gather(
                *(self._cancel_record(record, deadline=deadline) for record in records),
                return_exceptions=True,
            )
        # Do not report shutdown complete while a canceled dispatch still owns
        # its context serialization or while an auxiliary cancellation child
        # is pending. The adapter retains this coroutine as deferred cleanup,
        # which keeps reconnect fail-closed until every child has been reaped.
        while True:
            async with self._runs_guard:
                pending_records = list(self._runs.values())
            cleanup = [task for task in self._owned_cleanup_tasks if not task.done()]
            waits = [asyncio.create_task(record.settled.wait()) for record in pending_records]
            if not waits and not cleanup:
                return
            try:
                # asyncio.wait never propagates cancellation into the owned
                # cleanup tasks. A caller may time out shutdown, but that must
                # not abandon or re-cancel the children it is supervising.
                await asyncio.wait({*waits, *cleanup})
            finally:
                for wait in waits:
                    if not wait.done():
                        wait.cancel()
                await asyncio.sleep(0)
                for wait in waits:
                    self._consume_task(wait)

    def active_session_sources(self) -> tuple[SessionSource, ...]:
        return tuple(record.source for record in self._runs.values() if record.terminal is None)
