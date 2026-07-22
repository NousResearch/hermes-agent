"""Hermes implementation of the A2A ``AgentExecutor``.

This is the bridge between the a2a-sdk request lifecycle and Hermes' agent loop:
resolve the conversation context, run one ``AIAgent`` turn in a worker thread,
stream its progress as task-status updates, and deliver the final answer as an
artifact.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, InvalidParamsError, Part, TaskState, TextPart
from a2a.utils import new_task
from a2a.utils.errors import ServerError

from .config import DEFAULT_TOOL_IO, load_a2a_settings
from .events import (
    make_reasoning_cb,
    make_step_cb,
    make_stream_delta_cb,
    make_tool_progress_cb,
)
from .sessions import ContextSessionStore

logger = logging.getLogger(__name__)


class HermesAgentExecutor(AgentExecutor):
    """Runs Hermes turns in response to A2A ``message/send`` and ``message/stream``."""

    def __init__(
        self,
        store: ContextSessionStore | None = None,
        *,
        max_concurrency: int | None = None,
        tool_io_mode: str | None = None,
    ):
        settings = load_a2a_settings()
        self._store = store or ContextSessionStore()
        self._max_concurrency = max_concurrency or settings.max_concurrency
        self._tool_io_mode = tool_io_mode or settings.tool_io or DEFAULT_TOOL_IO
        # Dedicated, bounded pool so A2A turns neither saturate nor are starved
        # by asyncio's shared default executor. Created lazily on first turn.
        self._turn_pool: ThreadPoolExecutor | None = None
        self._admission_lock = threading.Lock()
        self._active_turns = 0
        self._closed = False

    def _pool(self) -> ThreadPoolExecutor:
        if self._turn_pool is None:
            self._turn_pool = ThreadPoolExecutor(
                max_workers=self._max_concurrency,
                thread_name_prefix="hermes-a2a-turn",
            )
        return self._turn_pool

    def _submit_admitted_turn(
        self,
        context_id: str,
        user_text: str,
        task_id: str,
        callbacks: dict[str, Any],
    ):
        """Lease a session and submit its turn atomically against shutdown.

        An admitted coroutine can await protocol event delivery before it is
        ready to start a worker.  Holding the admission lock across the final
        closed check, session acquisition, and pool submission ensures
        ``aclose()`` either sees and drains that worker or closes first and
        prevents any post-close session/pool recreation.
        """
        with self._admission_lock:
            if self._closed:
                raise ServerError(
                    error=InternalError(message="A2A server is stopping.")
                )
            session = self._store.acquire(context_id)
            try:
                worker_future = self._pool().submit(
                    session.run_turn,
                    user_text,
                    task_id,
                    callbacks=callbacks,
                )
            except Exception:
                self._store.release(session)
                raise
        return session, worker_future

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Reject empty/blank input before constructing a task. ``new_task``
        # itself raises on an empty TextPart, so guarding here avoids an
        # unhandled error and returns a proper JSON-RPC error to the client.
        user_text = context.get_user_input()
        if not user_text.strip():
            raise ServerError(
                error=InvalidParamsError(message="Message contains no text to act on.")
            )

        with self._admission_lock:
            if self._closed:
                raise ServerError(
                    error=InternalError(message="A2A server is stopping.")
                )
            if self._active_turns >= self._max_concurrency:
                raise ServerError(
                    error=InternalError(
                        message="A2A server is at turn capacity; retry later."
                    )
                )
            self._active_turns += 1

        reservation = {"transferred": False}
        try:
            await self._execute_admitted(
                context,
                event_queue,
                user_text,
                reservation=reservation,
            )
        finally:
            if not reservation["transferred"]:
                self._release_admission()

    async def _execute_admitted(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        user_text: str,
        *,
        reservation: dict[str, bool],
    ) -> None:
        task = context.current_task
        if task is None:
            message = context.message
            if message is None:
                raise ServerError(
                    error=InvalidParamsError(
                        message="A2A request has no message to start a task from."
                    )
                )
            task = new_task(message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        loop = asyncio.get_running_loop()

        # AIAgent's callbacks are bound onto the (shared, per-context) agent
        # *inside* run_turn under the session lock — never here on the shared
        # instance — so two concurrent turns on the same context can't overwrite
        # each other's TaskUpdater. ``thinking_callback`` is silenced (local
        # "kawaii" status spam, not for A2A).
        callbacks = {
            "stream_delta_callback": make_stream_delta_cb(updater, loop),
            "reasoning_callback": make_reasoning_cb(updater, loop),
            "tool_progress_callback": make_tool_progress_cb(
                updater, loop, tool_io_mode=self._tool_io_mode
            ),
            "step_callback": make_step_cb(
                updater, loop, tool_io_mode=self._tool_io_mode
            ),
            "thinking_callback": None,
        }

        try:
            session, worker_future = self._submit_admitted_turn(
                task.context_id,
                user_text,
                task.id,
                callbacks,
            )
            reservation["transferred"] = True

            def release_worker_resources(_future):
                self._store.release(session)
                self._release_admission()

            worker_future.add_done_callback(release_worker_resources)
            result = await asyncio.wrap_future(worker_future)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface any agent error as task failure
            logger.exception("Hermes turn failed for task %s", task.id)
            await self._safe_terminal(
                updater,
                updater.failed(
                    updater.new_agent_message([
                        Part(root=TextPart(text=self._peer_error_message(exc)))
                    ])
                ),
            )
            return

        # Map AIAgent's turn outcome onto the A2A task state. ``run_conversation``
        # catches its own failures and *returns* a dict (failed/interrupted/error)
        # rather than raising, so we must inspect it — otherwise a failed or
        # truncated turn would be reported to the peer agent as a successful,
        # empty completion.
        if session.cancel_event.is_set() or result.get("interrupted"):
            await self._safe_terminal(
                updater, updater.update_status(TaskState.canceled, final=True)
            )
            return

        final_text = str(result.get("final_response") or "").strip()
        err = str(result.get("error") or "").strip()

        # A turn is unsuccessful when it set an explicit ``failed`` flag, OR
        # carries an ``error`` string, OR produced no usable text. Several
        # degraded/partial early-returns in run_conversation (thinking-budget
        # exhausted, response truncation) set ``error`` + a human-readable
        # ``final_response`` but never reach ``finalize_turn``, so the dict has
        # no ``failed`` key — without the ``err`` check those reach the peer as a
        # "completed" task whose artifact is really an error notice.
        if result.get("failed") or err or not final_text:
            if final_text and err:
                text = f"{final_text}\n\n(error: {err})"
            elif final_text:
                text = final_text
            elif err:
                text = f"Agent turn failed: {err}"
            else:
                text = "Agent produced no response."
            await self._safe_terminal(
                updater,
                updater.failed(
                    updater.new_agent_message([Part(root=TextPart(text=text))])
                ),
            )
            return

        await updater.add_artifact(
            [Part(root=TextPart(text=final_text))], name="response", last_chunk=True
        )
        await self._safe_terminal(updater, updater.complete())

    def _release_admission(self) -> None:
        with self._admission_lock:
            self._active_turns = max(0, self._active_turns - 1)

    async def aclose(self) -> None:
        """Stop accepting work, interrupt turns, and release agent resources."""
        with self._admission_lock:
            if self._closed:
                return
            self._closed = True
        self._store.cancel_all()
        pool = self._turn_pool
        if pool is not None:
            await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)
            self._turn_pool = None
        self._store.close()

    @staticmethod
    def _peer_error_message(exc: Exception) -> str:
        """A peer-safe failure message.

        The full traceback is logged server-side; the remote peer gets only the
        exception *type*, never the raw ``str(exc)`` — which can carry file
        paths, prompts, or other internal detail to an untrusted caller.
        """
        return (
            f"The agent encountered an internal error ({type(exc).__name__}) "
            "while processing the task."
        )

    @staticmethod
    async def _safe_terminal(updater: TaskUpdater, coro) -> None:
        """Await a terminal TaskUpdater call, tolerating an already-terminal task.

        ``cancel()`` may drive the same task to a terminal state from a separate
        TaskUpdater; swallow the resulting ``RuntimeError`` instead of crashing
        the turn.
        """
        try:
            await coro
        except RuntimeError:
            logger.debug("Task %s already terminal; skipping update", updater.task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        # Prefer the task's context_id (the key the store is indexed by in
        # execute); fall back to the request context_id.
        context_id = (
            task.context_id if task is not None else None
        ) or context.context_id
        if context_id:
            session = self._store.get(context_id)
            if session is not None:
                # Scope the cancel to this specific task so it can't interrupt a
                # different concurrent turn on the same context.
                session.cancel(task.id if task is not None else None)

        if task is not None:
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            try:
                await updater.update_status(TaskState.canceled, final=True)
            except Exception:
                logger.debug("cancel status update failed", exc_info=True)
