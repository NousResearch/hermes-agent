"""Federation task executor relay — execute tasks claimed from peers.

When a device claims a task via consensus, this module:
1. Downloads task context (what the original device was doing)
2. Resumes execution from the checkpoint
3. Streams progress back to all peers in real-time
4. Reports final result to the federation

This is the "接力" (relay) mechanism — seamless task handoff.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from gateway.federation.federation_protocol import (
    FedMessage,
    MessageType,
)
from gateway.federation.federation_consensus import FederationConsensus

logger = logging.getLogger(__name__)


@dataclass
class TaskCheckpoint:
    """Snapshot of task execution state for relay."""

    task_id: str
    executor_device: str
    checkpoint_id: str
    timestamp: float = field(default_factory=time.time)

    # What was being done
    current_step: str = ""
    step_index: int = 0
    total_steps: int = 0

    # Context needed to resume
    context: dict = field(default_factory=dict)

    # Progress
    progress: float = 0.0  # 0.0 - 1.0
    note: str = ""


@dataclass
class TaskExecutionState:
    """Full state of a task being executed locally."""

    task_id: str
    title: str
    description: str = ""
    priority: int = 3

    # Execution
    status: str = "pending"  # pending | claimed | in_progress | completed | failed | relayed
    executor_device: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0

    # Progress
    progress: float = 0.0
    current_step: str = ""
    progress_note: str = ""

    # Checkpoints for relay
    checkpoints: list[TaskCheckpoint] = field(default_factory=list)
    context_snapshot: dict = field(default_factory=dict)

    # Result
    result_data: dict = field(default_factory=dict)
    error_info: str = ""

    # Relay tracking
    relay_count: int = 0
    previous_executors: list[str] = field(default_factory=list)


class TaskExecutorRelay:
    """Executes claimed tasks with checkpoint/relay support.

    Usage:
        relay = TaskExecutorRelay(federation_adapter, consensus)
        await relay.claim_and_execute(task_id, context)
    """

    def __init__(
        self,
        device_id: str,
        adapter: Any,  # FederationAdapter
        consensus: FederationConsensus,
        progress_interval: float = 10.0,
        checkpoint_interval: float = 30.0,
    ):
        self.device_id = device_id
        self.adapter = adapter
        self.consensus = consensus
        self.progress_interval = progress_interval
        self.checkpoint_interval = checkpoint_interval

        self._active_tasks: Dict[str, TaskExecutionState] = {}
        self._task_handlers: Dict[str, Callable] = {}
        self._running = False

    # ----------------------------------------------------------------
    # Task claiming and execution
    # ----------------------------------------------------------------

    async def claim_and_execute(
        self,
        task_id: str,
        title: str = "",
        description: str = "",
        priority: int = 3,
        context_snapshot: Optional[dict] = None,
        handler: Optional[Callable] = None,
    ) -> TaskExecutionState:
        """Claim a task via consensus and execute it.

        This is the main entry point — handles the full lifecycle:
        1. Initiate consensus claim
        2. If accepted, start execution
        3. Stream progress to peers
        4. Report result on completion
        """
        # Create execution state
        state = TaskExecutionState(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            context_snapshot=context_snapshot or {},
            executor_device=self.device_id,
        )
        self._active_tasks[task_id] = state

        # Step 1: Claim via consensus
        claimed = await self.consensus.initiate_claim(task_id)
        if not claimed:
            state.status = "failed"
            state.error_info = "Claim rejected by consensus (another device claimed first)"
            logger.warning(
                "Federation relay: task %s claim rejected", task_id,
            )
            return state

        state.status = "claimed"
        state.started_at = time.time()
        logger.info(
            "Federation relay: task %s claimed, starting execution", task_id,
        )

        # Step 2: Execute
        try:
            state.status = "in_progress"
            await self.adapter.report_progress(task_id, 0.0, "Starting execution")

            # Start progress streaming
            progress_task = asyncio.create_task(
                self._stream_progress(task_id)
            )

            # Start checkpointing
            checkpoint_task = asyncio.create_task(
                self._save_checkpoints(task_id)
            )

            # Run the actual task handler
            task_handler = handler or self._task_handlers.get(task_id)
            if task_handler:
                result = await task_handler(state)
            else:
                result = await self._default_handler(state)

            # Stop background tasks
            progress_task.cancel()
            checkpoint_task.cancel()

            # Step 3: Report result
            state.status = "completed"
            state.completed_at = time.time()
            state.progress = 1.0
            state.result_data = result or {}
            await self.adapter.report_result(task_id, True, result)

            logger.info(
                "Federation relay: task %s completed (%.1fs)",
                task_id, state.completed_at - state.started_at,
            )

        except asyncio.CancelledError:
            state.status = "failed"
            state.error_info = "Task cancelled"
            await self.adapter.report_result(
                task_id, False, error_info="Task cancelled",
            )
            raise

        except Exception as e:
            state.status = "failed"
            state.error_info = str(e)
            await self.adapter.report_result(task_id, False, error_info=str(e))
            logger.error(
                "Federation relay: task %s failed: %s", task_id, e,
            )

        finally:
            # Send task completion notification
            await self.adapter.send_task_heartbeat(task_id)

        return state

    # ----------------------------------------------------------------
    # Progress streaming
    # ----------------------------------------------------------------

    async def _stream_progress(self, task_id: str) -> None:
        """Stream progress updates to all peers periodically."""
        while self._running:
            state = self._active_tasks.get(task_id)
            if not state or state.status not in ("in_progress", "claimed"):
                break

            await self.adapter.report_progress(
                task_id,
                state.progress,
                state.progress_note or state.current_step,
            )
            await asyncio.sleep(self.progress_interval)

    # ----------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------

    async def _save_checkpoints(self, task_id: str) -> None:
        """Save execution checkpoints periodically for relay support."""
        checkpoint_idx = 0

        while self._running:
            state = self._active_tasks.get(task_id)
            if not state or state.status not in ("in_progress", "claimed"):
                break

            checkpoint = TaskCheckpoint(
                task_id=task_id,
                executor_device=self.device_id,
                checkpoint_id=f"cp-{task_id}-{checkpoint_idx}",
                current_step=state.current_step,
                step_index=checkpoint_idx,
                total_steps=0,  # Unknown for generic handler
                context=state.context_snapshot.copy(),
                progress=state.progress,
                note=state.progress_note,
            )
            state.checkpoints.append(checkpoint)

            # Store latest checkpoint in context for relay
            state.context_snapshot["last_checkpoint"] = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "current_step": checkpoint.current_step,
                "progress": checkpoint.progress,
                "context": checkpoint.context,
            }

            checkpoint_idx += 1
            await asyncio.sleep(self.checkpoint_interval)

    # ----------------------------------------------------------------
    # Task handoff (relay)
    # ----------------------------------------------------------------

    async def handoff_task(
        self,
        task_id: str,
        reason: str = "device going offline",
    ) -> Optional[TaskExecutionState]:
        """Hand off a running task to another device.

        Called when this device needs to stop executing (going offline,
        resource pressure, etc.). Creates a final checkpoint and broadcasts
        the task as available for relay.
        """
        state = self._active_tasks.get(task_id)
        if not state:
            return None

        # Create final checkpoint
        final_checkpoint = TaskCheckpoint(
            task_id=task_id,
            executor_device=self.device_id,
            checkpoint_id=f"cp-{task_id}-final",
            current_step=state.current_step,
            step_index=len(state.checkpoints),
            context=state.context_snapshot.copy(),
            progress=state.progress,
            note=f"Handoff: {reason}",
        )
        state.checkpoints.append(final_checkpoint)
        state.context_snapshot["last_checkpoint"] = {
            "checkpoint_id": final_checkpoint.checkpoint_id,
            "current_step": final_checkpoint.current_step,
            "progress": final_checkpoint.progress,
            "context": final_checkpoint.context,
            "handoff_reason": reason,
        }

        state.status = "pending"  # Back to pending for another device to claim
        state.previous_executors.append(self.device_id)
        state.relay_count += 1

        # Remove from active tasks
        del self._active_tasks[task_id]

        # Broadcast as available for relay
        await self.adapter.submit_task(
            task_id=task_id,
            title=f"[RELAY #{state.relay_count}] {state.title}",
            description=state.description,
            priority=state.priority,
            context_snapshot=state.context_snapshot,
        )

        logger.info(
            "Federation relay: task %s handed off (reason: %s, relay #%d)",
            task_id, reason, state.relay_count,
        )

        return state

    async def resume_from_checkpoint(
        self,
        task_id: str,
        checkpoint: dict,
        handler: Optional[Callable] = None,
    ) -> TaskExecutionState:
        """Resume a task from a relay checkpoint.

        Called when this device claims a relayed task.
        Restores context from checkpoint and continues execution.
        """
        # Extract checkpoint data
        progress = checkpoint.get("progress", 0.0)
        current_step = checkpoint.get("current_step", "")
        context = checkpoint.get("context", {})
        handoff_reason = checkpoint.get("handoff_reason", "")

        # Create execution state with restored context
        state = TaskExecutionState(
            task_id=task_id,
            title=f"[RELAY] {task_id}",
            description=f"Resumed after handoff: {handoff_reason}",
            priority=3,
            context_snapshot=context,
            executor_device=self.device_id,
            progress=progress,
            current_step=current_step,
        )
        self._active_tasks[task_id] = state

        # Execute from checkpoint
        return await self.claim_and_execute(
            task_id=task_id,
            title=state.title,
            description=state.description,
            context_snapshot=context,
            handler=handler,
        )

    # ----------------------------------------------------------------
    # Handler registration
    # ----------------------------------------------------------------

    def register_handler(self, task_id: str, handler: Callable) -> None:
        """Register a custom handler for a specific task."""
        self._task_handlers[task_id] = handler

    def register_handler_pattern(
        self, pattern: str, handler: Callable
    ) -> None:
        """Register a handler for tasks matching a pattern (future)."""
        # TODO: Implement pattern matching for task routing
        pass

    # ----------------------------------------------------------------
    # Default handler
    # ----------------------------------------------------------------

    async def _default_handler(self, state: TaskExecutionState) -> dict:
        """Bridge a federated task into the local Kanban dispatcher.

        Creates a Kanban task with the same task_id as the federation task
        (idempotency key ensures no duplicates on relay retry).  The local
        _kanban_dispatcher_watcher will see the 'ready' task on its next tick
        and spawn a real worker — this keeps federation and Kanban fully
        decoupled while still sharing the same execution pool.
        """
        logger.info(
            "Federation relay [%s]: bridging task to Kanban — %s",
            self.device_id, state.task_id,
        )

        try:
            import sqlite3
            from pathlib import Path
            from hermes_cli.kanban_db import create_task, kanban_home

            # Locate the local Kanban DB for this profile
            kanban_path = kanban_home() / "kanban" / "kanban.db"
            if not kanban_path.exists():
                logger.warning(
                    "Federation relay [%s]: Kanban DB not found at %s — "
                    "task %s will not be dispatched",
                    self.device_id, kanban_path, state.task_id,
                )
                return {"error": "kanban_db_not_found", "path": str(kanban_path)}

            conn = sqlite3.connect(kanban_path)
            try:
                # Use the federation task_id as idempotency_key so re-relayed
                # tasks don't create duplicates in Kanban.
                kanban_task_id = create_task(
                    conn,
                    title=state.title or f"[Federation] {state.task_id}",
                    body=state.description,
                    priority=state.priority,
                    initial_status="ready",
                    board=None,  # use default board
                    idempotency_key=f"fed:{state.task_id}",
                    session_id=state.context_snapshot.get("session_id"),
                )
                logger.info(
                    "Federation relay [%s]: task %s → Kanban task %s created",
                    self.device_id, state.task_id, kanban_task_id,
                )
                return {
                    "kanban_task_id": kanban_task_id,
                    "federation_task_id": state.task_id,
                    "status": "bridged_to_kanban",
                }
            finally:
                conn.close()

        except ImportError as exc:
            logger.warning(
                "Federation relay [%s]: hermes_cli.kanban_db not importable "
                "(%s) — task %s will not be dispatched",
                self.device_id, exc, state.task_id,
            )
            return {"error": "kanban_import_failed", "detail": str(exc)}
        except Exception as exc:
            logger.error(
                "Federation relay [%s]: failed to bridge task %s → Kanban: %s",
                self.device_id, state.task_id, exc, exc_info=True,
            )
            return {"error": "kanban_bridge_failed", "detail": str(exc)}

    # ----------------------------------------------------------------
    # State queries
    # ----------------------------------------------------------------

    def get_active_task(self, task_id: str) -> Optional[TaskExecutionState]:
        """Get state for an active task."""
        return self._active_tasks.get(task_id)

    def get_all_active_tasks(self) -> Dict[str, TaskExecutionState]:
        """Get all active tasks."""
        return dict(self._active_tasks)

    @property
    def active_task_count(self) -> int:
        """Number of currently active tasks."""
        return len(self._active_tasks)

    async def start(self) -> None:
        """Start the task executor relay."""
        self._running = True
        logger.info("Federation relay: task executor started")

    async def stop(self) -> None:
        """Stop the task executor relay — hand off all active tasks."""
        self._running = False

        # Hand off all active tasks before stopping
        for task_id in list(self._active_tasks):
            await self.handoff_task(task_id, reason="executor shutting down")

        logger.info("Federation relay: task executor stopped")
