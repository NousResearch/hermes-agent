"""Phase 5 Step 1 — AgentTaskRegistry (central-brain-openclaw.md §11 "Phase 5").

The single in-process ledger for live and completed agent tasks.  Step 1 is
purely additive: nothing imports this module yet.  Step 2 dual-writes from
``tools/delegate_tool.py``; later steps route the ``subagent.interrupt`` /
``delegation.pause`` RPCs and the ``task.status`` / ``task.cancel`` surface
through it.

Design rulings this module encodes (doc §"Open questions", decided 2026-06-10):

* Q1 — standalone module, NOT a BrainHost tenant: dual-write must work on the
  default path while ``HERMES_BRAIN_HOST`` is off.  BrainHost may hold a
  reference later.
* Q2 — ephemeral: records and the replay store live in memory only.  Tasks
  that were RUNNING when the process died are simply gone — honest-status
  forbids replaying a result we cannot prove completed.
* Q3 (option B) — ``TaskStatus`` here carries RUNNING; ``contract.Status``
  stays terminal-only and is not imported for the record's own lifecycle.
* Q6 — in-process, but all storage access goes through methods so a
  shared-store backend can replace the dicts without touching callers.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from action_runtime.contract import ExecutionResult

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Lifecycle of an AgentTaskRecord.

    Deliberately separate from ``contract.Status`` (Q3 ruling): an
    ``ExecutionResult`` is always a terminal answer, while a record can be in
    flight.  The five terminal values mirror ``contract.Status`` by string
    value so a completed record's status equals its result's status.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"
    BLOCKED = "blocked"


TERMINAL_STATUSES = frozenset(s for s in TaskStatus if s is not TaskStatus.RUNNING)


@dataclass
class AgentTaskRecord:
    """One live or completed agent task.

    ``task_id`` reuses the existing subagent id scheme (``sa-{i}-{8hex}``) or
    the caller-supplied ``task.submit`` id — no rename needed.
    ``agent_ref`` holds the live AIAgent (a ``weakref.ref`` or the object
    itself) for interrupt routing; it is dropped on completion and never
    serialized.  ``session_id`` ties the record to its gateway session so
    ``complete()`` can append the snapshot to that session's
    ``_tasks.jsonl`` (Step 6a); ``None`` means no persistence.
    """

    task_id: str
    parent_task_id: Optional[str] = None
    session_id: Optional[str] = None
    depth: int = 0
    goal: str = ""
    intent: str = ""
    model: Optional[str] = None
    started_at: float = 0.0
    finished_at: Optional[float] = None
    status: TaskStatus = TaskStatus.RUNNING
    agent_ref: Optional[Any] = None
    tool_count: int = 0
    last_tool: Optional[str] = None
    result: Optional[ExecutionResult] = None
    idempotency_key: Optional[str] = None

    def snapshot(self) -> dict:
        """Serializable view — no ``agent_ref``; result as the rich wire dict."""
        from action_runtime.adapters import result_to_wire_rich

        snap = {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "depth": self.depth,
            "goal": self.goal,
            "intent": self.intent,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status.value,
            "tool_count": self.tool_count,
            "last_tool": self.last_tool,
            "result": result_to_wire_rich(self.result) if self.result else None,
        }
        if self.session_id is not None:
            # Only when set: task.status / task.list spread snapshot() onto
            # the wire, so records without a session must keep the exact
            # pre-Step-6a shape (additive-first ruling).
            snap["session_id"] = self.session_id
        return snap


# Per-session append-only ledger of completed-task snapshots (Step 6a),
# living next to the TUI's legacy ``_index.jsonl`` in the same session dir
# (server.py ``_spawn_tree_session_dir``).  Same no-lock append pattern as
# ``_append_spawn_tree_index``: one ``f.write`` of the full line per open
# in append mode, so concurrent completes from worker threads stay
# crash-tolerant (doc risk R4).
_TASKS_JSONL = "_tasks.jsonl"


def _persist_task_snapshot(session_id: str, snapshot: dict) -> None:
    """Append *snapshot* to ``$HERMES_HOME/spawn-trees/<session_id>/_tasks.jsonl``.

    Best-effort observability: the whole body is guarded — a persistence
    failure can never break ``complete()``.  Imports are lazy so the module
    stays import-light on paths that never persist.  The snapshot is built
    by the caller under the registry lock so the line is never torn; the
    append itself is a single unbuffered byte write, because snapshots embed
    full rich results that can exceed the text-layer chunk size and a
    buffered text write could interleave between threads.
    """
    try:
        import json

        from hermes_constants import get_hermes_home

        # Same directory-name sanitization as server._spawn_tree_session_dir,
        # so registry lines land in the session dir the TUI already uses.
        safe = (
            "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
            or "unknown"
        )
        session_dir = get_hermes_home() / "spawn-trees" / safe
        session_dir.mkdir(parents=True, exist_ok=True)
        line_bytes = (json.dumps(snapshot, ensure_ascii=False) + "\n").encode("utf-8")
        with (session_dir / _TASKS_JSONL).open("ab", buffering=0) as f:
            f.write(line_bytes)
    except Exception as exc:
        logger.debug("task snapshot persist failed for %s: %s", session_id, exc)


class AgentTaskRegistry:
    """Process-wide task ledger.  Thread-safe; storage is private to methods
    so the in-memory dicts can later be swapped for a shared store (Q6).
    """

    REPLAY_TTL_S = 600.0
    REPLAY_CAP = 1024
    RECORDS_TERMINAL_CAP = 1024

    def __init__(self) -> None:
        self._records: dict[str, AgentTaskRecord] = {}
        self._replay: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._spawns_paused = False

    # -- records ----------------------------------------------------------

    def register(self, record: AgentTaskRecord) -> None:
        if not record.started_at:
            record.started_at = time.time()
        with self._lock:
            self._records[record.task_id] = record

    def get(self, task_id: str) -> Optional[AgentTaskRecord]:
        with self._lock:
            return self._records.get(task_id)

    def list_active(self) -> list[AgentTaskRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.status is TaskStatus.RUNNING]

    def get_snapshot(self, task_id: str) -> Optional[dict]:
        """Locked snapshot for the RPC layer — never a half-mutated record."""
        with self._lock:
            record = self._records.get(task_id)
            return record.snapshot() if record is not None else None

    def list_active_snapshots(self) -> list[dict]:
        """Locked snapshots of RUNNING records for the RPC layer."""
        with self._lock:
            return [
                r.snapshot()
                for r in self._records.values()
                if r.status is TaskStatus.RUNNING
            ]

    def complete(self, task_id: str, result: Optional[ExecutionResult]) -> bool:
        """Transition to a terminal status.

        With a result, the record's status mirrors the result's terminal
        status.  ``result=None`` is the timeout/interrupt path and maps to
        FAILED (doc ruling: interrupted == FAILED, ErrorType.TRANSPORT — the
        ExecError itself is built by the Step 2 caller).  Returns False for
        an unknown task_id or one already terminal — a late second
        ``complete()`` must never falsify the first result or double-append
        to ``_tasks.jsonl``.
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status is not TaskStatus.RUNNING:
                return False
            record.result = result
            record.status = (
                TaskStatus(result.status.value) if result is not None else TaskStatus.FAILED
            )
            record.finished_at = time.time()
            record.agent_ref = None
            key = record.idempotency_key
            session_id = record.session_id
            # One snapshot, built under the lock so it can never be torn by
            # a concurrent mutation; reused for the replay write and the
            # file append below (the I/O itself stays outside the lock).
            snapshot = record.snapshot()
            self._evict_terminal_records()
        if key is not None and result is not None:
            # task.submit replays this dict directly as the RPC result, so
            # it must be the rich wire shape — snapshot()["result"] is
            # exactly result_to_wire_rich(result).
            self.remember(key, snapshot["result"])
        if session_id:
            _persist_task_snapshot(session_id, snapshot)
        return True

    def _evict_terminal_records(self) -> None:
        # caller holds self._lock.  Bound terminal retention so a long-lived
        # gateway never grows _records without limit; RUNNING records are
        # never evicted.
        terminal = [
            r for r in self._records.values() if r.status is not TaskStatus.RUNNING
        ]
        excess = len(terminal) - self.RECORDS_TERMINAL_CAP
        if excess <= 0:
            return
        terminal.sort(key=lambda r: r.finished_at or 0.0)
        for record in terminal[:excess]:
            del self._records[record.task_id]

    def update_progress(
        self,
        task_id: str,
        tool_count: Optional[int] = None,
        last_tool: Optional[str] = None,
    ) -> bool:
        """Live-progress field update from the child progress callback.

        Lock-held so task.status/task.list snapshots never see a torn
        record.  No-op (False) on unknown or already-terminal records — a
        late callback after complete() must never mutate a terminal record.
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status is not TaskStatus.RUNNING:
                return False
            if tool_count is not None:
                record.tool_count = tool_count
            if last_tool is not None:
                record.last_tool = last_tool
            return True

    def interrupt(self, task_id: str, reason: Optional[str] = None) -> bool:
        """Interrupt the live agent behind *task_id*.

        Same mechanic as today's ``interrupt_subagent``: resolve the stored
        agent and call ``agent.interrupt()`` — outside the registry lock, so
        agent code never runs under it.  *reason* is forwarded as the
        agent's optional interrupt message when given.  Returns False when
        the task is unknown, already terminal, its agent is gone, or the
        interrupt itself raises (parity with ``interrupt_subagent``, which
        swallows exceptions so a flaky agent can't kill the RPC dispatcher).
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.status is not TaskStatus.RUNNING:
                return False
            ref = record.agent_ref
        agent = ref() if isinstance(ref, weakref.ref) else ref
        if agent is None or not hasattr(agent, "interrupt"):
            return False
        try:
            if reason is not None:
                agent.interrupt(reason)
            else:
                agent.interrupt()
        except Exception as exc:
            logger.debug("interrupt(%s) failed: %s", task_id, exc)
            return False
        return True

    # -- spawn pause flag (replaces delegate_tool._spawn_paused in Step 3) --

    def pause_spawns(self, paused: bool) -> None:
        with self._lock:
            self._spawns_paused = bool(paused)

    def spawns_paused(self) -> bool:
        with self._lock:
            return self._spawns_paused

    # -- idempotency replay store (folds in server._TASK_RESULTS at Step 5) --

    def remember(self, key: str, wire: dict, now: Optional[float] = None) -> None:
        """Store a terminal wire dict for replay.  Ephemeral by ruling (Q2).

        TTL bookkeeping runs on ``time.monotonic()`` — same timebase as the
        legacy ``server._TASK_RESULTS`` it folds in — so a wall-clock jump
        can't mass-evict or immortalize entries.
        """
        ts = time.monotonic() if now is None else now
        with self._lock:
            self._evict_expired(ts)
            if len(self._replay) >= self.REPLAY_CAP:
                oldest = min(self._replay, key=lambda k: self._replay[k][0])
                del self._replay[oldest]
            self._replay[key] = (ts, wire)

    def recall(self, key: str, now: Optional[float] = None) -> Optional[dict]:
        ts = time.monotonic() if now is None else now
        with self._lock:
            self._evict_expired(ts)
            entry = self._replay.get(key)
            return entry[1] if entry else None

    def _evict_expired(self, now: float) -> None:
        # caller holds self._lock
        cutoff = now - self.REPLAY_TTL_S
        expired = [k for k, (ts, _) in self._replay.items() if ts < cutoff]
        for k in expired:
            del self._replay[k]


_instance: Optional[AgentTaskRegistry] = None
_instance_lock = threading.Lock()


def get_registry() -> AgentTaskRegistry:
    """Return (or create) the process-level registry singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = AgentTaskRegistry()
    return _instance
