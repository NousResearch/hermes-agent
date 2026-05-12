"""Focused task registry substrate for Hermes (orchestrator Phase 3).

Hermes is moving toward a concierge / front-desk / butler orchestrator: it stays
accountable for user intent, prioritisation, synthesis and final output, but
heavy work may eventually be delegated to workers, and late user follow-ups must
still find their way back to the *task* they belong to -- not merely the chat
session that produced them.  Today there is nowhere to write down "which focused
task is Woo actually on right now, what state is it in, and what follow-ups have
piled up against it".

This module is that place.  :class:`FocusedTask` is a small, serialisable record
of one focused user task -- its identity (``task_id`` / ``session_key`` /
``origin``), its lifecycle ``status``, the pending follow-ups attached to it
(Phase 2 :class:`~agent.pending_turn_queue.PendingTurnItem` objects, in arrival
order), an optional worker linkage (``active_worker_id`` / ``worker_kind``) for
whoever runs the work later, plus free-form ``artifacts`` and ``notes``.
:class:`TaskRegistry` is a tiny in-memory collection with create / get / list /
status-update / cancel / attach helpers, JSON serialisation, and *optional*
atomic-write file persistence.

Scope discipline (Phase 3 is the *substrate*, not the behaviour):

* This is **not** the Ralph / focused-agent runtime, **not** a follow-up
  classifier or intent model, **not** a worker lane or background-delegation
  mechanism, and **not** automatic routing of CLI/Telegram messages into tasks.
  Those are Phase 4/5.  Nothing here decides whether a given follow-up is an
  append, a correction, a steer, a status query, or a new task -- it only gives
  later phases somewhere to *record* such a decision once it is made.
* This module is a leaf: it imports only the standard library and the Phase 2
  :mod:`agent.pending_turn_queue`, holds no global mutable state, and everything
  produced by :meth:`TaskRegistry.to_dict` is plain JSON-safe data.  Pending
  follow-ups are serialised via :meth:`PendingTurnItem.to_dict`, which drops the
  one local-process passthrough (``PendingTurnItem.raw``) without touching it;
  no raw payload is ever serialised or deep-copied.
* File persistence is intentionally minimal: a single JSON document written
  atomically (temp file + ``os.replace``).  It is optional -- a registry with no
  bound path is a perfectly good in-memory registry with explicit serialisation
  helpers; durable multi-writer storage (SQLite, per-task files, ...) is
  explicitly Phase 4/5 if it is ever needed.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field, fields, replace
from typing import Any

from agent.pending_turn_queue import PendingTurnItem, from_legacy_cli_payload

__all__ = [
    "STATUS_PROPOSED",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "STATUS_STEERABLE",
    "STATUS_BLOCKED",
    "STATUS_DONE",
    "STATUS_ERROR",
    "STATUS_CANCELLED",
    "TASK_STATUSES",
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "WORKER_DELEGATE",
    "WORKER_CLAUDE_CODE",
    "WORKER_TERMINAL",
    "WORKER_RALPH",
    "TaskOrigin",
    "FocusedTask",
    "TaskRegistry",
]

# --------------------------------------------------------------------------
# Task lifecycle vocabulary.
#
# Unlike the pending-turn ``kind`` / ``boundary`` strings (which are deliberately
# open -- unknown future values are tolerated), the task status set is *closed*:
# a status outside this set is a bug, so it is rejected with a clear error
# wherever a status is assigned (construction, ``update_status``, ``from_dict``).
# --------------------------------------------------------------------------
STATUS_PROPOSED = "proposed"     # created, not yet accepted/queued
STATUS_QUEUED = "queued"         # accepted, waiting to start
STATUS_RUNNING = "running"       # actively being worked (by Hermes or a worker)
STATUS_STEERABLE = "steerable"   # running and at a point where steering is safe
STATUS_BLOCKED = "blocked"       # needs user input / an external unblock
STATUS_DONE = "done"             # finished successfully
STATUS_ERROR = "error"           # finished with a failure
STATUS_CANCELLED = "cancelled"   # cancelled / reclaimed before completion

TASK_STATUSES = frozenset(
    {
        STATUS_PROPOSED,
        STATUS_QUEUED,
        STATUS_RUNNING,
        STATUS_STEERABLE,
        STATUS_BLOCKED,
        STATUS_DONE,
        STATUS_ERROR,
        STATUS_CANCELLED,
    }
)
# Terminal statuses: a task in one of these is finished and is excluded from the
# "active tasks" view by default.
TERMINAL_STATUSES = frozenset({STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED})
ACTIVE_STATUSES = frozenset(TASK_STATUSES - TERMINAL_STATUSES)

# Documented ``worker_kind`` hints for the worker-linkage fields.  These are
# *documentation*, not a rejecting enum -- ``worker_kind`` is a free-form string
# (or None) so Phase 4 can name new lanes without touching this module.  Nothing
# in Phase 3 starts a worker; these names just make the linkage intent concrete.
WORKER_DELEGATE = "delegate_task"   # an in-process delegate_task subagent
WORKER_CLAUDE_CODE = "claude_code"  # a Claude Code CLI worker
WORKER_TERMINAL = "terminal"        # a detached terminal / background process
WORKER_RALPH = "ralph"              # the future focused-agent ("Ralph") runtime


def _new_task_id() -> str:
    return f"task-{uuid.uuid4().hex}"


def _validate_status(status: Any) -> str:
    if status not in TASK_STATUSES:
        raise ValueError(
            f"unknown task status {status!r}; expected one of {sorted(TASK_STATUSES)}"
        )
    return status


def _as_float(value: Any, default: float) -> float:
    # ``bool`` is an ``int`` subclass; reject it so ``created_at: true`` from
    # mangled data does not silently become ``1.0``.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _json_safe_copy(value: Any, *, label: str) -> Any:
    """Return a detached JSON-safe copy of *value* or raise TypeError.

    The registry promises JSON-safe serialization for task metadata.  Validate
    local artifact/note-style payloads at insertion time so persistence cannot
    fail much later, and use a JSON round-trip to detach nested mutable values.
    This helper is only used for registry metadata; it never touches
    ``PendingTurnItem.raw``.
    """
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-serializable") from exc


# --------------------------------------------------------------------------
# TaskOrigin
# --------------------------------------------------------------------------
@dataclass
class TaskOrigin:
    """Where a focused task came from.  All fields are plain strings or ``None``."""

    platform: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None
    user_id: str | None = None
    session_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "chat_id": self.chat_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "session_key": self.session_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TaskOrigin:
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------
# FocusedTask
# --------------------------------------------------------------------------
@dataclass
class FocusedTask:
    """One focused user task: identity, lifecycle state, follow-ups, linkage.

    Constructed via :meth:`TaskRegistry.create_task` in normal use.  Every field
    is JSON-serialisable except the per-follow-up ``PendingTurnItem.raw``
    passthrough, which :meth:`PendingTurnItem.to_dict` drops -- so :meth:`to_dict`
    is always safe to log, persist, or move across a process boundary.
    """

    task_id: str
    user_goal: str
    status: str = STATUS_PROPOSED
    session_key: str | None = None
    origin: TaskOrigin = field(default_factory=TaskOrigin)
    active_worker_id: str | None = None
    worker_kind: str | None = None
    pending_followups: list[PendingTurnItem] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _validate_status(self.status)

    # -- introspection ----------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self.status not in TERMINAL_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def touch(self, *, now: float | None = None) -> None:
        """Bump ``updated_at`` -- to *now* if given, else the current time."""
        self.updated_at = time.time() if now is None else float(now)

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict.

        ``pending_followups`` are serialised via :meth:`PendingTurnItem.to_dict`
        (which drops ``raw`` without touching it). ``artifacts`` are returned as
        detached JSON-round-trip copies so nested metadata cannot alias live task
        state; ``notes`` are copied as a fresh list.
        """
        return {
            "task_id": self.task_id,
            "user_goal": self.user_goal,
            "status": self.status,
            "session_key": self.session_key,
            "origin": self.origin.to_dict(),
            "active_worker_id": self.active_worker_id,
            "worker_kind": self.worker_kind,
            "pending_followups": [it.to_dict() for it in self.pending_followups],
            "artifacts": [_json_safe_copy(a, label="artifact") for a in self.artifacts],
            "notes": list(self.notes),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FocusedTask:
        """Rebuild from :meth:`to_dict` output (unknown keys ignored).

        An invalid ``status`` is rejected (via ``__post_init__``).  Pending
        follow-ups are rebuilt with :meth:`PendingTurnItem.from_dict`, so they
        come back with ``raw is None`` -- the local passthrough does not survive a
        serialisation round-trip, by design.
        """
        origin = TaskOrigin.from_dict(data.get("origin"))
        session_key = data.get("session_key")
        if session_key is None:
            session_key = origin.session_key
        followups = [
            PendingTurnItem.from_dict(d)
            for d in (data.get("pending_followups") or [])
            if isinstance(d, dict)
        ]
        artifacts = [
            _json_safe_copy(a, label="artifact")
            for a in (data.get("artifacts") or [])
            if isinstance(a, dict)
        ]
        notes = [str(n) for n in (data.get("notes") or [])]
        created_at = _as_float(data.get("created_at"), time.time())
        updated_at = _as_float(data.get("updated_at"), created_at)
        return cls(
            task_id=str(data.get("task_id") or _new_task_id()),
            user_goal=str(data.get("user_goal") or ""),
            status=data.get("status", STATUS_PROPOSED),
            session_key=session_key,
            origin=origin,
            active_worker_id=data.get("active_worker_id"),
            worker_kind=data.get("worker_kind"),
            pending_followups=followups,
            artifacts=artifacts,
            notes=notes,
            created_at=created_at,
            updated_at=updated_at,
        )


# --------------------------------------------------------------------------
# TaskRegistry
# --------------------------------------------------------------------------
class TaskRegistry:
    """An ordered, in-memory collection of :class:`FocusedTask` records.

    Not thread-safe: callers that touch it from multiple threads / event loops
    should hold their own lock (mirroring
    :class:`~agent.pending_turn_queue.PendingTurnQueue`).  An optional bound
    *path* enables :meth:`save` / :meth:`load` JSON persistence; a registry with
    ``path=None`` is in-memory only, and :meth:`save` then requires an explicit
    path argument.
    """

    SCHEMA_VERSION = 1

    def __init__(self, *, path: str | os.PathLike[str] | None = None) -> None:
        self._tasks: dict[str, FocusedTask] = {}
        self._path: str | None = os.fspath(path) if path is not None else None

    # -- container-ish ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._tasks

    @property
    def path(self) -> str | None:
        """The bound persistence path, or ``None`` for an in-memory registry."""
        return self._path

    # -- internals --------------------------------------------------------
    def _require(self, task_id: str) -> FocusedTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task id: {task_id!r}")
        return task

    # -- creation ---------------------------------------------------------
    def create_task(
        self,
        user_goal: str,
        *,
        session_key: str | None = None,
        origin: TaskOrigin | dict[str, Any] | None = None,
        status: str = STATUS_PROPOSED,
        active_worker_id: str | None = None,
        worker_kind: str | None = None,
        task_id: str | None = None,
    ) -> FocusedTask:
        """Create and register a new focused task.

        *origin* may be a :class:`TaskOrigin`, a plain dict (lifted via
        :meth:`TaskOrigin.from_dict`), or ``None``.  A passed-in :class:`TaskOrigin`
        is shallow-copied so the registry does not alias the caller's object.  If
        *session_key* is given it becomes ``task.session_key`` (and fills
        ``origin.session_key`` when that is empty); otherwise ``task.session_key``
        falls back to ``origin.session_key``.  An invalid *status* raises
        ``ValueError``; a duplicate *task_id* raises ``ValueError``.
        """
        _validate_status(status)
        if isinstance(origin, dict):
            resolved_origin = TaskOrigin.from_dict(origin)
        elif isinstance(origin, TaskOrigin):
            resolved_origin = replace(origin)  # private copy; do not alias caller's
        elif origin is None:
            resolved_origin = TaskOrigin()
        else:
            raise TypeError("origin must be a TaskOrigin, a dict, or None")

        if session_key is not None:
            if resolved_origin.session_key is None:
                resolved_origin.session_key = session_key
            effective_session_key: str | None = session_key
        else:
            effective_session_key = resolved_origin.session_key

        tid = task_id or _new_task_id()
        if tid in self._tasks:
            raise ValueError(f"task id already exists: {tid!r}")

        task = FocusedTask(
            task_id=tid,
            user_goal=user_goal,
            status=status,
            session_key=effective_session_key,
            origin=resolved_origin,
            active_worker_id=active_worker_id,
            worker_kind=worker_kind,
        )
        self._tasks[tid] = task
        return task

    # -- lookup -----------------------------------------------------------
    def get_task(self, task_id: str) -> FocusedTask | None:
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        *,
        session_key: str | None = None,
        active_only: bool = False,
    ) -> list[FocusedTask]:
        """Return tasks in creation order, optionally filtered.

        *session_key* (when given) keeps only tasks whose ``session_key`` matches
        exactly.  *active_only* drops terminal tasks (``done`` / ``error`` /
        ``cancelled``).
        """
        out = list(self._tasks.values())
        if session_key is not None:
            out = [t for t in out if t.session_key == session_key]
        if active_only:
            out = [t for t in out if t.is_active]
        return out

    # -- mutation ---------------------------------------------------------
    def update_status(
        self,
        task_id: str,
        status: str,
        *,
        note: str | None = None,
    ) -> FocusedTask:
        """Set *task_id*'s status (rejecting unknown statuses) and bump ``updated_at``.

        An optional *note* is appended (string-coerced) to the task's notes.
        """
        _validate_status(status)
        task = self._require(task_id)
        task.status = status
        if note is not None:
            task.notes.append(str(note))
        task.touch()
        return task

    def assign_worker(
        self,
        task_id: str,
        worker_id: str,
        *,
        worker_kind: str | None = None,
    ) -> FocusedTask:
        """Record a worker linkage on *task_id*.  Does **not** start anything."""
        task = self._require(task_id)
        task.active_worker_id = worker_id
        task.worker_kind = worker_kind
        task.touch()
        return task

    def clear_worker(self, task_id: str) -> FocusedTask:
        """Drop *task_id*'s worker linkage (``active_worker_id`` / ``worker_kind`` -> None)."""
        task = self._require(task_id)
        task.active_worker_id = None
        task.worker_kind = None
        task.touch()
        return task

    def attach_followup(self, task_id: str, item: Any) -> PendingTurnItem:
        """Append a pending follow-up to *task_id*, preserving arrival order.

        *item* may be:

        * a :class:`PendingTurnItem` -- stored as-is, so an in-memory ``raw``
          passthrough is kept locally (it is still dropped on serialisation);
        * a ``dict`` -- rebuilt via :meth:`PendingTurnItem.from_dict`;
        * a legacy CLI payload -- a ``str``, a ``(caption, [paths])`` tuple, or an
          ``(INTEGRATED_BUSY_PAYLOAD, text)`` tag -- which is lifted with
          :func:`from_legacy_cli_payload` (threading the task's ``session_key``).

        Returns the stored :class:`PendingTurnItem`.
        """
        task = self._require(task_id)
        if isinstance(item, PendingTurnItem):
            pending = item
        elif isinstance(item, dict):
            pending = PendingTurnItem.from_dict(item)
        else:
            pending = from_legacy_cli_payload(item, session_key=task.session_key)
        task.pending_followups.append(pending)
        task.touch()
        return pending

    def attach_artifact(self, task_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
        """Append an artifact record (a dict) to *task_id*; return the stored copy.

        The artifact is copied into a detached JSON-safe representation on the
        way in so later mutation of the caller's dict (including nested values)
        does not reach into the task.  Non-JSON-serializable keys/values raise
        ``TypeError`` immediately rather than failing later during persistence.
        """
        if not isinstance(artifact, dict):
            raise TypeError("artifact must be a dict")
        task = self._require(task_id)
        stored = _json_safe_copy(artifact, label="artifact")
        task.artifacts.append(stored)
        task.touch()
        return stored

    def add_note(self, task_id: str, note: str) -> FocusedTask:
        """Append a (string-coerced) note to *task_id*."""
        task = self._require(task_id)
        task.notes.append(str(note))
        task.touch()
        return task

    def cancel_task(self, task_id: str, *, reason: str | None = None) -> FocusedTask:
        """Mark *task_id* ``cancelled``; record *reason* as a note when given."""
        task = self._require(task_id)
        task.status = STATUS_CANCELLED
        if reason is not None:
            task.notes.append(f"cancelled: {reason}")
        task.touch()
        return task

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe document: ``{"version": N, "tasks": [...]}``.

        ``tasks`` preserves creation order; each task is serialised via
        :meth:`FocusedTask.to_dict` (so no ``raw`` payload is included).
        """
        return {
            "version": self.SCHEMA_VERSION,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        path: str | os.PathLike[str] | None = None,
    ) -> TaskRegistry:
        """Rebuild a registry from :meth:`to_dict` output (unknown keys ignored).

        An invalid ``status`` in any task raises ``ValueError``.  If two tasks
        share a ``task_id`` the later one wins.  The returned registry is bound to
        *path* (if given) so a later :meth:`save` round-trips.
        """
        reg = cls(path=path)
        if data:
            for raw_task in (data.get("tasks") or []):
                if not isinstance(raw_task, dict):
                    continue
                task = FocusedTask.from_dict(raw_task)
                reg._tasks[task.task_id] = task
        return reg

    # -- persistence (optional) -------------------------------------------
    def save(self, path: str | os.PathLike[str] | None = None) -> str:
        """Atomically write the registry to *path* (or the bound path) as JSON.

        Writes a temp file in the destination directory, ``fsync``s it, then
        ``os.replace``s it over the target, so a concurrent reader never sees a
        half-written file.  Passing an explicit *path* also binds it for later
        ``save()`` calls.  Returns the path written.  Raises ``ValueError`` when
        no path is available.
        """
        target = os.fspath(path) if path is not None else self._path
        if not target:
            raise ValueError("no path given and this registry has no bound path")
        directory = os.path.dirname(target) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".task_registry_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        if path is not None:
            self._path = target
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> TaskRegistry:
        """Load a registry from *path*; an absent file yields an empty registry.

        A present-but-corrupt file raises (``json``/``ValueError`` propagate) -- a
        truncated or malformed registry should be visible, not silently dropped.
        The returned registry is bound to *path* so a later :meth:`save`
        round-trips.
        """
        target = os.fspath(path)
        if not os.path.exists(target):
            return cls(path=target)
        with open(target, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"task registry file is not a JSON object: {target!r}")
        return cls.from_dict(data, path=target)
