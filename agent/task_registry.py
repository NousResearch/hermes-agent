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
  append, a correction, a steer, explicit status, or a new task -- it only gives
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
    "RESULT_SUCCEEDED",
    "RESULT_FAILED",
    "RESULT_CANCELLED",
    "REVIEW_PENDING",
    "REVIEW_PASSED",
    "REVIEW_FAILED",
    "REVIEW_NEEDS_ITERATION",
    "REVIEW_BLOCKED",
    "FRONTDESK_QUEUED",
    "FRONTDESK_RUNNING_WORKER",
    "FRONTDESK_WORKER_DONE_PENDING_REVIEW",
    "FRONTDESK_RUNNING_REVIEW",
    "FRONTDESK_REVIEW_PASSED",
    "FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION",
    "FRONTDESK_BLOCKED_USER_INPUT",
    "FRONTDESK_CANCEL_REQUESTED",
    "FRONTDESK_CANCELLED",
    "FRONTDESK_DONE_PRESENTED",
    "FRONTDESK_ERROR",
    "FRONTDESK_STATES",
    "FRONTDESK_TERMINAL_STATES",
    "STAGE_NOT_STARTED",
    "STAGE_QUEUED",
    "STAGE_RUNNING",
    "STAGE_DONE",
    "STAGE_FAILED",
    "STAGE_BLOCKED",
    "STAGE_CANCELLED",
    "STAGE_STATUSES",
    "REVIEW_ACTION_PRESENT",
    "REVIEW_ACTION_ITERATE",
    "REVIEW_ACTION_ASK_USER",
    "REVIEW_ACTION_CANCEL",
    "ReviewResultArtifact",
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

RESULT_SUCCEEDED = "succeeded"
RESULT_FAILED = "failed"
RESULT_CANCELLED = "cancelled"
RESULT_STATUSES = frozenset({RESULT_SUCCEEDED, RESULT_FAILED, RESULT_CANCELLED})
REVIEW_PENDING = "pending_review"

REVIEW_PASSED = "passed"
REVIEW_FAILED = "failed"
REVIEW_NEEDS_ITERATION = "needs_iteration"
REVIEW_BLOCKED = "blocked"
REVIEW_STATUSES = frozenset(
    {REVIEW_PASSED, REVIEW_FAILED, REVIEW_NEEDS_ITERATION, REVIEW_BLOCKED}
)

REVIEW_ACTION_PRESENT = "present"
REVIEW_ACTION_ITERATE = "iterate"
REVIEW_ACTION_ASK_USER = "ask_user"
REVIEW_ACTION_CANCEL = "cancel"
REVIEW_ACTIONS = frozenset(
    {
        REVIEW_ACTION_PRESENT,
        REVIEW_ACTION_ITERATE,
        REVIEW_ACTION_ASK_USER,
        REVIEW_ACTION_CANCEL,
    }
)

FRONTDESK_QUEUED = "queued"
FRONTDESK_RUNNING_WORKER = "running_worker"
FRONTDESK_WORKER_DONE_PENDING_REVIEW = "worker_done_pending_review"
FRONTDESK_RUNNING_REVIEW = "running_review"
FRONTDESK_REVIEW_PASSED = "review_passed"
FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION = "review_failed_needs_iteration"
FRONTDESK_BLOCKED_USER_INPUT = "blocked_user_input"
FRONTDESK_CANCEL_REQUESTED = "cancel_requested"
FRONTDESK_CANCELLED = "cancelled"
FRONTDESK_DONE_PRESENTED = "done_presented"
FRONTDESK_ERROR = "error"
FRONTDESK_STATES = frozenset(
    {
        FRONTDESK_QUEUED,
        FRONTDESK_RUNNING_WORKER,
        FRONTDESK_WORKER_DONE_PENDING_REVIEW,
        FRONTDESK_RUNNING_REVIEW,
        FRONTDESK_REVIEW_PASSED,
        FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION,
        FRONTDESK_BLOCKED_USER_INPUT,
        FRONTDESK_CANCEL_REQUESTED,
        FRONTDESK_CANCELLED,
        FRONTDESK_DONE_PRESENTED,
        FRONTDESK_ERROR,
    }
)
FRONTDESK_TERMINAL_STATES = frozenset(
    {FRONTDESK_CANCELLED, FRONTDESK_DONE_PRESENTED, FRONTDESK_ERROR}
)

STAGE_NOT_STARTED = "not_started"
STAGE_QUEUED = "queued"
STAGE_RUNNING = "running"
STAGE_DONE = "done"
STAGE_FAILED = "failed"
STAGE_BLOCKED = "blocked"
STAGE_CANCELLED = "cancelled"
STAGE_STATUSES = frozenset(
    {
        STAGE_NOT_STARTED,
        STAGE_QUEUED,
        STAGE_RUNNING,
        STAGE_DONE,
        STAGE_FAILED,
        STAGE_BLOCKED,
        STAGE_CANCELLED,
    }
)

_MISSING = object()


def _new_task_id() -> str:
    return f"task-{uuid.uuid4().hex}"


def _validate_status(status: Any) -> str:
    if status not in TASK_STATUSES:
        raise ValueError(
            f"unknown task status {status!r}; expected one of {sorted(TASK_STATUSES)}"
        )
    return status


def _validate_frontdesk_state(state: Any) -> str | None:
    if state is None:
        return None
    if state not in FRONTDESK_STATES:
        raise ValueError(
            f"unknown frontdesk state {state!r}; expected one of {sorted(FRONTDESK_STATES)}"
        )
    return str(state)


def _validate_stage(stage: Any, *, label: str) -> str | None:
    if stage is None:
        return None
    if stage not in STAGE_STATUSES:
        raise ValueError(
            f"unknown {label} {stage!r}; expected one of {sorted(STAGE_STATUSES)}"
        )
    return str(stage)


def _validate_review_status(status: Any) -> str:
    if status not in REVIEW_STATUSES:
        raise ValueError(
            f"unknown review status {status!r}; expected one of {sorted(REVIEW_STATUSES)}"
        )
    return str(status)


def _validate_review_action(action: Any) -> str:
    if action not in REVIEW_ACTIONS:
        raise ValueError(
            f"unknown recommended next action {action!r}; expected one of {sorted(REVIEW_ACTIONS)}"
        )
    return str(action)


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


def _json_safe_clean(value: Any) -> Any:
    """Return a JSON-safe representation of *value*, dropping unsafe metadata.

    Worker results may be assembled near subprocesses, callbacks, exception
    instances, and other process-local objects.  Result snapshots intentionally
    keep only JSON data; unsupported nested values are omitted instead of being
    stringified wholesale into misleading metadata.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise TypeError("value is not finite")
        return value
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            try:
                cleaned[key] = _json_safe_clean(item)
            except TypeError:
                continue
        return cleaned
    if isinstance(value, (list, tuple)):
        cleaned_list = []
        for item in value:
            try:
                cleaned_list.append(_json_safe_clean(item))
            except TypeError:
                continue
        return cleaned_list
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def _coerce_result_status(status: Any) -> str:
    if status in RESULT_STATUSES:
        return str(status)
    if status == STATUS_DONE or status == "done":
        return RESULT_SUCCEEDED
    if status == STATUS_ERROR or status == "error":
        return RESULT_FAILED
    if status == STATUS_CANCELLED or status == "cancelled":
        return RESULT_CANCELLED
    raise ValueError(
        f"unknown worker result status {status!r}; expected one of {sorted(RESULT_STATUSES)}"
    )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, BaseException):
        text = str(value)
    else:
        text = value if isinstance(value, str) else str(value)
    return text if text else None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = _string_or_none(item)
        if text:
            out.append(text)
    return out


@dataclass(frozen=True)
class ReviewResultArtifact:
    """Structured result from a background reviewer.

    This is intentionally small and JSON-only so main/frontdesk can decide
    whether a worker result may be presented without parsing prose.
    """

    review_status: str
    reviewer_model: str = "gpt-5.5"
    summary: str = ""
    tests_run: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    recommended_next_action: str = REVIEW_ACTION_PRESENT
    artifact_paths: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "review_status", _validate_review_status(self.review_status))
        object.__setattr__(
            self,
            "recommended_next_action",
            _validate_review_action(self.recommended_next_action),
        )
        object.__setattr__(self, "reviewer_model", _string_or_none(self.reviewer_model) or "gpt-5.5")
        object.__setattr__(self, "summary", _string_or_none(self.summary) or "")
        object.__setattr__(self, "tests_run", _string_list(self.tests_run))
        object.__setattr__(self, "changed_files", _string_list(self.changed_files))
        object.__setattr__(self, "risks", _string_list(self.risks))
        object.__setattr__(self, "artifact_paths", _string_list(self.artifact_paths))

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_status": self.review_status,
            "reviewer_model": self.reviewer_model,
            "summary": self.summary,
            "tests_run": list(self.tests_run),
            "changed_files": list(self.changed_files),
            "risks": list(self.risks),
            "recommended_next_action": self.recommended_next_action,
            "artifact_paths": list(self.artifact_paths),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewResultArtifact:
        if not isinstance(data, dict):
            raise TypeError("review artifact must be a dict")
        return cls(
            review_status=data.get("review_status"),
            reviewer_model=_string_or_none(data.get("reviewer_model")) or "gpt-5.5",
            summary=_string_or_none(data.get("summary")) or "",
            tests_run=_string_list(data.get("tests_run")),
            changed_files=_string_list(data.get("changed_files")),
            risks=_string_list(data.get("risks")),
            recommended_next_action=data.get("recommended_next_action"),
            artifact_paths=_string_list(data.get("artifact_paths")),
        )


def _review_result_artifact(value: Any) -> ReviewResultArtifact:
    if isinstance(value, ReviewResultArtifact):
        return value
    if isinstance(value, dict):
        return ReviewResultArtifact.from_dict(value)
    raise TypeError("review artifact must be a ReviewResultArtifact or dict")


def _worker_result_snapshot(
    *,
    task_id: str,
    result: Any = None,
    worker_id: str | None = None,
    status: str | None = None,
    summary: Any = None,
    artifacts: Any = None,
    tests: Any = None,
    error: Any = None,
    review_status: str = REVIEW_PENDING,
) -> dict[str, Any]:
    if hasattr(result, "to_dict") and not isinstance(result, dict):
        result = result.to_dict()

    source = result if isinstance(result, dict) else {}
    resolved_worker_id = worker_id or _string_or_none(source.get("worker_id"))
    resolved_status = _coerce_result_status(status or source.get("status"))
    resolved_task_id = _string_or_none(source.get("task_id")) or task_id
    resolved_summary = summary
    if resolved_summary is None:
        resolved_summary = source.get("summary")
    if resolved_summary is None and not isinstance(result, dict):
        resolved_summary = result
    if resolved_summary is None:
        resolved_summary = source.get("result")
    resolved_error = error if error is not None else source.get("error")

    snapshot: dict[str, Any] = {
        "worker_id": resolved_worker_id or "",
        "task_id": resolved_task_id,
        "status": resolved_status,
        "summary": _string_or_none(resolved_summary) or "",
        "review_status": _string_or_none(source.get("review_status")) or review_status,
    }

    resolved_artifacts = artifacts if artifacts is not None else source.get("artifacts")
    if resolved_artifacts is not None:
        if isinstance(resolved_artifacts, dict):
            resolved_artifacts = [resolved_artifacts]
        if isinstance(resolved_artifacts, list):
            cleaned_artifacts = []
            for item in resolved_artifacts:
                if not isinstance(item, dict):
                    continue
                cleaned = _json_safe_clean(item)
                if cleaned:
                    cleaned_artifacts.append(cleaned)
            if cleaned_artifacts:
                snapshot["artifacts"] = cleaned_artifacts

    resolved_tests = tests if tests is not None else source.get("tests")
    if resolved_tests is not None:
        try:
            cleaned_tests = _json_safe_clean(resolved_tests)
        except TypeError:
            cleaned_tests = None
        if cleaned_tests not in (None, {}, []):
            snapshot["tests"] = cleaned_tests

    error_text = _string_or_none(resolved_error)
    if error_text:
        snapshot["error"] = error_text

    return _json_safe_copy(snapshot, label="worker result")


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
    frontdesk_state: str | None = None
    worker_stage: str | None = None
    reviewer_stage: str | None = None
    worker_process_id: str | None = None
    worker_session_id: str | None = None
    last_message_path: str | None = None
    summary_artifact_path: str | None = None
    review_artifact_path: str | None = None
    review_verdict: str | None = None
    blocked_reason: str | None = None
    awaiting_user_input: bool = False
    pending_followups: list[PendingTurnItem] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    review_result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        _validate_status(self.status)
        self.frontdesk_state = _validate_frontdesk_state(self.frontdesk_state)
        self.worker_stage = _validate_stage(self.worker_stage, label="worker stage")
        self.reviewer_stage = _validate_stage(self.reviewer_stage, label="reviewer stage")
        if self.review_verdict is not None:
            self.review_verdict = _validate_review_status(self.review_verdict)

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
            "frontdesk_state": self.frontdesk_state,
            "worker_stage": self.worker_stage,
            "reviewer_stage": self.reviewer_stage,
            "worker_process_id": self.worker_process_id,
            "worker_session_id": self.worker_session_id,
            "last_message_path": self.last_message_path,
            "summary_artifact_path": self.summary_artifact_path,
            "review_artifact_path": self.review_artifact_path,
            "review_verdict": self.review_verdict,
            "blocked_reason": self.blocked_reason,
            "awaiting_user_input": bool(self.awaiting_user_input),
            "pending_followups": [it.to_dict() for it in self.pending_followups],
            "artifacts": [_json_safe_copy(a, label="artifact") for a in self.artifacts],
            "notes": list(self.notes),
            "result": _json_safe_copy(self.result, label="worker result") if self.result else None,
            "review_result": (
                _json_safe_copy(self.review_result, label="review result")
                if self.review_result
                else None
            ),
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
        result = data.get("result")
        if result is not None and not isinstance(result, dict):
            result = None
        if result is not None:
            result = _worker_result_snapshot(
                task_id=str(data.get("task_id") or ""),
                result=result,
            )
        review_result = data.get("review_result")
        if review_result is not None:
            if isinstance(review_result, dict):
                review_result = _review_result_artifact(review_result).to_dict()
            else:
                review_result = None
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
            frontdesk_state=data.get("frontdesk_state"),
            worker_stage=data.get("worker_stage"),
            reviewer_stage=data.get("reviewer_stage"),
            worker_process_id=_string_or_none(data.get("worker_process_id")),
            worker_session_id=_string_or_none(data.get("worker_session_id")),
            last_message_path=_string_or_none(data.get("last_message_path")),
            summary_artifact_path=_string_or_none(data.get("summary_artifact_path")),
            review_artifact_path=_string_or_none(data.get("review_artifact_path")),
            review_verdict=data.get("review_verdict"),
            blocked_reason=_string_or_none(data.get("blocked_reason")),
            awaiting_user_input=bool(data.get("awaiting_user_input")),
            pending_followups=followups,
            artifacts=artifacts,
            notes=notes,
            result=result,
            review_result=review_result,
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
        frontdesk_state: str | None = None,
        worker_stage: str | None = None,
        reviewer_stage: str | None = None,
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
            frontdesk_state=frontdesk_state,
            worker_stage=worker_stage,
            reviewer_stage=reviewer_stage,
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

    def update_frontdesk_metadata(
        self,
        task_id: str,
        *,
        frontdesk_state: Any = _MISSING,
        worker_stage: Any = _MISSING,
        reviewer_stage: Any = _MISSING,
        worker_process_id: Any = _MISSING,
        worker_session_id: Any = _MISSING,
        last_message_path: Any = _MISSING,
        summary_artifact_path: Any = _MISSING,
        review_artifact_path: Any = _MISSING,
        review_verdict: Any = _MISSING,
        blocked_reason: Any = _MISSING,
        awaiting_user_input: Any = _MISSING,
        note: str | None = None,
    ) -> FocusedTask:
        """Update explicit frontdesk worker/reviewer metadata on a task.

        The coarse ``status`` field remains the existing task primitive; these
        fields preserve the production-shaped background lifecycle so status
        views do not have to guess whether a ``running`` task is in worker,
        review, blocked-input, or presentable-review state.
        """
        task = self._require(task_id)
        if frontdesk_state is not _MISSING:
            task.frontdesk_state = _validate_frontdesk_state(frontdesk_state)
        if worker_stage is not _MISSING:
            task.worker_stage = _validate_stage(worker_stage, label="worker stage")
        if reviewer_stage is not _MISSING:
            task.reviewer_stage = _validate_stage(reviewer_stage, label="reviewer stage")
        if worker_process_id is not _MISSING:
            task.worker_process_id = _string_or_none(worker_process_id)
        if worker_session_id is not _MISSING:
            task.worker_session_id = _string_or_none(worker_session_id)
        if last_message_path is not _MISSING:
            task.last_message_path = _string_or_none(last_message_path)
        if summary_artifact_path is not _MISSING:
            task.summary_artifact_path = _string_or_none(summary_artifact_path)
        if review_artifact_path is not _MISSING:
            task.review_artifact_path = _string_or_none(review_artifact_path)
        if review_verdict is not _MISSING:
            task.review_verdict = (
                _validate_review_status(review_verdict) if review_verdict is not None else None
            )
        if blocked_reason is not _MISSING:
            task.blocked_reason = _string_or_none(blocked_reason)
        if awaiting_user_input is not _MISSING:
            task.awaiting_user_input = bool(awaiting_user_input)
        if note is not None:
            task.notes.append(str(note))
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

    def append_followup(
        self,
        task_id: str,
        text: str,
        *,
        source: str = "user",
        session_key: str | None = None,
    ) -> PendingTurnItem:
        """Append a plain-text follow-up to *task_id*.

        This is the explicit frontdesk-facing helper for late user guidance.  It
        keeps the lower-level :meth:`attach_followup` flexibility while making
        the common text path spell out provenance and task linkage.
        """
        task = self._require(task_id)
        effective_session_key = session_key if session_key is not None else task.session_key
        return self.attach_followup(
            task_id,
            PendingTurnItem(
                source=source,
                text=str(text),
                session_key=effective_session_key,
                task_hint=task_id,
            ),
        )

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

    def attach_worker_result(
        self,
        task_id: str,
        result: Any = None,
        *,
        worker_id: str | None = None,
        status: str | None = None,
        summary: Any = None,
        artifacts: Any = None,
        tests: Any = None,
        error: Any = None,
        review_status: str = REVIEW_PENDING,
    ) -> dict[str, Any]:
        """Attach a JSON-safe worker result snapshot to *task_id*.

        Only the review/import gate fields are retained: worker/task identity,
        normalized result status, summary, optional artifacts/tests/error, and
        ``review_status``.  Unknown keys and non-JSON local-process metadata are
        dropped, so task serialization never captures subprocess objects,
        callbacks, exception instances, file handles, or raw payloads.
        """
        task = self._require(task_id)
        snapshot = _worker_result_snapshot(
            task_id=task_id,
            result=result,
            worker_id=worker_id,
            status=status,
            summary=summary,
            artifacts=artifacts,
            tests=tests,
            error=error,
            review_status=review_status,
        )
        task.result = snapshot
        task.touch()
        return snapshot

    def attach_review_result(
        self,
        task_id: str,
        review: ReviewResultArtifact | dict[str, Any],
        *,
        artifact_path: str | None = None,
    ) -> dict[str, Any]:
        """Attach a normalized reviewer artifact and update review lifecycle fields."""
        task = self._require(task_id)
        artifact = _review_result_artifact(review).to_dict()
        task.review_result = artifact
        if artifact_path is not None:
            task.review_artifact_path = str(artifact_path)
        elif artifact["artifact_paths"]:
            task.review_artifact_path = artifact["artifact_paths"][0]
        task.review_verdict = artifact["review_status"]
        task.reviewer_stage = STAGE_DONE
        verdict = artifact["review_status"]
        action = artifact["recommended_next_action"]
        if verdict == REVIEW_PASSED and action == REVIEW_ACTION_PRESENT:
            task.frontdesk_state = FRONTDESK_REVIEW_PASSED
            task.awaiting_user_input = False
            task.blocked_reason = None
        elif verdict == REVIEW_BLOCKED or action == REVIEW_ACTION_ASK_USER:
            task.frontdesk_state = FRONTDESK_BLOCKED_USER_INPUT
            task.awaiting_user_input = True
            task.blocked_reason = artifact["summary"] or "reviewer requested user input"
            task.status = STATUS_BLOCKED
        elif verdict in {REVIEW_FAILED, REVIEW_NEEDS_ITERATION} or action == REVIEW_ACTION_ITERATE:
            task.frontdesk_state = FRONTDESK_REVIEW_FAILED_NEEDS_ITERATION
            task.awaiting_user_input = False
        elif action == REVIEW_ACTION_CANCEL:
            task.frontdesk_state = FRONTDESK_CANCELLED
            task.status = STATUS_CANCELLED
        task.touch()
        return artifact

    def cancel_task(self, task_id: str, *, reason: str | None = None) -> FocusedTask:
        """Mark *task_id* ``cancelled``; record *reason* as a note when given."""
        task = self._require(task_id)
        task.status = STATUS_CANCELLED
        task.frontdesk_state = FRONTDESK_CANCELLED
        task.worker_stage = (
            STAGE_CANCELLED if task.worker_stage in {STAGE_QUEUED, STAGE_RUNNING} else task.worker_stage
        )
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
