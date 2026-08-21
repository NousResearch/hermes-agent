"""
Autonomous initiation path for AUTONOMOUS_INITIATION_PATH_V1.

Public API:

  attempt_autonomous_initiation(objective_spec, policy_context=None)
      -> InitiationResult

This is the single bridge between "an autonomous trigger arrives" and
"a Kanban task appears". It uses hermes_cli.kanban_db.create_task as the
canonical task creation API (the same one used by the CLI and the
dashboard). No raw SQL, no parallel table, no second scheduler.

The initiator consults the policy gate (agent.autonomy.policy.evaluate)
BEFORE writing anything. It also computes an idempotency key derived
from (objective_id, trigger_id) and passes it to create_task, which
guarantees exactly-one semantics on duplicate triggers.

Every successful admission records the active task id in
agent.autonomy.state so the post-run accounting (and the success
evaluator) can confirm exactly-one task was created.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from . import state
from .policy import evaluate, Verdict


@dataclass
class InitiationResult:
    """Structured result of an attempt_autonomous_initiation call.

    decision is the policy decision (verbatim from policy.evaluate).
    reason mirrors policy.evaluate(...).reason.
    task_id is the id of the Kanban task created (or None if no
    task was created).
    duplicate is True iff this call was a no-op because an earlier call
    with the same idempotency key already created the task.
    provenance is the structured provenance dict stored on the task
    body, exposed here for tests and the audit JSON.
    initiated_at is the unix timestamp of the successful admission, or
    None if no task was created.
    raw_policy is the full Verdict object (for tests and audit).
    """

    decision: str
    reason: str
    task_id: Optional[str] = None
    duplicate: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)
    initiated_at: Optional[float] = None
    raw_policy: Optional[Verdict] = None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def attempt_autonomous_initiation(
    objective_spec: Dict[str, Any],
    policy_context: Optional[Dict[str, Any]] = None,
) -> InitiationResult:
    """Attempt to admit an autonomous objective.

    Side effects: at most one Kanban task is created, and only if all
    policy gates pass. The canonical API is hermes_cli.kanban_db.create_task.
    """
    verdict = evaluate(objective_spec, policy_context)

    if verdict.decision != "admit":
        return InitiationResult(
            decision=verdict.decision,
            reason=verdict.reason,
            task_id=None,
            duplicate=False,
            raw_policy=verdict,
        )

    # Build the provenance payload
    trigger_id = objective_spec.get("trigger_id") or _generate_trigger_id()
    objective_id = objective_spec["objective_id"]
    initiated_at = time.time()
    provenance = {
        "OBJECTIVE_ID": objective_id,
        "ORIGIN": "autonomous_initiator",
        "POLICY_VERSION": objective_spec["policy_version"],
        "RISK_CLASS": objective_spec["risk_class"],
        "TRIGGER_ID": trigger_id,
        "INITIATED_AT": initiated_at,
        "RUNNING_MODE": "AUTONOMOUS_A1",
    }
    title = objective_spec.get("title") or f"[autonomous] {objective_id}"
    body_text = _render_task_body(provenance, objective_spec.get("body"))
    idempotency_key = f"autonomous-{objective_id}-{trigger_id}"

    # Use the canonical task creation API. The kanban_db is the public
    # surface used by `hermes kanban create` and the dashboard plugin; we
    # import lazily to keep this module importable in tests that mock the
    # database.
    try:
        from hermes_cli import kanban_db as kb
        with _autonomous_admission_lock(kb):
            with kb.connect_closing() as conn:
                existing_id = _find_existing(conn, idempotency_key)
                if existing_id is not None:
                    # Duplicate trigger; do NOT create another task. The active
                    # objective id, if any, is not changed.
                    return InitiationResult(
                        decision="duplicate_suppressed",
                        reason=(
                            f"idempotency_key={idempotency_key!r} already exists "
                            f"as task {existing_id!r}; duplicate trigger suppressed"
                        ),
                        task_id=existing_id,
                        duplicate=True,
                        provenance=provenance,
                        raw_policy=verdict,
                    )
                active_id = _find_active_autonomous_task(conn)
                if active_id is not None:
                    return InitiationResult(
                        decision="denied_concurrency",
                        reason=(
                            "another autonomous objective is already active "
                            f"(task={active_id!r}); max_concurrent_autonomous_objectives=1"
                        ),
                        task_id=None,
                        duplicate=False,
                        provenance=provenance,
                        raw_policy=verdict,
                    )
                task_id = kb.create_task(
                    conn,
                    title=title,
                    body=body_text,
                    assignee=objective_spec.get("assignee"),
                    created_by="autonomous_initiator",
                    idempotency_key=idempotency_key,
                    priority=objective_spec.get("priority", 0),
                    initial_status="running",
                    max_runtime_seconds=objective_spec.get("max_runtime_seconds"),
                    skills=objective_spec.get("skills"),
                    # V1.3-A: the canonical Kanban task store already supports
                    # ``model_override`` / ``provider_override`` and the dispatcher
                    # already consumes them. The autonomous bridge merely
                    # preserves the spec keys the caller already provided.
                    # Absent key -> None. Explicit None -> None. No hidden
                    # default, no MiniMax hardcoding, no config lookup.
                    model_override=objective_spec.get("model_override"),
                    provider_override=objective_spec.get("provider_override"),
                    # V2: STRICT-READONLY Kanban worker capability flag,
                    # opt-in from the objective spec only. Default is
                    # False (writable). Never inferred from
                    # ``created_by == "autonomous_initiator"`` —
                    # provenance is NOT capability.
                    strict_readonly=objective_spec.get("strict_readonly"),
                )
    except sqlite3.IntegrityError as e:
        # Race: another caller created the same idempotency_key between
        # our existence check and the create_task call. Look it up and
        # return the existing id.
        try:
            from hermes_cli import kanban_db as kb
            with kb.connect_closing() as conn:
                existing_id = _find_existing(conn, idempotency_key)
        except Exception:
            existing_id = None
        return InitiationResult(
            decision="duplicate_suppressed",
            reason=(
                f"idempotency_key={idempotency_key!r} already exists "
                f"(race detected: {e!s})"
            ),
            task_id=existing_id,
            duplicate=True,
            provenance=provenance,
            raw_policy=verdict,
        )
    except Exception as e:
        # Any other error from the DB layer is propagated as a structured
        # denial. No task was created.
        return InitiationResult(
            decision="denied_runtime_error",
            reason=f"create_task raised {type(e).__name__}: {e}",
            task_id=None,
            duplicate=False,
            provenance=provenance,
            raw_policy=verdict,
        )

    # Mark this objective as the active one. The active slot is what
    # makes the next attempt with a different objective_id fail
    # concurrency.
    state.reserve_active(objective_id, task_id)

    return InitiationResult(
        decision="admit",
        reason=verdict.reason,
        task_id=task_id,
        duplicate=False,
        provenance=provenance,
        initiated_at=initiated_at,
        raw_policy=verdict,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _generate_trigger_id() -> str:
    """Default trigger id: a uuid4 hex string, used only when the spec
    does not provide one. Tests should always pass a stable trigger_id
    so that idempotency_key derivation is deterministic."""
    return uuid.uuid4().hex


def _render_provenance_body(provenance: Dict[str, Any]) -> str:
    """Render the provenance as a deterministic body string. Each line is
    ``KEY=VALUE`` for easy grep and audit."""
    return "\n".join(f"{k}={v}" for k, v in sorted(provenance.items()))


def _render_task_body(provenance: Dict[str, Any], custom_body: Any) -> str:
    """Render durable provenance plus any caller-supplied task body.

    The provenance block is always first and delimited so it remains
    recoverable from the canonical Kanban task body even when the caller
    provides their own human-readable objective text.
    """
    parts = [
        "AUTONOMOUS_PROVENANCE_BEGIN",
        _render_provenance_body(provenance),
        "AUTONOMOUS_PROVENANCE_END",
    ]
    if custom_body is not None:
        body = str(custom_body)
        if body:
            parts.extend(["", "CUSTOM_BODY_BEGIN", body, "CUSTOM_BODY_END"])
    return "\n".join(parts)


def _find_existing(conn, idempotency_key: str) -> Optional[str]:
    """Return the id of the most recent non-archived task with the given
    idempotency_key, or None. Mirrors the behavior in kanban_db.create_task
    so that duplicate detection here is consistent with duplicate detection
    inside the create path itself.
    """
    cur = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? "
        "AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
        (idempotency_key,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row[0]


def _find_active_autonomous_task(conn) -> Optional[str]:
    """Return an already-active autonomous task id, if one exists.

    This makes the Kanban task table the durable cross-process source of
    truth for max_concurrent_autonomous_objectives=1. A completed or archived
    autonomous task no longer occupies the slot; every other lifecycle state
    does.
    """
    cur = conn.execute(
        "SELECT id FROM tasks WHERE created_by = ? "
        "AND status NOT IN ('done', 'archived') "
        "ORDER BY created_at ASC LIMIT 1",
        ("autonomous_initiator",),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return row[0]


@contextlib.contextmanager
def _autonomous_admission_lock(kb, timeout_seconds: float = 5.0) -> Iterator[None]:
    """Serialize autonomous admission across processes with a bounded lock."""
    path = _autonomous_admission_lock_path(kb)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                _try_lock_file(handle)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring autonomous admission lock at {path}"
                    )
                time.sleep(0.02)
        try:
            yield
        finally:
            _unlock_file(handle)


def _autonomous_admission_lock_path(kb) -> Path:
    override = os.environ.get("HERMES_KANBAN_DB", "").strip()
    if override:
        return Path(override).expanduser().parent / ".autonomous_admission.lock"
    return kb.kanban_home() / "kanban" / ".autonomous_admission.lock"


def _try_lock_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc


def _unlock_file(handle) -> None:
    if os.name == "nt":
        import msvcrt

        with contextlib.suppress(OSError):
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ----------------------------------------------------------------------
# Convenience for tests and dispatcher hooks
# ----------------------------------------------------------------------


def summarize(result: InitiationResult) -> Dict[str, Any]:
    """Return a JSON-safe summary of the result. Used by the audit JSON."""
    return {
        "decision": result.decision,
        "reason": result.reason,
        "task_id": result.task_id,
        "duplicate": result.duplicate,
        "provenance": result.provenance,
        "initiated_at": result.initiated_at,
    }
