"""Phase 5 Worker Mapping — pure kanban_task -> TaskState + WorkerRegistryEntry.

This module is the **only** place that translates a Phase 4B
Kanban Task into the (TaskState, WorkerRegistryEntry) inputs of
``agent.orchestrator.dispatcher.Dispatcher.dispatch`` and
``agent.orchestrator.batch_runner.BatchRunner.run_batch``.

It does NOT spawn workers, dispatch, or modify any kanban DB.
It only reads task metadata and produces pure-data structures.

The mapping is **idempotent**: given the same kanban Task, it
produces the same (TaskState, WorkerRegistryEntry) tuple.

Forbidden APIs (PROHIBITED):
* ``hermes_cli.kanban.kanban_command``
* ``hermes_cli.kanban._cmd_create`` / ``_cmd_swarm``
* ``hermes_cli.kanban_db.create_task`` / ``delete_task``
* ``hermes_cli.kanban_decompose.*`` / ``kanban_specify.*`` / ``kanban_swarm.*``
* ``hermes_cli.write_approval_commands``
* ``agent.execution_router.ExecutionRouter``
* ``agent.execution_dispatcher.ExecutionDispatcher``
* ``agent.orchestrator_interface.OrchestratorInterface.execute``
* ``delegate_task`` / ``execute()`` / ``worker_runner.real`` /
  ``pilot_bridge.real`` / ``batch_runner.real``
* Any LLM call (anthropic, openai, auxiliary_client, urllib, requests, httpx)
* Any subprocess / os.system / os.popen
* Any DB DDL (CREATE TABLE / ALTER TABLE / CREATE INDEX)
* gbrain / obsidian / notebooklm
"""

from __future__ import annotations

import json
import hashlib
from typing import Any, FrozenSet, Optional, Tuple

# Phase 4B types (reused; never duplicated).
from .types import (
    KanbanApplyResult,
    KanbanTaskLink,
)


# ── Pure mapping (no I/O) ────────────────────────────────────────────────

# Map Kanban task.status -> TaskState.state (orchestrator's vocabulary).
_STATUS_TO_STATE: dict = {
    "ready": "ready",
    "running": "running",
    "todo": "ready",
    "triage": "ready",
    "done": "done",
    "archived": "done",
    "blocked": "blocked",
    "failed": "failed",
}


def _field(t, name, default=None):
    """Return a field from either a dict or an object."""
    if isinstance(t, dict):
        return t.get(name, default)
    return getattr(t, name, default)


def kanban_task_to_task_state(task: Any) -> dict:
    """Pure: kanban Task (dict-like or object) -> TaskState-shaped dict.

    The output dict matches the field names of
    ``agent.orchestrator.dispatcher.TaskState``. The Dispatcher's
    ``to_orchestrator_state(task)`` is the canonical conversion
    that accepts this dict.

    This function does NOT import ``agent.orchestrator.dispatcher``
    (that would create a circular dependency). The mapping is
    documented and the orchestrator's conversion is the
    contract; if the orchestrator's TaskState gains a new field,
    this function must be updated to set it.
    """
    status = _field(task, "status", None) or "ready"
    state = _STATUS_TO_STATE.get(status, "ready")
    return {
        "task_id": _field(task, "id", None) or "",
        "state": state,
        "last_worker_id": _field(task, "assignee", None),
        "failure_count": int(_field(task, "consecutive_failures", 0) or 0),
        "started_at": _field(task, "started_at", None),
    }


def worker_registry_from_kanban_tasks(
    tasks: list,
    *,
    base_command: Optional[list] = None,
) -> list:
    """Pure: list of kanban Task -> list of WorkerRegistryEntry-shaped dicts.

    One worker entry per unique ``task.assignee``. Tasks with
    ``assignee is None`` are skipped (the dispatcher treats
    them as ineligible).

    The output dicts match the field names of
    ``agent.orchestrator.dispatcher.WorkerRegistryEntry``.
    """
    base = list(base_command) if base_command else ["hermes", "--dispatch"]
    seen: set = set()
    workers: list = []
    for task in tasks:
        worker_id = _field(task, "assignee", None)
        if not worker_id or worker_id in seen:
            continue
        seen.add(worker_id)
        capabilities: list = [str(worker_id).lower()]
        skills = _field(task, "skills", None)
        if isinstance(skills, list):
            for s in skills:
                if s:
                    capabilities.append(str(s).lower())
        # Per-worker command (orchestrator appends task_id at dispatch time).
        cmd = list(base) + ["--worker-id", str(worker_id)]
        workers.append({
            "worker_id": str(worker_id),
            "capabilities": capabilities,
            "command": cmd,
        })
    return workers


def _task_id(t):
    """Return the task id from either a dict or an object."""
    if isinstance(t, dict):
        return t.get("id")
    return getattr(t, "id", None)


def build_batch_inputs(
    task_ids: Tuple[str, ...],
    *,
    kanban_tasks: list,
) -> Tuple[list, list]:
    """Pure-ish: read kanban_tasks (a list of pre-fetched kb.Task objects)
    and produce (task_states, workers).

    No I/O; the caller must pre-fetch ``kanban_tasks``. ``task_ids``
    is the canonical ordering (Phase 4B's apply_record.task_ids).

    Accepts both dict-like tasks (``{"id": ..., "assignee": ...}``)
    and object-like tasks (``task.id``, ``task.assignee``).
    """
    by_id = {_task_id(t): t for t in kanban_tasks if _task_id(t) is not None}
    ordered_tasks: list = []
    for tid in task_ids:
        t = by_id.get(tid)
        if t is not None:
            ordered_tasks.append(t)
    task_states = [kanban_task_to_task_state(t) for t in ordered_tasks]
    workers = worker_registry_from_kanban_tasks(ordered_tasks)
    return task_states, workers


def derive_restrictions(policy_decision: Any) -> set:
    """Pure: PolicyDecision -> set of restriction strings.

    Restrictions are advisory strings that the orchestrator's
    Engine may use to disqualify workers (e.g. "no_external").

    Phase 5 contract:
    * At R6 (External_call), no "no_external" is added because
      the approval has already authorized external calls.
    * If the policy is fully autonomous (approval_required=False),
      restrict to safe workers.
    """
    restrictions: set = set()
    risk_level = getattr(policy_decision, "risk_level", None)
    risk_int = int(risk_level) if risk_level is not None else 0
    approval_required = bool(getattr(policy_decision, "approval_required", True))
    if risk_int >= 6:
        # R6 = external. Don't add the "no_external" restriction:
        # the policy already approved external calls.
        pass
    if not approval_required:
        restrictions.add("autonomous_only")
    return restrictions


# ── Fingerprint (stable sha256 of canonical inputs) ──────────────────────

def compute_dispatch_fingerprint(
    task_ids: Tuple[str, ...],
    *,
    restrictions: FrozenSet[str],
    decision_fingerprint: str,
    request_fingerprint: str,
    kanban_apply_fingerprint: str,
) -> str:
    """Compute a stable sha256 fingerprint for the dispatch.

    Pure: deterministic for the same inputs. Excludes `created_at`.
    """
    payload = {
        "task_ids": sorted(task_ids),
        "restrictions": sorted(restrictions),
        "decision_fingerprint": decision_fingerprint,
        "request_fingerprint": request_fingerprint,
        "kanban_apply_fingerprint": kanban_apply_fingerprint,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "kanban_task_to_task_state",
    "worker_registry_from_kanban_tasks",
    "build_batch_inputs",
    "derive_restrictions",
    "compute_dispatch_fingerprint",
]
