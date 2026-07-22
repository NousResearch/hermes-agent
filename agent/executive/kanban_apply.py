"""Phase 4B Kanban Apply — engine facade.

Phase 4B consumes a Phase 3 ``OrchestratorPlanPreview`` + a Phase 4A
``ApprovalRequest`` and produces real Kanban tasks via the existing
``hermes_cli.kanban_db.create_task`` API. It does NOT spawn workers,
does NOT call the dispatcher, and does NOT use any of the prohibited
APIs (``kanban_command``, ``_cmd_create``, ``create_swarm``,
``kanban_decompose``, ``kanban_specify``, ``kanban_swarm``,
``agent/orchestrator/kanban_adapter``, etc.).

Public surface:

* ``KanbanApplyEngine.build_apply_preview(...)`` — pure compute.
* ``KanbanApplyEngine.apply(...)`` — re-validates 8 approval gates
  + creates tasks + persists linkage.
* ``KanbanApplyEngine.rollback(...)`` — best-effort, idempotent
  cleanup of created tasks.
* ``kanban_apply(...)`` — module-level convenience.
* ``kanban_rollback(...)`` — module-level convenience.

The ``kanban_db_create_fn`` injection point allows tests to fake
``kb.create_task`` without touching the real kanban DB.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from .approval_gates import evaluate_approval_gates
from .goalmanager_bridge import BridgeApprovalError, BridgeMappingError
from .kanban_mapping import (
    DEFAULT_CREATED_BY,
    build_kanban_apply_preview as _build_kwargs_list,
    compute_idempotency_key,
)
from .state_storage import ObjectiveStateStorage
from .types import (
    ApprovalRequest,
    KanbanApplyPreview,
    KanbanApplyResult,
    KanbanRollbackPlan,
    PolicyDecision,
    compute_kanban_apply_fingerprint,
    compute_kanban_result_fingerprint,
    now_iso8601,
    objective_kanban_apply_key,
    objective_kanban_tasks_key,
)


# ──────────────────────────────────────────────────────────────────────
# Error types
# ──────────────────────────────────────────────────────────────────────

class KanbanApplyError(RuntimeError):
    """Base error for Phase 4B kanban apply."""


class KanbanLinkageConflictError(KanbanApplyError):
    """Raised when the persisted apply record conflicts with a new apply
    (different fingerprint than the in-memory preview)."""


class KanbanStorageError(KanbanApplyError):
    """Raised when the kanban DB or state_meta write fails."""


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _ensure_required_artifacts(
    *,
    plan,
    orchestrator_preview,
    policy_decision,
    approval_request,
    objective_id: str,
) -> None:
    """Verify all upstream artifacts exist. Raises BridgeMappingError."""
    if plan is None:
        raise BridgeMappingError(
            f"kanban_apply: objective_plan missing (objective_id={objective_id})"
        )
    if orchestrator_preview is None:
        raise BridgeMappingError(
            f"kanban_apply: orchestrator_preview missing (objective_id={objective_id})"
        )
    if policy_decision is None:
        raise BridgeMappingError(
            f"kanban_apply: policy_decision missing (objective_id={objective_id})"
        )
    if approval_request is None:
        raise BridgeMappingError(
            f"kanban_apply: approval_request missing (objective_id={objective_id})"
        )


def _revalidate_approvals(
    policy_decision: PolicyDecision,
    approval_request: ApprovalRequest,
    *,
    approver_id: Optional[str],
    kanban_approver_id: Optional[str],
    worker_approver_id: Optional[str],
    external_approver_id: Optional[str],
    cross_session: bool,
    session_id: Optional[str],
) -> None:
    """Re-validate the 8 Phase 4A approval gates + Phase 4B lineage check.

    Raises ``BridgeApprovalError`` on the first failing gate.
    Raises ``KanbanLinkageConflictError`` on fingerprint mismatch.
    """
    if approval_request.policy_decision_fingerprint != policy_decision.decision_fingerprint:
        raise KanbanLinkageConflictError(
            f"ApprovalRequest.policy_decision_fingerprint "
            f"({approval_request.policy_decision_fingerprint}) does not match "
            f"PolicyDecision.decision_fingerprint "
            f"({policy_decision.decision_fingerprint})"
        )
    result = evaluate_approval_gates(
        policy_decision,
        approver_id=approver_id,
        approval_token=approval_request.approval_token,
        kanban_approver_id=kanban_approver_id,
        worker_approver_id=worker_approver_id,
        external_approver_id=external_approver_id,
        cross_session=cross_session,
        session_id=session_id,
        expiry=approval_request.expiry,
        renewal=False,
    )
    if not result.approved:
        raise BridgeApprovalError(
            f"approval re-validation failed (layer {result.failure_layer}: "
            f"{result.failure_reason})"
        )


# ──────────────────────────────────────────────────────────────────────
# KanbanApplyEngine
# ──────────────────────────────────────────────────────────────────────

class KanbanApplyEngine:
    """Phase 4B: apply an approved OrchestratorPlanPreview to Kanban.

    Single entry point: ``apply(objective_id, ...)``. Side effects:

    * ``kb.create_task`` (existing API) — 1 call per task in the preview.
    * ``state_meta[objective_kanban_apply:<oid>]`` — JSON of
      ``KanbanApplyResult`` (linkage to Phase 4A decision + request).
    * ``state_meta[objective_kanban_tasks:<oid>]`` — JSON of the
      ``task_ids`` tuple.

    Does NOT:

    * Spawn workers (Phase 5+).
    * Assign tasks to profiles (Phase 5+).
    * Touch ``agent/orchestrator/*`` (Phase 5+).
    * Make any network / provider / API call.
    * Use prohibited Kanban APIs (``kanban_command``, ``_cmd_create``,
      ``_cmd_swarm``, ``create_swarm``, ``kanban_decompose``,
      ``kanban_specify``, ``kanban_swarm``).
    * Use ``agent/orchestrator/kanban_adapter.py``.
    """

    SCHEMA_VERSION = "phase4b.v1"

    def __init__(
        self,
        *,
        state_storage: Optional[ObjectiveStateStorage] = None,
        kanban_create_fn: Optional[Callable] = None,
        kanban_connect_fn: Optional[Callable] = None,
    ) -> None:
        self._storage = state_storage or ObjectiveStateStorage()
        # Default: use the real kb.create_task via lazy import.
        # The injection point lets tests fake kb.create_task.
        self._kanban_create_fn = kanban_create_fn
        self._kanban_connect_fn = kanban_connect_fn

    def _create_task(self, kwargs: dict) -> str:
        """Call kb.create_task (or the injected fake)."""
        if self._kanban_create_fn is not None:
            return self._kanban_create_fn(kwargs)
        # Default path: import kb lazily so tests can patch
        # hermes_cli.kanban_db if needed.
        from hermes_cli import kanban_db as _kb
        return _kb.create_task(**kwargs)

    # ── Mode 1: build_apply_preview (pure) ─────────────────────

    def build_apply_preview(
        self,
        objective_id: str,
        *,
        board: Optional[str] = None,
        created_by: str = DEFAULT_CREATED_BY,
    ) -> KanbanApplyPreview:
        """Compute a KanbanApplyPreview. NO state_meta writes. NO Kanban writes.

        Loads Phase 1+2+3+4A artifacts from storage and computes a
        pure preview of the apply. Raises ``BridgeMappingError`` if
        any required artifact is missing.
        """
        plan = self._storage.get_objective_plan(objective_id)
        orchestrator_preview = self._storage.get_objective_orchestrator_preview(
            objective_id
        )
        policy_decision = self._storage.get_objective_policy_decision(objective_id)
        approval_request = self._storage.get_objective_approval_request(
            objective_id
        )

        _ensure_required_artifacts(
            plan=plan,
            orchestrator_preview=orchestrator_preview,
            policy_decision=policy_decision,
            approval_request=approval_request,
            objective_id=objective_id,
        )

        # Now safe to assert types.
        assert orchestrator_preview is not None  # for mypy
        assert policy_decision is not None
        assert approval_request is not None

        task_specs = list(orchestrator_preview.task_specs)
        task_kwargs_list = _build_kwargs_list(
            task_specs,
            objective_id=objective_id,
            created_by=created_by,
            board=board,
        )

        warnings: list[str] = []
        if not task_kwargs_list:
            warnings.append("OrchestratorPlanPreview has no task_specs.")

        kanban_apply_fingerprint = compute_kanban_apply_fingerprint(
            objective_id=objective_id,
            task_ids=tuple(),
            decision_fingerprint=policy_decision.decision_fingerprint,
            request_fingerprint=approval_request.request_fingerprint,
            kanban_apply_fingerprint="",
        )

        return KanbanApplyPreview(
            objective_id=objective_id,
            task_specs=tuple(json.dumps(s, sort_keys=True) for s in task_specs),
            task_kwargs_list=tuple(
                json.dumps(k, sort_keys=True) for k in task_kwargs_list
            ),
            kanban_apply_fingerprint=kanban_apply_fingerprint,
            warnings=tuple(warnings),
            created_at=now_iso8601(),
        )

    # ── Mode 2: apply (writes Kanban tasks + state_meta) ────────

    def apply(
        self,
        objective_id: str,
        *,
        board: Optional[str] = None,
        created_by: str = DEFAULT_CREATED_BY,
        approver_id: Optional[str] = None,
        kanban_approver_id: Optional[str] = None,
        worker_approver_id: Optional[str] = None,
        external_approver_id: Optional[str] = None,
        cross_session: bool = False,
        session_id: Optional[str] = None,
    ) -> KanbanApplyResult:
        """Re-validate Phase 4A approval + create tasks + persist linkage.

        Side effects (only after all 8 gates pass):

        1. ``kb.create_task`` — 1 call per task in the preview.
        2. ``state_meta[objective_kanban_apply:<oid>]``.
        3. ``state_meta[objective_kanban_tasks:<oid>]``.

        Raises ``BridgeApprovalError`` on the first failing gate.
        Raises ``BridgeMappingError`` if any Phase 1+2+3+4A artifact is
        missing. Raises ``KanbanLinkageConflictError`` if the persisted
        apply record has a different fingerprint than the new preview.
        """
        plan = self._storage.get_objective_plan(objective_id)
        orchestrator_preview = self._storage.get_objective_orchestrator_preview(
            objective_id
        )
        policy_decision = self._storage.get_objective_policy_decision(objective_id)
        approval_request = self._storage.get_objective_approval_request(
            objective_id
        )

        _ensure_required_artifacts(
            plan=plan,
            orchestrator_preview=orchestrator_preview,
            policy_decision=policy_decision,
            approval_request=approval_request,
            objective_id=objective_id,
        )
        assert orchestrator_preview is not None
        assert policy_decision is not None
        assert approval_request is not None

        # Re-validate 8 Phase 4A approval gates.
        _revalidate_approvals(
            policy_decision,
            approval_request,
            approver_id=approver_id,
            kanban_approver_id=kanban_approver_id,
            worker_approver_id=worker_approver_id,
            external_approver_id=external_approver_id,
            cross_session=cross_session,
            session_id=session_id,
        )

        # Compute task kwargs.
        task_specs = list(orchestrator_preview.task_specs)
        task_kwargs_list = _build_kwargs_list(
            task_specs,
            objective_id=objective_id,
            created_by=created_by,
            board=board,
        )

        # Idempotency check 1: state_meta dedup.
        existing = self._storage.get_objective_kanban_apply(objective_id)
        if existing is not None:
            if (
                existing.decision_fingerprint
                == policy_decision.decision_fingerprint
                and existing.request_fingerprint
                == approval_request.request_fingerprint
                and len(existing.task_ids) == len(task_kwargs_list)
            ):
                # Duplicate apply with matching inputs.
                return KanbanApplyResult(
                    objective_id=objective_id,
                    task_ids=existing.task_ids,
                    preview_fingerprint=existing.preview_fingerprint,
                    decision_fingerprint=existing.decision_fingerprint,
                    request_fingerprint=existing.request_fingerprint,
                    result_fingerprint=existing.result_fingerprint,
                    duplicate=True,
                    created_at=existing.created_at,
                    created_by=existing.created_by,
                    board=existing.board,
                )
            raise KanbanLinkageConflictError(
                f"persisted apply record for {objective_id} has a different "
                f"fingerprint (existing.preview_fingerprint="
                f"{existing.preview_fingerprint}, task_ids="
                f"{list(existing.task_ids)})"
            )

        # Idempotency check 2: kb.create_task with idempotency_key handles
        # retries inside the loop. We just need to create the tasks.
        task_ids: list[str] = []
        for i, kwargs in enumerate(task_kwargs_list):
            # Resolve linear parent linkage: task N has task N-1 as parent.
            if i > 0:
                # Build a fresh kwargs dict to avoid mutating the cached one.
                new_kwargs = dict(kwargs)
                new_kwargs["parents"] = (task_ids[i - 1],)
                kwargs = new_kwargs
            else:
                new_kwargs = dict(kwargs)
                new_kwargs["parents"] = ()
                kwargs = new_kwargs
            task_id = self._create_task(kwargs)
            task_ids.append(task_id)

        # Persist linkage.
        preview_fingerprint = compute_kanban_apply_fingerprint(
            objective_id=objective_id,
            task_ids=tuple(task_ids),
            decision_fingerprint=policy_decision.decision_fingerprint,
            request_fingerprint=approval_request.request_fingerprint,
            kanban_apply_fingerprint="",
        )
        result_fingerprint = compute_kanban_result_fingerprint(
            objective_id=objective_id,
            task_ids=tuple(task_ids),
            preview_fingerprint=preview_fingerprint,
            decision_fingerprint=policy_decision.decision_fingerprint,
            request_fingerprint=approval_request.request_fingerprint,
        )
        result = KanbanApplyResult(
            objective_id=objective_id,
            task_ids=tuple(task_ids),
            preview_fingerprint=preview_fingerprint,
            decision_fingerprint=policy_decision.decision_fingerprint,
            request_fingerprint=approval_request.request_fingerprint,
            result_fingerprint=result_fingerprint,
            duplicate=False,
            created_at=now_iso8601(),
            created_by=created_by,
            board=board,
        )
        self._storage.set_objective_kanban_apply(result)
        self._storage.set_objective_kanban_tasks(objective_id, tuple(task_ids))
        return result

    # ── Mode 3: rollback (best-effort, idempotent) ──────────────

    def rollback(
        self,
        objective_id: str,
        *,
        hard_delete: bool = True,
    ) -> bool:
        """Delete all Kanban tasks created by this apply.

        ``hard_delete=True``: ``kb.delete_task`` (idempotent).
        ``hard_delete=False``: ``kb.archive_task`` (soft delete).

        Returns ``True`` if at least one task was removed/archived.
        Idempotent: a second call returns ``False``.
        """
        existing = self._storage.get_objective_kanban_apply(objective_id)
        if existing is None:
            # Cleanup state_meta anyway (idempotent).
            self._storage.delete_objective_kanban_tasks(objective_id)
            return False

        plan = KanbanRollbackPlan.from_apply_result(
            existing, mode="hard_delete" if hard_delete else "soft_archive"
        )
        removed_any = False
        # Open a single connection and reuse it for all rollback calls.
        from hermes_cli import kanban_db as _kb
        with _kb.connect_closing() as conn:
            for task_id in plan.task_ids:
                try:
                    if hard_delete:
                        if _kb.delete_task(conn, task_id):
                            removed_any = True
                    else:
                        if _kb.archive_task(conn, task_id):
                            removed_any = True
                except Exception:
                    # Best-effort: continue with next task.
                    continue

        # Cleanup state_meta.
        self._storage.delete_objective_kanban_apply(objective_id)
        self._storage.delete_objective_kanban_tasks(objective_id)
        return removed_any


# ──────────────────────────────────────────────────────────────────────
# Module-level functions (the spec's primary entry points)
# ──────────────────────────────────────────────────────────────────────

def build_kanban_apply_preview(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    board: Optional[str] = None,
    created_by: str = DEFAULT_CREATED_BY,
) -> KanbanApplyPreview:
    """Module-level wrapper: pure compute. NO state_meta writes. NO Kanban writes."""
    engine = KanbanApplyEngine(state_storage=storage)
    return engine.build_apply_preview(
        objective_id, board=board, created_by=created_by
    )


def kanban_apply(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    board: Optional[str] = None,
    created_by: str = DEFAULT_CREATED_BY,
    approver_id: Optional[str] = None,
    kanban_approver_id: Optional[str] = None,
    worker_approver_id: Optional[str] = None,
    external_approver_id: Optional[str] = None,
    cross_session: bool = False,
    session_id: Optional[str] = None,
) -> KanbanApplyResult:
    """Module-level wrapper: re-validate + create + persist."""
    engine = KanbanApplyEngine(state_storage=storage)
    return engine.apply(
        objective_id,
        board=board,
        created_by=created_by,
        approver_id=approver_id,
        kanban_approver_id=kanban_approver_id,
        worker_approver_id=worker_approver_id,
        external_approver_id=external_approver_id,
        cross_session=cross_session,
        session_id=session_id,
    )


def kanban_rollback(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    hard_delete: bool = True,
) -> bool:
    """Module-level wrapper: best-effort, idempotent rollback."""
    engine = KanbanApplyEngine(state_storage=storage)
    return engine.rollback(objective_id, hard_delete=hard_delete)


# ──────────────────────────────────────────────────────────────────────
# Read helpers
# ──────────────────────────────────────────────────────────────────────

def get_apply_result_for_objective(
    objective_id: str,
    storage: Optional[ObjectiveStateStorage] = None,
) -> Optional[KanbanApplyResult]:
    """Return the persisted KanbanApplyResult for the objective_id, or None."""
    store = storage or ObjectiveStateStorage()
    return store.get_objective_kanban_apply(objective_id)


def get_task_ids_for_objective(
    objective_id: str,
    storage: Optional[ObjectiveStateStorage] = None,
) -> tuple:
    """Return the persisted task_ids for the objective_id, or empty tuple."""
    store = storage or ObjectiveStateStorage()
    raw = store.get_objective_kanban_tasks(objective_id)
    return raw or ()


__all__ = [
    "KanbanApplyError",
    "KanbanLinkageConflictError",
    "KanbanStorageError",
    "KanbanApplyEngine",
    "build_kanban_apply_preview",
    "kanban_apply",
    "kanban_rollback",
    "get_apply_result_for_objective",
    "get_task_ids_for_objective",
]