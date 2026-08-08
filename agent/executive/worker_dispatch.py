"""Phase 5 Worker Dispatch — engine facade.

Phase 5 consumes a Phase 4B ``KanbanApplyResult`` + a Phase 4A
``ApprovalRequest`` / ``PolicyDecision`` and dispatches the kanban
tasks to real workers via the existing
``agent.orchestrator.{dispatcher, batch_runner, handlers, worker_runner, kanban_adapter}``
infrastructure. It does NOT spawn workers directly and does NOT
duplicate the dispatcher, scheduler, or worker_runner.

The engine has three modes:

* ``dry_run`` — pure compute; produces a ``WorkerDispatchPreview``
  with no kanban writes and no ``BatchRunner.run_batch`` call.
* ``apply`` — re-validates the 8 Phase 4A approval gates (incl.
  Layer 6 Worker_spawn R5 which requires ``worker_approver_id``),
  wires the canonical ``make_handlers(adapter)`` handlers, calls
  ``BatchRunner.run_batch`` exactly once, and persists the
  dispatch record to state_meta.
* ``rollback`` — best-effort, idempotent; calls
  ``kb.archive_task`` (or ``kb.delete_task`` if ``hard_delete=True``)
  for each task in the dispatch record, and clears state_meta.

Forbidden APIs (PROHIBITED in this module):

* ``hermes_cli.kanban.kanban_command`` / ``_cmd_create`` / ``_cmd_swarm``
* ``hermes_cli.kanban_db.create_task`` / ``delete_task`` (Phase 4B only)
* ``hermes_cli.kanban_decompose.*`` / ``kanban_specify.*`` / ``kanban_swarm.*``
* ``hermes_cli.write_approval_commands`` (Phase 1+2+3 ad-hoc)
* ``agent.execution_router.ExecutionRouter``
* ``agent.execution_dispatcher.ExecutionDispatcher``
* ``agent.orchestrator_interface.OrchestratorInterface.execute``
* ``delegate_task`` / ``execute()``
* Any LLM call (anthropic / openai / auxiliary_client / urllib / requests / httpx)
* Any subprocess / os.system / os.popen
* Any DB DDL (CREATE TABLE / ALTER TABLE / CREATE INDEX)
* gbrain / obsidian / notebooklm
"""

from __future__ import annotations

import json
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, FrozenSet, List, Optional, Tuple

from .types import (
    ApprovalRequest,
    KanbanApplyResult,
    PolicyDecision,
    WorkerDispatchPreview,
    WorkerDispatchResult,
    WorkerDispatchRollbackPlan,
    WorkerDispatchTaskLink,
    objective_worker_dispatch_key,
    objective_worker_dispatch_tasks_key,
    compute_dispatch_fingerprint,
    RiskLevel,
)
from .state_storage import ObjectiveStateStorage
from .approval_gates import evaluate_approval_gates, ApprovalGateResult
from .worker_mapping import (
    build_batch_inputs,
    compute_dispatch_fingerprint as _compute_dispatch_fingerprint,
    derive_restrictions,
    kanban_task_to_task_state,
    worker_registry_from_kanban_tasks,
)
from . import (
    goalmanager_bridge as gmb,
)
from .kanban_apply import KanbanLinkageConflictError

log = logging.getLogger("executive.worker_dispatch")

SCHEMA_VERSION = "phase5.v1"


# ── Errors (reused from Phase 2/4A/4B; do not duplicate) ──────────────
BridgeMappingError = gmb.BridgeMappingError
BridgeApprovalError = gmb.BridgeApprovalError


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Optional dependency: agent.orchestrator.* ──────────────────────
# Imported lazily inside apply() so the dry-run path does not pull in
# the orchestrator module. If the orchestrator is not importable (e.g.
# during tests without a fully-initialized env), apply() raises a
# clear BridgeMappingError.


def _try_import_orchestrator():
    """Lazy import of the canonical orchestrator. Returns None on failure."""
    try:
        from agent.orchestrator.dispatcher import (
            Dispatcher,
            DispatchResult,
        )
        from agent.orchestrator.batch_runner import BatchRunner
        from agent.orchestrator.handlers import make_handlers
        from agent.orchestrator.kanban_adapter import KanbanAdapter, KanbanTask
        from agent.orchestrator.worker_runner import (
            run_worker_subprocess,
            WorkerRunResult,
        )
        return {
            "Dispatcher": Dispatcher,
            "DispatchResult": DispatchResult,
            "BatchRunner": BatchRunner,
            "make_handlers": make_handlers,
            "KanbanAdapter": KanbanAdapter,
            "KanbanTask": KanbanTask,
            "run_worker_subprocess": run_worker_subprocess,
            "WorkerRunResult": WorkerRunResult,
        }
    except ImportError as e:
        log.warning("orchestrator import failed: %s", e)
        return None


def _try_import_kanban_db():
    try:
        import hermes_cli.kanban_db as kb
        return kb
    except ImportError as e:
        log.warning("kanban_db import failed: %s", e)
        return None


# ── Engine ────────────────────────────────────────────────────────

class WorkerDispatchEngine:
    """High-level facade for Phase 5 Worker Dispatch.

    Constructor takes an optional ``state_storage`` and
    ``board_root`` (passed through to ``KanbanAdapter`` when
    applying). All side effects are gated on:
    1. The 8 Phase 4A approval gates (re-validated in apply).
    2. The Phase 5 lineage check (ApprovalRequest.fingerprint
       matches PolicyDecision.fingerprint).
    3. The Layer 6 (Worker_spawn R5) gate which requires
       ``worker_approver_id``.

    ``dry_run`` is pure compute. ``apply`` is the only path that
    calls ``BatchRunner.run_batch`` and persists state_meta.
    ``rollback`` is best-effort, idempotent.
    """

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(
        self,
        *,
        state_storage: Optional[ObjectiveStateStorage] = None,
        board_root: Optional[Path] = None,
        orchestrator_factory=None,
        kanban_db_factory=None,
    ) -> None:
        self._storage = state_storage or ObjectiveStateStorage()
        self._board_root = board_root
        # Injection points for tests: callable that returns the
        # orchestrator modules (or None if not importable). Tests
        # inject fakes; production calls _try_import_orchestrator.
        self._orchestrator_factory = orchestrator_factory or _try_import_orchestrator
        self._kanban_db_factory = kanban_db_factory or _try_import_kanban_db

    # ── pure compute ──────────────────────────────────────────────

    def dry_run(
        self,
        objective_id: str,
        *,
        restrictions: Optional[FrozenSet[str]] = None,
    ) -> WorkerDispatchPreview:
        """Pure compute: build WorkerDispatchPreview. No state_meta writes,
        no BatchRunner.run_batch, no subprocess, no workers spawned.

        Reads the kanban DB via the injected ``kanban_db_factory`` to
        fetch task metadata. If the kanban DB is not importable, an
        empty preview is returned (still pure, but with empty
        task_states and workers).
        """
        preview_state = self._build_preview_inputs(
            objective_id,
            restrictions=restrictions,
        )
        return preview_state

    def _build_preview_inputs(
        self,
        objective_id: str,
        *,
        restrictions: Optional[FrozenSet[str]] = None,
    ) -> WorkerDispatchPreview:
        """Internal: build the preview, shared by dry_run and apply."""
        # 1. Load Phase 3+4A+4B artifacts.
        plan = self._storage.get_objective_plan(objective_id)
        preview = self._storage.get_objective_orchestrator_preview(objective_id)
        decision = self._storage.get_objective_policy_decision(objective_id)
        request = self._storage.get_objective_approval_request(objective_id)
        apply_record = self._storage.get_objective_kanban_apply(objective_id)

        if plan is None or preview is None:
            raise BridgeMappingError(
                f"Phase 3 plan/preview missing for objective_id={objective_id!r}"
            )
        if decision is None or request is None:
            raise BridgeMappingError(
                f"Phase 4A decision/request missing for objective_id={objective_id!r}"
            )
        if apply_record is None:
            raise BridgeMappingError(
                f"Phase 4B apply record missing for objective_id={objective_id!r} "
                f"(no kanban tasks to dispatch)"
            )

        # 2. Compute decision/request/kanban apply fingerprints.
        decision_fp = str(getattr(decision, "decision_fingerprint", "") or "")
        request_fp = str(getattr(request, "request_fingerprint", "") or "")
        kanban_apply_fp = str(
            getattr(apply_record, "preview_fingerprint", "")
            or getattr(apply_record, "dispatch_fingerprint", "")
            or ""
        )

        # 3. Read kanban tasks (in canonical order from apply_record).
        task_ids = tuple(apply_record.task_ids or ())
        kanban_tasks: list = []
        kb = self._kanban_db_factory()
        if kb is not None and task_ids:
            for tid in task_ids:
                try:
                    t = kb.get_task(tid)
                except Exception as e:
                    log.warning("kb.get_task(%s) failed: %s", tid, e)
                    t = None
                if t is not None:
                    kanban_tasks.append(t)

        # 4. Build (TaskState[], WorkerRegistryEntry[]) via the
        #    pure mapping in worker_mapping.
        task_states, workers = build_batch_inputs(task_ids, kanban_tasks=kanban_tasks)

        # 5. Derive restrictions.
        warnings: list = []
        if restrictions is None:
            restrictions = derive_restrictions(decision)
        if not task_states:
            warnings.append("no_task_states_built")

        # 6. Compute dispatch fingerprint.
        dispatch_fp = _compute_dispatch_fingerprint(
            task_ids,
            restrictions=restrictions,
            decision_fingerprint=decision_fp,
            request_fingerprint=request_fp,
            kanban_apply_fingerprint=kanban_apply_fp,
        )

        return WorkerDispatchPreview(
            objective_id=objective_id,
            kanban_task_ids=task_ids,
            task_states=tuple(task_states),
            workers=tuple(workers),
            restrictions=frozenset(restrictions),
            warnings=tuple(warnings),
            dispatch_fingerprint=dispatch_fp,
            created_at=_now_iso8601(),
        )

    # ── apply ────────────────────────────────────────────────────

    def apply(
        self,
        objective_id: str,
        *,
        approver_id: Optional[str] = None,
        approval_token: Optional[str] = None,
        kanban_approver_id: Optional[str] = None,
        worker_approver_id: Optional[str] = None,
        external_approver_id: Optional[str] = None,
        cross_session: bool = False,
        session_id: Optional[str] = None,
    ) -> WorkerDispatchResult:
        """Re-validate approval + build dispatcher + run batch.

        Raises:
        * ``BridgeApprovalError`` on the first failing approval gate.
        * ``KanbanLinkageConflictError`` on fingerprint mismatch.
        * ``BridgeMappingError`` if Phase 3/4A/4B artifacts are missing.

        Side effects (only after all gates pass):
        * 0 or 1 ``BatchRunner.run_batch`` call.
        * state_meta writes (objective_worker_dispatch and
          objective_worker_dispatch_tasks).
        """
        # 1. Re-validate approval gates.
        self._revalidate_approvals(
            objective_id,
            approver_id=approver_id,
            approval_token=approval_token,
            kanban_approver_id=kanban_approver_id,
            worker_approver_id=worker_approver_id,
            external_approver_id=external_approver_id,
            cross_session=cross_session,
            session_id=session_id,
        )

        # 2. Load artifacts (after re-validation, so a failed gate
        #    does not block state_meta reads).
        decision = self._storage.get_objective_policy_decision(objective_id)
        request = self._storage.get_objective_approval_request(objective_id)
        apply_record = self._storage.get_objective_kanban_apply(objective_id)

        # 3. Build the preview (pure).
        preview = self._build_preview_inputs(objective_id, restrictions=None)

        # 4. Idempotency check.
        existing = self._storage.get_objective_worker_dispatch(objective_id)
        if existing is not None:
            if (
                existing.dispatch_fingerprint == preview.dispatch_fingerprint
                and existing.decision_fingerprint == str(getattr(decision, "decision_fingerprint", "") or "")
                and existing.request_fingerprint == str(getattr(request, "request_fingerprint", "") or "")
                and existing.kanban_apply_fingerprint == (
                    str(getattr(apply_record, "preview_fingerprint", "") or "")
                    or str(getattr(apply_record, "dispatch_fingerprint", "") or "")
                )
            ):
                # Persisted result matches the new request: return
                # the persisted record with duplicate=True.
                return WorkerDispatchResult(
                    objective_id=existing.objective_id,
                    task_ids=existing.task_ids,
                    worker_runs=existing.worker_runs,
                    worker_runs_started=existing.worker_runs_started,
                    worker_runs_failed=existing.worker_runs_failed,
                    dispatch_fingerprint=existing.dispatch_fingerprint,
                    decision_fingerprint=existing.decision_fingerprint,
                    request_fingerprint=existing.request_fingerprint,
                    kanban_apply_fingerprint=existing.kanban_apply_fingerprint,
                    duplicate=True,
                    errors=existing.errors,
                    created_at=existing.created_at,
                )
            else:
                raise KanbanLinkageConflictError(
                    f"existing dispatch record has different fingerprints: "
                    f"existing.dispatch_fingerprint={existing.dispatch_fingerprint!r} "
                    f"vs new={preview.dispatch_fingerprint!r}"
                )

        # 5. Build dispatcher + BatchRunner.
        orch = self._orchestrator_factory()
        if orch is None:
            raise BridgeMappingError(
                "agent.orchestrator.* not importable; cannot run apply"
            )
        KanbanAdapter = orch["KanbanAdapter"]
        Dispatcher = orch["Dispatcher"]
        BatchRunner = orch["BatchRunner"]
        make_handlers = orch["make_handlers"]

        # Build the adapter (board_root from engine or from
        # kanban_task if it carries one; fall back to current).
        board_root = self._board_root
        if board_root is None:
            # Try to derive from the apply_record's first task.
            kb = self._kanban_db_factory()
            if kb is not None and preview.kanban_task_ids:
                first = kb.get_task(preview.kanban_task_ids[0])
                if first is not None:
                    board_root = getattr(first, "board_root", None) or getattr(first, "workspace_path", None)
        if board_root is None:
            board_root = Path(".")

        adapter = KanbanAdapter(board_root=board_root)
        dispatcher = Dispatcher(handlers=make_handlers(adapter))
        batch_runner = BatchRunner(dispatcher=dispatcher, adapter=adapter)

        # 6. Run the batch (single call).
        # Build the lists (already dicts in preview; orchestrator accepts dicts).
        try:
            batch_result = batch_runner.run_batch(
                list(preview.task_states),
                list(preview.workers),
                restrictions=set(preview.restrictions),
            )
        except Exception as e:
            log.error("BatchRunner.run_batch failed: %s", e)
            # Best-effort: record the error in the result, do not
            # raise (the caller can inspect errors and decide).
            batch_result = _FakeBatchResult(errors=[{
                "task_id": None,
                "error_type": type(e).__name__,
                "error_repr": repr(e),
            }])

        # 7. Serialize the batch results.
        worker_runs: list = []
        for r in getattr(batch_result, "results", []) or ():
            if hasattr(r, "to_dict"):
                worker_runs.append(r.to_dict())
            else:
                worker_runs.append(dict(r) if isinstance(r, dict) else {"raw": str(r)})
        worker_runs_started = int(getattr(batch_result, "worker_runs_started", 0) or 0)
        # worker_runs_failed = number of results with action_executed starting
        # with "NO_HANDLER_FOR_" or "BLOCK".
        worker_runs_failed = sum(
            1
            for r in worker_runs
            if isinstance(r, dict)
            and isinstance(r.get("action_executed"), str)
            and (r["action_executed"].startswith("NO_HANDLER_FOR_") or r["action_executed"] == "BLOCK")
        )
        errors: list = []
        for e in getattr(batch_result, "errors", []) or ():
            if isinstance(e, dict):
                errors.append(e)
            else:
                errors.append({"raw": str(e)})

        # 8. Build the result.
        result = WorkerDispatchResult(
            objective_id=objective_id,
            task_ids=preview.kanban_task_ids,
            worker_runs=tuple(worker_runs),
            worker_runs_started=worker_runs_started,
            worker_runs_failed=worker_runs_failed,
            dispatch_fingerprint=preview.dispatch_fingerprint,
            decision_fingerprint=str(getattr(decision, "decision_fingerprint", "") or ""),
            request_fingerprint=str(getattr(request, "request_fingerprint", "") or ""),
            kanban_apply_fingerprint=str(
                getattr(apply_record, "preview_fingerprint", "")
                or getattr(apply_record, "dispatch_fingerprint", "")
                or ""
            ),
            duplicate=False,
            errors=tuple(errors),
            created_at=_now_iso8601(),
        )

        # 9. Persist to state_meta.
        self._storage.set_objective_worker_dispatch(result)
        self._storage.set_objective_worker_dispatch_tasks(
            objective_id, result.task_ids, result.dispatch_fingerprint
        )
        return result

    # ── rollback ─────────────────────────────────────────────────

    def rollback(
        self,
        objective_id: str,
        *,
        hard_delete: bool = False,
    ) -> bool:
        """Best-effort, idempotent rollback. Returns True if at least
        one task was archived/deleted.

        Default mode is "archive" (soft delete via
        ``kb.archive_task``). Setting ``hard_delete=True`` uses
        ``kb.delete_task`` (NOT recommended for dispatched tasks;
        the worker may still be running).
        """
        record = self._storage.get_objective_worker_dispatch(objective_id)
        if record is None:
            # Idempotent: no record, nothing to do. Still try to
            # clean up any orphaned state_meta rows.
            self._storage.delete_objective_worker_dispatch(objective_id)
            self._storage.delete_objective_worker_dispatch_tasks(objective_id)
            return False

        plan = WorkerDispatchRollbackPlan.from_dispatch_record(
            record, mode="hard_delete" if hard_delete else "archive"
        )

        kb = self._kanban_db_factory()
        if kb is None:
            log.warning("kanban_db not importable; rollback is a no-op")
            return False

        any_action = False
        for task_id in plan.task_ids:
            try:
                if hard_delete:
                    kb.delete_task(task_id)
                else:
                    kb.archive_task(task_id)
                any_action = True
            except Exception as e:
                # best-effort: log and continue
                log.warning("rollback %s on %s failed: %s",
                            "delete" if hard_delete else "archive", task_id, e)

        # Cleanup state_meta (best-effort).
        self._storage.delete_objective_worker_dispatch(objective_id)
        self._storage.delete_objective_worker_dispatch_tasks(objective_id)
        return any_action

    # ── approvals ────────────────────────────────────────────────

    def _revalidate_approvals(
        self,
        objective_id: str,
        *,
        approver_id: Optional[str],
        approval_token: Optional[str],
        kanban_approver_id: Optional[str],
        worker_approver_id: Optional[str],
        external_approver_id: Optional[str],
        cross_session: bool,
        session_id: Optional[str],
    ) -> None:
        """Re-validate the 8 Phase 4A approval gates + Phase 5 lineage check.

        Raises:
        * ``KanbanLinkageConflictError`` on fingerprint mismatch.
        * ``BridgeApprovalError`` on the first failing gate.
        """
        decision = self._storage.get_objective_policy_decision(objective_id)
        request = self._storage.get_objective_approval_request(objective_id)
        if decision is None or request is None:
            raise BridgeMappingError(
                f"Phase 4A decision/request missing for objective_id={objective_id!r}"
            )

        # Phase 5 lineage: ApprovalRequest.policy_decision_fingerprint
        # must match PolicyDecision.decision_fingerprint.
        d_fp = str(getattr(decision, "decision_fingerprint", "") or "")
        r_fp = str(getattr(request, "policy_decision_fingerprint", "") or "")
        if d_fp and r_fp and d_fp != r_fp:
            raise KanbanLinkageConflictError(
                f"ApprovalRequest.policy_decision_fingerprint ({r_fp}) does not "
                f"match PolicyDecision.decision_fingerprint ({d_fp})"
            )

        # 8-layer re-validation (Phase 4A's evaluate_approval_gates).
        result = evaluate_approval_gates(
            decision,
            approver_id=approver_id,
            approval_token=approval_token or getattr(request, "approval_token", None),
            kanban_approver_id=kanban_approver_id,
            worker_approver_id=worker_approver_id,
            external_approver_id=external_approver_id,
            cross_session=cross_session,
            session_id=session_id,
            expiry=getattr(request, "expiry", None),
            renewal=False,
        )
        if not result.approved:
            raise BridgeApprovalError(
                f"approval re-validation failed (layer {result.failure_layer}: "
                f"{result.failure_reason})"
            )


# ── Module-level wrappers ──────────────────────────────────────────

def worker_dispatch_dry_run(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    board_root: Optional[Path] = None,
    orchestrator_factory=None,
    kanban_db_factory=None,
) -> WorkerDispatchPreview:
    """Module-level wrapper: pure compute dry-run. No state_meta writes."""
    eng = WorkerDispatchEngine(
        state_storage=storage,
        board_root=board_root,
        orchestrator_factory=orchestrator_factory,
        kanban_db_factory=kanban_db_factory,
    )
    return eng.dry_run(objective_id)


def worker_dispatch_apply(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    board_root: Optional[Path] = None,
    approver_id: Optional[str] = None,
    approval_token: Optional[str] = None,
    kanban_approver_id: Optional[str] = None,
    worker_approver_id: Optional[str] = None,
    external_approver_id: Optional[str] = None,
    cross_session: bool = False,
    session_id: Optional[str] = None,
    orchestrator_factory=None,
    kanban_db_factory=None,
) -> WorkerDispatchResult:
    """Module-level wrapper: re-validate + build + dispatch + persist."""
    eng = WorkerDispatchEngine(
        state_storage=storage,
        board_root=board_root,
        orchestrator_factory=orchestrator_factory,
        kanban_db_factory=kanban_db_factory,
    )
    return eng.apply(
        objective_id,
        approver_id=approver_id,
        approval_token=approval_token,
        kanban_approver_id=kanban_approver_id,
        worker_approver_id=worker_approver_id,
        external_approver_id=external_approver_id,
        cross_session=cross_session,
        session_id=session_id,
    )


def worker_dispatch_rollback(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    hard_delete: bool = False,
    kanban_db_factory=None,
) -> bool:
    """Module-level wrapper: best-effort, idempotent rollback."""
    eng = WorkerDispatchEngine(
        state_storage=storage,
        kanban_db_factory=kanban_db_factory,
    )
    return eng.rollback(objective_id, hard_delete=hard_delete)


# ── Internal helper ───────────────────────────────────────────────

class _FakeBatchResult:
    """Empty batch result used when BatchRunner.run_batch raises."""

    def __init__(self, *, results=None, errors=None, worker_runs_started=0):
        self.results = list(results or [])
        self.errors = list(errors or [])
        self.worker_runs_started = int(worker_runs_started or 0)


__all__ = [
    "WorkerDispatchEngine",
    "worker_dispatch_dry_run",
    "worker_dispatch_apply",
    "worker_dispatch_rollback",
    "WorkerDispatchPreview",
    "WorkerDispatchResult",
    "WorkerDispatchRollbackPlan",
    "WorkerDispatchTaskLink",
    "BridgeMappingError",
    "BridgeApprovalError",
    "KanbanLinkageConflictError",
    "SCHEMA_VERSION",
]
