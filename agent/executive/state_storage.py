"""ObjectiveStateStorage — wrapper over state_meta for objectives.

Persists ObjectiveStateData to SessionDB.state_meta under the
``objective:<oid>`` key. Phase 1 does NOT create any new table; the
namespace is parallel to GoalManager's ``goal:<session_id>``.

Read-only + write to state_meta only. No objective_db. No migrations.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable

from .types import (
    ObjectiveState,
    ObjectiveStateData,
    ApprovalRequest,
    EvaluationReport,
    GoalLinkage,
    KanbanApplyResult,
    KanbanRollbackPlan,
    ObjectivePlan,
    OrchestratorPlanPreview,
    PolicyDecision,
    RecoveryDiagnosis,
    RecoveryPlanPreview,
    SuccessReport,
    WorkerDispatchResult,
    objective_archive_key,
    objective_approval_request_key,
    objective_evaluation_key,
    objective_evidence_pack_key,
    objective_goal_link_key,
    objective_kanban_apply_key,
    objective_kanban_tasks_key,
    objective_key,
    objective_orchestrator_preview_key,
    objective_plan_key,
    objective_policy_decision_key,
    objective_recovery_diagnosis_key,
    objective_recovery_plan_key,
    objective_success_report_key,
    objective_worker_dispatch_key,
    objective_worker_dispatch_tasks_key,
)

DEFAULT_STATE_DB_PATH = Path.home() / ".hermes" / "state.db"


def _now_iso8601() -> str:
    """ISO 8601 UTC timestamp (no microseconds, Z suffix)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")




class StateStorageError(RuntimeError):
    """Raised when state_meta is unavailable or write fails."""


class ObjectiveStateStorage:
    """Read-only + write wrapper around state_meta for objectives.

    Phase 1: single-objective read/write. Phase 2+ may add batch ops.
    """

    def __init__(self, *, db_factory: Callable | None = None) -> None:
        self._factory = db_factory
        self._owns_db = db_factory is None

    def _get_db(self) -> Any:
        if self._factory is not None:
            return self._factory()
        try:
            from hermes_state import SessionDB
            from hermes_constants import get_hermes_home

            db_path = get_hermes_home() / "state.db"
            db = SessionDB(db_path=db_path, read_only=False)
            return db
        except Exception as exc:
            raise StateStorageError(f"SessionDB unavailable: {exc}") from exc

    def _close_db(self, db: Any) -> None:
        if not self._owns_db:
            return
        try:
            db.close()
        except Exception:
            pass

    def save(self, state: ObjectiveStateData) -> None:
        db = self._get_db()
        try:
            payload = json.dumps(state.to_dict(), default=str, sort_keys=True)
            db.set_meta(objective_key(state.objective_id), payload)
        except Exception as exc:
            raise StateStorageError(f"save failed: {exc}") from exc
        finally:
            self._close_db(db)

    def load(self, objective_id: str) -> ObjectiveStateData | None:
        db = self._get_db()
        try:
            raw = db.get_meta(objective_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return ObjectiveStateData.from_dict(data)
        except Exception as exc:
            raise StateStorageError(f"load failed: {exc}") from exc
        finally:
            self._close_db(db)

    def exists(self, objective_id: str) -> bool:
        db = self._get_db()
        try:
            raw = db.get_meta(objective_key(objective_id))
            return raw is not None
        except Exception:
            return False
        finally:
            self._close_db(db)

    def list_active(self) -> list[str]:
        db = self._get_db()
        try:
            keys = self._list_keys_with_prefix(db, "objective:")
        except Exception:
            return []
        finally:
            self._close_db(db)
        return [
            k.removeprefix("objective:")
            for k in keys
            if not k.startswith("objective_archive:")
        ]

    def list_all(self, *, include_archived: bool = False) -> list[str]:
        db = self._get_db()
        try:
            # Query both namespaces and merge.
            active_keys = self._list_keys_with_prefix(db, "objective:")
            archived_keys = (
                self._list_keys_with_prefix(db, "objective_archive:")
                if include_archived else []
            )
        except Exception:
            return []
        finally:
            self._close_db(db)
        out: list[str] = []
        for k in active_keys:
            if k.startswith("objective_archive:"):
                continue
            out.append(k.removeprefix("objective:"))
        for k in archived_keys:
            out.append(k.removeprefix("objective_archive:"))
        return out

    def archive(self, objective_id: str) -> None:
        db = self._get_db()
        try:
            raw = db.get_meta(objective_key(objective_id))
            if raw is None:
                return
            db.set_meta(objective_archive_key(objective_id), raw)
            # Best-effort: also delete the active key.
            try:
                if hasattr(db, "delete_meta"):
                    db.delete_meta(objective_key(objective_id))
                else:
                    self._fallback_delete(objective_key(objective_id))
            except Exception:
                pass
        except Exception as exc:
            raise StateStorageError(f"archive failed: {exc}") from exc
        finally:
            self._close_db(db)

    def _list_keys_with_prefix(self, db: Any, prefix: str) -> list[str]:
        """Best-effort: use list_meta_keys if available, else fall back
        to a SQLite query on state_meta directly.
        """
        if hasattr(db, "list_meta_keys"):
            try:
                return list(db.list_meta_keys(prefix=prefix) or [])
            except Exception:
                pass
        # Fallback: open state.db read-only and run a LIKE query.
        return self._sqlite_fallback_list(prefix)

    def _sqlite_fallback_list(self, prefix: str) -> list[str]:
        if not DEFAULT_STATE_DB_PATH.exists():
            return []
        try:
            conn = sqlite3.connect(
                f"file:{DEFAULT_STATE_DB_PATH}?mode=ro", uri=True
            )
        except Exception:
            return []
        try:
            cursor = conn.execute(
                "SELECT key FROM state_meta WHERE key LIKE ?", (prefix + "%",)
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _fallback_delete(self, key: str) -> None:
        if not DEFAULT_STATE_DB_PATH.exists():
            return
        try:
            conn = sqlite3.connect(str(DEFAULT_STATE_DB_PATH))
            conn.execute("DELETE FROM state_meta WHERE key = ?", (key,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    # ── Phase 2 GoalManager Bridge link methods ───────────────────

    def set_objective_goal_link(self, link: GoalLinkage) -> None:
        """Persist a Phase 2 objective↔goal linkage."""
        db = self._get_db()
        try:
            payload = json.dumps(link.to_dict(), default=str, sort_keys=True)
            db.set_meta(objective_goal_link_key(link.objective_id), payload)
        except Exception as exc:
            raise StateStorageError(f"set_objective_goal_link failed: {exc}") from exc
        finally:
            self._close_db(db)

    def get_objective_goal_link(
        self, objective_id: str
    ) -> GoalLinkage | None:
        """Load a Phase 2 linkage for the given objective_id, or None."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_goal_link_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return GoalLinkage.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_goal_link failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_goal_link(self, objective_id: str) -> bool:
        """Delete a Phase 2 linkage. Returns True on success.

        Best-effort. Idempotent: returns False on missing or failure.
        """
        db = self._get_db()
        try:
            if hasattr(db, "delete_meta"):
                db.delete_meta(objective_goal_link_key(objective_id))
                return True
            self._fallback_delete(objective_goal_link_key(objective_id))
            return True
        except Exception:
            return False
        finally:
            self._close_db(db)

    # ── Phase 3 Planner / Orchestrator Bridge methods ────────────

    def set_objective_plan(self, plan: ObjectivePlan) -> None:
        """Persist a Phase 3 plan."""
        db = self._get_db()
        try:
            payload = json.dumps(plan.to_dict(), default=str, sort_keys=True)
            db.set_meta(objective_plan_key(plan.objective_id), payload)
        except Exception as exc:
            raise StateStorageError(f"set_objective_plan failed: {exc}") from exc
        finally:
            self._close_db(db)

    def get_objective_plan(
        self, objective_id: str
    ) -> ObjectivePlan | None:
        """Load a Phase 3 plan for the given objective_id, or None."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_plan_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return ObjectivePlan.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_plan failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_plan(self, objective_id: str) -> bool:
        """Delete a Phase 3 plan. Idempotent. Best-effort."""
        db = self._get_db()
        try:
            if hasattr(db, "delete_meta"):
                db.delete_meta(objective_plan_key(objective_id))
                return True
            self._fallback_delete(objective_plan_key(objective_id))
            return True
        except Exception:
            return False
        finally:
            self._close_db(db)

    def set_objective_orchestrator_preview(
        self, preview: OrchestratorPlanPreview
    ) -> None:
        """Persist a Phase 3 orchestrator preview."""
        db = self._get_db()
        try:
            payload = json.dumps(preview.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_orchestrator_preview_key(preview.objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_orchestrator_preview failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_orchestrator_preview(
        self, objective_id: str
    ) -> OrchestratorPlanPreview | None:
        """Load a Phase 3 preview, or None."""
        db = self._get_db()
        try:
            raw = db.get_meta(
                objective_orchestrator_preview_key(objective_id)
            )
            if raw is None:
                return None
            data = json.loads(raw)
            return OrchestratorPlanPreview.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_orchestrator_preview failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_orchestrator_preview(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 3 preview. Idempotent. Best-effort."""
        db = self._get_db()
        try:
            if hasattr(db, "delete_meta"):
                db.delete_meta(
                    objective_orchestrator_preview_key(objective_id)
                )
                return True
            self._fallback_delete(
                objective_orchestrator_preview_key(objective_id)
            )
            return True
        except Exception:
            return False
        finally:
            self._close_db(db)

    # ── Phase 4A Policy / Approval Gates methods ─────────────

    def set_objective_policy_decision(
        self, decision: PolicyDecision
    ) -> None:
        """Persist a Phase 4A PolicyDecision.

        Writes to ``state_meta[objective_policy_decision:<oid>]``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(decision.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_policy_decision_key(decision.objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_policy_decision failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_policy_decision(
        self, objective_id: str
    ) -> PolicyDecision | None:
        """Load a Phase 4A PolicyDecision for the given objective_id,
        or ``None`` when no row exists."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_policy_decision_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return PolicyDecision.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_policy_decision failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_policy_decision(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 4A PolicyDecision. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion; ``False``
        otherwise. Used by ``policy_rollback``.
        """
        db = self._get_db()
        try:
            key = objective_policy_decision_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    def set_objective_approval_request(
        self, request: ApprovalRequest
    ) -> None:
        """Persist a Phase 4A ApprovalRequest.

        Writes to ``state_meta[objective_approval_request:<oid>]``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(request.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_approval_request_key(request.objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_approval_request failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_approval_request(
        self, objective_id: str
    ) -> ApprovalRequest | None:
        """Load a Phase 4A ApprovalRequest for the given objective_id,
        or ``None`` when no row exists."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_approval_request_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return ApprovalRequest.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_approval_request failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_approval_request(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 4A ApprovalRequest. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion; ``False``
        otherwise. Used by ``policy_rollback``.
        """
        db = self._get_db()
        try:
            key = objective_approval_request_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    # ── Phase 7 B1 Knowledge Discovery storage methods (Gate C; default OFF) ──

    def save_evidence_pack(self, objective_id: str, pack_dict: dict) -> str:
        """Persist a B1 EvidencePack (as dict) for an objective.

        Writes to ``state_meta[objective_knowledge_discovery:<oid>:v2]``.
        Returns the state_meta key written.

        The caller is responsible for passing an EvidencePack-like dict
        (e.g. ``EvidencePack.to_dict()``). The engine routes the
        call through its injected storage; this public method exists
        for direct callers and for tests.
        """
        db = self._get_db()
        try:
            payload = json.dumps(pack_dict, default=str, sort_keys=True)
            key = objective_evidence_pack_key(objective_id)
            db.set_meta(key, payload)
            return key
        except Exception as exc:
            raise StateStorageError(f"save_evidence_pack failed: {exc}") from exc
        finally:
            self._close_db(db)

    def load_evidence_pack(self, objective_id: str) -> dict | None:
        """Load the B1 EvidencePack dict for an objective, or ``None``.

        Tolerant: if the stored value is a JSON string it is parsed; if
        already a dict it is returned as a copy; if missing or
        unparseable, returns ``None``.
        """
        db = self._get_db()
        try:
            raw = db.get_meta(objective_evidence_pack_key(objective_id))
            if raw is None:
                return None
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except ValueError:
                    return None
            if isinstance(raw, dict):
                return dict(raw)
            return None
        except Exception as exc:
            raise StateStorageError(f"load_evidence_pack failed: {exc}") from exc
        finally:
            self._close_db(db)

    def delete_evidence_pack(self, objective_id: str) -> bool:
        """Delete the B1 EvidencePack for an objective. Idempotent.

        Returns ``True`` if a pack existed before deletion; ``False``
        otherwise. Used by rollback paths.
        """
        db = self._get_db()
        try:
            key = objective_evidence_pack_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    # ── Phase 4B Kanban Apply methods ───────────────────

    def set_objective_kanban_apply(
        self, result: KanbanApplyResult
    ) -> None:
        """Persist a Phase 4B KanbanApplyResult.

        Writes to ``state_meta[objective_kanban_apply:<oid>]``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(result.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_kanban_apply_key(result.objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_kanban_apply failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_kanban_apply(
        self, objective_id: str
    ) -> KanbanApplyResult | None:
        """Load a Phase 4B KanbanApplyResult, or ``None`` when missing."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_kanban_apply_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return KanbanApplyResult.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_kanban_apply failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_kanban_apply(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 4B KanbanApplyResult. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion.
        """
        db = self._get_db()
        try:
            key = objective_kanban_apply_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    def set_objective_kanban_tasks(
        self,
        objective_id: str,
        task_ids: tuple,
    ) -> None:
        """Persist a Phase 4B task list (parallel to the apply record).

        Writes to ``state_meta[objective_kanban_tasks:<oid>]``.
        Stored as a JSON dict ``{"objective_id": ..., "task_ids": [...]}``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(
                {
                    "objective_id": str(objective_id),
                    "task_ids": list(task_ids),
                },
                default=str,
                sort_keys=True,
            )
            db.set_meta(
                objective_kanban_tasks_key(objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_kanban_tasks failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_kanban_tasks(
        self, objective_id: str
    ) -> tuple | None:
        """Load a Phase 4B task list, or ``None`` when missing.

        Returns the raw ``task_ids`` tuple (or ``None``).
        """
        db = self._get_db()
        try:
            raw = db.get_meta(objective_kanban_tasks_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return tuple(data.get("task_ids") or ())
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_kanban_tasks failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    # ── Phase 7 Objective Recovery storage ───────────────────

    def set_objective_recovery_diagnosis(
        self, record: RecoveryDiagnosis
    ) -> None:
        """Persist a Phase 7 RecoveryDiagnosis for an objective.

        Writes to ``state_meta[objective_recovery_diagnosis:<oid>]``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(record.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_recovery_diagnosis_key(record.objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_recovery_diagnosis failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_recovery_diagnosis(
        self, objective_id: str
    ) -> RecoveryDiagnosis | None:
        """Load a Phase 7 RecoveryDiagnosis, or ``None`` when no row exists."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_recovery_diagnosis_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return RecoveryDiagnosis.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_recovery_diagnosis failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_recovery_diagnosis(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 7 RecoveryDiagnosis. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion.
        """
        db = self._get_db()
        try:
            key = objective_recovery_diagnosis_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    def set_objective_recovery_plan(
        self,
        objective_id: str,
        plan: RecoveryPlanPreview,
    ) -> None:
        """Persist a Phase 7 RecoveryPlanPreview for an objective.

        Writes to ``state_meta[objective_recovery_plan:<oid>]``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(plan.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_recovery_plan_key(objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_recovery_plan failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_recovery_plan(
        self, objective_id: str
    ) -> RecoveryPlanPreview | None:
        """Load a Phase 7 RecoveryPlanPreview, or ``None`` when no row exists."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_recovery_plan_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return RecoveryPlanPreview.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_recovery_plan failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_recovery_plan(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 7 RecoveryPlanPreview. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion.
        """
        db = self._get_db()
        try:
            key = objective_recovery_plan_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    # ── Phase 6 Success Evaluator storage ────────────────────

    def set_objective_evaluation(
        self, record: EvaluationReport
    ) -> None:
        """Persist a Phase 6 EvaluationReport for an objective.

        Writes to ``state_meta[objective_evaluation:<oid>]``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(record.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_evaluation_key(record.objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_evaluation failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_evaluation(
        self, objective_id: str
    ) -> EvaluationReport | None:
        """Load a Phase 6 EvaluationReport, or ``None`` when no row exists."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_evaluation_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return EvaluationReport.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_evaluation failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_evaluation(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 6 EvaluationReport. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion.
        """
        db = self._get_db()
        try:
            key = objective_evaluation_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    def set_objective_success_report(
        self,
        objective_id: str,
        report: SuccessReport,
    ) -> None:
        """Persist a Phase 6 slim SuccessReport for an objective.

        Writes to ``state_meta[objective_success_report:<oid>]``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(report.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_success_report_key(objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_success_report failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_success_report(
        self, objective_id: str
    ) -> SuccessReport | None:
        """Load a Phase 6 slim SuccessReport, or ``None`` when no row exists."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_success_report_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return SuccessReport.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_success_report failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_success_report(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 6 slim SuccessReport. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion.
        """
        db = self._get_db()
        try:
            key = objective_success_report_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    # ── Phase 5 Worker Dispatch storage ────────────────────────────

    def set_objective_worker_dispatch(
        self, record: WorkerDispatchResult
    ) -> None:
        """Persist a Phase 5 WorkerDispatchResult for an objective.

        Writes to ``state_meta[objective_worker_dispatch:<oid>]``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(record.to_dict(), default=str, sort_keys=True)
            db.set_meta(
                objective_worker_dispatch_key(record.objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_worker_dispatch failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_worker_dispatch(
        self, objective_id: str
    ) -> WorkerDispatchResult | None:
        """Load a Phase 5 WorkerDispatchResult, or ``None`` when no row exists."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_worker_dispatch_key(objective_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return WorkerDispatchResult.from_dict(data)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_worker_dispatch failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_worker_dispatch(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 5 WorkerDispatchResult. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion.
        """
        db = self._get_db()
        try:
            key = objective_worker_dispatch_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)

    def set_objective_worker_dispatch_tasks(
        self,
        objective_id: str,
        task_ids,
        dispatch_fingerprint: str,
    ) -> None:
        """Persist the task_ids list of a Phase 5 worker dispatch.

        Writes to ``state_meta[objective_worker_dispatch_tasks:<oid>]``
        as a JSON object ``{task_ids: [...], dispatch_fingerprint: ...}``.
        """
        db = self._get_db()
        try:
            payload = json.dumps(
                {
                    "objective_id": objective_id,
                    "task_ids": list(task_ids),
                    "dispatch_fingerprint": dispatch_fingerprint,
                    "created_at": _now_iso8601(),
                },
                default=str,
                sort_keys=True,
            )
            db.set_meta(
                objective_worker_dispatch_tasks_key(objective_id),
                payload,
            )
        except Exception as exc:
            raise StateStorageError(
                f"set_objective_worker_dispatch_tasks failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def get_objective_worker_dispatch_tasks(
        self, objective_id: str
    ) -> dict | None:
        """Load the task_ids list of a Phase 5 worker dispatch, or ``None``."""
        db = self._get_db()
        try:
            raw = db.get_meta(objective_worker_dispatch_tasks_key(objective_id))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            raise StateStorageError(
                f"get_objective_worker_dispatch_tasks failed: {exc}"
            ) from exc
        finally:
            self._close_db(db)

    def delete_objective_worker_dispatch_tasks(
        self, objective_id: str
    ) -> bool:
        """Delete the task_ids list of a Phase 5 worker dispatch. Idempotent."""
        db = self._get_db()
        try:
            key = objective_worker_dispatch_tasks_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)


    def delete_objective_kanban_tasks(
        self, objective_id: str
    ) -> bool:
        """Delete a Phase 4B task list. Idempotent. Best-effort.

        Returns ``True`` only if the row existed before deletion.
        """
        db = self._get_db()
        try:
            key = objective_kanban_tasks_key(objective_id)
            existed = db.get_meta(key) is not None
            if hasattr(db, "delete_meta"):
                db.delete_meta(key)
            else:
                self._fallback_delete(key)
            return existed
        except Exception:
            return False
        finally:
            self._close_db(db)
