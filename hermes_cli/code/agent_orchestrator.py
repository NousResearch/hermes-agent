#!/usr/bin/env python3
"""AgentOrchestrator state machine foundation for Hermes Code Mode."""

from __future__ import annotations

import time
import uuid
from typing import Any

from hermes_cli.code.artifact_ledger import ArtifactLedger
from hermes_cli.code.event_bus import get_code_event_bus
from hermes_state import SessionDB


class OrchestratorState:
    INTAKE = "intake"
    DISCOVERY = "discovery"
    PRODUCT_FRAMING = "product_framing"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    APPROVAL = "approval"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    REVIEW = "review"
    READY_FOR_PR = "ready_for_pr"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    ALL = frozenset(
        {
            INTAKE,
            DISCOVERY,
            PRODUCT_FRAMING,
            ARCHITECTURE,
            PLANNING,
            APPROVAL,
            IMPLEMENTATION,
            VALIDATION,
            REVIEW,
            READY_FOR_PR,
            COMPLETED,
            CANCELLED,
            FAILED,
        }
    )
    TERMINAL = frozenset({COMPLETED, CANCELLED, FAILED})


TRANSITIONS: dict[str, frozenset[str]] = {
    OrchestratorState.INTAKE: frozenset({OrchestratorState.DISCOVERY, OrchestratorState.PRODUCT_FRAMING, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.DISCOVERY: frozenset({OrchestratorState.PRODUCT_FRAMING, OrchestratorState.ARCHITECTURE, OrchestratorState.PLANNING, OrchestratorState.APPROVAL, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.PRODUCT_FRAMING: frozenset({OrchestratorState.ARCHITECTURE, OrchestratorState.PLANNING, OrchestratorState.APPROVAL, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.ARCHITECTURE: frozenset({OrchestratorState.PLANNING, OrchestratorState.APPROVAL, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.PLANNING: frozenset({OrchestratorState.APPROVAL, OrchestratorState.IMPLEMENTATION, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.APPROVAL: frozenset({OrchestratorState.IMPLEMENTATION, OrchestratorState.PLANNING, OrchestratorState.ARCHITECTURE, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.IMPLEMENTATION: frozenset({OrchestratorState.VALIDATION, OrchestratorState.APPROVAL, OrchestratorState.REVIEW, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.VALIDATION: frozenset({OrchestratorState.REVIEW, OrchestratorState.IMPLEMENTATION, OrchestratorState.APPROVAL, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.REVIEW: frozenset({OrchestratorState.READY_FOR_PR, OrchestratorState.IMPLEMENTATION, OrchestratorState.VALIDATION, OrchestratorState.COMPLETED, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.READY_FOR_PR: frozenset({OrchestratorState.COMPLETED, OrchestratorState.IMPLEMENTATION, OrchestratorState.CANCELLED, OrchestratorState.FAILED}),
    OrchestratorState.COMPLETED: frozenset(),
    OrchestratorState.CANCELLED: frozenset(),
    OrchestratorState.FAILED: frozenset(),
}


def validate_transition(from_state: str, to_state: str) -> tuple[bool, str]:
    if from_state not in OrchestratorState.ALL:
        return False, f"Unknown from_state: {from_state}"
    if to_state not in OrchestratorState.ALL:
        return False, f"Unknown to_state: {to_state}"
    if from_state in OrchestratorState.TERMINAL:
        return False, f"State {from_state} is terminal"
    if to_state not in TRANSITIONS.get(from_state, frozenset()):
        return False, f"Invalid transition {from_state} -> {to_state}"
    return True, ""


class AgentOrchestrator:
    def __init__(self, db_path=None):
        self._db_path = db_path

    def _db(self) -> SessionDB:
        return SessionDB(db_path=self._db_path) if self._db_path else SessionDB()

    def _ledger(self) -> ArtifactLedger:
        return ArtifactLedger(db_path=self._db_path)

    def _emit(self, event_type: str, payload: dict[str, Any], *, workspace_id: str | None = None, code_session_id: str | None = None, orchestrated_run_id: str | None = None) -> None:
        bus = get_code_event_bus(self._db_path)
        bus.publish(
            event_type,
            payload=payload,
            workspace_id=workspace_id,
            code_session_id=code_session_id,
            orchestrated_run_id=orchestrated_run_id,
            metadata={"source": "orchestrator"},
            source="orchestrator",
        )

    def create_run(
        self,
        *,
        title: str | None = None,
        goal: str | None = None,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        create_intake_artifact: bool = True,
    ) -> dict[str, Any]:
        now = time.time()
        run = {
            "id": str(uuid.uuid4()),
            "title": title or "Untitled Run",
            "goal": goal or "",
            "state": OrchestratorState.INTAKE,
            "workspace_id": workspace_id,
            "code_session_id": code_session_id,
            "current_phase": OrchestratorState.INTAKE,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        db = self._db()
        try:
            db.create_code_orchestrated_run(run)
        finally:
            db.close()
        self._emit(
            "code.orchestrator.run_created",
            payload={"run_id": run["id"], "state": run["state"], "title": run["title"]},
            workspace_id=workspace_id,
            code_session_id=code_session_id,
            orchestrated_run_id=run["id"],
        )
        self._emit(
            "orchestrator.run.created",
            payload={"run_id": run["id"], "state": run["state"], "title": run["title"]},
            workspace_id=workspace_id,
            code_session_id=code_session_id,
            orchestrated_run_id=run["id"],
        )

        if create_intake_artifact and goal:
            self._ledger().create_artifact(
                "task_intake",
                goal,
                title=title or "Task Intake",
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                orchestrated_run_id=run["id"],
            )
        return run

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        db = self._db()
        try:
            return db.get_code_orchestrated_run(run_id)
        finally:
            db.close()

    def list_runs(
        self,
        *,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        state: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        db = self._db()
        try:
            return db.list_code_orchestrated_runs(
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                state=state,
                limit=limit,
                offset=offset,
            )
        finally:
            db.close()

    def transition_run(
        self,
        run_id: str,
        to_state: str,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        db = self._db()
        try:
            run = db.get_code_orchestrated_run(run_id)
            if not run:
                raise ValueError(f"Run not found: {run_id}")
            from_state = run["state"]
            valid, msg = validate_transition(from_state, to_state)
            if not valid:
                raise ValueError(msg)
            now = time.time()
            updates: dict[str, Any] = {
                "state": to_state,
                "current_phase": to_state,
                "updated_at": now,
            }
            if metadata:
                current_metadata = run.get("metadata") or {}
                if isinstance(current_metadata, dict):
                    current_metadata = dict(current_metadata)
                    current_metadata.update(metadata)
                    updates["metadata"] = current_metadata
            if to_state in OrchestratorState.TERMINAL:
                updates["completed_at"] = now
            db.update_code_orchestrated_run(run_id, updates)
            db.add_code_run_transition(
                {
                    "id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "from_state": from_state,
                    "to_state": to_state,
                    "reason": reason or f"{from_state} -> {to_state}",
                    "created_at": now,
                }
            )
            updated = db.get_code_orchestrated_run(run_id)
            out = updated or run
        finally:
            db.close()
        self._emit(
            "code.orchestrator.transitioned",
            payload={
                "run_id": run_id,
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
            },
            workspace_id=out.get("workspace_id"),
            code_session_id=out.get("code_session_id"),
            orchestrated_run_id=run_id,
        )
        self._emit(
            "orchestrator.run.transitioned",
            payload={
                "run_id": run_id,
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
            },
            workspace_id=out.get("workspace_id"),
            code_session_id=out.get("code_session_id"),
            orchestrated_run_id=run_id,
        )
        if to_state == OrchestratorState.COMPLETED:
            self._emit(
                "code.orchestrator.completed",
                payload={"run_id": run_id, "state": to_state, "reason": reason},
                workspace_id=out.get("workspace_id"),
                code_session_id=out.get("code_session_id"),
                orchestrated_run_id=run_id,
            )
        if to_state == OrchestratorState.FAILED:
            self._emit(
                "code.orchestrator.failed",
                payload={"run_id": run_id, "state": to_state, "reason": reason},
                workspace_id=out.get("workspace_id"),
                code_session_id=out.get("code_session_id"),
                orchestrated_run_id=run_id,
            )
        return out

    def list_transitions(self, run_id: str) -> list[dict[str, Any]]:
        db = self._db()
        try:
            return db.list_code_run_transitions(run_id=run_id)
        finally:
            db.close()
