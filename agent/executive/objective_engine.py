"""ObjectiveEngine — Phase 1 standalone state machine.

Accepts an objective_text, normalizes, classifies, discovers,
generates a contract, persists to state_meta. Does NOT execute, does
NOT call Planner, Orchestrator, GoalManager, or any LLM.

Default-off. The engine is enabled via resolve_v2_enabled().
"""

from __future__ import annotations

import logging
from typing import Any

from .contract import build_execution_contract_v1
from .flag import resolve_v2_enabled
from .normalizer import normalize_objective
from .capability_discovery_p0_p1 import discover_capabilities_p0_p1
from .state_storage import ObjectiveStateStorage
from .types import (
    ObjectiveState,
    ObjectiveStateData,
    new_uuid,
    now_iso8601,
)

logger = logging.getLogger(__name__)


class PermissionError_(Exception):  # noqa: N801
    """Raised when the engine is disabled."""


class StateTransitionError(Exception):
    """Raised when a transition is invalid."""


class ObjectiveEngine:
    """Standalone Objective Engine for Phase 1.

    Pipeline: submit -> normalize -> classify -> discover ->
    generate_contract -> persist.
    """

    def __init__(
        self,
        *,
        user_id: str,
        enabled: bool | None = None,
        storage: ObjectiveStateStorage | None = None,
        agent: Any | None = None,
    ) -> None:
        self._user_id = user_id
        self._enabled = (
            enabled if enabled is not None else resolve_v2_enabled(agent)
        )
        self._storage = storage or ObjectiveStateStorage()
        self._states: dict[str, ObjectiveStateData] = {}
        self._transition_log: list[dict] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def user_id(self) -> str:
        return self._user_id

    def _new_state(
        self, objective_text: str, constraints: list[str] | None
    ) -> ObjectiveStateData:
        now = now_iso8601()
        return ObjectiveStateData(
            objective_id=new_uuid(),
            state=ObjectiveState.DRAFT,
            objective_text=objective_text,
            constraints=list(constraints or []),
            user_id=self._user_id,
            created_at=now,
            last_transition_at=now,
            last_transition_id=new_uuid(),
        )

    def _transition(
        self,
        state: ObjectiveStateData,
        new_state: ObjectiveState,
        *,
        patch: dict | None = None,
    ) -> bool:
        """Apply a state transition. Returns True if applied, False if
        already in new_state (idempotent)."""
        if state.state == new_state:
            return False
        state.state = new_state
        state.last_transition_at = now_iso8601()
        state.last_transition_id = new_uuid()
        if patch:
            for k, v in patch.items():
                setattr(state, k, v)
        self._transition_log.append({
            "objective_id": state.objective_id,
            "new_state": new_state.value,
            "at": state.last_transition_at,
            "transition_id": state.last_transition_id,
        })
        logger.debug(
            "objective %s -> %s", state.objective_id, new_state.value
        )
        return True

    def _fail(self, state: ObjectiveStateData, error: str) -> None:
        state.last_error = error
        state.state = ObjectiveState.FAILED
        state.last_transition_at = now_iso8601()
        state.last_transition_id = new_uuid()
        logger.warning(
            "objective %s FAILED: %s", state.objective_id, error
        )

    def submit(
        self, objective_text: str, *, constraints: list[str] | None = None
    ) -> str:
        """Submit a new objective. Returns objective_id.

        Raises PermissionError_ if the engine is disabled.
        Raises ValueError if the input is invalid.
        """
        if not self._enabled:
            raise PermissionError_(
                "Executive v2 is disabled. Set HERMES_EXECUTIVE_V2_ENABLED=1 "
                "or agent._executive_v2_enabled = True to enable."
            )
        if not objective_text or not objective_text.strip():
            raise ValueError("objective_text must be non-empty")
        if len(objective_text) > 10_000:
            objective_text = objective_text[:10_000]
        if not self._user_id:
            raise ValueError("user_id must be non-empty")

        state = self._new_state(objective_text, constraints)
        self._states[state.objective_id] = state
        return state.objective_id

    def normalize(self, objective_id: str) -> None:
        """DRAFT -> NORMALIZED."""
        state = self._require_state(objective_id)
        if state.state != ObjectiveState.DRAFT:
            return
        try:
            from .types import NormalizedObjective as _N
            normalized = normalize_objective(
                state.objective_text,
                constraints=state.constraints,
                user_id=self._user_id,
            )
            # Serialize manually to avoid asdict/tuple issues; use
            # __dict__ which respects frozen=True and avoids recursion.
            norm_dict = {
                "objective_id": normalized.objective_id,
                "goal_class": normalized.goal_class.value,
                "constraints": list(normalized.constraints),
                "success_criteria": list(normalized.success_criteria),
                "human_constraints": list(normalized.human_constraints),
                "approval_requirements": list(normalized.approval_requirements),
                "risk_profile": normalized.risk_profile.value,
                "estimated_complexity": normalized.estimated_complexity.value,
                "knowledge_requirements": list(normalized.knowledge_requirements),
                "execution_requirements": dict(normalized.execution_requirements),
                "created_at": normalized.created_at,
                "created_by": normalized.created_by,
                "parent_objective_id": normalized.parent_objective_id,
                "session_id": normalized.session_id,
                "fingerprint": normalized.fingerprint,
                "schema_version": normalized.schema_version,
            }
            self._transition(
                state,
                ObjectiveState.NORMALIZED,
                patch={
                    "normalized": norm_dict,
                    "fingerprint": normalized.fingerprint,
                },
            )
        except Exception as exc:
            self._fail(state, f"normalize: {exc}")

    def classify(self, objective_id: str) -> None:
        """NORMALIZED -> CLASSIFIED."""
        state = self._require_state(objective_id)
        if state.state != ObjectiveState.NORMALIZED:
            return
        try:
            from .types import NormalizedObjective, ClassifiedObjective
            from .normalizer import tokenize as _tokenize
            from .classifier import classify_objective as _classify
            assert state.normalized is not None
            normalized = NormalizedObjective(**state.normalized)
            tokens = _tokenize(state.objective_text)
            classified = _classify(tokens)
            self._transition(
                state,
                ObjectiveState.CLASSIFIED,
                patch={"classified": classified.__dict__},
            )
        except Exception as exc:
            self._fail(state, f"classify: {exc}")

    def discover(self, objective_id: str) -> None:
        """CLASSIFIED -> DISCOVERED."""
        state = self._require_state(objective_id)
        if state.state != ObjectiveState.CLASSIFIED:
            return
        try:
            from .types import NormalizedObjective
            assert state.normalized is not None
            normalized = NormalizedObjective(**state.normalized)
            discovered = discover_capabilities_p0_p1(
                normalized, objective_id=objective_id
            )
            self._transition(
                state,
                ObjectiveState.DISCOVERED,
                patch={"discovered": discovered.__dict__},
            )
        except Exception as exc:
            self._fail(state, f"discover: {exc}")

    def generate_contract(self, objective_id: str) -> None:
        """DISCOVERED -> CONTRACT_DRAFT."""
        state = self._require_state(objective_id)
        if state.state != ObjectiveState.DISCOVERED:
            return
        try:
            from .types import (
                ClassifiedObjective,
                CapabilityDiscovery,
                NormalizedObjective,
            )
            assert state.normalized is not None
            assert state.classified is not None
            assert state.discovered is not None
            normalized = NormalizedObjective(**state.normalized)
            classified = ClassifiedObjective(**state.classified)
            discovered = CapabilityDiscovery(**state.discovered)
            contract = build_execution_contract_v1(
                normalized, classified, discovered, user_id=self._user_id
            )
            self._transition(
                state,
                ObjectiveState.CONTRACT_DRAFT,
                patch={"contract": contract.__dict__},
            )
        except Exception as exc:
            self._fail(state, f"generate_contract: {exc}")

    def persist(self, objective_id: str) -> None:
        """CONTRACT_DRAFT -> PERSISTED. Save to state_meta."""
        state = self._require_state(objective_id)
        if state.state != ObjectiveState.CONTRACT_DRAFT:
            return
        try:
            self._storage.save(state)
            self._transition(state, ObjectiveState.PERSISTED)
        except Exception as exc:
            self._fail(state, f"persist: {exc}")

    def get_state(self, objective_id: str) -> ObjectiveStateData:
        return self._require_state(objective_id)

    def list_active(self) -> list[str]:
        return list(self._states.keys())

    def list_persisted(self) -> list[str]:
        try:
            return self._storage.list_active()
        except Exception:
            return []

    def archive(self, objective_id: str) -> None:
        try:
            self._storage.archive(objective_id)
        except Exception:
            pass
        self._states.pop(objective_id, None)

    def _require_state(self, objective_id: str) -> ObjectiveStateData:
        if not self._enabled:
            raise PermissionError_(
                "Executive v2 is disabled."
            )
        if objective_id not in self._states:
            try:
                loaded = self._storage.load(objective_id)
                if loaded is not None:
                    self._states[objective_id] = loaded
            except Exception:
                pass
        if objective_id not in self._states:
            raise StateTransitionError(
                f"unknown objective_id: {objective_id}"
            )
        return self._states[objective_id]

    def run_pipeline(
        self,
        objective_text: str,
        *,
        constraints: list[str] | None = None,
        persist_to_state_meta: bool = False,
    ) -> str:
        """Run the full pipeline: submit -> normalize -> classify ->
        discover -> generate_contract. Optionally persist.

        Returns the objective_id.
        """
        oid = self.submit(objective_text, constraints=constraints)
        self.normalize(oid)
        if self._states[oid].state == ObjectiveState.NORMALIZED:
            self.classify(oid)
        if self._states[oid].state == ObjectiveState.CLASSIFIED:
            self.discover(oid)
        if self._states[oid].state == ObjectiveState.DISCOVERED:
            self.generate_contract(oid)
        if persist_to_state_meta and self._states[oid].state == ObjectiveState.CONTRACT_DRAFT:
            self.persist(oid)
        return oid
