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
    objective_evidence_pack_key,
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
        evidence_engine: Any | None = None,  # EvidencePackEngine | None (B1 Gate C)
    ) -> None:
        self._user_id = user_id
        self._enabled = (
            enabled if enabled is not None else resolve_v2_enabled(agent)
        )
        self._storage = storage or ObjectiveStateStorage()
        self._states: dict[str, ObjectiveStateData] = {}
        self._transition_log: list[dict] = []
        # ── B1 Knowledge Discovery wiring (Gate C; default OFF) ──
        # The engine stays inert when ``evidence_engine is None`` or
        # when the env-var / per-instance flag is not truthy. With
        # both conditions unmet, ``discover_evidence_pack`` is a no-op
        # and ``run_pipeline(..., evidence_pack=True)`` short-circuits.
        try:
            from .knowledge_discovery.flag import (
                resolve_knowledge_discovery_enabled,
            )
            self._evidence_discovery_enabled = bool(
                resolve_knowledge_discovery_enabled(agent)
                and evidence_engine is not None
            )
        except Exception:
            self._evidence_discovery_enabled = False
        self._evidence_engine = evidence_engine

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

    def discover_evidence_pack(self, objective_id: str) -> dict | None:
        """DISCOVERED -> DISCOVERED (no state change) with pack persisted.

        Optional Phase 1.5 step: invoke the injected EvidencePackEngine
        to build a knowledge pack for the objective. Returns the pack
        dict (as written to state_meta) on success; ``None`` on no-op
        (feature off) or engine error (state transitions to FAILED).

        Preconditions:
        - ``self._evidence_discovery_enabled`` is True
        - state is in ``ObjectiveState.DISCOVERED`` (or beyond)

        Postconditions (success):
        - ``state.evidence_pack_ref`` set to the state_meta key
        - ``state.evidence_pack_summary`` set to the pack summary
          (truncated to 200 chars)
        - state is NOT transitioned (still DISCOVERED)

        Postconditions (disabled): no-op; returns None; state unchanged.
        Postconditions (error): ``state.last_error`` set; state -> FAILED.
        """
        state = self._require_state(objective_id)
        if not self._evidence_discovery_enabled or self._evidence_engine is None:
            return None
        if state.state not in (
            ObjectiveState.DISCOVERED,
            ObjectiveState.CONTRACT_DRAFT,
            ObjectiveState.PERSISTED,
        ):
            return None
        try:
            from .types import NormalizedObjective, ClassifiedObjective
            from .knowledge_discovery import (
                EvidencePack,
                KnowledgeQuery,
            )
            assert state.normalized is not None
            assert state.classified is not None
            classified_dict = state.classified or {}
            normalized = NormalizedObjective(**state.normalized)
            # Build a KnowledgeQuery from the normalized/classified state.
            try:
                goal_class_val = (
                    classified_dict.get("goal_class", "OTHER")
                )
            except Exception:
                goal_class_val = "OTHER"
            try:
                risk_profile_val = (
                    classified_dict.get("risk_profile", "low")
                )
            except Exception:
                risk_profile_val = "low"
            try:
                complexity_val = (
                    classified_dict.get("estimated_complexity", "S")
                )
            except Exception:
                complexity_val = "S"
            query = KnowledgeQuery(
                objective_id=state.objective_id,
                objective_text=state.objective_text,
                goal_class=goal_class_val,
                risk_profile=risk_profile_val,
                complexity=complexity_val,
            )
            pack = self._evidence_engine.discover(
                objective_id=state.objective_id,
                objective_text=state.objective_text,
                goal_class=goal_class_val,
                risk_profile=risk_profile_val,
                complexity=complexity_val,
            )
            # Idempotency: if the engine already wrote a pack for this
            # query, do NOT overwrite. The engine's storage has the
            # canonical copy; just refresh the in-state reference.
            is_idempotent = bool(
                getattr(pack, "is_idempotent_reuse", False)
            )
            if not is_idempotent:
                # Persist via storage; tolerate storage failure gracefully.
                try:
                    pack_dict = (
                        pack.to_dict() if hasattr(pack, "to_dict") else dict(pack)
                    )
                    self._storage.save_evidence_pack(state.objective_id, pack_dict)
                except Exception as exc:  # noqa: BLE001
                    self._fail(state, f"evidence_pack: {exc}")
                    return None
            # Populate state fields.
            summary_text = getattr(pack, "summary_text", "") or ""
            state.evidence_pack_ref = objective_evidence_pack_key(
                state.objective_id
            )
            state.evidence_pack_summary = summary_text[:200]
            return (
                pack.to_dict() if hasattr(pack, "to_dict") else dict(pack)
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(state, f"evidence_pack: {exc}")
            return None

    def generate_contract(self, objective_id: str) -> None:
        """DISCOVERED -> CONTRACT_DRAFT.

        When a B1 EvidencePack was discovered for this objective, the
        pack is passed to ``build_execution_contract_v1`` so the
        contract carries the new evidence_pack_* fields and the
        appropriate approval requirements. When the feature is off
        (default), the call is byte-identical to the pre-wiring
        version.
        """
        state = self._require_state(objective_id)
        if state.state != ObjectiveState.DISCOVERED:
            return
        try:
            from .types import (
                ClassifiedObjective,
                CapabilityDiscovery,
                NormalizedObjective,
            )
            from .knowledge_discovery import EvidencePack as _EP
            assert state.normalized is not None
            assert state.classified is not None
            assert state.discovered is not None
            normalized = NormalizedObjective(**state.normalized)
            classified = ClassifiedObjective(**state.classified)
            discovered = CapabilityDiscovery(**state.discovered)
            # Optional evidence pack (Gate C; default None).
            evidence_pack_obj: Any = None
            if state.evidence_pack_summary is not None and self._evidence_discovery_enabled:
                # Reconstruct the EvidencePack object from state_meta
                # storage so the contract builder can read
                # .summary_text, .overall_confidence, etc.
                try:
                    pack_dict = self._storage.load_evidence_pack(
                        state.objective_id
                    )
                    if pack_dict:
                        evidence_pack_obj = _EP(
                            objective_id=pack_dict.get("objective_id", state.objective_id),
                            query_fingerprint=pack_dict.get("query_fingerprint", ""),
                            sources_queried=pack_dict.get("sources_queried", []),
                            sources_failed=pack_dict.get("sources_failed", []),
                            hits=pack_dict.get("hits", []),
                            citations=pack_dict.get("citations", []),
                            conflicts=pack_dict.get("conflicts", []),
                            missing_information=pack_dict.get("missing_information", []),
                            overall_freshness_score=pack_dict.get("overall_freshness_score", 0.0),
                            overall_confidence=pack_dict.get("overall_confidence", 0.0),
                            summary_text=pack_dict.get("summary_text", "")[:2000],
                            summary_fingerprint=pack_dict.get("summary_fingerprint", ""),
                            duration_ms=pack_dict.get("duration_ms", 0),
                            created_at=pack_dict.get("created_at", ""),
                            schema_version=pack_dict.get("schema_version", "evidence_pack.v1"),
                            is_idempotent_reuse=pack_dict.get("is_idempotent_reuse", False),
                            total_hits=pack_dict.get("total_hits", 0),
                        )
                except Exception:
                    evidence_pack_obj = None
            contract = build_execution_contract_v1(
                normalized, classified, discovered, user_id=self._user_id,
                evidence_pack=evidence_pack_obj,
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
        evidence_pack: bool = True,  # B1 Gate C; default True, gated by env+engine
    ) -> str:
        """Run the full pipeline: submit -> normalize -> classify ->
        discover -> [discover_evidence_pack] -> generate_contract.
        Optionally persist.

        Returns the objective_id. When the B1 feature is off
        (default), ``evidence_pack=True`` is a no-op and the
        pre-wiring behavior is preserved.
        """
        oid = self.submit(objective_text, constraints=constraints)
        self.normalize(oid)
        if self._states[oid].state == ObjectiveState.NORMALIZED:
            self.classify(oid)
        if self._states[oid].state == ObjectiveState.CLASSIFIED:
            self.discover(oid)
        # ── B1 evidence discovery (Gate C; default OFF) ──
        if (
            evidence_pack
            and self._evidence_discovery_enabled
            and self._states[oid].state == ObjectiveState.DISCOVERED
        ):
            self.discover_evidence_pack(oid)
            if self._states[oid].state == ObjectiveState.FAILED:
                return oid
        if self._states[oid].state == ObjectiveState.DISCOVERED:
            self.generate_contract(oid)
        if persist_to_state_meta and self._states[oid].state == ObjectiveState.CONTRACT_DRAFT:
            self.persist(oid)
        return oid
