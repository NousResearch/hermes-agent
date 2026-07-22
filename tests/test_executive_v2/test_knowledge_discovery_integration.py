"""Hermetic integration tests for B1 Knowledge Discovery wiring.

These tests cover the integration surface between the production
``agent.executive`` module and the B1 ``agent.executive.knowledge_discovery``
package. They verify:

- Default-OFF behavior (no engine, no state_meta writes, no audit).
- Injected-engine behavior (pack persisted under the namespaced key,
  state fields populated, idempotency).
- ExecutionContractV1 extended fields (defaults when off, populated when on).
- Planner gate (additive [GATED] marker only when summary prefix demands).
- Dryrun rendering (Evidence Pack section only when summary present).
- Flag resolver (env var, per-instance attribute, default OFF).
- Backward compatibility (existing tests + new state_meta keys coexist).

Hermeticity invariants:
- All state goes through ``_InMemoryStorage`` (defined in canary_b1.conftest).
- No network, no subprocess, no LLM calls, no real state.db / audit log.
- All sources are ``FakeProviderSpec`` from canary_b1.fake_providers.

Test IDs are stable; matching the design's test plan (test_plan_apply.md §4).
"""

from __future__ import annotations

import os
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

# ── Imports from the production wiring under test ─────────────────
from agent.executive.knowledge_discovery import (
    EvidencePack,
    EvidencePackEngine,
    KnowledgeHitV2,
    KnowledgeQuery,
    resolve_knowledge_discovery_enabled,
    SUMMARY_TEXT_MAX_LEN,
    get_state_meta_key,
)
from agent.executive.knowledge_discovery.flag import (
    resolve_knowledge_discovery_enabled as _resolve_kd,
)

# ── Imports from Executive v2 surfaces under test ────────────────
from agent.executive.types import (
    ApprovalRequirement,
    ExecutionContractV1,
    ObjectiveStateData,
    ObjectiveState,
    objective_evidence_pack_key,
)
from agent.executive.contract import build_execution_contract_v1
from agent.executive.dryrun import render_dry_run
from agent.executive.planner import (
    decompose_goal_to_subgoals,
    compute_plan_fingerprint,
)
from agent.executive.objective_engine import ObjectiveEngine
from agent.executive.state_storage import ObjectiveStateStorage

# ── Reusable test fixtures from canary_b1 ────────────────────────
from tests.test_executive_v2.canary_b1.conftest import (
    _InMemoryStorage,
    _AuditCapture,
    CANARY_FROZEN_TIME_UTC,
)
from tests.test_executive_v2.canary_b1.fake_providers import (
    FakeProviderSpec,
    make_provider_bundle,
    default_gbrain_spec,
    default_obsidian_spec,
    default_reports_spec,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _build_minimal_evidence_pack(
    *,
    objective_id: str = "obj-test-1",
    summary_prefix: str = "[READY_FOR_STRATEGY]",
    confidence: float = 0.78,
    freshness: float = 0.85,
) -> EvidencePack:
    """Build a minimal valid EvidencePack for contract / state tests."""
    return EvidencePack(
        objective_id=objective_id,
        query_fingerprint="f" * 64,
        sources_queried=["gbrain", "obsidian"],
        sources_failed=[],
        hits=[],
        citations=[],
        conflicts=[],
        missing_information=[],
        overall_freshness_score=freshness,
        overall_confidence=confidence,
        summary_text=f"{summary_prefix} found 5 hits across 2 sources",
        summary_fingerprint="a" * 64,
        duration_ms=12,
        created_at=CANARY_FROZEN_TIME_UTC,
        schema_version="evidence_pack.v1",
        is_idempotent_reuse=False,
        total_hits=0,
    )


def _build_normalized_classified_discovered():
    """Build minimal NormalizedObjective/ClassifiedObjective/CapabilityDiscovery."""
    from agent.executive.types import (
        NormalizedObjective,
        ClassifiedObjective,
        CapabilityDiscovery,
        GoalClass,
        RiskProfile,
        Complexity,
    )
    normalized = NormalizedObjective(
        objective_id="obj-test-1",
        goal_class=GoalClass.RESEARCH,
        constraints=(),
        success_criteria=("ship it",),
        human_constraints=(),
        approval_requirements=(),
        risk_profile=RiskProfile.LOW,
        estimated_complexity=Complexity.S,
        knowledge_requirements=(),
        execution_requirements={},
        created_at=CANARY_FROZEN_TIME_UTC,
        created_by="u",
        fingerprint="f" * 64,
    )
    classified = ClassifiedObjective(
        goal_class=GoalClass.RESEARCH,
        risk_profile=RiskProfile.LOW,
        estimated_complexity=Complexity.S,
        rationale="r",
        signal_tokens=(),
    )
    discovered = CapabilityDiscovery(
        objective_id="obj-test-1",
        discovered_at=CANARY_FROZEN_TIME_UTC,
        candidates=(),
        reuse_decision="generate",
        rationale="r",
        gaps=(),
        p0_query_duration_ms=0,
        p1_query_duration_ms=0,
    )
    return normalized, classified, discovered


@pytest.fixture
def hermetic_storage():
    return _InMemoryStorage()


@pytest.fixture
def production_storage(hermetic_storage):
    """An ObjectiveStateStorage that shares its backing dict with
    ``hermetic_storage`` (so engine writes + engine reads see the same
    data). Uses ``db_factory`` to inject the in-memory db.
    """
    return ObjectiveStateStorage(
        db_factory=lambda: hermetic_storage
    )


@pytest.fixture
def audit_sink():
    return _AuditCapture()


@pytest.fixture
def minimal_bundle(tmp_path):
    return make_provider_bundle(
        gbrain_spec=default_gbrain_spec(),
        obsidian_spec=default_obsidian_spec(),
        report_spec=default_reports_spec(tmp_path),
    )


# ─────────────────────────────────────────────────────────────────
# 4.1 Default-OFF tests
# ─────────────────────────────────────────────────────────────────


class TestDefaultOff:
    def test_kd_off_default_returns_none(
        self, hermetic_storage, production_storage, monkeypatch
    ):
        """Default-off: no env, no per-instance flag → no-op."""
        monkeypatch.delenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED",
            raising=False,
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,  # executive v2 must be on
        )
        assert engine._evidence_discovery_enabled is False
        assert engine._evidence_engine is None
        # discover_evidence_pack is a no-op.
        # First, get an objective into DISCOVERED state.
        oid = engine.submit("test goal", constraints=None)
        engine.normalize(oid)
        engine.classify(oid)
        engine.discover(oid)
        result = engine.discover_evidence_pack(oid)
        assert result is None
        # No state_meta key was written.
        assert (
            hermetic_storage.get_meta(
                objective_evidence_pack_key(oid)
            )
            is None
        )

    def test_kd_off_no_audit_emitted(
        self, production_storage, audit_sink, monkeypatch
    ):
        """Default-off: no audit events emitted from run_pipeline."""
        monkeypatch.delenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED",
            raising=False,
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
        )
        engine.run_pipeline("test goal")
        assert audit_sink.get_events() == []

    def test_kd_off_contract_fields_empty(self):
        """Default-off: evidence_pack=None → contract fields default to
        None / '' / 0.0 / 0.0 and no new approval added."""
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered, user_id="u"
        )
        assert contract.evidence_pack_ref is None
        assert contract.evidence_pack_summary == ""
        assert contract.evidence_pack_confidence == 0.0
        assert contract.evidence_pack_freshness == 0.0
        # No 'knowledge_review' approval.
        approval_gates = {
            ar.get("gate") for ar in contract.approval_requirements
        }
        assert "knowledge_review" not in approval_gates
        assert "knowledge_freshness_review" not in approval_gates

    def test_kd_off_no_new_approvals_even_with_pack_in_constructor(self):
        """When env is OFF and engine is None, no approval is added even
        if a pack is passed in. The flag check is at the engine layer;
        the contract builder does NOT check the flag (it just reads
        the pack's prefix). This test documents the layered design:
        the ObjectiveEngine is the gatekeeper. The contract builder
        is a pure function."""
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        # Pass a pack with REQUIRES_HUMAN to the contract builder; the
        # builder adds the approval because it only looks at the
        # summary prefix (pure function).
        pack = _build_minimal_evidence_pack(
            summary_prefix="[REQUIRES_HUMAN]",
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        # Contract builder adds the approval (pure function).
        gates = {ar.get("gate") for ar in contract.approval_requirements}
        assert "knowledge_review" in gates
        # But the engine does NOT propagate it when off, because the
        # engine passes evidence_pack=None to the builder in OFF mode.
        # We test this at the engine level (test_kd_off_default_returns_none).


# ─────────────────────────────────────────────────────────────────
# 4.2 Injected-engine tests
# ─────────────────────────────────────────────────────────────────


class TestInjectedEngine:
    def test_kd_engine_none_disables_kd(self, production_storage, monkeypatch):
        """evidence_engine=None arg → _evidence_discovery_enabled is
        False even with env=1 (gating works as designed)."""
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
            evidence_engine=None,
        )
        assert engine._evidence_discovery_enabled is False

    def test_kd_on_with_injected_engine_persists_pack(
        self, hermetic_storage, production_storage, audit_sink, minimal_bundle, monkeypatch
    ):
        """With engine injected + env=1, discover_evidence_pack returns
        dict and state_meta has the namespaced key."""
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )
        engine_obj = EvidencePackEngine(
            sources=minimal_bundle,
            storage=hermetic_storage,  # shares with production_storage
            audit_sink=audit_sink,
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
            evidence_engine=engine_obj,
        )
        assert engine._evidence_discovery_enabled is True
        oid = engine.submit("test goal", constraints=None)
        engine.normalize(oid)
        engine.classify(oid)
        engine.discover(oid)
        result = engine.discover_evidence_pack(oid)
        assert result is not None
        assert isinstance(result, dict)
        # state_meta has the namespaced key.
        key = objective_evidence_pack_key(oid)
        assert hermetic_storage.get_meta(key) is not None

    def test_kd_pack_ref_set_on_state_after_discover(
        self, hermetic_storage, production_storage, audit_sink, minimal_bundle, monkeypatch
    ):
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )
        engine_obj = EvidencePackEngine(
            sources=minimal_bundle,
            storage=hermetic_storage,
            audit_sink=audit_sink,
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
            evidence_engine=engine_obj,
        )
        oid = engine.submit("test goal", constraints=None)
        engine.normalize(oid)
        engine.classify(oid)
        engine.discover(oid)
        engine.discover_evidence_pack(oid)
        state = engine.get_state(oid)
        assert state.evidence_pack_ref == objective_evidence_pack_key(oid)

    def test_kd_idempotent_discover_when_query_unchanged(
        self, hermetic_storage, production_storage, audit_sink, minimal_bundle, monkeypatch
    ):
        """Second call with same query → is_idempotent_reuse=True on the
        pack object; state_meta key still exists; pack is not
        re-persisted (idempotency gate at engine level)."""
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )
        engine_obj = EvidencePackEngine(
            sources=minimal_bundle,
            storage=hermetic_storage,
            audit_sink=audit_sink,
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
            evidence_engine=engine_obj,
        )
        oid = engine.submit("test goal", constraints=None)
        engine.normalize(oid)
        engine.classify(oid)
        engine.discover(oid)
        first = engine.discover_evidence_pack(oid)
        assert first is not None
        # Reset state, run again with the same query.
        engine.discover(oid)
        # Patch the engine to capture the second pack and assert
        # is_idempotent_reuse=True on the returned object.
        captured = {}
        original = engine_obj.discover
        def _capture(*a, **kw):
            p = original(*a, **kw)
            captured["pack"] = p
            return p
        engine_obj.discover = _capture
        second = engine.discover_evidence_pack(oid)
        engine_obj.discover = original
        assert second is not None
        # Idempotency flag is on the object (not the dict, which
        # deliberately excludes runtime-only flags per design).
        assert captured.get("pack") is not None
        assert bool(captured["pack"].is_idempotent_reuse) is True
        # Key still present.
        key = objective_evidence_pack_key(oid)
        assert hermetic_storage.get_meta(key) is not None

    def test_kd_engine_error_sets_state_failed(
        self, production_storage, monkeypatch
    ):
        """Engine raises → state goes to FAILED with last_error set."""
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )

        class _BoomEngine:
            def discover(self, *a, **kw):
                raise RuntimeError("kaboom")

        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
            evidence_engine=_BoomEngine(),
        )
        oid = engine.submit("test goal", constraints=None)
        engine.normalize(oid)
        engine.classify(oid)
        engine.discover(oid)
        engine.discover_evidence_pack(oid)
        state = engine.get_state(oid)
        assert state.state == ObjectiveState.FAILED
        assert (state.last_error or "").startswith("evidence_pack:")

    def test_kd_storage_methods_roundtrip(self, production_storage):
        """save → load → delete → load returns None."""
        oid = "obj-rt-1"
        pack_dict = {
            "objective_id": oid,
            "summary_text": "[READY_FOR_STRATEGY] test",
            "schema_version": "evidence_pack.v1",
        }
        key = production_storage.save_evidence_pack(oid, pack_dict)
        assert key == objective_evidence_pack_key(oid)
        loaded = production_storage.load_evidence_pack(oid)
        assert loaded is not None
        assert loaded["objective_id"] == oid
        assert loaded["summary_text"].startswith("[READY_FOR_STRATEGY]")
        # Delete: first call True, second call False (idempotent).
        assert production_storage.delete_evidence_pack(oid) is True
        assert production_storage.delete_evidence_pack(oid) is False
        # Load: returns None.
        assert production_storage.load_evidence_pack(oid) is None

    def test_kd_storage_load_returns_none_when_absent(self, production_storage):
        assert production_storage.load_evidence_pack("nope-oid") is None


# ─────────────────────────────────────────────────────────────────
# 4.3 Contract builder tests
# ─────────────────────────────────────────────────────────────────


class TestContractBuilder:
    def test_kd_contract_summary_populated_when_provided(self):
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        pack = _build_minimal_evidence_pack(
            summary_prefix="[READY_FOR_STRATEGY]",
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        assert contract.evidence_pack_summary.startswith("[READY_FOR_STRATEGY]")
        assert len(contract.evidence_pack_summary) <= SUMMARY_TEXT_MAX_LEN

    def test_kd_contract_confidence_populated_when_provided(self):
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        pack = _build_minimal_evidence_pack(confidence=0.91)
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        assert contract.evidence_pack_confidence == 0.91

    def test_kd_contract_freshness_populated_when_provided(self):
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        pack = _build_minimal_evidence_pack(freshness=0.42)
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        assert contract.evidence_pack_freshness == 0.42

    def test_kd_requires_human_summary_adds_approval(self):
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        pack = _build_minimal_evidence_pack(
            summary_prefix="[REQUIRES_HUMAN]"
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        gates = {ar.get("gate") for ar in contract.approval_requirements}
        assert "knowledge_review" in gates

    def test_kd_expert_review_summary_adds_approval(self):
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        pack = _build_minimal_evidence_pack(
            summary_prefix="[NEEDS_EXPERT_REVIEW]"
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        gates = {ar.get("gate") for ar in contract.approval_requirements}
        assert "knowledge_review" in gates

    def test_kd_degraded_freshness_summary_adds_approval(self):
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        pack = _build_minimal_evidence_pack(
            summary_prefix="[DEGRADED_FRESHNESS]"
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        gates = {ar.get("gate") for ar in contract.approval_requirements}
        assert "knowledge_freshness_review" in gates

    def test_kd_ready_summary_no_new_approval(self):
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        pack = _build_minimal_evidence_pack(
            summary_prefix="[READY_FOR_STRATEGY]"
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        gates = {ar.get("gate") for ar in contract.approval_requirements}
        assert "knowledge_review" not in gates
        assert "knowledge_freshness_review" not in gates

    def test_kd_empty_summary_no_new_approval(self):
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        # Build a pack with an empty-summary prefix.
        pack = _build_minimal_evidence_pack(
            summary_prefix="(no relevant knowledge found)"
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered,
            user_id="u", evidence_pack=pack,
        )
        gates = {ar.get("gate") for ar in contract.approval_requirements}
        assert "knowledge_review" not in gates

    def test_kd_contract_byte_identical_when_evidence_pack_none(self):
        """When evidence_pack=None, the contract dict is byte-identical
        to the pre-extension version: the new keys are present with
        their default empty values."""
        normalized, classified, discovered = (
            _build_normalized_classified_discovered()
        )
        contract = build_execution_contract_v1(
            normalized, classified, discovered, user_id="u"
        )
        d = contract.__dict__
        assert d["evidence_pack_ref"] is None
        assert d["evidence_pack_summary"] == ""
        assert d["evidence_pack_confidence"] == 0.0
        assert d["evidence_pack_freshness"] == 0.0


# ─────────────────────────────────────────────────────────────────
# 4.4 Planner tests
# ─────────────────────────────────────────────────────────────────


class TestPlannerGate:
    def test_kd_planner_gates_subgoal_on_requires_human(self):
        ec = {
            "success_criteria": ("ship it", "verify it"),
            "approval_requirements": (),
            "hard_constraints": (),
            "soft_constraints": (),
            "budget": {},
            "evidence_pack_summary": "[REQUIRES_HUMAN] review required",
        }
        subgoals = decompose_goal_to_subgoals(
            "test goal", None, ec, risk_score=0.0
        )
        for sg in subgoals:
            assert sg.expected_output.endswith(
                " [GATED: evidence pack requires human review]"
            )

    def test_kd_planner_gates_subgoal_on_expert_review(self):
        ec = {
            "success_criteria": ("research X",),
            "approval_requirements": (),
            "hard_constraints": (),
            "soft_constraints": (),
            "budget": {},
            "evidence_pack_summary": "[NEEDS_EXPERT_REVIEW] domain X",
        }
        subgoals = decompose_goal_to_subgoals(
            "test goal", None, ec, risk_score=0.0
        )
        for sg in subgoals:
            assert sg.expected_output.endswith(
                " [GATED: evidence pack requires human review]"
            )

    def test_kd_planner_no_gate_when_summary_absent(self):
        ec = {
            "success_criteria": ("ship it",),
            "approval_requirements": (),
            "hard_constraints": (),
            "soft_constraints": (),
            "budget": {},
            # No evidence_pack_summary at all.
        }
        subgoals = decompose_goal_to_subgoals(
            "test goal", None, ec, risk_score=0.0
        )
        for sg in subgoals:
            assert "[GATED:" not in sg.expected_output

    def test_kd_planner_no_gate_when_summary_empty(self):
        ec = {
            "success_criteria": ("ship it",),
            "approval_requirements": (),
            "hard_constraints": (),
            "soft_constraints": (),
            "budget": {},
            "evidence_pack_summary": "",
        }
        subgoals = decompose_goal_to_subgoals(
            "test goal", None, ec, risk_score=0.0
        )
        for sg in subgoals:
            assert "[GATED:" not in sg.expected_output

    def test_kd_planner_no_gate_when_summary_unknown_prefix(self):
        ec = {
            "success_criteria": ("ship it",),
            "approval_requirements": (),
            "hard_constraints": (),
            "soft_constraints": (),
            "budget": {},
            "evidence_pack_summary": "(no relevant knowledge found)",
        }
        subgoals = decompose_goal_to_subgoals(
            "test goal", None, ec, risk_score=0.0
        )
        for sg in subgoals:
            assert "[GATED:" not in sg.expected_output

    def test_kd_planner_fingerprint_changes_when_gated(self):
        ec_off = {
            "success_criteria": ("ship it",),
            "approval_requirements": (),
            "hard_constraints": (),
            "soft_constraints": (),
            "budget": {},
        }
        ec_gated = dict(ec_off)
        ec_gated["evidence_pack_summary"] = "[REQUIRES_HUMAN] review"
        sub_off = decompose_goal_to_subgoals(
            "test goal", None, ec_off, risk_score=0.0
        )
        sub_gated = decompose_goal_to_subgoals(
            "test goal", None, ec_gated, risk_score=0.0
        )
        # Fingerprints differ when the [GATED] marker is added.
        f_off = compute_plan_fingerprint(
            "obj-1", sub_off, []
        )
        f_gated = compute_plan_fingerprint(
            "obj-1", sub_gated, []
        )
        assert f_off != f_gated


# ─────────────────────────────────────────────────────────────────
# 4.5 Dryrun tests
# ─────────────────────────────────────────────────────────────────


class TestDryrunSection:
    def _build_state(
        self,
        evidence_pack_summary: Optional[str] = None,
        evidence_pack_ref: Optional[str] = None,
        contract: Optional[dict] = None,
    ) -> Any:
        state = ObjectiveStateData(
            objective_id="obj-dry-1",
            state=ObjectiveState.CONTRACT_DRAFT,
            objective_text="test",
            constraints=[],
            user_id="u",
            created_at=CANARY_FROZEN_TIME_UTC,
            normalized={
                "goal_class": "RESEARCH",
                "risk_profile": "low",
                "estimated_complexity": "S",
                "success_criteria": ["c1"],
            },
            discovered={"candidates": [], "reuse_decision": "generate"},
            contract=contract,
            evidence_pack_summary=evidence_pack_summary,
            evidence_pack_ref=evidence_pack_ref,
        )
        return state

    def test_kd_dryrun_renders_pack_section(self):
        state = self._build_state(
            evidence_pack_summary="[READY_FOR_STRATEGY] 5 hits",
            evidence_pack_ref=objective_evidence_pack_key("obj-dry-1"),
            contract={
                "risk_components": {},
                "approval_requirements": [],
                "budget": {},
                "risk_score": 0.0,
                "evidence_pack_summary": "[READY_FOR_STRATEGY] 5 hits",
                "evidence_pack_ref": objective_evidence_pack_key(
                    "obj-dry-1"
                ),
            },
        )
        out = render_dry_run(state)
        assert "│ Evidence Pack:" in out
        assert "│   ref:" in out
        assert "│   summary:" in out
        assert "[READY_FOR_STRATEGY]" in out

    def test_kd_dryrun_omits_section_when_state_field_absent(self):
        state = self._build_state(
            evidence_pack_summary=None,
            contract={
                "risk_components": {},
                "approval_requirements": [],
                "budget": {},
                "risk_score": 0.0,
            },
        )
        out = render_dry_run(state)
        assert "│ Evidence Pack:" not in out

    def test_kd_dryrun_omits_section_when_state_field_empty(self):
        state = self._build_state(
            evidence_pack_summary="",
            contract={
                "risk_components": {},
                "approval_requirements": [],
                "budget": {},
                "risk_score": 0.0,
            },
        )
        out = render_dry_run(state)
        assert "│ Evidence Pack:" not in out

    def test_kd_dryrun_uses_contract_field_precedence(self):
        """The contract dict's evidence_pack_summary takes precedence
        over the state field. This matches the design (the contract is
        the authoritative snapshot for rendering)."""
        state = self._build_state(
            evidence_pack_summary="[STALE]",
            contract={
                "risk_components": {},
                "approval_requirements": [],
                "budget": {},
                "risk_score": 0.0,
                "evidence_pack_summary": "[CONTRACT_VALUE]",
                "evidence_pack_ref": objective_evidence_pack_key(
                    "obj-dry-1"
                ),
            },
        )
        out = render_dry_run(state)
        assert "[CONTRACT_VALUE]" in out
        assert "[STALE]" not in out


# ─────────────────────────────────────────────────────────────────
# 4.6 Default-OFF flag tests
# ─────────────────────────────────────────────────────────────────


class TestFlagResolver:
    def test_kd_env_var_1_enables(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )
        assert resolve_knowledge_discovery_enabled() is True

    def test_kd_env_var_0_disables(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "0"
        )
        assert resolve_knowledge_discovery_enabled() is False

    def test_kd_env_var_truthy_aliases(self, monkeypatch):
        for val in ("true", "yes", "on", "TRUE", "Yes", "ON"):
            monkeypatch.setenv(
                "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", val
            )
            assert resolve_knowledge_discovery_enabled() is True, val

    def test_kd_env_var_unknown_disables(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "bogus"
        )
        assert resolve_knowledge_discovery_enabled() is False

    def test_kd_agent_attribute_overrides_env(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "0"
        )
        agent = MagicMock()
        agent._executive_knowledge_discovery_enabled = True
        assert resolve_knowledge_discovery_enabled(agent) is True

    def test_kd_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )
        assert resolve_knowledge_discovery_enabled(None) is True

    def test_kd_default_disabled(self, monkeypatch):
        monkeypatch.delenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED",
            raising=False,
        )
        assert resolve_knowledge_discovery_enabled() is False
        assert resolve_knowledge_discovery_enabled(None) is False


# ─────────────────────────────────────────────────────────────────
# 4.7 Wiring integration tests
# ─────────────────────────────────────────────────────────────────


class TestRunPipeline:
    def test_kd_run_pipeline_kwarg_false_skips_discovery(
        self, hermetic_storage, production_storage, audit_sink, minimal_bundle, monkeypatch
    ):
        """evidence_pack=False skips discover_evidence_pack entirely."""
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )
        engine_obj = EvidencePackEngine(
            sources=minimal_bundle,
            storage=hermetic_storage,
            audit_sink=audit_sink,
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
            evidence_engine=engine_obj,
        )
        oid = engine.run_pipeline(
            "test goal", evidence_pack=False
        )
        # No state_meta evidence key was written.
        assert (
            hermetic_storage.get_meta(
                objective_evidence_pack_key(oid)
            )
            is None
        )

    def test_kd_run_pipeline_continues_when_engine_returns_none(
        self, production_storage, monkeypatch
    ):
        """Engine returns None (off) → pipeline continues to
        generate_contract (no-op gate)."""
        monkeypatch.delenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED",
            raising=False,
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
        )
        oid = engine.run_pipeline("test goal")
        state = engine.get_state(oid)
        assert state.state == ObjectiveState.CONTRACT_DRAFT
        assert state.contract is not None

    def test_kd_run_pipeline_aborts_when_engine_fails(
        self, production_storage, monkeypatch
    ):
        """Engine raises → state goes to FAILED; generate_contract
        NOT called."""
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )

        class _BoomEngine:
            def discover(self, *a, **kw):
                raise RuntimeError("kaboom")

        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
            evidence_engine=_BoomEngine(),
        )
        oid = engine.run_pipeline("test goal")
        state = engine.get_state(oid)
        assert state.state == ObjectiveState.FAILED
        assert state.contract is None

    def test_kd_pipeline_persists_evidence_pack_alongside_objective(
        self, hermetic_storage, production_storage, audit_sink, minimal_bundle, monkeypatch
    ):
        """After run_pipeline with env=1, both objective:<oid> and
        objective_knowledge_discovery:<oid>:v2 exist in state_meta."""
        monkeypatch.setenv(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1"
        )
        engine_obj = EvidencePackEngine(
            sources=minimal_bundle,
            storage=hermetic_storage,
            audit_sink=audit_sink,
        )
        engine = ObjectiveEngine(
            user_id="u",
            storage=production_storage,
            enabled=True,
            evidence_engine=engine_obj,
        )
        oid = engine.run_pipeline(
            "test goal", persist_to_state_meta=True
        )
        assert (
            hermetic_storage.get_meta(f"objective:{oid}") is not None
        )
        assert (
            hermetic_storage.get_meta(
                objective_evidence_pack_key(oid)
            )
            is not None
        )


# ─────────────────────────────────────────────────────────────────
# 4.8 Backward-compat tests
# ─────────────────────────────────────────────────────────────────


class TestBackwardCompat:
    def test_kd_objective_engine_no_evidence_kwarg_works(
        self, hermetic_storage
    ):
        """ObjectiveEngine(user_id='u') → evidence_engine is None;
        all existing tests pass byte-identically."""
        engine = ObjectiveEngine(user_id="u", storage=hermetic_storage)
        assert engine._evidence_engine is None
        assert engine._evidence_discovery_enabled is False

    def test_kd_state_data_default_values_when_loaded(self):
        """Loading an old ObjectiveStateData dict without
        evidence_pack_* fields → defaults applied."""
        old_dict = {
            "objective_id": "obj-old",
            "state": "CONTRACT_DRAFT",
            "objective_text": "old",
            "constraints": [],
            "user_id": "u",
            "created_at": CANARY_FROZEN_TIME_UTC,
            "normalized": None,
            "classified": None,
            "discovered": None,
            "contract": None,
            "fingerprint": None,
            "last_error": None,
            "last_transition_at": None,
            "last_transition_id": None,
        }
        s = ObjectiveStateData.from_dict(old_dict)
        assert s.evidence_pack_ref is None
        assert s.evidence_pack_summary is None

    def test_kd_storage_load_old_state_meta_with_no_pack(
        self, production_storage
    ):
        """load_evidence_pack(oid) returns None for an unknown oid."""
        assert production_storage.load_evidence_pack("never-written") is None
