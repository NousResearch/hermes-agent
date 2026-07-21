"""Hermetic fixtures for the B1 E2E canary (READONLY, in-memory).

Layered on top of the existing canary_b1 + b1_tests conftest fixtures
(provider bundle, frozen_time, in_memory_storage, audit capture). Adds
the canary-specific glue:

* ``objective_engine_with_b1_engine`` — real ``ObjectiveEngine`` with
  the B1 ``EvidencePackEngine`` injected and the B1 env flag ON.
* ``objective_engine_b1_disabled`` — same engine but with the env flag
  OFF; used to assert the default-off contract.
* ``human_gate_provider_bundle`` — variant of the provider bundle whose
  assembled pack produces a ``[REQUIRES_HUMAN]`` prefix; the contract
  planner must propagate the human review gate to subgoals.
* ``objective_engine_disabled`` — engine with HERMES_EXECUTIVE_V2_ENABLED=0;
  asserts submit() raises PermissionError_ (proves B1 path is not
  reached when master engine is off).
* ``render_helper`` — wraps ``render_dry_run`` against an
  ``ObjectiveStateData`` built from real pipeline output, so tests can
  assert deterministic snapshot.

No real dependency is touched. All effects are in-memory storage,
in-memory audit sink, frozen time, and a fake provider bundle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

# Repo root on sys.path so absolute imports work without installation.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-export existing fixtures so tests can compose without re-importing
from tests.test_executive_v2.conftest import (  # noqa: F401
    in_memory_storage,
)
from tests.test_executive_v2.canary_b1.conftest import (  # noqa: F401
    CANARY_FROZEN_TIME_UTC,
    audit_capture,
    frozen_time,
)


# ─────────────────────────────────────────────────────────────────────
# B1 env flag toggles
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def b1_env_enabled(monkeypatch):
    """Enable both Executive v2 master flag and B1 evidence flag."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    monkeypatch.setenv("HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1")
    return monkeypatch


@pytest.fixture
def b1_env_master_only(monkeypatch):
    """Executive v2 master on, B1 evidence OFF (default-off path)."""
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    monkeypatch.delenv("HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", raising=False)
    return monkeypatch


@pytest.fixture
def b1_env_completely_off(monkeypatch):
    """Both flags off; engine.submit must raise PermissionError_."""
    monkeypatch.delenv("HERMES_EXECUTIVE_V2_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", raising=False)
    return monkeypatch


# ─────────────────────────────────────────────────────────────────────
# Engines: real ObjectiveEngine wired with the real EvidencePackEngine
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# Self-contained 5-source fake bundle for the default canary flow
# ─────────────────────────────────────────────────────────────────────


def _build_default_provider_bundle(
    *,
    frozen_time: str,
    in_memory_storage: Any,
    audit_capture: Any,
):
    """Build an EvidencePackEngine with 5 fake sources, one canned hit
    each, current freshness, distinct tokens.

    Produces a [READY_FOR_STRATEGY] summary (no high-severity
    conflicts) so the contract does NOT trigger a human review gate by
    default — this is the baseline for most tests in this canary.
    """
    from tests.test_executive_v2.canary_b1.evidence_pack import (
        EvidencePackEngine,
        _make_hit_v2,
        SOURCE_TTL_DAYS,
    )

    observed = frozen_time
    updated = frozen_time  # all sources current at observed_at

    # KEY DESIGN: snippets are nearly-identical across all 5 sources so
    # the near-dup Jaccard filter (>= 0.85) collapses them to one hit
    # before pairwise conflict detection runs. The resulting pack has:
    #   - total_hits = 1
    #   - conflicts = []
    #   - summary prefix = [READY_FOR_STRATEGY]
    # This is the "clean" baseline used by most canary tests.
    _SHARED_TOKENS = (
        "canary hermetic e2e evidence pack clean shared content tokens"
    )

    def _shared_token_hit(*, source: str, hit_id: str):
        def _provider(query, *, max_hits: int = 5, observed_at: str):
            return [_make_hit_v2(
                source=source,
                hit_id=hit_id,
                title=f"canary {source} {hit_id}",
                relevance_score=0.85,
                snippet=_SHARED_TOKENS,
                source_uri=f"{source}://canary-e2e/{hit_id}",
                source_updated_at=updated,
                retrieval_mode="metadata_only",
                observed_at=observed,
                ttl_days=SOURCE_TTL_DAYS[source],
            )]
        return _provider

    sources = {
        "policy": _shared_token_hit(
            source="policy", hit_id="canary-e2e-policy-clean",
        ),
        "obsidian": _shared_token_hit(
            source="obsidian", hit_id="canary-e2e-obsidian-clean",
        ),
        "gbrain": _shared_token_hit(
            source="gbrain", hit_id="canary-e2e-gbrain-clean",
        ),
        "contract": _shared_token_hit(
            source="contract", hit_id="canary-e2e-contract-clean",
        ),
        "report": _shared_token_hit(
            source="report", hit_id="canary-e2e-report-clean",
        ),
    }
    return EvidencePackEngine(
        sources=sources,
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )


@pytest.fixture
def objective_engine_b1(
    b1_env_enabled,
    in_memory_storage,
    audit_capture,
    frozen_time,
):
    """Real ObjectiveEngine + self-contained EvidencePackEngine (B1 ON).

    Returns ``(objective_engine, evidence_engine)`` tuple. The 5
    source bundle produces a [READY_FOR_STRATEGY] summary by default
    (no high-severity conflicts), so the resulting contract does NOT
    add a human review gate. Tests that want the human gate use the
    ``objective_engine_human_gate`` fixture below.
    """
    from agent.executive.objective_engine import ObjectiveEngine

    evidence_engine = _build_default_provider_bundle(
        frozen_time=frozen_time,
        in_memory_storage=in_memory_storage,
        audit_capture=audit_capture,
    )
    engine = ObjectiveEngine(
        user_id="canary-e2e-b1-user",
        storage=in_memory_storage,
        evidence_engine=evidence_engine,
    )
    return engine, evidence_engine


@pytest.fixture
def objective_engine_b1_off(
    b1_env_master_only,
    in_memory_storage,
):
    """Real ObjectiveEngine, master on, B1 OFF (no evidence engine injected).

    Used to assert the default-off contract: contract has no
    evidence_pack_ref, state.evidence_pack_ref stays None.
    """
    from agent.executive.objective_engine import ObjectiveEngine

    return ObjectiveEngine(
        user_id="canary-e2e-b1-off-user",
        storage=in_memory_storage,
        # evidence_engine intentionally omitted
    )


@pytest.fixture
def objective_engine_disabled(b1_env_completely_off, in_memory_storage):
    """Real ObjectiveEngine with master flag OFF.

    submit() must raise PermissionError_. Used to prove the B1 path is
    not reachable when the master gate is closed (B1 cannot bypass v2
    master gate).
    """
    from agent.executive.objective_engine import ObjectiveEngine

    return ObjectiveEngine(
        user_id="canary-e2e-b1-disabled-user",
        storage=in_memory_storage,
    )


# ─────────────────────────────────────────────────────────────────────
# Degradation engine — produces [DEGRADED_FRESHNESS] prefix
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def objective_engine_degraded(
    b1_env_enabled,
    in_memory_storage,
    audit_capture,
    frozen_time,
):
    """Engine with one very-stale gbrain hit only.

    The remaining 4 sources return no hits at all (drop them from the
    bundle). The single stale hit produces overall_freshness < 0.5,
    yielding [DEGRADED_FRESHNESS] prefix and the
    ``knowledge_freshness_review`` approval requirement on the
    contract.
    """
    from tests.test_executive_v2.canary_b1.evidence_pack import (
        EvidencePackEngine,
        _make_hit_v2,
        SOURCE_TTL_DAYS,
    )
    from agent.executive.objective_engine import ObjectiveEngine

    observed = frozen_time

    def _stale_gbrain(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="gbrain",
            hit_id="canary-e2e-degraded-gbrain-001",
            title="canary degraded gbrain stub",
            relevance_score=0.5,
            snippet="canary degraded gbrain stub content",
            source_uri="gbrain://canary-e2e-degraded-gbrain-001",
            source_updated_at="2020-01-01T00:00:00+00:00",  # 6+ yrs old
            retrieval_mode="metadata_only",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["gbrain"],
        )]

    evidence_engine = EvidencePackEngine(
        sources={"gbrain": _stale_gbrain},
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )
    engine = ObjectiveEngine(
        user_id="canary-e2e-degraded-user",
        storage=in_memory_storage,
        evidence_engine=evidence_engine,
    )
    return engine, evidence_engine


# ─────────────────────────────────────────────────────────────────────
# Human-gate provider bundle — packs that trip [REQUIRES_HUMAN]
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def human_gate_provider_bundle(
    frozen_time,
    in_memory_storage,
    audit_capture,
):
    """Provider bundle engineered to produce a policy_vs_goal conflict
    (severity=high) so the engine's summary prefix is [REQUIRES_HUMAN].

    The conflict requires one ``policy`` hit paired with one
    ``obsidian`` hit on distinct hit_ids with low token overlap. The
    classifier lands the pair on policy_vs_goal (high) and the engine
    propagates that to ``summary_text``. The ExecutionContract must add
    a ``knowledge_review`` approval requirement and propagate a
    ``[GATED: ...]`` suffix to every subgoal.
    """
    from tests.test_executive_v2.canary_b1.evidence_pack import (
        EvidencePackEngine,
        _make_hit_v2,
        SOURCE_TTL_DAYS,
    )

    observed = frozen_time  # 2026-07-08T20:00:00+00:00
    updated = "2026-07-08T20:00:00+00:00"

    # Each source provides one canned hit; pair {policy, obsidian}
    # triggers policy_vs_goal → severity=high → summary
    # prefix [REQUIRES_HUMAN].
    def _policy(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="policy",
            hit_id="canary-e2e-policy-001",
            title="canary policy decision alpha",
            relevance_score=0.9,
            snippet="canary decision alpha approved",
            source_uri="state_meta[objective_policy_decision:canary-e2e-policy-001]",
            source_updated_at=updated,
            retrieval_mode="metadata_only",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["policy"],
        )]

    def _obsidian(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="obsidian",
            hit_id="canary-e2e-obsidian-001",
            title="canary obsidian diary beta",
            relevance_score=0.9,
            snippet="canary diary beta rejected",
            source_uri="file://obsidian/canary-e2e-obsidian-001",
            source_updated_at=updated,
            retrieval_mode="snippet",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["obsidian"],
        )]

    def _gbrain(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="gbrain",
            hit_id="canary-e2e-gbrain-001",
            title="canary gbrain context",
            relevance_score=0.7,
            snippet="canary gbrain context",
            source_uri="gbrain://canary-e2e-gbrain-001",
            source_updated_at=updated,
            retrieval_mode="metadata_only",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["gbrain"],
        )]

    def _contract(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="contract",
            hit_id="canary-e2e-contract-001",
            title="canary contract fixture",
            relevance_score=0.7,
            snippet="canary contract constraints",
            source_uri="state_meta[objective:<canary-e2e-contract-001>].contract",
            source_updated_at=updated,
            retrieval_mode="metadata_only",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["contract"],
        )]

    def _report(query, *, max_hits: int = 5, observed_at: str):
        return [_make_hit_v2(
            source="report",
            hit_id="canary-e2e-report-001",
            title="canary report fixture",
            relevance_score=0.6,
            snippet="canary report canary content",
            source_uri="canary-e2e-report-001",
            source_updated_at=updated,
            retrieval_mode="full_document",
            observed_at=observed,
            ttl_days=SOURCE_TTL_DAYS["report"],
        )]

    sources = {
        "policy": _policy,
        "obsidian": _obsidian,
        "gbrain": _gbrain,
        "contract": _contract,
        "report": _report,
    }
    return EvidencePackEngine(
        sources=sources,
        storage=in_memory_storage,
        audit_sink=audit_capture,
    )


@pytest.fixture
def objective_engine_human_gate(
    b1_env_enabled,
    in_memory_storage,
    human_gate_provider_bundle,
):
    """Real ObjectiveEngine + human-gate provider bundle wired on.

    Produces a contract + state with ``[REQUIRES_HUMAN]`` prefix and a
    ``knowledge_review`` approval requirement.
    """
    from agent.executive.objective_engine import ObjectiveEngine

    engine = ObjectiveEngine(
        user_id="canary-e2e-human-user",
        storage=in_memory_storage,
        evidence_engine=human_gate_provider_bundle,
    )
    return engine, human_gate_provider_bundle


# ─────────────────────────────────────────────────────────────────────
# Synthetic objective payload (deterministic across all tests)
# ─────────────────────────────────────────────────────────────────────


CANARY_OBJECTIVE_TEXT = (
    "canary hermetic end-to-end B1 evidence pipeline: "
    "research the knowledge discovery canary and deliver an evidence pack"
)
CANARY_SUCCESS_CRITERIA = [
    "objective: produce an evidence pack",
    "objective: derive execution contract from the pack",
    "objective: decompose into subgoals respecting the human gate",
]
CANARY_CONSTRAINTS = [
    "forbidden:network",
    "forbidden:subprocess",
    "forbidden:real_gbrain",
    "forbidden:real_obsidian",
    "limit:subgoals<=3",
]


@pytest.fixture
def objective_payload():
    """Deterministic synthetic objective payload.

    Shared by all E2E tests so byte-determinism in ``render`` is
    reproducible. Constraints cover the operator scope guards and are
    preserved verbatim by the Execution Contract.
    """
    return {
        "objective_text": CANARY_OBJECTIVE_TEXT,
        "constraints": list(CANARY_CONSTRAINTS),
        "success_criteria": list(CANARY_SUCCESS_CRITERIA),
        "max_subgoals": 3,
    }


# ─────────────────────────────────────────────────────────────────────
# Render helper
# ─────────────────────────────────────────────────────────────────────


def _state_data_to_dict(state: Any) -> dict:
    """Snapshot state.contract / state.discovered / state.normalized into a
    dict shaped like the ObjectiveStateData dict that ``render_dry_run``
    reads via state.X. Falls back to getting attributes off state itself.
    """
    norm = state.normalized or {}
    disc = state.discovered or {}
    cont = state.contract or {}
    return {
        "objective_id": state.objective_id,
        "fingerprint": state.fingerprint,
        "state": state.state,
        "normalized": norm,
        "discovered": disc,
        "contract": cont,
        "evidence_pack_ref": getattr(state, "evidence_pack_ref", None),
        "evidence_pack_summary": getattr(state, "evidence_pack_summary", None),
    }


class _StubState:
    """Minimal state-shaped object for render_dry_run hermetic snapshot."""

    def __init__(self, **kw: Any) -> None:
        self.objective_id = kw.get("objective_id", "obj-canary-e2e")
        self.fingerprint = kw.get("fingerprint", "canary-fp")
        self.normalized = kw.get("normalized")
        self.discovered = kw.get("discovered")
        self.contract = kw.get("contract")
        self.evidence_pack_ref = kw.get("evidence_pack_ref")
        self.evidence_pack_summary = kw.get("evidence_pack_summary")
        state_value = kw.get("state_value", "draft")
        # state.value is read by render_dry_run
        self.state_value = state_value

    # render_dry_run reads ``state.state.value`` (Enum)
    class _Enum:
        def __init__(self, v: str) -> None:
            self.value = v

    @property
    def state(self) -> "_StubState._Enum":
        return _StubState._Enum(self.state_value)


def make_stub_state_from_objective_state(state: Any) -> _StubState:
    """Build a _StubState from a real ObjectiveStateData so render works."""
    return _StubState(**_state_data_to_dict(state))


@pytest.fixture
def render_helper():
    """Return a callable that renders a real ObjectiveStateData."""
    from agent.executive.dryrun import render_dry_run

    def _render(state: Any) -> str:
        return render_dry_run(make_stub_state_from_objective_state(state))
    return _render
