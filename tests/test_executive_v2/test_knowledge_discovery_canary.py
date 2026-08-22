"""Knowledge Discovery canary tests (READONLY + hermetic).

Demonstrates the Phase 1.5 KnowledgeDiscoveryEngine and its
read-only behavior over the 3 immediate sources:

* ``policy``   — ``state_meta[objective_policy_decision:*]``
* ``report``   — ``~/.hermes/reports/**/*.md``  (filesystem, read-only)
* ``contract`` — ``state_meta[objective:*].contract``

This canary does NOT invoke the E2E submit->SUCCESS pipeline. The
Controlled Execution End-to-End Wiring baseline (8/8 in
``test_e2e_canary.py``) remains untouched. Knowledge Discovery
exists as an additive production module that the Executive pipeline
does NOT call yet.

Test cases:

1. ``test_kd_discovers_policy_from_prior_objective``
   Discovers a PolicyDecision from a prior objective and returns it
   as a hit with source="policy".

2. ``test_kd_discovers_report_in_reports_dir``
   Discovers a markdown report in a temp ``~/.hermes/reports/`` tree
   whose tokens overlap with the objective text.

3. ``test_kd_discovers_execution_contract_from_prior_objective``
   Discovers an Execution Contract from a prior objective via
   state_meta.

4. ``test_kd_persists_report_in_state_meta``
   ``discover()`` writes ``objective_knowledge_discovery:<oid>``
   exactly once.

5. ``test_kd_idempotent_reuse_on_second_call``
   A second ``discover()`` call with the same inputs returns
   ``is_idempotent_reuse=True`` and does NOT re-query.

6. ``test_kd_no_writes_to_sources``
   The 3 sources (state.db policy rows, reports dir, contract rows)
   are unchanged before/after ``discover()``.

7. ``test_kd_dry_run_is_pure``
   ``dry_run()`` produces a report but does NOT write to state_meta.

8. E2E canary coverage is selected by pytest/CI alongside this file;
   this Knowledge Discovery canary never shells out to pytest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _seed_policy(
    storage, objective_id: str, *, goal_class: str, risk_level: int, warnings: tuple
):
    """Seed a PolicyDecision in state_meta via the existing storage method.

    Also seeds the underlying ``objective:<oid>`` state so that the
    ``list_active()`` iteration in the policy provider can find the
    prior objective. (Without the state row, ``list_active()`` returns
    no IDs and the policy is unreachable.)
    """
    from agent.executive.types import (
        ObjectiveState,
        ObjectiveStateData,
        PolicyDecision,
        RiskLevel,
    )

    # Seed the underlying objective state (required for list_active()).
    state = ObjectiveStateData(
        objective_id=objective_id,
        state=ObjectiveState.DRAFT,
        objective_text="seeded prior objective",
        constraints=[],
        user_id="canary-user",
        created_at="2026-07-02T00:00:00+00:00",
        contract={
            "risk_score": 0.45,
            "risk_components": {},
            "hard_constraints": [],
            "soft_constraints": [],
            "success_criteria": [],
            "approval_requirements": [],
        },
    )
    storage.save(state)

    # Then seed the policy decision.
    decision = PolicyDecision(
        objective_id=objective_id,
        risk_level=RiskLevel(risk_level),
        allowed_actions=("read_state_meta",),
        forbidden_actions=("spawn_worker",),
        approval_required=True,
        warnings=warnings,
        approval_requirements=(),
        risk_score=0.45,
        risk_components={},
        created_at="2026-07-02T00:00:00+00:00",
        decision_fingerprint="canary-policy-fp-" + objective_id[:6],
    )
    storage.set_objective_policy_decision(decision)


def _seed_contract(
    storage,
    objective_id: str,
    *,
    success_criteria: tuple,
    risk_score: float = 0.3,
    risk_components: dict | None = None,
):
    """Seed an ObjectiveStateData with a contract via the existing storage method."""
    from agent.executive.types import ObjectiveState, ObjectiveStateData

    if risk_components is None:
        risk_components = {}
    state = ObjectiveStateData(
        objective_id=objective_id,
        state=ObjectiveState.DRAFT,
        objective_text="seeded prior objective",
        constraints=[],
        user_id="canary-user",
        created_at="2026-07-02T00:00:00+00:00",
        contract={
            "risk_score": risk_score,
            "risk_components": risk_components,
            "hard_constraints": [],
            "soft_constraints": [],
            "success_criteria": list(success_criteria),
            "approval_requirements": [],
        },
    )
    storage.save(state)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def test_kd_discovers_policy_from_prior_objective(
    in_memory_storage, clean_env_executive
):
    """KD reads state_meta[objective_policy_decision:*] for prior objectives."""
    # Seed a prior policy whose warnings token-overlap with the
    # current objective text.
    _seed_policy(
        in_memory_storage,
        "prior-obj-001",
        goal_class="DOCUMENT",
        risk_level=3,
        warnings=("STRATEGIC: hermes index of files modified in last 7 days",),
    )
    from agent.executive.phase15_knowledge_discovery import KnowledgeDiscoveryEngine

    eng = KnowledgeDiscoveryEngine(storage=in_memory_storage)
    report = eng.dry_run(
        "canary-obj-001",
        "compile a hermes-archives index: list files modified in last 7 days",
        goal_class="DOCUMENT",
        risk_profile="low",
        complexity="S",
    )

    sources = set(h.source for h in report.hits_by_source)
    assert "policy" in sources, (
        f"expected 'policy' in sources, got {sources}"
    )
    policy_hits = [h for h in report.hits_by_source if h.source == "policy"]
    assert policy_hits, "no policy hit"
    assert policy_hits[0].hit_id == "prior-obj-001"
    assert policy_hits[0].relevance_score > 0.0


def test_kd_discovers_report_in_reports_dir(
    in_memory_storage, tmp_path, monkeypatch
):
    """KD reads ~/.hermes/reports/**/*.md (read-only)."""
    # Create a temporary reports dir and put a matching report there.
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    target = reports_dir / "matching_report.md"
    target.write_text(
        "# Matching report\n\nThis report discusses hermes-archives index of files "
        "modified in the last 7 days, grouped by directory.\n",
        encoding="utf-8",
    )
    # Also create an unrelated report that should NOT match.
    (reports_dir / "unrelated_report.md").write_text(
        "# Unrelated\n\nThis is about something completely different: cats and dogs.\n",
        encoding="utf-8",
    )

    from agent.executive.phase15_knowledge_discovery import KnowledgeDiscoveryEngine

    eng = KnowledgeDiscoveryEngine(
        storage=in_memory_storage, reports_root=reports_dir
    )
    report = eng.dry_run(
        "canary-obj-002",
        "compile a hermes-archives index of files modified in last 7 days",
        goal_class="DOCUMENT",
        risk_profile="low",
        complexity="S",
    )

    report_hits = [h for h in report.hits_by_source if h.source == "report"]
    titles = [h.title for h in report_hits]
    assert "matching_report.md" in titles, (
        f"expected 'matching_report.md' in report hits, got {titles}"
    )
    # The unrelated report must NOT match (no token overlap).
    assert "unrelated_report.md" not in titles


def test_kd_discovers_execution_contract_from_prior_objective(
    in_memory_storage
):
    """KD reads state_meta[objective:*].contract for prior objectives."""
    _seed_contract(
        in_memory_storage,
        "prior-obj-002",
        success_criteria=("list files modified in last 7 days",),
        risk_score=0.4,
    )
    from agent.executive.phase15_knowledge_discovery import KnowledgeDiscoveryEngine

    eng = KnowledgeDiscoveryEngine(storage=in_memory_storage)
    report = eng.dry_run(
        "canary-obj-003",
        "compile a hermes-archives index of files modified in last 7 days",
        goal_class="DOCUMENT",
        risk_profile="low",
        complexity="S",
    )

    contract_hits = [h for h in report.hits_by_source if h.source == "contract"]
    assert contract_hits, "no contract hit"
    assert contract_hits[0].hit_id == "prior-obj-002"
    assert contract_hits[0].relevance_score > 0.0


def test_kd_persists_report_in_state_meta(in_memory_storage):
    """discover() writes objective_knowledge_discovery:<oid> exactly once."""
    _seed_contract(
        in_memory_storage,
        "prior-obj-003",
        success_criteria=("prior criteria",),
    )
    from agent.executive.phase15_knowledge_discovery import KnowledgeDiscoveryEngine
    from agent.executive.types import objective_knowledge_discovery_key

    eng = KnowledgeDiscoveryEngine(storage=in_memory_storage)
    report = eng.discover(
        "canary-obj-004",
        "compile a hermes-archives index of files modified in last 7 days",
        goal_class="DOCUMENT",
        risk_profile="low",
        complexity="S",
    )
    assert report.is_idempotent_reuse is False

    # Verify state_meta write.
    loaded = in_memory_storage.get_objective_knowledge_discovery("canary-obj-004")
    assert loaded is not None, (
        "expected state_meta[objective_knowledge_discovery:canary-obj-004]"
    )
    assert loaded.summary_fingerprint == report.summary_fingerprint
    assert loaded.knowledge_query_fingerprint == report.knowledge_query_fingerprint


def test_kd_idempotent_reuse_on_second_call(in_memory_storage):
    """Second discover() with same inputs returns is_idempotent_reuse=True."""
    _seed_contract(
        in_memory_storage,
        "prior-obj-004",
        success_criteria=("prior criteria",),
    )
    from agent.executive.phase15_knowledge_discovery import KnowledgeDiscoveryEngine

    eng = KnowledgeDiscoveryEngine(storage=in_memory_storage)
    args = dict(
        objective_id="canary-obj-005",
        objective_text="compile a hermes-archives index of files modified in last 7 days",
        goal_class="DOCUMENT",
        risk_profile="low",
        complexity="S",
    )
    report_1 = eng.discover(**args)
    assert report_1.is_idempotent_reuse is False

    report_2 = eng.discover(**args)
    assert report_2.is_idempotent_reuse is True
    assert report_1.summary_fingerprint == report_2.summary_fingerprint


def test_kd_no_writes_to_sources(in_memory_storage, tmp_path, monkeypatch):
    """The 3 sources are unchanged before/after discover()."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    target = reports_dir / "report.md"
    original_content = "# original content\n"
    target.write_text(original_content, encoding="utf-8")
    original_size = target.stat().st_size
    original_mtime = target.stat().st_mtime

    # Seed policy + contract as "prior" sources.
    _seed_policy(
        in_memory_storage,
        "prior-obj-006",
        goal_class="DOCUMENT",
        risk_level=3,
        warnings=("STRATEGIC: hermes index of files modified in last 7 days",),
    )
    _seed_contract(
        in_memory_storage,
        "prior-obj-006b",
        success_criteria=("list files modified in last 7 days",),
    )

    # Snapshot the state.db content for the policy and contract rows.
    policy_key = "objective_policy_decision:prior-obj-006"
    contract_key = "objective:prior-obj-006b"
    policy_before = in_memory_storage._get_db().get_meta(policy_key)
    contract_before = in_memory_storage._get_db().get_meta(contract_key)
    assert policy_before is not None
    assert contract_before is not None

    from agent.executive.phase15_knowledge_discovery import KnowledgeDiscoveryEngine

    eng = KnowledgeDiscoveryEngine(
        storage=in_memory_storage, reports_root=reports_dir
    )
    eng.discover(
        "canary-obj-006",
        "compile a hermes-archives index of files modified in last 7 days",
    )

    # Verify state.db unchanged for sources.
    policy_after = in_memory_storage._get_db().get_meta(policy_key)
    contract_after = in_memory_storage._get_db().get_meta(contract_key)
    assert policy_after == policy_before, "policy row was modified"
    assert contract_after == contract_before, "contract row was modified"

    # Verify report file unchanged (size, mtime, content).
    assert target.stat().st_size == original_size
    assert target.stat().st_mtime == original_mtime
    assert target.read_text(encoding="utf-8") == original_content


def test_kd_dry_run_is_pure(in_memory_storage):
    """dry_run() builds the report but does NOT write to state_meta."""
    _seed_contract(
        in_memory_storage,
        "prior-obj-007",
        success_criteria=("prior criteria",),
    )
    from agent.executive.phase15_knowledge_discovery import KnowledgeDiscoveryEngine
    from agent.executive.types import objective_knowledge_discovery_key

    eng = KnowledgeDiscoveryEngine(storage=in_memory_storage)
    report = eng.dry_run(
        "canary-obj-007",
        "compile a hermes-archives index of files modified in last 7 days",
    )
    assert report.is_idempotent_reuse is False
    # No state_meta write.
    loaded = in_memory_storage.get_objective_knowledge_discovery("canary-obj-007")
    assert loaded is None



def test_kd_no_prohibited_imports():
    """KD does NOT import GBrain, Obsidian, NotebookLM, network, or providers."""
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    src = _REPO_ROOT / "agent/executive/phase15_knowledge_discovery.py"
    content = src.read_text(encoding="utf-8")
    prohibited = [
        "urllib", "httpx", "requests", "aiohttp",
        "gbrain", "obsidian", "notebooklm",
        "anthropic", "openai", "litellm",
        "subprocess" + ".run", "subprocess" + ".Popen", "os.system",
    ]
    found = []
    for term in prohibited:
        # Match "import X", "from X", or direct module usage like
        # "urllib.request.urlopen" but exclude docstrings/comments.
        if term in content:
            # Skip matches inside triple-quoted strings (docstrings).
            in_docstring = False
            for line in content.split("\n"):
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                if term in line and not in_docstring:
                    found.append((term, line.strip()))
                    break
    assert not found, (
        f"prohibited term(s) found in knowledge_discovery.py:\n"
        + "\n".join(f"  {term}: {line}" for term, line in found)
    )