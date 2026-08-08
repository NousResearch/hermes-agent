"""Tests for Executive v2 capability discovery: P0/P1, no GBrain."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from agent.executive.capability_discovery_p0_p1 import (
    _discover_p0_codebase,
    _discover_p0_reports,
    _discover_p1_memory,
    _discover_p1_state_meta,
    _extract_keywords_from_normalized,
    _jaccard,
    discover_capabilities_p0_p1,
)
from agent.executive.types import GoalClass, NormalizedObjective, RiskProfile, Complexity


def _make_normalized(**overrides) -> NormalizedObjective:
    defaults = dict(
        objective_id="oid",
        goal_class=GoalClass.RESEARCH,
        constraints=(),
        success_criteria=("Information about research is documented",),
        human_constraints=(),
        approval_requirements=(),
        risk_profile=RiskProfile.LOW,
        estimated_complexity=Complexity.XS,
        knowledge_requirements=(),
        execution_requirements={},
        created_at="2026-01-01",
        created_by="u",
    )
    defaults.update(overrides)
    return NormalizedObjective(**defaults)


def test_jaccard_identical_sets():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets():
    assert _jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_partial_overlap():
    assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5


def test_extract_keywords_includes_constraints_and_criteria():
    n = _make_normalized(
        constraints=("forbidden:stripe",),
        success_criteria=("Analysis of data is complete",),
        knowledge_requirements=("kb:financial",),
    )
    kws = _extract_keywords_from_normalized(n)
    assert "stripe" in kws
    assert "data" in kws
    assert "financial" in kws


def test_discover_p0_reports_empty_when_no_reports_dir(monkeypatch, tmp_path):
    """If reports dir doesn't exist, P0 reports returns []."""
    from agent.executive import capability_discovery_p0_p1 as cd
    monkeypatch.setattr(cd, "DEFAULT_REPORTS_DIR", tmp_path / "nonexistent")
    assert _discover_p0_reports({"x"}) == []


def test_discover_capabilities_returns_capability_discovery(monkeypatch):
    n = _make_normalized()
    discovery = discover_capabilities_p0_p1(n, objective_id="oid")
    assert discovery.objective_id == "oid"
    assert discovery.reuse_decision in ("reuse", "generate", "hybrid")
    assert discovery.p0_query_duration_ms >= 0
    assert discovery.p1_query_duration_ms >= 0


def test_capability_discovery_does_not_import_gbrain():
    """AST check: capability_discovery module does NOT import GBrain."""
    import ast
    from agent.executive import capability_discovery_p0_p1 as cd
    src = Path(cd.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)
    for mod in imported_modules:
        # No submodule of these banned namespaces.
        assert "gbrain" not in mod.lower(), f"Banned import: {mod}"
        assert "obsidian" not in mod.lower(), f"Banned import: {mod}"
        assert "notebooklm" not in mod.lower(), f"Banned import: {mod}"


def test_capability_discovery_does_not_read_obsidian():
    """String scan: no path access to Obsidian vault in source."""
    from agent.executive import capability_discovery_p0_p1 as cd
    src = Path(cd.__file__).read_text(encoding="utf-8")
    # No path access to Obsidian vault.
    assert ".hermes/Obsidian" not in src
    assert "Obsidian/Hermes" not in src
    # No NotebookLM integration.
    assert "notebooklm_adapter" not in src


def test_decision_logic_reuse_when_no_candidates():
    """Empty candidates -> generate decision."""
    from agent.executive.capability_discovery_p0_p1 import _decide
    decision, _, gaps = _decide([])
    assert decision == "generate"
    assert "no_capability" in gaps


def test_decision_logic_with_high_score_candidate():
    from agent.executive.capability_discovery_p0_p1 import _decide
    from agent.executive.types import CapabilityCandidate
    c = CapabilityCandidate(
        kind="tool", id="x", name="x", source_path="/x", description="",
        keywords=(), match_score=0.9, match_reasons=(),
    )
    decision, _, _ = _decide([c])
    assert decision == "reuse"
