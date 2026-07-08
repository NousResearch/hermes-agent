"""Shared fixtures for the KD-Kanban canary pilot.

Hermetic. No network, no subprocess, no real state.db. Uses in-memory
storage via the existing ``in_memory_storage`` conftest in the parent
directory (tests/test_executive_v2/conftest.py).

B1 wiring (Knowledge Discovery) is enabled by:
  1. monkeypatching HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED=1
  2. injecting an evidence_engine stub into ObjectiveEngine(...)

The objective_engine_in_memory fixture below does both, and is the
only sanctioned path for tests in this directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is importable (mirror parent conftest behaviour).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _StubEvidenceEngine:
    """Stand-in EvidencePackEngine for the pilot pipeline path.

    The engine's discover_evidence_pack calls
    ``evidence_engine.discover(objective_id, objective_text, goal_class,
    risk_profile, complexity)`` and expects back an object with
    ``to_dict()`` and ``is_idempotent_reuse``. We return a real
    EvidencePack constructed from the canary_b1 shim so the contract
    builder downstream has a valid pack to embed.
    """

    def __init__(self):
        from tests.test_executive_v2.canary_b1.evidence_pack import (
            EvidencePack,
        )

        # Sentinel pack used for every discover() call in the pilot.
        self._pack = EvidencePack(
            objective_id="obj-pilot-stub",
            query_fingerprint="pilot-stub-fp",
            sources_queried=["tmp_path:report_snapshot"],
            sources_failed=[],
            hits=[],
            citations=[],
            conflicts=[],
            missing_information=["kanban_snapshot"],
            overall_freshness_score=1.0,
            overall_confidence=0.0,
            summary_text=(
                "[NO_EVIDENCE_YET] pilot stub; awaiting kanban_snapshot."
            ),
            summary_fingerprint="pilot-stub-summary-fp",
            duration_ms=0,
            created_at="2026-07-09T00:00:00Z",
            schema_version="evidence_pack.v1",
            is_idempotent_reuse=False,
            total_hits=0,
        )

    def discover(
        self,
        objective_id: str,
        objective_text: str,
        goal_class: str = "OTHER",
        risk_profile: str = "low",
        complexity: str = "S",
    ):
        return self._pack


@pytest.fixture
def b1_env(monkeypatch):
    """Enable B1 wiring for the pilot.

    Two env vars must be on for the pilot to exercise the full chain:
      1. HERMES_EXECUTIVE_V2_ENABLED=1 — master gate for Executive v2
      2. HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED=1 — B1 sub-gate

    The engine also needs an injected ``evidence_engine`` so the
    B1 gate resolves True (env alone is not sufficient per the
    resolve_knowledge_discovery_enabled contract).
    """
    monkeypatch.setenv("HERMES_EXECUTIVE_V2_ENABLED", "1")
    monkeypatch.setenv("HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "1")
    return monkeypatch


@pytest.fixture
def frozen_evidence_pack():
    """Frozen EvidencePack-shaped object for build_execution_contract_v1."""
    from tests.test_executive_v2.canary_b1.evidence_pack import EvidencePack

    pack = EvidencePack(
        objective_id="obj-pilot-001",
        query_fingerprint="pilot-fp-001",
        sources_queried=["tmp_path:report_snapshot"],
        sources_failed=[],
        hits=[],
        citations=[],
        conflicts=[],
        missing_information=["kanban_snapshot"],
        overall_freshness_score=1.0,
        overall_confidence=0.0,
        summary_text="[NO_EVIDENCE_YET] pilot dryrun; awaiting kanban_snapshot.",
        summary_fingerprint="pilot-summary-fp",
        duration_ms=0,
        created_at="2026-07-09T00:00:00Z",
        schema_version="evidence_pack.v1",
        is_idempotent_reuse=False,
        total_hits=0,
    )
    return pack


@pytest.fixture
def kanban_snapshot_dict():
    """Frozen in-process kanban_snapshot (plain dict, JSON-serialisable)."""
    return {
        "schema_version": "kanban_snapshot.v1",
        "captured_at": "2026-07-09T00:00:00Z",
        "frozen": True,
        "candidates": (
            {
                "id": "cand-001",
                "title": "audit reports dir entropy",
                "risk_profile": "low",
                "complexity": "XS",
            },
            {
                "id": "cand-002",
                "title": "verify branch-protection state",
                "risk_profile": "low",
                "complexity": "XS",
            },
        ),
    }


@pytest.fixture
def tmp_audit_dir(tmp_path: Path) -> Path:
    """tmp_path isolated audit dir; nothing written outside this."""
    audit = tmp_path / "canary_pilot_audit"
    audit.mkdir(parents=True, exist_ok=True)
    return audit


@pytest.fixture
def objective_engine_in_memory(in_memory_storage, b1_env):
    """Real ObjectiveEngine with in-memory storage and B1 flag enabled.

    No state.db, no SessionDB, no Kanban writes, no LLM, no real
    GBrain/Obsidian. The evidence_engine stub is injected so that
    ``self._evidence_discovery_enabled`` resolves True and the
    pipeline reaches ``discover_evidence_pack``.
    """
    from agent.executive.objective_engine import ObjectiveEngine

    engine = ObjectiveEngine(
        user_id="pilot-user",
        storage=in_memory_storage,
        evidence_engine=_StubEvidenceEngine(),
    )
    return engine