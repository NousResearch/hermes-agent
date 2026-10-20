"""Behavior contracts for the generated security-assurance matrix."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_security_parity import catalog, render  # noqa: E402


def test_catalog_has_unique_grounded_guarantees():
    guarantees = catalog()
    ids = [guarantee.id for guarantee in guarantees]

    assert len(ids) == len(set(ids))
    assert guarantees
    for guarantee in guarantees:
        assert guarantee.claim.strip()
        assert guarantee.status in {"proven", "partial", "gap"}
        assert guarantee.proofs or guarantee.status == "gap"
        for proof in guarantee.proofs:
            assert (REPO_ROOT / proof).is_file(), f"missing proof: {proof}"


def test_render_is_deterministic_and_exposes_limits():
    first = render(catalog())
    second = render(catalog())

    assert first == second
    assert "# Hermes security assurance matrix" in first
    assert "| Guarantee | Status | Evidence | Honest limit |" in first
    assert "present but not yet proven uniformly" in first
    assert "No security or compliance guarantee" in first


def test_committed_matrix_matches_generator():
    committed = REPO_ROOT / "website/docs/developer-guide/security-assurance-matrix.md"

    assert committed.read_text(encoding="utf-8") == render(catalog())
