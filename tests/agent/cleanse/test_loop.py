"""Tests for agent.cleanse.loop — main repair loop."""

from pathlib import Path

import pytest

from agent.cleanse.loop import run_cleanse
from agent.cleanse.parser import Diagnostic


def test_run_cleanse_empty_project(tmp_path):
    """run_cleanse() handles empty project cleanly."""
    result = run_cleanse(tmp_path, max_workers=2, run_tests=False)
    assert result.ok
    assert result.initial_count == 0
    assert result.fixed_count == 0
    assert len(result.remaining_diagnostics) == 0


def test_run_cleanse_writes_evidence(tmp_path):
    """run_cleanse() writes verification_evidence.json."""
    result = run_cleanse(tmp_path, max_workers=2, run_tests=False)
    evidence_path = tmp_path / "verification_evidence.json"
    assert evidence_path.exists()

    import json

    with evidence_path.open() as f:
        evidence = json.load(f)
    assert "status" in evidence
    assert evidence["status"] == "clean"


def test_run_cleanse_to_dict(tmp_path):
    """CleanseResult.to_dict() serializes cleanly."""
    result = run_cleanse(tmp_path, max_workers=2, run_tests=False)
    data = result.to_dict()
    assert "ok" in data
    assert "initial_count" in data
    assert "fixed_count" in data
    assert "remaining_count" in data
    assert "remaining_diagnostics" in data
    assert "verification_evidence" in data
