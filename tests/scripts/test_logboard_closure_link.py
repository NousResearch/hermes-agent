"""Tests for scripts/logboard_closure_link.py — open→ack→close lifecycle.

P05 Batch 1: proves the Logboard alert/issue lifecycle with states
open/ack/closed and a closure-link to a review packet.  Validates
transitions.  Uses a temporary HERMES_HOME — never touches live ``~/.hermes``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "logboard_closure_link_under_test",
        str(SCRIPTS / "logboard_closure_link.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_logboard(tmp_path):
    lb = tmp_path / "hermes" / "governance" / "logboard"
    lb.mkdir(parents=True)
    return lb


class TestStateTransitions:
    def test_open_to_ack(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack")
        assert artifact["state"] == "ack"

    def test_ack_to_closed(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack")
        mod.close_with_link(artifact, review_packet_id="rp_001")
        assert artifact["state"] == "closed"

    def test_open_to_closed_rejected(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        with pytest.raises(mod.TransitionError, match="invalid transition"):
            mod.transition(artifact, "closed")

    def test_closed_to_ack_rejected(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack")
        mod.close_with_link(artifact, review_packet_id="rp_001")
        with pytest.raises(mod.TransitionError, match="invalid transition"):
            mod.transition(artifact, "ack")

    def test_ack_to_open_rejected(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack")
        with pytest.raises(mod.TransitionError, match="invalid transition"):
            mod.transition(artifact, "open")

    def test_invalid_target_rejected(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        with pytest.raises(mod.TransitionError, match="invalid target"):
            mod.transition(artifact, "bogus")

    def test_double_ack_rejected(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack")
        with pytest.raises(mod.TransitionError, match="invalid transition"):
            mod.transition(artifact, "ack")


class TestClosureLink:
    def test_close_requires_review_packet(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack")
        with pytest.raises(mod.TransitionError, match="review_packet_id"):
            mod.close_with_link(artifact, review_packet_id="")

    def test_close_with_link_sets_closure_link(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack")
        mod.close_with_link(artifact, review_packet_id="rp_2026Q3")
        assert artifact["closure_link"] == "rp_2026Q3"

    def test_close_without_ack_rejected(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        with pytest.raises(mod.TransitionError, match="invalid transition"):
            mod.close_with_link(artifact, review_packet_id="rp_001")


class TestLifecycleValidation:
    def test_valid_lifecycle_passes(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack", actor="denji")
        mod.close_with_link(artifact, review_packet_id="rp_001", actor="kensei")
        assert mod.validate_lifecycle(artifact) is True

    def test_missing_ack_fails(self, fake_logboard):
        mod = _load_script()
        # Build an artifact that went open→closed directly (shouldn't be
        # possible via transition(), but validate must catch it).
        artifact = mod._new_artifact(title="Test alert")
        # Manually tamper to simulate the invalid path
        artifact["state"] = "closed"
        artifact["transitions"] = [
            {"to": "open", "ts": "2026-01-01T00:00:00Z"},
            {"to": "closed", "ts": "2026-01-02T00:00:00Z"},
        ]
        artifact["closure_link"] = "rp_001"
        assert mod.validate_lifecycle(artifact) is False

    def test_no_closure_link_fails(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        mod.transition(artifact, "ack")
        artifact["state"] = "closed"
        artifact["transitions"].append({"to": "closed", "ts": "2026-01-01"})
        # No closure_link set
        assert mod.validate_lifecycle(artifact) is False

    def test_empty_transitions_fails(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(title="Test alert")
        artifact["transitions"] = []
        assert mod.validate_lifecycle(artifact) is False


class TestPersistence:
    def test_save_and_load_round_trip(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(artifact_id="alert-001", title="Test")
        path = fake_logboard / "alert-001.json"
        mod.save_artifact(path, artifact)
        loaded = mod.load_artifact(path)
        assert loaded["artifact_id"] == "alert-001"
        assert loaded["state"] == "open"

    def test_full_lifecycle_persisted(self, fake_logboard):
        mod = _load_script()
        artifact = mod._new_artifact(artifact_id="alert-002", title="Persisted")
        mod.transition(artifact, "ack", actor="denji")
        mod.close_with_link(artifact, review_packet_id="rp_persist", actor="kensei")
        path = fake_logboard / "alert-002.json"
        mod.save_artifact(path, artifact)
        loaded = mod.load_artifact(path)
        assert loaded["state"] == "closed"
        assert loaded["closure_link"] == "rp_persist"
        assert mod.validate_lifecycle(loaded) is True
        # Check transition history
        states = [t["to"] for t in loaded["transitions"]]
        assert states == ["open", "ack", "closed"]

    def test_load_invalid_state_rejected(self, fake_logboard):
        mod = _load_script()
        path = fake_logboard / "bad.json"
        path.write_text(json.dumps({"state": "bogus", "transitions": []}))
        with pytest.raises(mod.TransitionError, match="invalid state"):
            mod.load_artifact(path)
