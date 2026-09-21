"""
Tests for blast_radius module (P2-3).

Covers: edit cap, canary apply-observe-promote/revert, fleet-health tripwire,
state persistence.  All state-affecting tests use a patched TRIPWIRE_STATE_PATH
to prevent cross-test contamination.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.blast_radius import (
    EditGuard,
    CanaryState,
    CanaryStage,
    FleetHealth,
    TripwireStatus,
    EditResult,
    TRIPWIRE_STATE_PATH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockEvalRun:
    """Minimal mock of EvalRun for canary observation."""
    def __init__(self, pass_rate: float = 0.85):
        self.pass_rate = pass_rate


@pytest.fixture
def isolated_guard():
    """Create an EditGuard with an isolated temp state file."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = os.path.join(tmp, "tripwire_state.json")
        fleet_path = os.path.join(tmp, "fleet_health.json")
        with (
            patch("hermes_cli.blast_radius.TRIPWIRE_STATE_PATH", state_path),
            patch(
                "hermes_cli.blast_radius.EditGuard._state_path",
                return_value=state_path,
            ),
        ):
            # Also patch _write_fleet_health path
            guard = EditGuard(
                max_edits_per_cycle=5,
                canary_observe_runs=3,
            )
            # Override fleet health path
            guard._write_fleet_health = lambda h: None  # no-op for tests
            guard._read_fleet_health = lambda: FleetHealth(
                timestamp="t", tripwire=TripwireStatus.NORMAL,
            )
            yield guard


@pytest.fixture
def guard_single_observe():
    """Guard with canary_observe_runs=1 for fast promotion tests."""
    with tempfile.TemporaryDirectory() as tmp:
        state_path = os.path.join(tmp, "tripwire_state.json")
        with (
            patch("hermes_cli.blast_radius.TRIPWIRE_STATE_PATH", state_path),
        ):
            guard = EditGuard(
                max_edits_per_cycle=5,
                canary_observe_runs=1,
            )
            guard._write_fleet_health = lambda h: None
            guard._read_fleet_health = lambda: FleetHealth(
                timestamp="t", tripwire=TripwireStatus.NORMAL,
            )
            yield guard


# ---------------------------------------------------------------------------
# Edit cap
# ---------------------------------------------------------------------------


class TestEditCap:
    def test_first_edit_allowed(self, isolated_guard):
        guard = isolated_guard
        result = guard.try_edit("test-profile", "patch content", "summary")
        assert result.allowed is True
        assert result.canary is not None
        assert result.canary.profile == "test-profile"

    def test_edit_cap_exhausted(self, isolated_guard):
        guard = isolated_guard
        guard.max_edits_per_cycle = 3
        for i in range(3):
            r = guard.try_edit(f"p-{i}", f"patch-{i}", f"sum-{i}")
            assert r.allowed is True
            guard.apply_canary(r.canary)
        r4 = guard.try_edit("p-4", "patch-4", "sum-4")
        assert r4.allowed is False
        assert r4.reason == "cycle_cap_exceeded"

    def test_tripwire_blocks_all(self, isolated_guard):
        guard = isolated_guard
        guard._read_fleet_health = lambda: FleetHealth(
            timestamp="t", tripwire=TripwireStatus.TRIPPED,
        )
        result = guard.try_edit("p", "patch", "summary")
        assert result.allowed is False
        assert "tripped" in result.reason

    def test_pause_blocks_all(self, isolated_guard):
        guard = isolated_guard
        guard._read_fleet_health = lambda: FleetHealth(
            timestamp="t", tripwire=TripwireStatus.PAUSED,
        )
        result = guard.try_edit("p", "patch", "summary")
        assert result.allowed is False
        assert "paused" in result.reason


# ---------------------------------------------------------------------------
# Canary apply-observe-promote
# ---------------------------------------------------------------------------


class TestCanaryCommitWiring:
    """P2-3 close-the-loop: commit capture + active-canary accessor."""

    def test_attach_commit_persists(self, isolated_guard):
        guard = isolated_guard
        r = guard.try_edit("p", "patch", "sum")
        guard.apply_canary(r.canary)
        assert guard.attach_commit(r.canary.edit_id, "abc123def456") is True
        # Reload from disk to prove persistence (commit survives serialisation).
        guard._canaries = []
        guard._load_state()
        c = guard._find_canary(r.canary.edit_id)
        assert c is not None and c.commit == "abc123def456"

    def test_attach_commit_unknown_id(self, isolated_guard):
        assert isolated_guard.attach_commit("nope", "abc") is False

    def test_active_canaries_filters_terminal(self, isolated_guard):
        guard = isolated_guard
        r = guard.try_edit("p", "patch", "sum")
        guard.apply_canary(r.canary)
        active = guard.active_canaries()
        assert [c.edit_id for c in active] == [r.canary.edit_id]
        # Force it terminal; it drops out of the active set.
        guard.revert_canary(r.canary.edit_id, "test")
        assert guard.active_canaries() == []

    def test_eval_domain_for(self, monkeypatch):
        import hermes_cli.blast_radius as br
        monkeypatch.setattr(
            br, "load_eval_domains", lambda: {"coder": "code"}
        )
        assert br.eval_domain_for("coder") == "code"
        assert br.eval_domain_for("mystery") is None


class TestCanaryApply:
    def test_apply_increments_count(self, isolated_guard):
        guard = isolated_guard
        r = guard.try_edit("p", "patch", "summary")
        assert guard._cycle_count == 0
        guard.apply_canary(r.canary)
        assert guard._cycle_count == 1
        assert r.canary.stage == CanaryStage.APPLIED

    def test_observe_sets_observing_stage(self, isolated_guard):
        guard = isolated_guard
        r = guard.try_edit("p", "patch", "summary")
        guard.apply_canary(r.canary)
        canary = guard.observe_canary(r.canary.edit_id, MockEvalRun(0.85))
        assert canary.observed_runs == 1
        assert canary.baseline_pass_rate == 0.85
        assert canary.stage == CanaryStage.OBSERVING

    def test_promotes_after_enough_runs(self, guard_single_observe):
        guard = guard_single_observe
        r = guard.try_edit("p", "patch", "summary")
        guard.apply_canary(r.canary)
        canary = guard.observe_canary(r.canary.edit_id, MockEvalRun(0.90))
        assert canary.stage == CanaryStage.PROMOTED

    def test_reverts_on_regression(self, guard_single_observe):
        guard = guard_single_observe
        r = guard.try_edit("p", "patch", "summary")
        guard.apply_canary(r.canary)
        # First obs sets baseline to 0.90
        canary = guard.observe_canary(r.canary.edit_id, MockEvalRun(0.90))
        # With observe_runs=1, stage is already PROMOTED or REVERTED
        # Reset for regression test
        canary.stage = CanaryStage.APPLIED
        canary.observed_runs = 1  # preserve baseline from first obs
        canary.baseline_pass_rate = 0.90
        canary.required_runs = 1
        result = guard.observe_canary(r.canary.edit_id, MockEvalRun(0.50))
        assert result.stage == CanaryStage.REVERTED
        assert result.error is not None

    def test_force_promote(self, isolated_guard):
        guard = isolated_guard
        r = guard.try_edit("p", "patch", "summary")
        guard.apply_canary(r.canary)
        canary = guard.observe_canary(r.canary.edit_id, MockEvalRun(0.85), promote=True)
        assert canary.stage == CanaryStage.PROMOTED

    def test_promote_to_additional_profiles(self, guard_single_observe):
        guard = guard_single_observe
        r = guard.try_edit("p", "patch", "summary")
        guard.apply_canary(r.canary)
        guard.observe_canary(r.canary.edit_id, MockEvalRun(0.85))
        assert guard.promote_to_profiles(r.canary.edit_id, ["p2", "p3"])
        canary = guard._find_canary(r.canary.edit_id)
        assert "p2" in canary.promoted_to

    def test_revert_manually(self, isolated_guard):
        guard = isolated_guard
        r = guard.try_edit("p", "patch", "summary")
        guard.apply_canary(r.canary)
        canary = guard.revert_canary(r.canary.edit_id, "Bad")
        assert canary.stage == CanaryStage.REVERTED
        assert "Bad" in canary.error

    def test_find_missing_none(self, isolated_guard):
        guard = isolated_guard
        assert guard._find_canary("nonexistent") is None
        assert guard.observe_canary("nonexistent", MockEvalRun(0.5)) is None
        assert guard.revert_canary("nonexistent") is None
        assert guard.promote_to_profiles("nonexistent", []) is False


# ---------------------------------------------------------------------------
# Fleet-health tripwire
# ---------------------------------------------------------------------------


class TestFleetHealth:
    def test_normal_when_healthy(self, isolated_guard):
        guard = isolated_guard
        health = guard.check_fleet_health(success_rate=0.85, eval_pass_rate=0.80)
        assert health.tripwire == TripwireStatus.NORMAL

    def test_warning_approaching(self, isolated_guard):
        guard = isolated_guard
        guard.tripwire_pass_rate = 0.60
        health = guard.check_fleet_health(success_rate=0.65)
        assert health.tripwire == TripwireStatus.WARNING

    def test_tripped_below_threshold(self, isolated_guard):
        guard = isolated_guard
        guard.tripwire_pass_rate = 0.60
        health = guard.check_fleet_health(success_rate=0.30)
        assert health.tripwire == TripwireStatus.TRIPPED

    def test_eval_pass_rate_trips(self, isolated_guard):
        guard = isolated_guard
        guard.tripwire_eval_pass_rate = 0.70
        health = guard.check_fleet_health(success_rate=0.90, eval_pass_rate=0.40)
        assert health.tripwire == TripwireStatus.TRIPPED


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "tripwire_state.json")
            fleet_path = os.path.join(tmp, "fleet_health.json")
            with patch("hermes_cli.blast_radius.TRIPWIRE_STATE_PATH", state_path):
                guard = EditGuard(max_edits_per_cycle=5)
                guard._write_fleet_health = lambda h: None
                guard._read_fleet_health = lambda: FleetHealth(
                    timestamp="t", tripwire=TripwireStatus.NORMAL,
                )
                r = guard.try_edit("p", "patch", "summary")
                guard.apply_canary(r.canary)

                # New guard loads the state
                guard2 = EditGuard(max_edits_per_cycle=5)
                assert guard2._cycle_count == 1
                assert len(guard2._canaries) == 1
                assert guard2._canaries[0].profile == "p"

    def test_corrupted_state_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "tripwire_state.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w") as f:
                f.write("not json {{{")
            with patch("hermes_cli.blast_radius.TRIPWIRE_STATE_PATH", state_path):
                guard = EditGuard()
                assert guard._cycle_count == 0
