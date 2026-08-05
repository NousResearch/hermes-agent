"""M7 T42: Canary — root identity, egress denial, deadline, cleanup."""

import pytest
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_canary import (
    ALLOWED_CONSUMERS,
    CanaryError,
    CanaryScenarioResult,
    DEADLINES,
    LocalCaptureAdapter,
    cleanup_canary,
    prepare_canary,
    run_scenario_crash_recovery,
    run_scenario_normal,
    run_scenario_replay_owner_race,
    verify_canary,
)


def test_canary_isolation_and_egress():
    """T42: canary root identity/deny egress/deadline/cleanup.

    CSPRNG root, O_EXCL, reject existing/symlink.
    Egress hard-denied outside sink allowlist.
    Deadlines match §16.1.
    Cleanup removes root.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        # Prepare: CSPRNG root
        prepare = prepare_canary(
            run_id="test-001",
            source_hash="a" * 64,
            root_parent=tmpdir,
            now=1000,
        )
        assert prepare.identity.board_instance_id != "test-001"  # CSPRNG, not run_id
        assert len(prepare.identity.board_instance_id) == 32  # 16 bytes hex
        assert prepare.identity.tenant_scope == "orch-v4-canary-test-001"
        assert prepare.identity.schema_version == 4
        assert os.path.isdir(prepare.identity.root_path)

        # Reject existing root
        with pytest.raises(CanaryError, match="canary_root_exists"):
            prepare_canary(
                run_id="test-001",
                source_hash="a" * 64,
                root_parent=tmpdir,
            )

        # Deadlines match §16.1
        assert DEADLINES["startup"] == 30
        assert DEADLINES["scenario"] == 180
        assert DEADLINES["idle"] == 30
        assert DEADLINES["sigterm_drain"] == 15
        assert DEADLINES["sigkill_reap"] == 5
        assert DEADLINES["cleanup"] == 30

        # Egress denial: allowed consumers pass
        adapter = LocalCaptureAdapter()
        assert not adapter.egress_denied({"consumer_kind": "gateway_message_bridge"})
        assert not adapter.egress_denied({"consumer_kind": "cli_foreground_bridge"})

        # Egress denial: unknown consumers denied
        assert adapter.egress_denied({"consumer_kind": "malicious_caller"})
        assert adapter.egress_denied({"consumer_kind": ""})
        assert adapter.egress_denied({})  # missing consumer_kind

        # Local capture adapter records sends
        adapter.send({"consumer_kind": "gateway_message_bridge"}, b"payload")
        assert len(adapter.sends) == 1
        assert adapter.sends[0]["accepted"] is True
        assert adapter.sends[0]["provider_message_id"].startswith("local-")

        # Cleanup
        assert cleanup_canary(prepare)
        assert not os.path.exists(prepare.identity.root_path)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_canary_three_scenarios():
    """T42: three scenarios — normal, replay-owner-race, crash-recovery."""
    tmpdir = tempfile.mkdtemp()
    try:
        prepare = prepare_canary(
            run_id="test-002",
            source_hash="b" * 64,
            root_parent=tmpdir,
            now=2000,
        )
        adapter = LocalCaptureAdapter()

        # Normal scenario
        normal = run_scenario_normal(prepare, adapter)
        assert normal.scenario == "normal"
        assert normal.passed
        assert normal.exit_code == 0
        assert len(normal.findings) == 0

        # Replay-owner-race
        race = run_scenario_replay_owner_race(prepare, adapter, workers=8)
        assert race.scenario == "replay-owner-race"
        assert race.passed
        assert race.exit_code == 0

        # Crash-recovery
        crash = run_scenario_crash_recovery(prepare, adapter)
        assert crash.scenario == "crash-recovery"
        assert crash.passed
        assert crash.exit_code == 0

        cleanup_canary(prepare)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_canary_verify_aborts():
    """T42: verify aborts on identity drift, egress, active process, unknown."""
    tmpdir = tempfile.mkdtemp()
    try:
        prepare = prepare_canary(
            run_id="test-003",
            source_hash="c" * 64,
            root_parent=tmpdir,
            now=3000,
        )
        adapter = LocalCaptureAdapter()

        # Normal results
        results = [
            run_scenario_normal(prepare, adapter),
            run_scenario_replay_owner_race(prepare, adapter),
            run_scenario_crash_recovery(prepare, adapter),
        ]

        # Clean verify
        verify = verify_canary(prepare, results)
        assert verify.passed
        assert len(verify.abort_reasons) == 0

        # Identity drift
        verify = verify_canary(prepare, results, identity_drift=True)
        assert not verify.passed
        assert "identity_drift" in verify.abort_reasons

        # Egress violation
        verify = verify_canary(prepare, results, egress_violation=True)
        assert not verify.passed
        assert "egress_violation" in verify.abort_reasons

        # Active process after deadline
        verify = verify_canary(prepare, results, active_processes=2)
        assert not verify.passed
        assert "active_process_after_deadline" in verify.abort_reasons

        # Active lease after deadline
        verify = verify_canary(prepare, results, active_leases=1)
        assert not verify.passed
        assert "active_lease_after_deadline" in verify.abort_reasons

        # Unknown unresolved
        verify = verify_canary(prepare, results, unknown_resolved=False)
        assert not verify.passed
        assert "unknown_unresolved" in verify.abort_reasons

        # Scenario failure with duplicate finding
        bad_results = [
            CanaryScenarioResult(scenario="normal", passed=False, findings=["duplicate_lane"]),
        ]
        verify = verify_canary(prepare, bad_results)
        assert not verify.passed
        assert any("scenario_normal_failed" in r for r in verify.abort_reasons)
        assert any("duplicate" in r for r in verify.abort_reasons)

        cleanup_canary(prepare)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
