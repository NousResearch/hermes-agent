"""Sidecar integration tests: M1-M8 models wired to sidecar DB via bridge.

Tests prove:
1. Full ORCH lifecycle runs against sidecar persistence
2. Native DB never mutated (SHA-256 before == after)
3. Soft FK enforced (missing native task rejected)
4. Observer exit clean on valid lifecycle
5. Delivery satisfaction wired to completion
6. Reconcile events enqueued on sidecar
7. Supersession creates new generation
8. Canary + rollback models work in sidecar context
"""

import pytest
import os
import sys
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_integration import (
    IntegrationError,
    SidecarTestHarness,
    run_full_orch_lifecycle,
    submit_orch_request,
)
from hermes_cli.kanban_orch_lifecycle import supersede_request, Request as LifecycleRequest
from hermes_cli.kanban_orch_canary import (
    LocalCaptureAdapter,
    prepare_canary,
    run_scenario_normal,
    verify_canary,
    cleanup_canary,
)
from hermes_cli.kanban_orch_rollback import (
    create_rollback,
    advance_phase,
    check_no_early_reopen,
    authorize_reopen,
)
from hermes_cli.kanban_orch_observer import EXIT_CLEAN


@pytest.fixture
def harness():
    h = SidecarTestHarness.create()
    sha_before = h.native_sha256()
    yield h, sha_before
    h.cleanup()


def test_full_lifecycle_on_sidecar(harness):
    """Full ORCH lifecycle: submit→decompose→accept→synthesize→deliver→complete."""
    h, sha_before = harness
    results = run_full_orch_lifecycle(h)

    assert "request_key" in results["submit"]
    assert len(results["submit"]["request_key"]) == 64
    assert results["submit"]["parent_task_ref"]["id"] == "parent-1"
    assert results["plan"]["requirements"] == 2
    assert results["acceptances"]["run1"]["node_key"] == "lane-1"
    assert results["acceptances"]["run2"]["node_key"] == "lane-2"
    assert results["synthesis"]["state"] == "work_accepted"
    assert results["observer"]["exit_code"] == EXIT_CLEAN
    assert results["delivery"]["satisfied"] is True
    assert results["completion"]["state"] == "completed"
    assert results["completion"]["terminal"] is True
    assert results["reconcile"]["events"] == 1
    assert results["reconcile"]["queue_items"] == 1

    sha_after = h.native_sha256()
    assert sha_before == sha_after, "Native DB mutated during full lifecycle!"


def test_soft_fk_rejects_missing_task(harness):
    """Soft FK: submitting with nonexistent parent_task_id raises."""
    h, sha_before = harness
    with pytest.raises(IntegrationError, match="soft_fk_violation"):
        submit_orch_request(h, board_instance_id="board_0123456789abcdef",
                            tenant_scope="", orch_id="orch-bad", parent_task_id="nonexistent-task")
    assert sha_before == h.native_sha256()


def test_supersession_on_sidecar(harness):
    """Supersession: failed request→new generation on sidecar."""
    h, sha_before = harness
    submit_orch_request(h, board_instance_id="board_0123456789abcdef",
                        tenant_scope="", orch_id="orch-gen1", parent_task_id="parent-1")
    predecessor = LifecycleRequest("board_0123456789abcdef", "", "orch-gen1", "failed",
                                    lifecycle_revision=5, generation=1)
    successor = supersede_request(predecessor, new_orch_id="orch-gen2")
    assert successor.generation == 2
    assert successor.orch_id == "orch-gen2"
    assert successor.state == "submitted"
    submit_orch_request(h, board_instance_id="board_0123456789abcdef",
                        tenant_scope="", orch_id=successor.orch_id, parent_task_id="parent-1")
    assert sha_before == h.native_sha256()


def test_canary_in_sidecar_context(harness):
    """Canary model works in sidecar context (local sink only)."""
    h, sha_before = harness
    canary_parent = os.path.join(h.tmpdir, "canary-root")
    os.makedirs(canary_parent, exist_ok=True)
    prepare = prepare_canary(run_id="int-test-001", source_hash="f"*64, root_parent=canary_parent, now=1000)
    adapter = LocalCaptureAdapter()
    normal = run_scenario_normal(prepare, adapter)
    assert normal.passed
    verify = verify_canary(prepare, [normal])
    assert verify.passed
    cleanup_canary(prepare)
    assert sha_before == h.native_sha256()


def test_rollback_model_in_sidecar_context(harness):
    """Rollback model works in sidecar context (no live writer switch)."""
    h, sha_before = harness
    op = create_rollback(operation_id="rb-int-001", owner_token_hash="tok-hash",
                         source_live_hash="a"*64, source_code_hash="src"+"a"*61,
                         target_live_hash="b"*64, target_code_hash="tgt"+"b"*61)
    for phase in ["fence_draining", "workers_stopped", "leases_revoked",
                   "snapshot_sealed", "code_switched", "old_writer_receipts_verified", "verified"]:
        advance_phase(op, target_phase=phase, owner_token_hash="tok-hash", expected_revision=op.phase_revision)
        assert check_no_early_reopen(op) is True
    authorize_reopen(op, david_token="david-approval")
    advance_phase(op, target_phase="reopened", owner_token_hash="tok-hash", expected_revision=op.phase_revision)
    assert op.fence_generation == 0
    assert sha_before == h.native_sha256()


def test_sidecar_persists_board_identity(harness):
    """Sidecar DB persists board identity from submit."""
    h, sha_before = harness
    submit_orch_request(h, board_instance_id="board_persist_test_0123",
                        tenant_scope="", orch_id="orch-persist-1", parent_task_id="parent-1")
    row = h.sidecar_conn.execute(
        "SELECT board_instance_id FROM kanban_board_identity WHERE board_instance_id = ?",
        ("board_persist_test_0123",)).fetchone()
    assert row is not None
    assert row[0] == "board_persist_test_0123"
    assert sha_before == h.native_sha256()


def test_no_live_paths_in_integration(harness):
    """Integration tests must not use live kanban.db path."""
    h, _ = harness
    LIVE_PATHS = {"/home/claw/.hermes/kanban.db", "/home/claw/.hermes/orch_v4.db"}
    assert h.native_path not in LIVE_PATHS
    assert h.sidecar_path not in LIVE_PATHS
    assert h.tmpdir not in LIVE_PATHS
    assert "/home/claw/.hermes" not in h.tmpdir
