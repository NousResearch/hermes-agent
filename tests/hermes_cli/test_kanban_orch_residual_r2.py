"""Residual R2 hostile closures for ORCH V4 candidate."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_canonical import (
    CanonicalError,
    assert_digest_matches,
    digest,
    request_digest,
)
from hermes_cli.kanban_orch_lifecycle import (
    LifecycleError,
    Request,
    Task,
    Node,
    apply_transition,
    cancel_cascade,
    cancellation_satisfied,
    finish_cancellation,
)
from hermes_cli.kanban_orch_delivery import (
    DeliveryError,
    DeliveryObligation,
    DeliveryReceipt,
    claim_obligation,
    finish_attempt,
    process_receipt,
    start_attempt,
)
from hermes_cli.kanban_orch_bridge import BridgeError, OrchBridge, init_sidecar_db


def test_server_digest_recompute_rejects_forged_claim():
    value = {"schema_version": 4, "kind": "orch_route", "x": 1}
    real = digest(value)
    assert assert_digest_matches(value, real) == real
    with pytest.raises(CanonicalError, match="digest_mismatch"):
        assert_digest_matches(value, "b" * 64)


def test_lifecycle_evidence_gates():
    req = Request("b", "", "o", "waiting_lanes")
    with pytest.raises(LifecycleError, match="missing_required_lane_acceptances"):
        apply_transition(req, "required_set_accepted", "synthesizing")
    ok = apply_transition(req, "required_set_accepted", "synthesizing", accepted_required_lanes=2)
    assert ok.state == "synthesizing"
    with pytest.raises(LifecycleError, match="missing_synthesis_result"):
        apply_transition(ok, "result_accepted", "work_accepted")
    ok2 = apply_transition(ok, "result_accepted", "work_accepted", has_result=True)
    with pytest.raises(LifecycleError, match="board_only_origin_required"):
        apply_transition(ok2, "explicit_board_only", "completed")
    assert apply_transition(ok2, "explicit_board_only", "completed", origin_kind="board_only").state == "completed"


def test_cancel_running_task_must_drain():
    req = Request("b", "", "o", "waiting_lanes")
    tasks = [Task("b", "", "o", "t1", "running")]
    nodes = [Node("b", "", "o", "n1", True, "running")]
    cancel_cascade(
        req,
        tasks=tasks,
        task_runs=[],
        nodes=nodes,
        leases=[],
        obligations=[],
        attempts=[],
        now=1,
    )
    assert req.state == "cancelling"
    assert cancellation_satisfied(
        req, tasks=tasks, task_runs=[], nodes=nodes, leases=[], obligations=[], attempts=[]
    ) is False
    tasks[0].status = "cancelled"
    tasks[0].cancellation_requested_at = None
    nodes[0].state = "cancelled"
    assert cancellation_satisfied(
        req, tasks=tasks, task_runs=[], nodes=nodes, leases=[], obligations=[], attempts=[]
    ) is True
    finish_cancellation(
        req, tasks=tasks, task_runs=[], nodes=nodes, leases=[], obligations=[], attempts=[]
    )
    assert req.state == "cancelled"


def test_delivery_attempt_cas_and_rejected_cannot_ack():
    d = "a" * 64
    obl = DeliveryObligation("o1", "k1", True, d, "provider", "message_id")
    claim_obligation(obl, owner="w", token_hash="t", ttl=1, now=1)
    a1 = start_attempt(
        obl, attempt_id=1, send_nonce="n1", payload_digest=d, result_digest=d,
        claim_owner="w", claim_token_hash="t", claim_epoch=1, lifecycle_revision=0, cancel_epoch=0,
    )
    with pytest.raises(DeliveryError, match="attempt_already_open"):
        start_attempt(
            obl, attempt_id=2, send_nonce="n2", payload_digest=d, result_digest=d,
            claim_owner="w", claim_token_hash="t", claim_epoch=1, lifecycle_revision=0, cancel_epoch=0,
        )
    finish_attempt(obl, a1, terminal_state="rejected", now=2)
    with pytest.raises(DeliveryError, match="rejected_attempt_cannot_ack"):
        process_receipt(
            obl,
            a1,
            DeliveryReceipt(1, "o1", "n1", d, d, d, "provider", "message_id"),
            now=3,
        )


def test_bridge_board_tenant_scope_and_atomic_init(tmp_path):
    native = tmp_path / "n.db"
    side = tmp_path / "s.db"
    nc = sqlite3.connect(native)
    nc.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, "
        "created_at INTEGER NOT NULL, tenant TEXT)"
    )
    nc.execute("INSERT INTO tasks VALUES ('task-1','t','pending',1,'')")
    nc.commit()
    nc.close()

    init_sidecar_db(str(side))
    # second init must fail (exists)
    with pytest.raises(BridgeError, match="sidecar_exists"):
        init_sidecar_db(str(side))

    br = OrchBridge(str(native), str(side), board_instance_id="board_0123456789abcdef", tenant_scope="")
    br.bind_parent_task("board_0123456789abcdef", "", "orch-1", "task-1")
    with pytest.raises(BridgeError, match="board_tenant_scope_mismatch"):
        br.bind_parent_task("board_FOREIGN_12345678", "other", "orch-2", "task-1")
    br.close()
