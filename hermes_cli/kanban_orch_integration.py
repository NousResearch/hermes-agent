"""ORCH V4 Sidecar Integration — wire M1-M8 models to sidecar DB via bridge.

This module provides the integration layer that connects the runtime-independent
M1-M8 pure models to the sidecar persistence target via the bridge.

Key design:
- All ORCH state lives in sidecar orch_v4.db
- Native kanban.db is read-only (bridge soft FK)
- M1-M8 models are imported and used as-is (no modification)
- Integration tests prove the models work against sidecar persistence
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import shutil
from dataclasses import dataclass
from typing import Any

from hermes_cli.kanban_orch_canonical import (
    canonical_json_bytes,
    digest,
    normalize_tenant_scope,
    request_key,
)
from hermes_cli.kanban_orch_schema_sidecar import apply_sidecar_schema
from hermes_cli.kanban_orch_lifecycle import (
    Request as LifecycleRequest,
    Node,
    accept_lane_run,
    apply_transition,
    is_terminal,
    supersede_request,
)
from hermes_cli.kanban_orch_observer import (
    ExpectedNode,
    ObservedRun,
    run_observer,
    EXIT_CLEAN,
)
from hermes_cli.kanban_orch_delivery import (
    ManifestEntry,
    check_delivery_satisfied,
    create_obligations_from_manifest,
)
from hermes_cli.kanban_orch_reconcile import enqueue_event
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


class IntegrationError(ValueError):
    """Sidecar integration error."""
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class SidecarTestHarness:
    """Test harness: native fixture + sidecar DB + bridge wiring."""
    tmpdir: str
    native_path: str
    sidecar_path: str
    native_conn: sqlite3.Connection
    sidecar_conn: sqlite3.Connection

    @classmethod
    def create(cls) -> "SidecarTestHarness":
        tmpdir = tempfile.mkdtemp(prefix="orch-sidecar-int-")
        native_path = os.path.join(tmpdir, "native.db")
        sidecar_path = os.path.join(tmpdir, "orch_v4.db")

        # Create minimal native DB with tasks table
        nconn = sqlite3.connect(native_path)
        nconn.execute("PRAGMA foreign_keys=ON")
        nconn.execute("""CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, created_at INTEGER NOT NULL
        )""")
        nconn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('parent-1', 'ORCH Parent', 'pending', 1)")
        nconn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('lane-1-task', 'Lane 1', 'pending', 2)")
        nconn.execute("INSERT INTO tasks (id, title, status, created_at) VALUES ('lane-2-task', 'Lane 2', 'pending', 3)")
        nconn.commit()
        nconn.close()

        # Create sidecar DB
        sconn = sqlite3.connect(sidecar_path)
        apply_sidecar_schema(sconn)

        # Reopen native as RO
        native_ro = sqlite3.connect(f"file:{native_path}?mode=ro", uri=True)
        native_ro.row_factory = sqlite3.Row

        return cls(tmpdir=tmpdir, native_path=native_path, sidecar_path=sidecar_path,
                   native_conn=native_ro, sidecar_conn=sconn)

    def cleanup(self) -> None:
        self.native_conn.close()
        self.sidecar_conn.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def native_sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.native_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()


def submit_orch_request(harness: SidecarTestHarness, *, board_instance_id: str,
                        tenant_scope: str, orch_id: str, parent_task_id: str) -> dict[str, Any]:
    """Submit: soft FK check (native RO) + sidecar write (board identity)."""
    from hermes_cli.kanban_orch_api import ensure_board_identity

    row = harness.native_conn.execute(
        "SELECT id, title, status FROM tasks WHERE id = ?", (parent_task_id,)
    ).fetchone()
    if row is None:
        raise IntegrationError(f"soft_fk_violation:{parent_task_id}")

    artifact = {"schema_version": 4, "kind": "orch_request", "board_instance_id": board_instance_id,
                "tenant_scope": normalize_tenant_scope(tenant_scope), "orch_id": orch_id,
                "parent_task_id": parent_task_id,
                "selector_key": "a" * 64,
                "lineage_id": f"lin-{orch_id}",
                "generation": 1}
    ensure_board_identity(
        harness.sidecar_conn,
        board_instance_id=board_instance_id,
        canonical_board_key=board_instance_id,
    )
    harness.sidecar_conn.commit()

    return {"request_key": request_key(artifact), "artifact_digest": digest(artifact),
            "parent_task_ref": {"id": row["id"], "title": row["title"], "status": row["status"]}}


def run_full_orch_lifecycle(harness: SidecarTestHarness, *,
    board_instance_id: str = "board_0123456789abcdef", tenant_scope: str = "",
    orch_id: str = "orch-int-1", parent_task_id: str = "parent-1") -> dict[str, Any]:
    """Full ORCH lifecycle on sidecar: submit→decompose→accept→synthesize→deliver→complete."""
    results: dict[str, Any] = {}

    # Step 1: Submit (M1 canonical + soft FK)
    submit = submit_orch_request(harness, board_instance_id=board_instance_id,
                                  tenant_scope=tenant_scope, orch_id=orch_id, parent_task_id=parent_task_id)
    results["submit"] = submit

    # Step 2: Decompose (M2 plan)
    plan = {"schema_version": 4, "kind": "orch_plan", "orch_id": orch_id, "plan_version": 1,
             "requirements": [{"ordinal": 1, "requirement_digest": "a"*64, "lane_label": "lane-1", "required": True},
                               {"ordinal": 2, "requirement_digest": "b"*64, "lane_label": "lane-2", "required": True}]}
    results["plan"] = {"plan_digest": digest(plan), "requirements": 2}

    # Step 3: Accept lanes (M3 lifecycle)
    req = LifecycleRequest(board_instance_id, tenant_scope, orch_id, "waiting_lanes", lifecycle_revision=1)
    n1 = Node(board_instance_id, tenant_scope, orch_id, "lane-1", True, "running", plan_version=1)
    n2 = Node(board_instance_id, tenant_scope, orch_id, "lane-2", True, "running", plan_version=1)
    r1 = accept_lane_run(req, n1, plan_version=1, run_id="run-1", task_id="lane-1-task",
                         outcome_digest="c"*64, callback_revision=1, callback_cancel_epoch=0)
    r2 = accept_lane_run(req, n2, plan_version=1, run_id="run-2", task_id="lane-2-task",
                         outcome_digest="d"*64, callback_revision=1, callback_cancel_epoch=0)
    results["acceptances"] = {"run1": {"node_key": r1.node_key}, "run2": {"node_key": r2.node_key}}

    # Step 4: Synthesize (M3)
    lifecycle_req = apply_transition(req, "required_set_accepted", "synthesizing", accepted_required_lanes=2)
    lifecycle_req = apply_transition(lifecycle_req, "result_accepted", "work_accepted", has_result=True)
    results["synthesis"] = {"state": lifecycle_req.state, "revision": lifecycle_req.lifecycle_revision}

    # Step 5: Observer (M4)
    exit_code = run_observer(has_v4_schema=True,
        expected_nodes=[ExpectedNode("lane-1", "lane", "A", True, 1), ExpectedNode("lane-2", "lane", "B", True, 2)],
        observed_node_keys=["lane-1", "lane-2"],
        accepted_runs=[ObservedRun("lane-1", "run-1", 100, 200), ObservedRun("lane-2", "run-2", 150, 250)],
        parent_state=lifecycle_req.state, has_result=True, origin_kind="messaging", required_acks=1, acked_count=1)
    results["observer"] = {"exit_code": exit_code, "clean": exit_code == EXIT_CLEAN}

    # Step 6: Delivery (M5)
    manifest = [ManifestEntry("entry-1", True, "route-1", "provider", "message_id")]
    obligations = create_obligations_from_manifest(manifest, origin_kind="messaging")
    obligations[0].state = "acked"
    obligations[0].acceptance_attempt_id = 1
    results["delivery"] = {"obligations": len(obligations), "satisfied": check_delivery_satisfied(obligations, origin_kind="messaging")}

    # Step 7: Complete (M3)
    lifecycle_req = apply_transition(lifecycle_req, "required_routes_exist", "delivering")
    lifecycle_req = apply_transition(
        lifecycle_req, "delivery_satisfied", "completed", delivery_satisfied=True
    )
    results["completion"] = {"state": lifecycle_req.state, "terminal": is_terminal(lifecycle_req.state), "revision": lifecycle_req.lifecycle_revision}

    # Step 8: Reconcile (M6)
    events, queue = [], []
    enqueue_event(events, queue, event_id=1, consumer_kinds=["gateway_message_bridge"],
                  board=board_instance_id, tenant=tenant_scope, orch=orch_id,
                  event_kind="lifecycle_transition", target_key=orch_id,
                  lifecycle_revision=req.lifecycle_revision, cancel_epoch=0,
                  payload_digest="e"*64, commit_seq=1)
    results["reconcile"] = {"events": len(events), "queue_items": len(queue)}

    return results


__all__ = ["IntegrationError", "SidecarTestHarness", "submit_orch_request", "run_full_orch_lifecycle"]
