"""P1 closure tests for ORCH V4 candidate-hash review findings."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import hashlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.kanban_orch_capability import (
    CapabilityGrant,
    install_fail_closed_udf,
    install_test_open_udf,
)
from hermes_cli.kanban_orch_db import OrchDBError, assert_not_live_path, open_orch_db, close_orch_db, grant
from hermes_cli.kanban_orch_api import (
    apply_lifecycle_transition_db,
    bootstrap_board_only_request,
)
from hermes_cli.kanban_orch_observer import (
    EXIT_HARD_VIOLATION,
    ExpectedNode,
    check_delivery_satisfied,
    check_exact_multiset,
    run_observer,
)
from hermes_cli.kanban_orch_schema_v4 import apply_schema
from hermes_cli.kanban_orch_schema_sidecar import apply_sidecar_schema
from hermes_cli.kanban_orch_bridge import OrchBridge, init_sidecar_db


def test_capability_fail_closed_by_default():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        conn = sqlite3.connect(path)
        apply_schema(conn, test_open_capability=False)
        val = conn.execute("SELECT orch_capability_ok('x','b','t','o',0,0,'k')").fetchone()[0]
        assert val == 0
        conn.close()
    finally:
        os.unlink(path)


def test_capability_grant_required_for_mutation():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        conn = open_orch_db(path, create=True, test_open_capability=False)
        apply_schema(conn, test_open_capability=False)
        # Without grant, protected insert fails.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO kanban_board_identity "
                "(singleton, board_instance_id, canonical_board_key, created_at) "
                "VALUES (1, 'board_0123456789abcdef', 'default', 1)"
            )
        # With grant, insert succeeds.
        grant(conn, CapabilityGrant(kind="maintenance_identity", board="board_0123456789abcdef", target_key="default"))
        conn.execute(
            "INSERT INTO kanban_board_identity "
            "(singleton, board_instance_id, canonical_board_key, created_at) "
            "VALUES (1, 'board_0123456789abcdef', 'default', 1)"
        )
        conn.commit()
        close_orch_db(conn)
    finally:
        os.unlink(path)


def test_expected_side_duplicate_nodes_rejected():
    expected = [
        ExpectedNode("lane-1", "lane", "A", True, 1),
        ExpectedNode("lane-1", "lane", "A", True, 2),
    ]
    finding = check_exact_multiset(expected, ["lane-1"])
    assert finding is not None
    assert finding.code == "duplicate_expected_nodes"
    assert finding.exit_code == EXIT_HARD_VIOLATION


def test_observer_delivery_obligation_key_multiset():
    # Count-only can still work for compat.
    assert check_delivery_satisfied("messaging", 2, 2) is None
    # Key multiset missing one required key fails even if counts could be lied about.
    f = check_delivery_satisfied(
        "messaging",
        required_obligation_keys=["obl-a", "obl-b"],
        acked_obligation_keys=["obl-a", "obl-a"],
    )
    assert f is not None
    assert f.code in {"duplicate_acked_obligation_keys", "delivery_required_acks_missing"}

    assert (
        check_delivery_satisfied(
            "messaging",
            required_obligation_keys=["obl-a", "obl-b"],
            acked_obligation_keys=["obl-a", "obl-b"],
        )
        is None
    )

    exit_code = run_observer(
        has_v4_schema=True,
        expected_nodes=[
            ExpectedNode("lane-1", "lane", "A", True, 1),
            ExpectedNode("lane-2", "lane", "B", True, 2),
        ],
        observed_node_keys=["lane-1", "lane-2"],
        accepted_runs=[],
        parent_state="completed",
        has_result=True,
        origin_kind="messaging",
        required_obligation_keys=["obl-1"],
        acked_obligation_keys=[],
    )
    assert exit_code == EXIT_HARD_VIOLATION


def test_i3_surface_modules_exist():
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    for rel in [
        "hermes_cli/kanban_orch_api.py",
        "hermes_cli/kanban_orch_db.py",
        "scripts/orch_v4_migrate.py",
    ]:
        assert os.path.isfile(os.path.join(root, rel)), rel


def test_db_backed_lifecycle_cas():
    tmp = tempfile.mkdtemp()
    try:
        native = os.path.join(tmp, "native.db")
        side = os.path.join(tmp, "side.db")
        n = sqlite3.connect(native)
        n.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, created_at INTEGER NOT NULL)")
        n.execute("INSERT INTO tasks VALUES ('task-1','t','pending',1)")
        n.commit()
        n.close()
        init_sidecar_db(side)
        bridge = OrchBridge(native, side)
        bound = bridge.bind_parent_task("board_0123456789abcdef", "", "orch-cas-1", "task-1")
        assert bound.lifecycle_state == "submitted"
        out = apply_lifecycle_transition_db(
            bridge._sidecar,
            board_instance_id="board_0123456789abcdef",
            tenant_scope="",
            orch_id="orch-cas-1",
            event="claim_decomposition",
            to_state="decomposing",
        )
        assert out["to_state"] == "decomposing"
        assert out["lifecycle_revision"] == 1
        row = bridge._sidecar.execute(
            "SELECT lifecycle_state, lifecycle_revision FROM orch_requests WHERE orch_id=?",
            ("orch-cas-1",),
        ).fetchone()
        assert row["lifecycle_state"] == "decomposing"
        assert row["lifecycle_revision"] == 1
        bridge.close()
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_migrate_script_refuses_live_path():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "orch_v4_migrate.py"
    spec = importlib.util.spec_from_file_location("orch_v4_migrate_mod", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    rc = mod.main(["--db", "/home/claw/.hermes/kanban.db", "--mode", "sidecar"])
    assert rc == 2


def test_live_path_guard():
    with pytest.raises(OrchDBError):
        assert_not_live_path("/home/claw/.hermes/kanban.db")
