from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.kanban_orch_bridge import init_sidecar_db
from hermes_cli.kanban_orch_cmin import judge_board_only_to_fixed_point
from hermes_cli.kanban_orch_dual_bind import dual_bind_parent_task
from hermes_cli.kanban_orch_multilane import MultiLaneError, run_multilane_once
from hermes_cli.kanban_orch_writer_switch import open_live_bridge, writer_enabled


def _setup(tmp_path, monkeypatch, *, child_statuses=("done", "done")):
    native = tmp_path / "n.db"
    side = tmp_path / "s.db"
    cfg = tmp_path / "w.json"
    nc = sqlite3.connect(native)
    nc.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, "
        "created_at INTEGER NOT NULL, tenant TEXT)"
    )
    nc.execute(
        "CREATE TABLE task_links (parent_id TEXT NOT NULL, child_id TEXT NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'depends', PRIMARY KEY (parent_id, child_id))"
    )
    nc.execute("INSERT INTO tasks VALUES ('p1','parent','running',1,'')")
    for i, st in enumerate(child_statuses, start=1):
        cid = f"c{i}"
        nc.execute("INSERT INTO tasks VALUES (?,?,?,1,'')", (cid, f"child{i}", st))
        nc.execute("INSERT INTO task_links VALUES (?,?,?)", ("p1", cid, "depends"))
    nc.commit()
    nc.close()
    init_sidecar_db(str(side))
    cfg.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "enabled_default": True,
                "native_db": str(native),
                "sidecar_db": str(side),
                "native_writable": False,
                "canonical_board_key": "tmp",
                "tenant_scope": "",
            }
        )
    )
    monkeypatch.delenv("ORCH_V4_WRITER", raising=False)
    assert writer_enabled(cfg) is True
    bound = dual_bind_parent_task(task_id="p1", title="parent", cfg_path=cfg)
    assert bound.error is None, bound
    return native, side, cfg, bound


def test_multilane_materialize_accept_complete(tmp_path, monkeypatch):
    native, side, cfg, bound = _setup(tmp_path, monkeypatch)
    br = open_live_bridge(cfg)
    try:
        # first claim decomposition via board_only path
        mid = judge_board_only_to_fixed_point(
            br._sidecar,
            board_instance_id=bound.board_instance_id or "",
            tenant_scope=bound.tenant_scope,
            orch_id=bound.orch_id or "",
            native_status="running",
        )
        assert mid.after_state == "decomposing", mid

        children = [
            {"child_id": "c1", "status": "done", "title": "child1"},
            {"child_id": "c2", "status": "done", "title": "child2"},
        ]
        # materialize
        r1 = run_multilane_once(
            br._sidecar,
            board_instance_id=bound.board_instance_id or "",
            tenant_scope=bound.tenant_scope,
            orch_id=bound.orch_id or "",
            parent_task_id="p1",
            parent_title="parent",
            parent_status="running",
            children=children,
        )
        assert r1.after_state == "waiting_lanes", r1
        assert r1.plan_version == 1
        assert any(s.action == "materialize" for s in r1.steps)

        # accept + complete
        r2 = run_multilane_once(
            br._sidecar,
            board_instance_id=bound.board_instance_id or "",
            tenant_scope=bound.tenant_scope,
            orch_id=bound.orch_id or "",
            parent_task_id="p1",
            parent_title="parent",
            parent_status="running",
            children=children,
        )
        assert r2.after_state == "completed", r2
        assert r2.accepted_required_lanes >= 2

        # durable rows exist
        n_plans = br._sidecar.execute("SELECT count(*) FROM orch_plans").fetchone()[0]
        n_nodes = br._sidecar.execute("SELECT count(*) FROM orch_nodes").fetchone()[0]
        n_acc = br._sidecar.execute("SELECT count(*) FROM orch_node_acceptances").fetchone()[0]
        n_mat = br._sidecar.execute("SELECT count(*) FROM orch_plan_materializations").fetchone()[0]
        assert n_plans == 1
        assert n_nodes >= 3
        assert n_acc >= 2
        assert n_mat == 1
    finally:
        br.close()


def test_multilane_need_two_children(tmp_path, monkeypatch):
    native, side, cfg, bound = _setup(tmp_path, monkeypatch, child_statuses=("done",))
    br = open_live_bridge(cfg)
    try:
        judge_board_only_to_fixed_point(
            br._sidecar,
            board_instance_id=bound.board_instance_id or "",
            tenant_scope=bound.tenant_scope,
            orch_id=bound.orch_id or "",
            native_status="running",
        )
        r = run_multilane_once(
            br._sidecar,
            board_instance_id=bound.board_instance_id or "",
            tenant_scope=bound.tenant_scope,
            orch_id=bound.orch_id or "",
            parent_task_id="p1",
            parent_title="parent",
            parent_status="running",
            children=[{"child_id": "c1", "status": "done", "title": "c1"}],
        )
        assert r.skipped is True
        assert r.reason and "no_multilane_rule" in r.reason
    finally:
        br.close()
