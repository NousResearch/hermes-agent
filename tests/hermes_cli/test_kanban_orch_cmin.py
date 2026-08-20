from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.kanban_orch_bridge import init_sidecar_db
from hermes_cli.kanban_orch_cmin import (
    children_progress,
    judge_board_only_to_fixed_point,
    live_tick_once,
    read_native_children,
)
from hermes_cli.kanban_orch_dual_bind import dual_bind_parent_task
from hermes_cli.kanban_orch_lifecycle import LifecycleError, Request, apply_transition
from hermes_cli.kanban_orch_writer_switch import open_live_bridge, writer_enabled


def _setup_pair(tmp_path, monkeypatch, *, parent_status="running", children=None):
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
    nc.execute(f"INSERT INTO tasks VALUES ('t1','hello',?,1,'')", (parent_status,))
    for i, st in enumerate(children or [], start=1):
        cid = f"c{i}"
        nc.execute("INSERT INTO tasks VALUES (?,?,?,1,'')", (cid, f"child{i}", st))
        nc.execute("INSERT INTO task_links VALUES (?,?,?)", ("t1", cid, "depends"))
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
    bound = dual_bind_parent_task(task_id="t1", title="hello", cfg_path=cfg)
    assert bound.error is None, bound
    return native, side, cfg, bound


def test_board_only_parent_done_evidence_gate():
    req = Request("b", "", "o", "submitted")
    with pytest.raises(LifecycleError, match="board_only_parent_not_done"):
        apply_transition(req, "board_only_parent_done", "completed", origin_kind="board_only")
    ok = apply_transition(
        req,
        "board_only_parent_done",
        "completed",
        origin_kind="board_only",
        native_parent_done=True,
    )
    assert ok.state == "completed"
    ok2 = apply_transition(
        Request("b", "", "o", "decomposing", lifecycle_revision=1),
        "board_only_parent_done",
        "completed",
        origin_kind="board_only",
        children_all_done=True,
    )
    assert ok2.state == "completed"


def test_cmin_fixed_point_temp(tmp_path, monkeypatch):
    native, side, cfg, bound = _setup_pair(tmp_path, monkeypatch)
    br = open_live_bridge(cfg)
    try:
        mid = judge_board_only_to_fixed_point(
            br._sidecar,
            board_instance_id=bound.board_instance_id or "",
            tenant_scope=bound.tenant_scope,
            orch_id=bound.orch_id or "",
            native_status="running",
        )
        assert mid.after_state == "decomposing", mid
        nc = sqlite3.connect(native)
        nc.execute("UPDATE tasks SET status='done' WHERE id='t1'")
        nc.commit()
        nc.close()
        fin = judge_board_only_to_fixed_point(
            br._sidecar,
            board_instance_id=bound.board_instance_id or "",
            tenant_scope=bound.tenant_scope,
            orch_id=bound.orch_id or "",
            native_status="done",
        )
        assert fin.after_state == "completed", fin
        assert any(s.event == "board_only_parent_done" for s in fin.steps)
    finally:
        br.close()


def test_children_all_done_completes_while_parent_running(tmp_path, monkeypatch):
    native, side, cfg, bound = _setup_pair(
        tmp_path, monkeypatch, parent_status="running", children=["done", "done"]
    )
    br = open_live_bridge(cfg)
    try:
        kids = read_native_children(br._native, "t1")
        total, done, all_done = children_progress(kids)
        assert (total, done, all_done) == (2, 2, True)
        fin = judge_board_only_to_fixed_point(
            br._sidecar,
            board_instance_id=bound.board_instance_id or "",
            tenant_scope=bound.tenant_scope,
            orch_id=bound.orch_id or "",
            native_status="running",
            children_all_done=True,
            children_total=2,
            children_done=2,
        )
        assert fin.after_state == "completed", fin
        assert fin.children_all_done is True
    finally:
        br.close()


def test_live_tick_once_scans_open(tmp_path, monkeypatch):
    native, side, cfg, bound = _setup_pair(tmp_path, monkeypatch, parent_status="ready")
    from hermes_cli import kanban_orch_writer_switch as ws
    from hermes_cli import kanban_orch_cmin as cmin

    orig = ws.open_live_bridge

    def _open(cfg_path=None):
        return orig(cfg_path or cfg)

    monkeypatch.setattr(ws, "open_live_bridge", _open)
    monkeypatch.setattr(cmin, "open_live_bridge", _open)

    tick = live_tick_once(limit=10)
    assert tick.scanned >= 1
    assert tick.advanced >= 1
    assert any(r.after_state == "decomposing" for r in tick.results)

    # parent done → completed on next tick
    nc = sqlite3.connect(native)
    nc.execute("UPDATE tasks SET status='done' WHERE id='t1'")
    nc.commit()
    nc.close()
    tick2 = live_tick_once(limit=10)
    assert tick2.completed >= 1
    assert any(r.after_state == "completed" for r in tick2.results)
