from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.kanban_orch_bridge import init_sidecar_db
from hermes_cli.kanban_orch_cmin import judge_board_only_to_fixed_point
from hermes_cli.kanban_orch_dual_bind import dual_bind_parent_task
from hermes_cli.kanban_orch_lifecycle import LifecycleError, Request, apply_transition
from hermes_cli.kanban_orch_writer_switch import writer_enabled


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


def test_cmin_fixed_point_temp(tmp_path, monkeypatch):
    native = tmp_path / "n.db"
    side = tmp_path / "s.db"
    cfg = tmp_path / "w.json"
    nc = sqlite3.connect(native)
    nc.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, "
        "created_at INTEGER NOT NULL, tenant TEXT)"
    )
    nc.execute("INSERT INTO tasks VALUES ('t1','hello','running',1,'')")
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

    # open sidecar with same grants path via dual_bind already wrote request
    from hermes_cli.kanban_orch_writer_switch import open_live_bridge

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
        # mark native done and complete
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
