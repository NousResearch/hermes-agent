
from __future__ import annotations
import json, os, sqlite3, tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hermes_cli.kanban_orch_bridge import init_sidecar_db, OrchBridge
from hermes_cli.kanban_orch_dual_bind import dual_bind_parent_task, preflight_dual_bind
from hermes_cli.kanban_orch_writer_switch import writer_enabled

def test_dual_bind_temp(tmp_path, monkeypatch):
    native = tmp_path / "n.db"
    side = tmp_path / "s.db"
    cfg = tmp_path / "w.json"
    nc = sqlite3.connect(native)
    nc.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, status TEXT NOT NULL, created_at INTEGER NOT NULL, tenant TEXT)")
    nc.execute("INSERT INTO tasks VALUES ('t1','hello','pending',1,'')")
    nc.commit(); nc.close()
    init_sidecar_db(str(side))
    cfg.write_text(json.dumps({
        "schema_version": 4,
        "enabled_default": True,
        "native_db": str(native),
        "sidecar_db": str(side),
        "native_writable": False,
        "canonical_board_key": "tmp",
        "tenant_scope": "",
    }))
    monkeypatch.delenv("ORCH_V4_WRITER", raising=False)
    assert writer_enabled(cfg) is True
    pf = preflight_dual_bind(cfg)
    assert pf["ok"] is True
    res = dual_bind_parent_task(task_id="t1", title="hello", cfg_path=cfg)
    assert res.error is None, res
    assert res.skipped is False
    assert res.orch_id
    # kill switch
    monkeypatch.setenv("ORCH_V4_WRITER", "0")
    assert writer_enabled(cfg) is False
