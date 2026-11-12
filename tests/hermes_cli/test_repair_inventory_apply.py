"""Apply inventories tombstones without migrating them or disabling live fixes (#149132)."""
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.sessions_repair_profiles import RepairPlan, Store, _Session
from hermes_constants import mark_named_profile_deleted
from hermes_state import SessionDB


@pytest.mark.parametrize("schema", ["current", "legacy", "legacy-minimal"])
def test_apply_inventories_tombstone_read_only_and_repairs_live_target(tmp_path, monkeypatch, schema):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    science, ghost = (tmp_path / "profiles" / name for name in ("science", "ghost"))
    science.mkdir(parents=True)
    ghost.mkdir()
    local, target = SessionDB(tmp_path / "state.db"), SessionDB(science / "state.db")
    ghost_path = ghost / "state.db"
    try:
        if schema == "current":
            with SessionDB(ghost_path) as db:
                db.create_session("duplicated", "cli")
        else:
            with sqlite3.connect(ghost_path) as conn:
                if schema == "legacy":
                    conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY,parent_session_id TEXT,ended_at REAL,"
                                 "end_reason TEXT,profile_name TEXT,session_key TEXT,message_count INTEGER)")
                    conn.execute("INSERT INTO sessions(id) VALUES ('duplicated')")
                else:
                    conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,started_at REAL,ended_at REAL)")
                    conn.execute("INSERT INTO sessions VALUES ('duplicated','cli',1,NULL)")
        mark_named_profile_deleted(ghost)
        local._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('duplicated','cli',1)")
        local._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name,session_key) "
                             "VALUES ('unrelated','cli',1,'science','agent:science:local:dm:unrelated')")
        local._conn.commit()
        before = ghost_path.read_bytes()
        session = _Session(read_only=False)
        try:
            plan = RepairPlan([Store("default", tmp_path, routing=True), Store("science", science)], root=tmp_path).scan(session)
            assert any(f.kind == "dual_home" and f.subject == "duplicated" for f in plan.findings)
            result = plan.apply(session, snapshot=lambda store: None)
            assert result["failures"] == []
            assert local.get_session("unrelated") is None
            assert target.get_session("unrelated")["profile_name"] == "science"
        finally:
            session.close()
        assert ghost_path.read_bytes() == before
        with pytest.raises(FileNotFoundError):
            SessionDB(ghost_path)
    finally:
        local.close()
        target.close()


@pytest.mark.parametrize("action", ["relabel", "sever", "rekey"])
def test_linked_duplicate_appearing_after_plan_fences_each_in_place_action(tmp_path, monkeypatch, action):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    local, sibling = SessionDB(tmp_path / "state.db"), SessionDB(profile / "state.db")
    try:
        store = Store("science", profile)
        owner = "main" if action == "rekey" else "science"
        sibling._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name,session_key) "
                              "VALUES ('parent','cli',1,?,?)", (owner, f"agent:{owner}:local:dm:parent"))
        sibling._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name,session_key,parent_session_id) "
                              "VALUES ('child','cli',2,?,?,'parent')",
                              ("wrong" if action == "relabel" else "science",
                               "agent:main:local:dm:child" if action == "sever" else f"agent:{owner}:local:dm:child"))
        sibling._conn.commit()
        session = _Session(read_only=False)
        try:
            plan = RepairPlan([Store("default", tmp_path, routing=True), store], root=tmp_path,
                              legacy_main="rekey" if action == "rekey" else "report").scan(session)
            assert plan.repairable
            local._conn.execute("INSERT INTO sessions(id,source,started_at) VALUES ('parent','cli',1)")
            local._conn.commit()
            before = [[tuple(row) for row in db._conn.execute("SELECT * FROM sessions ORDER BY id")]
                      for db in (local, sibling)]
            result = plan.apply(session, snapshot=lambda store: None)
            assert result["failures"]
            assert [[tuple(row) for row in db._conn.execute("SELECT * FROM sessions ORDER BY id")]
                    for db in (local, sibling)] == before
        finally:
            session.close()
    finally:
        local.close()
        sibling.close()
