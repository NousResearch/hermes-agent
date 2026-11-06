"""Repair holds complete linked components around duplicate owners (#149132)."""
from pathlib import Path

import pytest

from hermes_cli.sessions_repair_profiles import RepairPlan, Store, _Session
from hermes_state import SessionDB


def _seed(db, sid, *, parent=None, profile="science", keyed=True):
    db._conn.execute("INSERT INTO sessions (id,source,started_at,profile_name,session_key,parent_session_id) "
                     "VALUES (?, 'cli', 1, ?, ?, ?)",
                     (sid, profile, f"agent:{profile}:local:dm:{sid}" if keyed else None, parent))
    db._conn.commit()


def _rows(db):
    return [tuple(row) for row in db._conn.execute("SELECT * FROM sessions ORDER BY id")]


@pytest.mark.parametrize("duplicate", ["parent", "child"])
@pytest.mark.parametrize("shape", ["foreign", "crossed", "unkeyed-bridge"])
def test_duplicate_owner_holds_both_sides_of_the_link(tmp_path, monkeypatch, duplicate, shape):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    local, target = SessionDB(tmp_path / "state.db"), SessionDB(profile / "state.db")
    try:
        _seed(local, "parent")
        if shape == "unkeyed-bridge":
            _seed(local, "bridge", parent="parent", keyed=False)
        _seed(local, "child", parent="bridge" if shape == "unkeyed-bridge" else "parent",
              profile="main" if shape == "crossed" else "science")
        _seed(target, duplicate)
        before = _rows(local), _rows(target)
        session = _Session(read_only=False)
        try:
            plan = RepairPlan([Store("default", tmp_path, routing=True), Store("science", profile)], root=tmp_path).scan(session)
            assert any(row.kind == "dual_home" and row.subject == duplicate for row in plan.findings)
            assert not [row for row in plan.repairable if row.subject in {"parent", "bridge", "child"}]
            plan.apply(session, snapshot=lambda store: None)
        finally:
            session.close()
        assert (_rows(local), _rows(target)) == before
    finally:
        local.close()
        target.close()


def test_unrelated_component_still_moves_when_a_lineage_is_held(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    local, target = SessionDB(tmp_path / "state.db"), SessionDB(profile / "state.db")
    try:
        _seed(local, "parent")
        _seed(local, "child", parent="parent")
        _seed(target, "child")
        _seed(local, "unrelated")
        before = {row["id"]: dict(row) for row in local._conn.execute("SELECT * FROM sessions WHERE id!='unrelated'")}
        session = _Session(read_only=False)
        try:
            plan = RepairPlan([Store("default", tmp_path, routing=True), Store("science", profile)], root=tmp_path).scan(session)
            assert {row.subject for row in plan.repairable} == {"unrelated"}
            result = plan.apply(session, snapshot=lambda store: None)
            assert result["failures"] == []
        finally:
            session.close()
        assert local.get_session("unrelated") is None
        assert target.get_session("unrelated") is not None
        assert {row["id"]: dict(row) for row in local._conn.execute("SELECT * FROM sessions")} == before
    finally:
        local.close()
        target.close()


def test_move_rechecks_connected_duplicate_owners_after_planning(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    local, target = SessionDB(tmp_path / "state.db"), SessionDB(profile / "state.db")
    try:
        _seed(local, "parent")
        _seed(local, "child", parent="parent", profile="main")
        session = _Session(read_only=False)
        try:
            plan = RepairPlan([Store("default", tmp_path, routing=True), Store("science", profile)], root=tmp_path).scan(session)
            _seed(target, "child")
            before = _rows(local), _rows(target)
            result = plan.apply(session, snapshot=lambda store: None)
            assert result["failures"]
            assert (_rows(local), _rows(target)) == before
        finally:
            session.close()
    finally:
        local.close()
        target.close()
