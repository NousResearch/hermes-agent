"""A compaction migration never archives the parent's /goal, /loop or /heartbeat when the child
copy was not persisted (#149594): the save helpers swallow ``set_meta`` failures, so archiving
unconditionally left the state on no session at all."""
import sqlite3

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _refuse_child_writes(monkeypatch, key):
    from hermes_cli import goals

    db = goals._get_session_db()
    assert db is not None
    real_set_meta = db.set_meta

    def _set_meta(meta_key, value):
        if meta_key == key:
            raise sqlite3.OperationalError("database is locked")
        real_set_meta(meta_key, value)

    monkeypatch.setattr(db, "set_meta", _set_meta)


@pytest.mark.parametrize("kind", ["goal", "loop", "heartbeat"])
def test_failed_child_write_keeps_parent_active(hermes_home, monkeypatch, kind):
    from hermes_cli import goals, heartbeat, loops

    if kind == "goal":
        goals.save_goal("parent-x", goals.GoalState(goal="keep me"))
        migrate, load = goals.migrate_goal_to_session, goals.load_goal
    elif kind == "loop":
        loops.save_loop("parent-x", loops.LoopState(prompt="tick", interval_seconds=60.0))
        migrate, load = loops.migrate_loop_to_session, loops.load_loop
    else:
        heartbeat.save_heartbeat("parent-x", heartbeat.HeartbeatState(prompt="beat", interval_seconds=60))
        migrate, load = heartbeat.migrate_heartbeat_to_session, heartbeat.load_heartbeat
    _refuse_child_writes(monkeypatch, f"{kind}:child-x")

    assert migrate("parent-x", "child-x") is False
    parent = load("parent-x")
    assert parent is not None and parent.status == "active"
