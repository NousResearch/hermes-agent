"""on_session_delete post-commit observer (#124511).

Delete paths must notify plugins after the commit so per-session on-disk
state can be cleaned up without polling the store. Failing observers fail
open — deletion itself never breaks.
"""

import pytest

from hermes_cli import lifecycle
from hermes_cli.plugins import VALID_HOOKS
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "test_state.db")
    yield session_db
    session_db.close()


@pytest.fixture(autouse=True)
def _no_fts_rebuild_throttle(monkeypatch):
    monkeypatch.setattr(SessionDB, "_FTS_REBUILD_MIN_PAUSE", 0.0)
    monkeypatch.setattr(SessionDB, "_FTS_REBUILD_DUTY_FACTOR", 0.0)


@pytest.fixture()
def fired(monkeypatch):
    calls = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(lifecycle, "notify_session_deleted", _spy)
    return calls


def test_hook_is_registered():
    assert "on_session_delete" in VALID_HOOKS


def test_delete_session_fires_once(db, fired):
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="user", content="hi")

    assert db.delete_session("s1") is True

    assert len(fired) == 1
    assert fired[0]["session_id"] == "s1"
    assert "deleted_ids" not in fired[0]


def test_delete_session_unknown_id_does_not_fire(db, fired):
    assert db.delete_session("missing") is False
    assert fired == []


def test_delete_sessions_bulk_fires_once_with_ids(db, fired):
    db.create_session(session_id="s1", source="cli")
    db.create_session(session_id="s2", source="cli")
    db.create_session(session_id="keep", source="cli")

    assert db.delete_sessions(["s1", "s2", "missing"]) == 2

    assert len(fired) == 1
    assert fired[0]["session_id"] == "s1"
    assert fired[0]["deleted_ids"] == ["s1", "s2"]


def test_delete_empty_sessions_carries_reason(db, fired):
    db.create_session(session_id="empty", source="cli")
    db.end_session("empty", "done")

    assert db.delete_empty_sessions() == 1

    assert len(fired) == 1
    assert fired[0]["session_id"] == "empty"
    assert fired[0]["reason"] == "empty_sweep"


def test_delete_moved_session_fires_with_reason(db, fired):
    db.create_session(session_id="moved1", source="cli")

    assert db.delete_moved_session("moved1") is True

    assert len(fired) == 1
    assert fired[0]["session_id"] == "moved1"
    assert fired[0]["reason"] == "profile_repair"


def test_delete_moved_session_unknown_id_does_not_fire(db, fired):
    assert db.delete_moved_session("missing") is False
    assert fired == []


def test_failing_observer_does_not_break_delete(db, monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("plugin down")

    monkeypatch.setattr(lifecycle, "notify_session_deleted", _boom)
    db.create_session(session_id="s1", source="cli")

    assert db.delete_session("s1") is True
    assert db.get_session("s1") is None


def test_notify_dispatches_hook_and_fails_open(monkeypatch):
    seen = []
    monkeypatch.setattr(
        lifecycle, "invoke_hook", lambda name, **kw: seen.append((name, kw)) or []
    )
    lifecycle.notify_session_deleted("s1", reason="prune")
    assert seen == [("on_session_delete", {"session_id": "s1", "reason": "prune"})]

    def _boom(name, **kw):
        raise RuntimeError("dispatch down")

    monkeypatch.setattr(lifecycle, "invoke_hook", _boom)
    assert lifecycle.notify_session_deleted("s1") == []
