from argparse import Namespace
from datetime import datetime

import pytest

from hermes_cli.sessions_cmd import cmd_sessions
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _create_at(db, session_id, started_at, source="cli", cwd=None):
    db.create_session(session_id, source, cwd=cwd)
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (started_at, session_id),
    )
    db._conn.commit()


def test_list_sessions_rich_applies_inclusive_after_and_exclusive_before(db):
    _create_at(db, "at-after", 100.0)
    _create_at(db, "inside", 150.0)
    _create_at(db, "at-before", 200.0)

    rows = db.list_sessions_rich(started_after=100.0, started_before=200.0)

    assert [row["id"] for row in rows] == ["inside", "at-after"]


def test_list_sessions_rich_composes_time_source_and_limit(db):
    _create_at(db, "cli-old", 100.0, cwd="/work/target-repo")
    _create_at(db, "other-source", 190.0, source="telegram", cwd="/work/target-repo")
    _create_at(db, "other-workspace", 190.0, cwd="/work/elsewhere")
    _create_at(db, "cli-new", 175.0, cwd="/work/target-repo/src")
    _create_at(db, "cli-too-new", 250.0, cwd="/work/target-repo")

    rows = db.list_sessions_rich(
        source="cli",
        started_after=100.0,
        started_before=200.0,
        workspace_query="TARGET-REPO",
        limit=1,
    )

    assert [row["id"] for row in rows] == ["cli-new"]


@pytest.mark.parametrize("order_by_last_active", [False, True])
def test_list_sessions_rich_filters_projected_compression_tip_workspace(
    db, order_by_last_active
):
    _create_at(db, "root", 100.0, cwd="C:/work/old-repo")
    db.end_session("root", "compression")
    _create_at(db, "tip", 150.0, cwd="C:/work/new-repo")
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        ("root", "tip"),
    )
    _create_at(db, "competitor", 200.0, cwd="C:/work/other-repo")
    db._conn.commit()

    matching = db.list_sessions_rich(
        workspace_query="new-repo",
        order_by_last_active=order_by_last_active,
        limit=1,
    )
    stale = db.list_sessions_rich(
        workspace_query="old-repo",
        order_by_last_active=order_by_last_active,
        limit=1,
    )

    assert [row["id"] for row in matching] == ["tip"]
    assert stale == []


def _list_args(**overrides):
    values = {
        "sessions_action": "list",
        "source": None,
        "limit": 20,
        "workspace": None,
        "after": None,
        "before": None,
    }
    values.update(overrides)
    return Namespace(**values)


class _RecordingDB:
    def __init__(self):
        self.list_kwargs = None

    def list_sessions_rich(self, **kwargs):
        self.list_kwargs = kwargs
        return []


def test_sessions_list_parses_time_bounds_and_preserves_other_filters(
    monkeypatch, capsys
):
    db = _RecordingDB()
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    result = cmd_sessions(
        _list_args(
            source="telegram",
            limit=7,
            workspace="repo",
            after="2026-08-01",
            before="2026-08-08",
        )
    )

    assert result is None
    assert db.list_kwargs == {
        "source": "telegram",
        "exclude_sources": None,
        "limit": 7,
        "started_after": pytest.approx(
            datetime.fromisoformat("2026-08-01").timestamp()
        ),
        "started_before": pytest.approx(
            datetime.fromisoformat("2026-08-08").timestamp()
        ),
        "workspace_query": "repo",
    }
    assert capsys.readouterr().out == "No sessions found.\n"


def test_sessions_list_rejects_empty_start_window(monkeypatch, capsys):
    db = _RecordingDB()
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    result = cmd_sessions(_list_args(after="2026-08-08", before="2026-08-01"))

    assert result == 2
    assert db.list_kwargs is None
    assert "Empty start-time window" in capsys.readouterr().out


def test_sessions_list_without_time_bounds_preserves_existing_query(monkeypatch):
    db = _RecordingDB()
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    cmd_sessions(_list_args())

    assert db.list_kwargs["started_after"] is None
    assert db.list_kwargs["started_before"] is None
    assert db.list_kwargs["workspace_query"] is None
    assert db.list_kwargs["exclude_sources"] == ["tool"]
