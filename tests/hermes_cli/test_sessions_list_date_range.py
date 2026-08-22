"""Date-range regressions for ``hermes sessions list`` (issue #91900)."""

from argparse import Namespace

import hermes_cli.sessions_cmd as sessions_cmd
import hermes_state
from hermes_state import SessionDB


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


def _seed_session(db, session_id, *, started_at, source="cli", cwd="/work/alpha"):
    db.create_session(session_id, source, cwd=cwd)
    db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (started_at, session_id),
    )
    db._conn.commit()


def test_list_date_bounds_are_inclusive_exclusive_and_compose(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        _seed_session(db, "at-after", started_at=100, source="cli")
        _seed_session(db, "inside", started_at=150, source="cli")
        _seed_session(db, "at-before", started_at=200, source="cli")
        _seed_session(db, "wrong-source", started_at=150, source="telegram")
        _seed_session(db, "wrong-workspace", started_at=150, cwd="/work/beta")
    finally:
        db.close()
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path))

    sessions_cmd.cmd_sessions(
        _list_args(
            after="1970-01-01T00:01:40+00:00",
            before="1970-01-01T00:03:20+00:00",
            source="cli",
            workspace="alpha",
        )
    )

    output = capsys.readouterr().out
    assert "at-after" in output
    assert "inside" in output
    assert "at-before" not in output
    assert "wrong-source" not in output
    assert "wrong-workspace" not in output


def test_list_filters_date_before_limit(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        _seed_session(db, "newer-out-of-range", started_at=300)
        _seed_session(db, "matching-newer", started_at=200)
        _seed_session(db, "matching-older", started_at=100)
    finally:
        db.close()
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: SessionDB(db_path))

    sessions_cmd.cmd_sessions(
        _list_args(before="1970-01-01T00:04:10+00:00", limit=2)
    )

    output = capsys.readouterr().out
    assert "newer-out-of-range" not in output
    assert "matching-newer" in output
    assert "matching-older" in output
