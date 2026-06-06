"""Regression tests for dashboard session resume targeting."""

from __future__ import annotations

import hermes_state


def _use_temp_state_db(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    return db_path


def _set_session_times(
    db,
    session_id: str,
    *,
    started_at: float,
    ended_at: float | None = None,
    end_reason: str | None = None,
) -> None:
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=?, end_reason=? WHERE id=?",
        (started_at, ended_at, end_reason, session_id),
    )
    db._conn.commit()


def test_dashboard_resume_does_not_follow_empty_non_compression_child(monkeypatch, tmp_path):
    """Embedded chat resume must not switch to an unrelated/empty child row.

    Branches, delegate/subagent sessions, and partial child rows can share
    parent_session_id with the visible conversation, but they are not the live
    continuation of that chat. Resuming the empty child makes the TUI render the
    new-chat empty state even though the user selected an existing thread.
    """
    _use_temp_state_db(monkeypatch, tmp_path)

    from hermes_cli.web_server import _session_latest_descendant
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("root", source="tui", model="test")
        _set_session_times(db, "root", started_at=1_000.0)
        db.append_message("root", role="user", content="resume this existing conversation")

        # A newer child exists, but the parent never ended by compression and
        # the child has no messages. The old newest-leaf resolver followed this
        # row and reopened an empty TUI transcript.
        db.create_session("empty-child", source="tui", model="test", parent_session_id="root")
        _set_session_times(db, "empty-child", started_at=1_100.0)
    finally:
        db.close()

    latest, path = _session_latest_descendant("root")

    assert latest == "root"
    assert path == ["root"]


def test_dashboard_resume_follows_real_compression_tip(monkeypatch, tmp_path):
    """Dashboard resume should still follow true compression continuations."""
    _use_temp_state_db(monkeypatch, tmp_path)

    from hermes_cli.web_server import _session_latest_descendant
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("root", source="tui", model="test")
        _set_session_times(db, "root", started_at=2_000.0, ended_at=2_100.0, end_reason="compression")
        db.append_message("root", role="user", content="before compression")

        db.create_session("tip", source="tui", model="test", parent_session_id="root")
        _set_session_times(db, "tip", started_at=2_101.0)
        db.append_message("tip", role="user", content="after compression")
    finally:
        db.close()

    latest, path = _session_latest_descendant("root")

    assert latest == "tip"
    assert path == ["root", "tip"]
