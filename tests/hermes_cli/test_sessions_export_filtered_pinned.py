"""Regression coverage for filtered exports retaining pinned sessions."""

import json
import sys
import time

import hermes_state
import hermes_cli.main as main_mod
from hermes_state import SessionDB


def _seed_session(db, session_id, title, *, pinned=False):
    db.create_session(session_id, source="cli")
    db.set_session_title(session_id, title)
    db.append_message(session_id=session_id, role="user", content=f"message for {session_id}")
    ended_at = time.time() - 100 * 86400
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, started_at=? WHERE id=?",
            (ended_at, ended_at, session_id),
        )
        db._conn.commit()
    if pinned:
        assert db.set_session_pinned(session_id, True)


def test_filtered_export_includes_pinned_session(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)

    db = SessionDB(db_path)
    try:
        _seed_session(db, "20260101_000000_aaaaaa", "keep me", pinned=True)
        _seed_session(db, "20260101_000001_bbbbbb", "delete me")
    finally:
        db.close()

    output = tmp_path / "filtered.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "sessions", "export", "--title", "me", str(output)],
    )

    main_mod.main()

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {record["id"] for record in records} == {
        "20260101_000000_aaaaaa",
        "20260101_000001_bbbbbb",
    }
