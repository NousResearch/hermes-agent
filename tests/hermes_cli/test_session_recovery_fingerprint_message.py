"""Regression test for #72291 — the source-fingerprint error must tell the
user that the parent interactive ``hermes`` CLI session counts as one of the
"Hermes processes" writing to the database.

Users who hit this error stop the gateway, retry, and fail again, because
their own interactive CLI session keeps writing session bookkeeping to
``state.db`` in the background. The check itself is correct; the guidance
needs to name the culprit and the two working escapes (fresh shell, or a
SQLite backup-API snapshot as ``--source``).
"""

import sqlite3

import pytest

from hermes_cli import session_recovery


def _make_source_db(path):
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE placeholder (x INTEGER)")
        conn.commit()
    finally:
        conn.close()


def test_fingerprint_error_names_parent_cli_session(tmp_path, monkeypatch):
    source = tmp_path / "state.db"
    _make_source_db(source)

    real_copy = session_recovery._copy_source_bundle

    def copy_then_mutate(src, snapshot_dir):
        result = real_copy(src, snapshot_dir)
        # Simulate a background bookkeeping write from the parent CLI
        # session landing while the bundle was being copied.
        with open(src, "ab") as handle:
            handle.write(b"\x00")
        return result

    monkeypatch.setattr(session_recovery, "_copy_source_bundle", copy_then_mutate)

    with pytest.raises(session_recovery.SessionRecoverySafetyError) as excinfo:
        session_recovery.inspect_session_database(source, work_dir=tmp_path)

    message = str(excinfo.value)
    # Original guidance is preserved…
    assert "changed while it was being copied" in message
    assert "Stop every Hermes process" in message
    # …and the new guidance names the parent CLI session and both escapes.
    assert "interactive `hermes` CLI session" in message
    assert "fresh shell" in message
    assert "backup" in message


def test_unchanged_source_does_not_trip_the_fingerprint_guard(tmp_path):
    source = tmp_path / "state.db"
    _make_source_db(source)

    # No writer touches the source: inspection must complete without raising
    # the safety error (the placeholder DB is simply not recoverable).
    report = session_recovery.inspect_session_database(source, work_dir=tmp_path)
    assert report["operation"] == "inspect"
    assert report["source_unchanged"] is True
