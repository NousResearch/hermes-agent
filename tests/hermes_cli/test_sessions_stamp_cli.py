"""``hermes sessions stamp`` CLI — the script/bot write path for the session label.

Mirrors the pin CLI test: the handler must write the SAME store the Desktop
sidebar and the dashboard read (SessionDB.set_session_stamp), not a
client-local list.
"""

from __future__ import annotations

import sys

import pytest

from hermes_state import SessionDB

_SID = "20260915_120000_stampcli"


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "state.db"
    seed = SessionDB(path)
    seed.create_session(_SID, source="cli")
    seed.close()
    return path


def _stored(db_path):
    db = SessionDB(db_path)
    try:
        return db.get_session_stamp(_SID)
    finally:
        db.close()


def _run(monkeypatch, capsys, argv_tail, db_path):
    """Dispatch through the real CLI entry point; each invocation opens (and closes)
    its own SessionDB, like the real command."""
    import hermes_cli.main as main_mod
    import hermes_state

    # ``sessions_cmd`` opens the store as ``SessionDB(read_only=observational)`` on
    # main, so the stub has to accept (and ignore) the constructor's kwargs the way
    # tests/hermes_cli/test_sessions_pin.py does.
    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: SessionDB(db_path))
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *argv_tail])
    try:
        main_mod.main()
        code = 0
    except SystemExit as e:  # non-zero exits propagate through main()
        code = e.code or 0
    return code, capsys.readouterr().out


def test_stamp_sets_and_reports_the_stored_label(monkeypatch, capsys, db_path):
    code, out = _run(monkeypatch, capsys, ["stamp", _SID, "Merged"], db_path)

    assert _stored(db_path) == "Merged"
    assert f"Session '{_SID}' stamped: Merged" in out
    assert code == 0


def test_stamp_clear_removes_it(monkeypatch, capsys, db_path):
    _run(monkeypatch, capsys, ["stamp", _SID, "WIP"], db_path)

    code, out = _run(monkeypatch, capsys, ["stamp", _SID, "--clear"], db_path)

    assert _stored(db_path) is None
    assert "Cleared stamp" in out
    assert code == 0


def test_stamp_rejects_over_long_label(monkeypatch, capsys, db_path):
    code, out = _run(monkeypatch, capsys, ["stamp", _SID, "x" * 25], db_path)

    assert _stored(db_path) is None
    assert "Error:" in out and "too long" in out
    assert code == 1
