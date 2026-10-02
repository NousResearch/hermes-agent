"""Invalid time bounds must stop session mutations before querying the store."""

import sys
from argparse import Namespace

import pytest

from hermes_cli.sessions_cmd import cmd_sessions
from hermes_state import SessionDB


@pytest.mark.parametrize("action", ["prune", "archive"])
@pytest.mark.parametrize("flag", ["older_than", "newer_than", "before", "after"])
@pytest.mark.parametrize(
    "value", ["9" * 400, "9" * 305 + "w"], ids=["float-overflow", "unit-overflow"]
)
def test_nonfinite_time_bounds_leave_sessions_unchanged(
    tmp_path, monkeypatch, capsys, action, flag, value
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("keep", source="cli")
        db.append_message("keep", "user", "preserve this conversation")
        db.end_session("keep", "done")
        before = db.get_session("keep")
        args = Namespace(sessions_action=action, yes=True, dry_run=False, **{flag: value})

        assert cmd_sessions(args) == 1
        assert f"Invalid value for --{flag.replace('_', '-')}" in capsys.readouterr().out
        assert db.get_session("keep") == before
        assert db.get_messages("keep")[0]["content"] == "preserve this conversation"
    finally:
        db.close()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CRT timestamp limitation")
@pytest.mark.parametrize("action", ["prune", "archive"])
def test_unrepresentable_iso_time_is_a_cli_error(tmp_path, monkeypatch, capsys, action):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    args = Namespace(sessions_action=action, before="0001-01-01", yes=True, dry_run=False)
    assert cmd_sessions(args) == 1
    assert "Invalid value for --before" in capsys.readouterr().out
