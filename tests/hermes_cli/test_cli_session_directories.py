"""Session lists identify saved workspaces without losing long directory paths."""

import os

import pytest

from cli import HermesCLI
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def _cli_for(db):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._session_db = db
    cli_obj.session_id = "current_session"
    cli_obj._agent_running = False
    cli_obj._pending_resume_sessions = None
    return cli_obj


@pytest.mark.parametrize("command", ["/sessions", "/resume"])
@pytest.mark.parametrize("width", [80, 140, 240])
def test_session_lists_preserve_full_saved_directories(
    session_db, tmp_path, monkeypatch, capsys, command, width
):
    directories = [
        tmp_path / "Projects" / ("long-codebase-" + "x" * 100) / "[bold] Hiring Fleet",
        tmp_path / "Projects" / "another-codebase" / "feature-Δ",
    ]
    ids = ["20260101_010000_abc123", "20260101_020000_def456"]
    for session_id, directory in zip(ids, directories):
        session_db.create_session(session_id, "cli", cwd=str(directory))
        session_db.set_session_title(session_id, f"Code work {session_id[-6:]}")
        session_db.append_message(session_id, "user", "hello")
    monkeypatch.setattr(
        "hermes_cli.cli_session_mixin.shutil.get_terminal_size",
        lambda fallback=(80, 24): os.terminal_size((width, 24)),
    )

    cli_obj = _cli_for(session_db)
    if command == "/sessions":
        cli_obj._handle_sessions_command(command)
    else:
        cli_obj._handle_resume_command(command)
    output = capsys.readouterr().out

    lines = output.splitlines()
    directory_start = next(line.index("Working") for line in lines if "Working" in line)
    # Reassemble only the last column: wrapping may move spaces to a line boundary,
    # but no path characters, including literal Rich markup, may disappear.
    rendered_directories = "".join(
        "".join(line[directory_start:].split()) for line in lines
    )
    assert "WorkingDirectory" in rendered_directories
    for directory in directories:
        assert "".join(str(directory).split()) in rendered_directories
    for session_id in ids:
        assert session_id in output
    assert "Example: /resume 2" in output
    assert all(len(line) <= width for line in lines if "Use /resume" not in line)
    if command == "/resume":
        assert cli_obj._pending_resume_sessions is not None
        assert cli_obj._pending_resume_sessions == cli_obj._list_recent_sessions(limit=10)


@pytest.mark.parametrize("cwd", [None, ""])
def test_unrecorded_directory_does_not_use_current_codebase(
    session_db, tmp_path, monkeypatch, capsys, cwd
):
    current_codebase = tmp_path / "unrelated-current-codebase"
    current_codebase.mkdir()
    monkeypatch.chdir(current_codebase)
    session_db.create_session("legacy_session", "cli", cwd=cwd)
    session_db.append_message("legacy_session", "user", "hello")

    _cli_for(session_db)._handle_sessions_command("/sessions")
    output = capsys.readouterr().out

    assert "not recorded" in output
    assert str(current_codebase) not in output
