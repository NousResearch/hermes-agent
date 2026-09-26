"""Export previews must select sessions without producing export side effects."""

import argparse

import pytest

import hermes_state
from hermes_cli.sessions_cmd import cmd_sessions
from hermes_cli.subcommands.sessions import build_sessions_parser
from hermes_state import SessionDB


@pytest.fixture
def session_export(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    sid = "preview-session-1234"
    with SessionDB(home / "state.db") as db:
        db.create_session(sid, "cli")
        db.set_session_title(sid, "Export preview fixture")
        db.append_message(sid, "assistant", "A synthetic export message.")
        db.end_session(sid, "normal")
    parser = argparse.ArgumentParser()
    build_sessions_parser(parser.add_subparsers(dest="command"), cmd_sessions=cmd_sessions)

    def run(*args):
        parsed = parser.parse_args(["sessions", "export", *args])
        return parsed.func(parsed)

    return sid, run


@pytest.mark.parametrize("fmt,extra", [
    ("jsonl", []), ("md", []), ("qmd", []), ("html", []),
    ("jsonl", ["--only", "user-prompts"]), ("md", ["--only", "user-prompts"]),
    ("trace", ["--no-redact"]),
])
def test_preview_preserves_output_until_normal_export(session_export, tmp_path, capsys, fmt, extra):
    sid, run = session_export
    output = tmp_path / "exports"
    options = [str(output) + "/", "--format", fmt, *extra]

    run(*options, "--session-id", "preview-session", "--dry-run")
    preview = capsys.readouterr().out
    assert "Would export 1 session" in preview and sid in preview
    assert not output.exists()

    run(*options, "--session-id", "missing-session", "--dry-run")
    assert "No session" in capsys.readouterr().out
    assert not output.exists()

    run(*options, "--source", "cli", "--dry-run")
    assert "Would export 1 session" in capsys.readouterr().out
    assert not output.exists()

    run(*options, "--session-id", sid)
    assert "Exported" in capsys.readouterr().out
    assert any(path.is_file() for path in output.rglob("*"))


@pytest.mark.parametrize("options", [
    ["--format", "trace", "--upload", "--session-id", "preview-session"],
    ["--format", "trace", "--upload"],
    ["--format", "md", "--delete-after-verified", "--yes", "--session-id", "preview-session"],
])
def test_preview_does_not_run_post_export_actions(session_export, tmp_path, monkeypatch, capsys, options):
    from agent import trace_upload

    sid, run = session_export

    def unexpected_action(*args, **kwargs):
        pytest.fail("dry-run must not upload or delete a session")

    monkeypatch.setattr(trace_upload, "upload_session_trace", unexpected_action)
    monkeypatch.setattr(SessionDB, "delete_session", unexpected_action)
    output = tmp_path / "exports"
    run(str(output) + "/", *options, "--dry-run")
    preview = capsys.readouterr().out
    assert "Would export 1 session" in preview and sid in preview
    assert not output.exists()
    with SessionDB() as db:
        assert db.get_session(sid) is not None
