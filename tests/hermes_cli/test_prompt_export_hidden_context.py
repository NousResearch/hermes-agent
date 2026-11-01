"""Prompt-only exports omit model scaffolding stored as hidden user rows."""

import json
import sys

import pytest

from hermes_cli.session_export import export_record_count, render_sessions_export
from hermes_state import SessionDB


@pytest.fixture
def stored_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("prompt-export", "cli")
        db.append_message("prompt-export", "user", "First real prompt")
        db.append_message("prompt-export", "user", "Internal model scaffolding", display_kind="hidden")
        db.append_message("prompt-export", "user", "A real correction", display_kind="steer")
        return db.export_session("prompt-export")


@pytest.mark.parametrize("fmt", ["jsonl", "md"])
def test_prompt_only_cli_omits_hidden_user_rows(stored_session, monkeypatch, capsys, fmt):
    from hermes_cli.main import main

    monkeypatch.setattr(sys, "argv", [
        "hermes", "sessions", "export", "-", "--session-id", "prompt-export",
        "--only", "user-prompts", "--format", fmt,
    ])
    main()
    output = capsys.readouterr().out
    assert "Internal model scaffolding" not in output
    assert "First real prompt" in output
    assert "A real correction" in output
    if fmt == "jsonl":
        records = [json.loads(line) for line in output.splitlines()]
        assert [record["index"] for record in records] == [1, 2]


def test_prompt_count_and_backup_keep_their_respective_projections(stored_session):
    assert export_record_count([stored_session], only="user-prompts") == (2, "prompt")
    backup = render_sessions_export([stored_session])
    assert "Internal model scaffolding" in backup
    assert len(json.loads(backup)["messages"]) == 3
