"""CLI command contract for Hermes-native non-blocking user input."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("cli-session", source="cli")
    try:
        yield database
    finally:
        database.close()


def test_answer_command_resolves_request_on_existing_agent(monkeypatch, db):
    import cli
    import hermes_cli.cli_commands_mixin as commands
    from tools.user_input_tool import request_user_input

    request = __import__("json").loads(request_user_input(
        questions=[{"id": "target", "text": "Which target?", "options": ["stable"], "default": "stable"}],
        session_id="cli-session", turn_id="turn-1", session_db=db,
    ))
    output = []
    monkeypatch.setattr(cli, "_cprint", lambda text: output.append(str(text)))
    monkeypatch.setattr(commands, "_cp", lambda *items: output.extend(str(item) for item in items))

    agent = SimpleNamespace(
        session_id="cli-session",
        _current_turn_id="turn-1",
        _inflight_turn_id="turn-1",
        _executing_tools=True,
        _interrupt_requested=False,
        steer=lambda text: True,
    )
    host = SimpleNamespace(agent=agent, _session_db=db, session_id="cli-session")

    commands.CLICommandsMixin._handle_answer_command(
        host, f"/answer {request['request_id']} {{\"target\": \"stable\"}}"
    )

    assert any("Answer recorded" in item for item in output)
    assert db.get_pending_user_input(request["request_id"], session_id="cli-session")["status"] == "answered"
