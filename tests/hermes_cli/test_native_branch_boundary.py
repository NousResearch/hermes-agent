"""Branch creation must succeed before fencing the current native session."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("creation_fails", [False, True])
def test_branch_fences_only_after_child_creation(monkeypatch, creation_fails):
    import cli
    import hermes_cli.cli_commands_mixin as commands

    events = []

    def create_session(**kwargs):
        events.append("create")
        if creation_fails:
            raise RuntimeError("synthetic creation failure")

    stub = SimpleNamespace(
        conversation_history=[{"role": "user", "content": "synthetic prompt"}],
        _session_db=SimpleNamespace(
            create_session=create_session,
            append_messages_batch=lambda *args, **kwargs: None,
            set_session_title=lambda *args: None,
        ),
        session_id="parent-session",
        model="synthetic-model",
        max_turns=10,
        reasoning_config=None,
        agent=None,
        _transfer_session_yolo=lambda *args: None,
    )

    def fence(reason):
        assert reason == "session_switch"
        assert stub.session_id == "parent-session"
        events.append("fence")

    stub._fence_native_turn_sources = fence
    monkeypatch.setattr(commands, "mint_session_id", lambda *args: "child-session")
    monkeypatch.setattr(commands, "_end_current_session", lambda *args: events.append("end"))
    monkeypatch.setattr(commands, "_sync_agent_to_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(commands, "_cp", lambda *args: None)
    monkeypatch.setattr(cli, "_sync_process_session_id", lambda *args: None)

    from typing import cast

    commands.CLICommandsMixin._handle_branch_command(
        cast(commands.CLICommandsMixin, stub), "/branch example"
    )

    if creation_fails:
        assert events == ["create"]
        assert stub.session_id == "parent-session"
    else:
        assert events == ["create", "fence", "end"]
        assert stub.session_id == "child-session"