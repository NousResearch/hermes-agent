"""Tests for /side and /back topic parking commands."""

import os
from datetime import datetime
from types import MethodType
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def session_db(tmp_path):
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / ".hermes" / "test_sessions.db")
    yield db
    db.close()


@pytest.fixture
def cli_instance(session_db):
    cli = MagicMock()
    cli._session_db = session_db
    cli.session_id = "20260403_120000_parent"
    cli.model = "anthropic/claude-sonnet-4.6"
    cli.max_turns = 90
    cli.reasoning_config = {"enabled": True, "effort": "medium"}
    cli.session_start = datetime.now()
    cli._pending_title = None
    cli._resumed = False
    cli.agent = None
    cli.conversation_history = [
        {"role": "user", "content": "Main topic"},
        {"role": "assistant", "content": "Main answer"},
    ]
    cli._confirm_destructive_slash = lambda *_a, **_kw: "once"
    cli._notify_session_boundary = MagicMock()

    from cli import HermesCLI
    cli._sync_agent_after_session_switch = MethodType(
        HermesCLI._sync_agent_after_session_switch, cli
    )

    session_db.create_session(session_id=cli.session_id, source="cli", model=cli.model)
    session_db.set_session_title(cli.session_id, "Main Session")
    for msg in cli.conversation_history:
        session_db.append_message(
            session_id=cli.session_id,
            role=msg["role"],
            content=msg["content"],
        )
    return cli


class TestSideCommandCLI:
    def test_side_creates_clean_child_session_without_copying_parent_history(self, cli_instance, session_db):
        from cli import HermesCLI

        parent_id = cli_instance.session_id
        HermesCLI._handle_side_command(cli_instance, "/side ncore quick question")

        side_id = cli_instance.session_id
        assert side_id != parent_id
        side = session_db.get_session(side_id)
        assert side is not None
        assert side["parent_session_id"] == parent_id
        assert session_db.get_session_title(side_id) == "ncore quick question"
        assert cli_instance.conversation_history == []
        assert session_db.get_messages_as_conversation(side_id) == []

        active = session_db.get_active_side_session(source="cli")
        assert active["parent_session_id"] == parent_id
        assert active["side_session_id"] == side_id
        assert active["status"] == "active"

    def test_side_without_title_leaves_session_eligible_for_auto_title(self, cli_instance, session_db):
        from cli import HermesCLI

        HermesCLI._handle_side_command(cli_instance, "/side")

        side_id = cli_instance.session_id
        assert session_db.get_session_title(side_id) is None

        active = session_db.get_active_side_session(source="cli")
        assert active["side_session_id"] == side_id
        assert active["title"] is None

    def test_side_with_title_sets_side_session_title(self, cli_instance, session_db):
        from cli import HermesCLI

        HermesCLI._handle_side_command(cli_instance, "/side ncore quick question")

        assert session_db.get_session_title(cli_instance.session_id) == "ncore quick question"

    def test_back_pops_active_side_session_and_restores_parent_history(self, cli_instance, session_db):
        from cli import HermesCLI

        parent_id = cli_instance.session_id
        HermesCLI._handle_side_command(cli_instance, "/side ncore quick question")
        side_id = cli_instance.session_id
        session_db.append_message(side_id, role="user", content="side only")
        cli_instance.conversation_history = [{"role": "user", "content": "side only"}]

        HermesCLI._handle_back_command(cli_instance, "/back")

        assert cli_instance.session_id == parent_id
        assert cli_instance.conversation_history == [
            {"role": "user", "content": "Main topic"},
            {"role": "assistant", "content": "Main answer"},
        ]
        assert session_db.get_active_side_session(source="cli") is None
        assert session_db.get_session(side_id)["end_reason"] == "side_session_returned"
        assert session_db.get_session(parent_id)["ended_at"] is None

    def test_side_and_back_notify_memory_manager_with_reset_semantics(self, cli_instance, session_db):
        from cli import HermesCLI

        agent = MagicMock()
        mm = MagicMock()
        agent._memory_manager = mm
        agent._last_flushed_db_idx = 99
        cli_instance.agent = agent
        parent_id = cli_instance.session_id

        HermesCLI._handle_side_command(cli_instance, "/side quick")
        side_id = cli_instance.session_id
        HermesCLI._handle_back_command(cli_instance, "/back")

        assert mm.on_session_switch.call_count == 2
        first_args, first_kwargs = mm.on_session_switch.call_args_list[0]
        assert first_args[0] == side_id
        assert first_kwargs["parent_session_id"] == parent_id
        assert first_kwargs["reset"] is True
        assert first_kwargs["reason"] == "side"

        second_args, second_kwargs = mm.on_session_switch.call_args_list[1]
        assert second_args[0] == parent_id
        assert second_kwargs["parent_session_id"] == side_id
        assert second_kwargs["reset"] is True
        assert second_kwargs["reason"] == "back"
        assert agent._last_flushed_db_idx == 2

    def test_back_without_active_side_session_is_noop(self, cli_instance):
        from cli import HermesCLI

        original_id = cli_instance.session_id
        HermesCLI._handle_back_command(cli_instance, "/back")

        assert cli_instance.session_id == original_id


class TestSideCommandDef:
    def test_side_and_back_are_registered(self):
        from hermes_cli.commands import COMMAND_REGISTRY

        names = [c.name for c in COMMAND_REGISTRY]
        assert "side" in names
        assert "back" in names

    def test_return_alias_resolves_to_back(self):
        from hermes_cli.commands import resolve_command

        result = resolve_command("return")
        assert result is not None
        assert result.name == "back"
