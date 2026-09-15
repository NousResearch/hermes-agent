"""Tests for the /branch (/fork) command — session branching.

Verifies that:
- Branching creates a new session with copied conversation history
- The original session is preserved (ended with "branched" reason)
- Auto-generated titles use lineage numbering
- Custom branch names are used when provided
- parent_session_id links are set correctly
- Structured reasoning fields survive the copy
- Edge cases: empty conversation, missing session DB
"""

import os
from datetime import datetime
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def session_db(tmp_path):
    """Create a real SessionDB for testing."""
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / ".hermes" / "test_sessions.db")
    yield db
    db.close()


@pytest.fixture
def cli_instance(tmp_path, session_db):
    """Create a minimal HermesCLI-like object for testing _handle_branch_command."""
    # We'll mock the CLI enough to test the branch logic without full init
    from unittest.mock import MagicMock

    cli = MagicMock()
    cli._session_db = session_db
    cli.session_id = "20260403_120000_abc123"
    cli.model = "anthropic/claude-sonnet-4.6"
    cli.max_turns = 90
    cli.reasoning_config = {"enabled": True, "effort": "medium"}
    cli.session_start = datetime.now()
    cli._pending_title = None
    cli._resumed = False
    cli.agent = None
    cli.conversation_history = [
        {"role": "user", "content": "Hello, can you help me?"},
        {"role": "assistant", "content": "Of course! How can I help?"},
        {"role": "user", "content": "Write a Python function to sort a list."},
        {"role": "assistant", "content": "def sort_list(lst): return sorted(lst)"},
    ]

    # Create the original session in the DB
    session_db.create_session(
        session_id=cli.session_id,
        source="cli",
        model=cli.model,
    )
    session_db.set_session_title(cli.session_id, "My Coding Session")

    return cli


class TestBranchCommandCLI:
    """Test the /branch command logic for the CLI."""

    def test_branch_creates_new_session(self, cli_instance, session_db):
        """Branching should create a new session in the DB."""
        from cli import HermesCLI

        # Call the real method on the mock, using the real implementation
        HermesCLI._handle_branch_command(cli_instance, "/branch")

        # Verify a new session was created
        assert cli_instance.session_id != "20260403_120000_abc123"
        new_session = session_db.get_session(cli_instance.session_id)
        assert new_session is not None

    def test_branch_copies_history(self, cli_instance, session_db):
        """Branching should copy all messages to the new session."""
        from cli import HermesCLI

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        messages = session_db.get_messages_as_conversation(cli_instance.session_id)
        assert len(messages) == 4  # All 4 messages copied



    def test_branch_with_custom_name(self, cli_instance, session_db):
        """Custom branch name should be used as the title."""
        from cli import HermesCLI

        HermesCLI._handle_branch_command(cli_instance, "/branch refactor approach")

        title = session_db.get_session_title(cli_instance.session_id)
        assert title == "refactor approach"



    def test_branch_no_session_db(self, cli_instance):
        """Branching without a session DB should show an error."""
        from cli import HermesCLI
        cli_instance._session_db = None

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        # session_id should not have changed
        assert cli_instance.session_id == "20260403_120000_abc123"


    def test_branch_sets_resumed_flag(self, cli_instance, session_db):
        """Branch should set _resumed=True to prevent auto-title generation."""
        from cli import HermesCLI

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        assert cli_instance._resumed is True


    def test_branch_fires_on_session_switch_hook(self, cli_instance, session_db):
        """The /branch command must notify memory providers of the rotation.

        Without this, providers that cache per-session state in
        initialize() keep writing under the old session_id. See #6672.
        """
        from cli import HermesCLI

        # Wire a real-ish agent object with a MagicMock memory_manager
        agent = MagicMock()
        mm = MagicMock()
        agent._memory_manager = mm
        cli_instance.agent = agent
        original_id = cli_instance.session_id

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        # Hook must have been called exactly once with the new session_id,
        # parent pointing at the branched-from session, reset=False, and
        # reason="branch" for diagnostics.
        assert mm.on_session_switch.call_count == 1
        _, kwargs = mm.on_session_switch.call_args
        assert mm.on_session_switch.call_args.args[0] == cli_instance.session_id
        assert kwargs["parent_session_id"] == original_id
        assert kwargs["reset"] is False
        assert kwargs["reason"] == "branch"



class TestBranchCommandDef:
    """Test the CommandDef registration for /branch."""

    def test_branch_in_registry(self):
        """The branch command should be in the command registry."""
        from hermes_cli.commands import COMMAND_REGISTRY
        names = [c.name for c in COMMAND_REGISTRY]
        assert "branch" in names


    def test_branch_in_session_category(self):
        """The branch command should be in the Session category."""
        from hermes_cli.commands import COMMAND_REGISTRY
        branch = next(c for c in COMMAND_REGISTRY if c.name == "branch")
        assert branch.category == "Session"


class TestBranchFlushesBeforeEndSession:
    """Regression for #47202: /branch must flush un-persisted messages to
    the session DB before ending the old session, just like /new and
    compress_context() already do."""

    def test_branch_flushes_when_agent_present(self, cli_instance, session_db):
        from cli import HermesCLI

        agent = MagicMock()
        cli_instance.agent = agent

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        agent._flush_messages_to_session_db.assert_called_once_with(
            cli_instance.conversation_history,
            conversation_history=cli_instance.conversation_history,
        )


REASONING_DETAILS = [
    {"type": "reasoning.text", "text": "sort in place instead", "format": "unknown"}
]
CODEX_REASONING_ITEMS = [
    {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque-blob"}
]
CODEX_MESSAGE_ITEMS = [
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}],
    }
]


class TestBranchPreservesReasoningFields:
    """The copy loop must carry the structured reasoning columns across.

    Only ``reasoning`` was forwarded to append_message, so a branch dropped
    the preserved Anthropic thinking blocks and the Codex
    encrypted-reasoning/message-item continuation state that the parent had
    accumulated — the branch then replayed without them, because every
    consumer gates on isinstance(..., list) and a missing field reads as None.
    """

    def test_reasoning_fields_survive_branch(self, cli_instance, session_db):
        from cli import HermesCLI

        cli_instance.conversation_history = [
            {"role": "user", "content": "Sort this list."},
            {
                "role": "assistant",
                "content": "def sort_list(lst): return sorted(lst)",
                "reasoning": "picked sorted()",
                "reasoning_details": REASONING_DETAILS,
                "codex_reasoning_items": CODEX_REASONING_ITEMS,
                "codex_message_items": CODEX_MESSAGE_ITEMS,
            },
        ]

        HermesCLI._handle_branch_command(cli_instance, "/branch")

        messages = session_db.get_messages_as_conversation(cli_instance.session_id)
        assistant = next(m for m in messages if m["role"] == "assistant")
        assert assistant["reasoning_details"] == REASONING_DETAILS
        assert assistant["codex_reasoning_items"] == CODEX_REASONING_ITEMS
        assert assistant["codex_message_items"] == CODEX_MESSAGE_ITEMS
