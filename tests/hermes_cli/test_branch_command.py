"""Tests for the /branch (/fork) command — session branching.

Verifies that:
- Branching creates a new session with copied conversation history
- The original session is preserved (ended with "branched" reason)
- Auto-generated titles use lineage numbering
- Custom branch names are used when provided
- parent_session_id links are set correctly
- Structured reasoning fields survive the copy
- Edge cases: empty conversation, missing session DB

The CLI's store is its ``GatewaySessionStore`` on a connection to a real session authority in this
process: the owner forks the transcript it holds and writes the rows the assertions read.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from tests.fixtures.gateway_owner import GatewayOwner
from tests.fixtures.storage_seed import seed_messages


def _mock_cli(store):
    """A minimal HermesCLI-like object for testing _handle_branch_command."""
    cli = MagicMock()
    cli._agent_running = False  # a bare MagicMock attribute is truthy and would trip the mid-turn guard
    from hermes_cli.cli_init_mixin import CLIInitMixin
    # The id seam: on an open gateway store the new session id is the owner's branch child.
    cli._new_session_id = lambda now, **kw: CLIInitMixin._new_session_id(cli, now, **kw)
    cli._session_db = store
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
    return cli


@pytest.fixture
def owner(tmp_path, monkeypatch):
    owner = GatewayOwner(tmp_path, monkeypatch)
    yield owner
    owner.close()


@pytest.fixture
def store(owner):
    """The classic CLI's session store, on its connection to the owner."""
    return owner.client()


@pytest.fixture
def gateway_cli(store):
    """The CLI on its gateway store: the original session is the one the owner created for it."""
    cli = _mock_cli(store)
    cli._gateway_session_open = True
    cli.session_id = store.open_session(cli)
    store.set_session_title(cli.session_id, "My Coding Session")
    return cli


def _persist_history(owner, cli):
    """The parent's transcript as the gateway holds it: the turns the CLI shows."""
    seed_messages(owner.persistence, cli.session_id, *cli.conversation_history)


class TestBranchCommandCLI:
    """Test the /branch command logic for the CLI."""

    def test_failed_branch_creation_leaves_original_session_open(self, gateway_cli, store):
        """Branching is child-first: when the gateway's branch fails the user stays on the original
        session, so it must not be marked ended as "branched" (#11030)."""
        from unittest.mock import patch
        from cli import HermesCLI

        original = gateway_cli.session_id
        with patch.object(store, "branch", side_effect=RuntimeError("boom")):
            HermesCLI._handle_branch_command(gateway_cli, "/branch")

        assert gateway_cli.session_id == original
        row = store.get_session(original)
        assert row["end_reason"] is None and row["ended_at"] is None

    def test_gateway_refusal_is_reported_and_leaves_the_session(self, gateway_cli, store, capsys):
        """A gateway /branch step that fails (e.g. the child's YOLO could not be set, CLI-20) prints
        main's branch error and keeps the user on the original session, never a traceback."""
        from cli import HermesCLI
        from hermes_cli.gateway_client import GatewayClientError

        def refused(now, **kw):
            raise GatewayClientError("Gateway did not answer config.set")
        gateway_cli._new_session_id = refused
        original = gateway_cli.session_id
        HermesCLI._handle_branch_command(gateway_cli, "/branch")

        assert gateway_cli.session_id == original
        assert "Gateway did not answer config.set" in capsys.readouterr().out
        assert store.get_session(original)["end_reason"] is None

    def test_branch_copies_history(self, owner, gateway_cli, store):
        """Branching should copy all messages to the new session."""
        from cli import HermesCLI

        _persist_history(owner, gateway_cli)
        HermesCLI._handle_branch_command(gateway_cli, "/branch")

        messages = store.get_messages_as_conversation(gateway_cli.session_id)
        assert len(messages) == 4  # All 4 messages copied



    def test_branch_with_custom_name(self, gateway_cli, store):
        """Custom branch name should be used as the title."""
        from cli import HermesCLI

        HermesCLI._handle_branch_command(gateway_cli, "/branch refactor approach")

        title = store.get_session_title(gateway_cli.session_id)
        assert title == "refactor approach"



    def test_branch_no_session_db(self, gateway_cli):
        """Branching without a session DB should show an error."""
        from cli import HermesCLI
        original = gateway_cli.session_id
        gateway_cli._session_db = None

        HermesCLI._handle_branch_command(gateway_cli, "/branch")

        # session_id should not have changed
        assert gateway_cli.session_id == original




    def test_branch_fires_on_session_switch_hook(self, gateway_cli):
        """The /branch command must notify memory providers of the rotation.

        Without this, providers that cache per-session state in
        initialize() keep writing under the old session_id. See #6672.
        """
        from cli import HermesCLI

        # Wire a real-ish agent object with a MagicMock memory_manager
        agent = MagicMock()
        mm = MagicMock()
        agent._memory_manager = mm
        gateway_cli.agent = agent
        original_id = gateway_cli.session_id

        HermesCLI._handle_branch_command(gateway_cli, "/branch")

        # Hook must have been called exactly once with the new session_id,
        # parent pointing at the branched-from session, reset=False, and
        # reason="branch" for diagnostics.
        assert mm.on_session_switch.call_count == 1
        _, kwargs = mm.on_session_switch.call_args
        assert mm.on_session_switch.call_args.args[0] == gateway_cli.session_id
        assert kwargs["parent_session_id"] == original_id
        assert kwargs["reset"] is False
        assert kwargs["reason"] == "branch"





class TestBranchFlushesBeforeEndSession:
    """Regression for #47202: /branch must flush un-persisted messages to
    the session DB before ending the old session, just like /new and
    compress_context() already do."""

    def test_branch_flushes_when_agent_present(self, gateway_cli):
        from cli import HermesCLI

        agent = MagicMock()
        gateway_cli.agent = agent

        HermesCLI._handle_branch_command(gateway_cli, "/branch")

        agent._flush_messages_to_session_db.assert_called_once_with(
            gateway_cli.conversation_history,
            conversation_history=gateway_cli.conversation_history,
        )

    def test_branch_child_row_carries_the_parent_system_prompt(self, owner, gateway_cli, store, monkeypatch):
        """The branch's first turn must send the bytes the parent already sends: with no stored prompt the
        child rebuilds (a fresh workspace probe) and the copied transcript's warm cache is lost at byte 0.
        The owner creates the child row; the CLI hands the prompt to the store's ``create_session``."""
        from cli import HermesCLI

        created = []
        monkeypatch.setattr(store, "create_session", lambda **kwargs: created.append(kwargs))
        parent_id = gateway_cli.session_id
        owner.persistence.sessions.update_system_prompt(parent_id, "PARENT PROMPT\n\nWorkspace snapshot: session start")
        # No live agent: the parent's persisted row is the source.
        HermesCLI._handle_branch_command(gateway_cli, "/branch")
        child = store.get_session(gateway_cli.session_id)
        assert child["parent_session_id"] == parent_id
        assert created[-1]["system_prompt"] == "PARENT PROMPT\n\nWorkspace snapshot: session start"

        # A live agent's cached prompt (what THIS process sends) wins over the row.
        gateway_cli.agent = MagicMock(_cached_system_prompt="LIVE PROMPT")
        HermesCLI._handle_branch_command(gateway_cli, "/branch")
        assert created[-1]["system_prompt"] == "LIVE PROMPT"


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

    def test_reasoning_fields_survive_branch(self, owner, gateway_cli, store):
        from cli import HermesCLI

        gateway_cli.conversation_history = [
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
        _persist_history(owner, gateway_cli)

        HermesCLI._handle_branch_command(gateway_cli, "/branch")

        messages = store.get_messages_as_conversation(gateway_cli.session_id)
        assistant = next(m for m in messages if m["role"] == "assistant")
        assert assistant["reasoning_details"] == REASONING_DETAILS
        assert assistant["codex_reasoning_items"] == CODEX_REASONING_ITEMS
        assert assistant["codex_message_items"] == CODEX_MESSAGE_ITEMS
