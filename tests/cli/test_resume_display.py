"""Tests for session resume history display — _display_resumed_history() and
_preload_resumed_session().

Verifies that resuming a session shows a compact recap of the previous
conversation with correct formatting, truncation, and config behavior.
"""

import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_cli(config_overrides=None, env_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking."""
    import cli as _cli_mod
    from cli import HermesCLI

    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all", "resume_display": "full"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in _clean_config and isinstance(_clean_config[k], dict):
                _clean_config[k].update(v)
            else:
                _clean_config[k] = v

    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    if env_overrides:
        clean_env.update(env_overrides)
    with (
        patch("cli.get_tool_definitions", return_value=[]),
        patch.dict("os.environ", clean_env, clear=False),
        patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
    ):
        return HermesCLI(**kwargs)


# ── Sample conversation histories for tests ──────────────────────────


def _simple_history():
    """Two-turn conversation: user → assistant → user → assistant."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."},
        {"role": "user", "content": "How do I install it?"},
        {"role": "assistant", "content": "You can install Python from python.org."},
    ]


def _tool_call_history():
    """Conversation with tool calls and tool results."""
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Search for Python tutorials"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"python tutorials"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "web_extract", "arguments": '{"urls":["https://example.com"]}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Found 5 results..."},
        {"role": "tool", "tool_call_id": "call_2", "content": "Page content..."},
        {"role": "assistant", "content": "Here are some great Python tutorials I found."},
    ]


def _large_history(n_exchanges=15):
    """Build a history with many exchanges to test truncation."""
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": f"Question #{i + 1}: What is item {i + 1}?"})
        msgs.append({"role": "assistant", "content": f"Answer #{i + 1}: Item {i + 1} is great."})
    return msgs


def _multimodal_history():
    """Conversation with multimodal (image) content."""
    return [
        {"role": "system", "content": "system prompt"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
            ],
        },
        {"role": "assistant", "content": "I see a cat in the image."},
    ]


# ── Tests for _display_resumed_history ───────────────────────────────


class TestDisplayResumedHistory:
    """_display_resumed_history() renders a Rich panel with conversation recap."""

    def _capture_display(self, cli_obj):
        """Run _display_resumed_history and capture the Rich console output."""
        buf = StringIO()
        cli_obj.console.file = buf
        cli_obj._display_resumed_history()
        return buf.getvalue()

    def test_simple_history_shows_user_and_assistant(self):
        cli = _make_cli()
        cli.conversation_history = _simple_history()
        output = self._capture_display(cli)

        assert "You:" in output
        assert "Hermes:" in output
        assert "What is Python?" in output
        assert "Python is a high-level programming language." in output
        assert "How do I install it?" in output

    def test_system_messages_hidden(self):
        cli = _make_cli()
        cli.conversation_history = _simple_history()
        output = self._capture_display(cli)

        assert "You are a helpful assistant" not in output

    def test_tool_messages_hidden(self):
        cli = _make_cli()
        cli.conversation_history = _tool_call_history()
        output = self._capture_display(cli)

        # Tool result content should NOT appear
        assert "Found 5 results" not in output
        assert "Page content" not in output

    def test_tool_calls_shown_as_summary(self):
        cli = _make_cli()
        cli.conversation_history = _tool_call_history()
        output = self._capture_display(cli)

        assert "2 tool calls" in output
        assert "web_search" in output
        assert "web_extract" in output

    def test_long_user_message_truncated(self):
        cli = _make_cli()
        long_text = "A" * 500
        cli.conversation_history = [
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": "OK."},
        ]
        output = self._capture_display(cli)

        # Should have truncation indicator and NOT contain the full 500 chars
        assert "..." in output
        assert "A" * 500 not in output
        # The 300-char truncated text is present but may be line-wrapped by
        # Rich's panel renderer, so check the total A count in the output
        a_count = output.count("A")
        assert 200 <= a_count <= 310  # roughly 300 chars (±panel padding)

    def test_long_assistant_message_truncated(self):
        cli = _make_cli()
        long_text = "B" * 400
        cli.conversation_history = [
            {"role": "user", "content": "Tell me a lot."},
            {"role": "assistant", "content": long_text},
        ]
        output = self._capture_display(cli)

        assert "..." in output
        assert "B" * 400 not in output

    def test_multiline_assistant_truncated(self):
        cli = _make_cli()
        multi = "\n".join([f"Line {i}" for i in range(20)])
        cli.conversation_history = [
            {"role": "user", "content": "Show me lines."},
            {"role": "assistant", "content": multi},
        ]
        output = self._capture_display(cli)

        # First 3 lines should be there
        assert "Line 0" in output
        assert "Line 1" in output
        assert "Line 2" in output
        # Line 19 should NOT be there (truncated after 3 lines)
        assert "Line 19" not in output

    def test_large_history_shows_truncation_indicator(self):
        cli = _make_cli()
        cli.conversation_history = _large_history(n_exchanges=15)
        output = self._capture_display(cli)

        # Should show "earlier messages" indicator
        assert "earlier messages" in output
        # Last question should still be visible
        assert "Question #15" in output

    def test_multimodal_content_handled(self):
        cli = _make_cli()
        cli.conversation_history = _multimodal_history()
        output = self._capture_display(cli)

        assert "What's in this image?" in output
        assert "[image]" in output

    def test_empty_history_no_output(self):
        cli = _make_cli()
        cli.conversation_history = []
        output = self._capture_display(cli)

        assert output.strip() == ""

    def test_minimal_config_suppresses_display(self):
        cli = _make_cli(config_overrides={"display": {"resume_display": "minimal"}})
        # resume_display is captured as an instance variable during __init__
        assert cli.resume_display == "minimal"
        cli.conversation_history = _simple_history()
        output = self._capture_display(cli)

        assert output.strip() == ""

    def test_panel_has_title(self):
        cli = _make_cli()
        cli.conversation_history = _simple_history()
        output = self._capture_display(cli)

        assert "Previous Conversation" in output

    def test_assistant_with_no_content_no_tools_skipped(self):
        """Assistant messages with no visible output (e.g. pure reasoning)
        are skipped in the recap."""
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None},
        ]
        output = self._capture_display(cli)

        # The assistant entry should be skipped, only the user message shown
        assert "You:" in output
        assert "Hermes:" not in output

    def test_only_system_messages_no_output(self):
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "system", "content": "You are helpful."},
        ]
        output = self._capture_display(cli)

        assert output.strip() == ""

    def test_reasoning_scratchpad_stripped(self):
        """<REASONING_SCRATCHPAD> blocks should be stripped from display."""
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": "Think about this"},
            {
                "role": "assistant",
                "content": (
                    "<REASONING_SCRATCHPAD>\nLet me think step by step.\n"
                    "</REASONING_SCRATCHPAD>\n\nThe answer is 42."
                ),
            },
        ]
        output = self._capture_display(cli)

        assert "REASONING_SCRATCHPAD" not in output
        assert "Let me think step by step" not in output
        assert "The answer is 42" in output

    def test_pure_reasoning_message_skipped(self):
        """Assistant messages that are only reasoning should be skipped."""
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "<REASONING_SCRATCHPAD>\nJust thinking...\n</REASONING_SCRATCHPAD>",
            },
            {"role": "assistant", "content": "Hi there!"},
        ]
        output = self._capture_display(cli)

        assert "Just thinking" not in output
        assert "Hi there!" in output

    def test_assistant_with_text_and_tool_calls(self):
        """When an assistant message has both text content AND tool_calls."""
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": "Do something complex"},
            {
                "role": "assistant",
                "content": "Let me search for that.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
                    }
                ],
            },
        ]
        output = self._capture_display(cli)

        assert "Let me search for that." in output
        assert "1 tool call" in output
        assert "terminal" in output


# ── Tests for _preload_resumed_session ──────────────────────────────


class TestPreloadResumedSession:
    """_preload_resumed_session() loads session from DB early."""

    def test_returns_false_when_not_resumed(self):
        cli = _make_cli()
        assert cli._preload_resumed_session() is False

    def test_returns_false_when_no_session_db(self):
        cli = _make_cli(resume="test_session_id")
        cli._session_db = None
        assert cli._preload_resumed_session() is False

    def test_returns_false_when_session_not_found(self):
        cli = _make_cli(resume="nonexistent_session")
        mock_db = MagicMock()
        mock_db.get_session.return_value = None
        cli._session_db = mock_db

        buf = StringIO()
        cli.console.file = buf
        result = cli._preload_resumed_session()

        assert result is False
        output = buf.getvalue()
        assert "Session not found" in output

    def test_returns_false_when_session_has_no_messages(self):
        cli = _make_cli(resume="empty_session")
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "empty_session", "title": None}
        mock_db.get_messages_as_conversation.return_value = []
        cli._session_db = mock_db

        buf = StringIO()
        cli.console.file = buf
        result = cli._preload_resumed_session()

        assert result is False
        output = buf.getvalue()
        assert "no messages" in output

    def test_loads_session_successfully(self):
        cli = _make_cli(resume="good_session")
        messages = _simple_history()
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "good_session", "title": "Test Session"}
        mock_db.get_messages_as_conversation.return_value = messages
        cli._session_db = mock_db

        buf = StringIO()
        cli.console.file = buf
        result = cli._preload_resumed_session()

        assert result is True
        assert cli.conversation_history == messages
        output = buf.getvalue()
        assert "Resumed session" in output
        assert "good_session" in output
        assert "Test Session" in output
        assert "2 user messages" in output

    def test_reopens_session_in_db(self):
        cli = _make_cli(resume="reopen_session")
        messages = [{"role": "user", "content": "hi"}]
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "reopen_session", "title": None}
        mock_db.get_messages_as_conversation.return_value = messages
        mock_conn = MagicMock()
        mock_db._conn = mock_conn
        cli._session_db = mock_db

        buf = StringIO()
        cli.console.file = buf
        cli._preload_resumed_session()

        # Should have executed UPDATE to clear ended_at
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "ended_at = NULL" in call_args[0][0]
        mock_conn.commit.assert_called_once()

    def test_singular_user_message_grammar(self):
        """1 user message should say 'message' not 'messages'."""
        cli = _make_cli(resume="one_msg_session")
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "one_msg_session", "title": None}
        mock_db.get_messages_as_conversation.return_value = messages
        mock_db._conn = MagicMock()
        cli._session_db = mock_db

        buf = StringIO()
        cli.console.file = buf
        cli._preload_resumed_session()

        output = buf.getvalue()
        assert "1 user message," in output
        assert "1 user messages" not in output


# ── Integration: _init_agent skips when preloaded ────────────────────


class TestInitAgentSkipsPreloaded:
    """_init_agent() should skip DB load when history is already populated."""

    def test_init_agent_skips_db_when_preloaded(self):
        """If conversation_history is already set, _init_agent should not
        reload from the DB."""
        cli = _make_cli(resume="preloaded_session")
        cli.conversation_history = _simple_history()

        mock_db = MagicMock()
        cli._session_db = mock_db

        # _init_agent will fail at credential resolution (no real API key),
        # but the session-loading block should be skipped entirely
        with patch.object(cli, "_ensure_runtime_credentials", return_value=False):
            cli._init_agent()

        # get_messages_as_conversation should NOT have been called
        mock_db.get_messages_as_conversation.assert_not_called()


# ── Config default tests ─────────────────────────────────────────────


class TestResumeDisplayConfig:
    """resume_display config option defaults and behavior."""

    def test_default_config_has_resume_display(self):
        """DEFAULT_CONFIG in hermes_cli/config.py includes resume_display."""
        from hermes_cli.config import DEFAULT_CONFIG
        display = DEFAULT_CONFIG.get("display", {})
        assert "resume_display" in display
        assert display["resume_display"] == "full"

    def test_cli_defaults_have_resume_display(self):
        """cli.py load_cli_config defaults include resume_display."""
        import cli as _cli_mod
        from cli import load_cli_config

        with (
            patch("pathlib.Path.exists", return_value=False),
            patch.dict("os.environ", {"LLM_MODEL": ""}, clear=False),
        ):
            config = load_cli_config()

        display = config.get("display", {})
        assert display.get("resume_display") == "full"
