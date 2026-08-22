"""Tests for session resume history display — _display_resumed_history() and
_preload_resumed_session().

Verifies that resuming a session shows a compact recap of the previous
conversation with correct formatting, truncation, and config behavior.
"""

from io import StringIO
from unittest.mock import MagicMock, patch

import cli as cli_mod



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


    def test_timeline_markers_render_as_events_not_user_input(self):
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": "opaque model context", "display_kind": "model_switch"},
            {"role": "user", "content": "opaque delegation context", "display_kind": "async_delegation_complete"},
            {"role": "user", "content": "opaque hidden context", "display_kind": "hidden"},
        ]

        output = self._capture_display(cli)

        assert "◈ model changed" in output
        assert "◈ background delegation completed" in output
        assert "You:" not in output
        assert "opaque" not in output



    def test_tool_only_message_skipped_by_default(self):
        """Assistant messages with only tool_calls (no text) are skipped when
        resume_skip_tool_only=True (the default). The summary line is hidden.
        """
        cli = _make_cli()
        cli.conversation_history = _tool_call_history()
        output = self._capture_display(cli)

        # The tool-only assistant entry should be skipped
        assert "2 tool calls" not in output
        # The final text reply should still appear
        assert "Here are some great Python tutorials" in output









    def test_empty_message_content_does_not_crash(self):
        """Regression: _display_resumed_history IndexError when a message has empty text.

        An assistant turn that produced no text (or a blank user message)
        makes ``text.splitlines()`` return ``[]``, crashing on ``msg_lines[0]``.
        The fix guards with ``or [""]`` so the message renders as a blank line.
        """
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Follow-up question"},
            {"role": "assistant", "content": "Real answer"},
        ]
        # Must not raise IndexError
        output = self._capture_display(cli)

        assert "Follow-up question" in output
        assert "Real answer" in output

    def test_whitespace_only_message_does_not_crash(self):
        """Whitespace-only messages should also not crash the resume display."""
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "\n\n"},
        ]
        output = self._capture_display(cli)
        # Should render without error
        assert "You:" in output

    def test_minimal_config_suppresses_display(self):
        cli = _make_cli(config_overrides={"display": {"resume_display": "minimal"}})
        # resume_display is captured as an instance variable during __init__
        assert cli.resume_display == "minimal"
        cli.conversation_history = _simple_history()
        output = self._capture_display(cli)

        assert output.strip() == ""






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





    def test_unclosed_think_tag_stripped(self):
        """Unclosed <think> (truncated generation) should not leak reasoning."""
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": "Truncated response"},
            {
                "role": "assistant",
                "content": "Some text before.\n<think>\nUnfinished reasoning...",
            },
        ]
        output = self._capture_display(cli)

        assert "<think>" not in output
        assert "Unfinished reasoning" not in output
        assert "Some text before" in output





# ── Tests for _preload_resumed_session ──────────────────────────────


class TestPreloadResumedSession:
    """_preload_resumed_session() loads session from DB early."""

    def test_returns_false_when_not_resumed(self):
        cli = _make_cli()
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



    def test_reopens_session_in_db(self):
        cli = _make_cli(resume="reopen_session")
        messages = [{"role": "user", "content": "hi"}]
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "reopen_session", "title": None}
        mock_db.get_resume_conversations.return_value = (messages, messages)
        mock_db.resolve_resume_session_id.return_value = "reopen_session"
        cli._session_db = mock_db

        buf = StringIO()
        cli.console.file = buf
        cli._preload_resumed_session()

        mock_db.reopen_session.assert_called_once_with("reopen_session")

    def test_rejects_runaway_transcript_before_history_load(self):
        from hermes_state import SessionResumeTooLargeError

        cli = _make_cli(resume="runaway-session")
        mock_db = MagicMock()
        mock_db.get_session.return_value = {
            "id": "runaway-session",
            "title": None,
        }
        mock_db.resolve_resume_session_id.return_value = "runaway-session"
        mock_db.assert_resume_safe = MagicMock(
            side_effect=SessionResumeTooLargeError(20_001)
        )
        cli._session_db = mock_db

        output = StringIO()
        cli.console.file = output
        result = cli._preload_resumed_session()

        assert result is False
        assert "safe resume limit is 20000" in output.getvalue()
        mock_db.get_resume_conversations.assert_not_called()



# ── Tests for _handle_resume_command recap display ───────────────────


class TestHandleResumeCommandRecap:
    """In-session /resume should show the same recap panel as startup resume."""

    def test_resume_command_displays_recap_when_messages_restored(self):
        cli = _make_cli()
        cli.session_id = "current_session"
        messages = _simple_history()

        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "target_session", "title": "Test Session"}
        mock_db.get_resume_conversations.return_value = (messages, messages)
        # resolve_resume_session_id passes the id through when no compression chain.
        mock_db.resolve_resume_session_id.return_value = "target_session"
        cli._session_db = mock_db

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="target_session"),
            patch.object(cli, "_display_resumed_history") as display_mock,
        ):
            cli._handle_resume_command("/resume test session")

        assert cli.session_id == "target_session"
        assert cli.conversation_history == messages
        assert cli._resume_display_history == messages
        mock_db.end_session.assert_called_once_with("current_session", "resumed_other")
        mock_db.reopen_session.assert_called_once_with("target_session")
        display_mock.assert_called_once_with()


    def test_resume_command_replaces_stale_display_history(self):
        """In-session /resume B after startup --resume A must show B's recap,
        not A's. The _resume_display_history attribute set by startup resume
        must be replaced, not retained."""
        cli = _make_cli(resume="session_a")
        cli.session_id = "session_a"
        # Simulate startup --resume A having populated both projections.
        messages_a = [{"role": "user", "content": "from session A"}]
        cli.conversation_history = messages_a
        cli._resume_display_history = messages_a

        messages_b = [{"role": "user", "content": "from session B"}]
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"id": "session_b", "title": "Session B"}
        mock_db.get_resume_conversations.return_value = (messages_b, messages_b)
        mock_db.resolve_resume_session_id.return_value = "session_b"
        cli._session_db = mock_db

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="session_b"),
            patch.object(cli, "_display_resumed_history") as display_mock,
        ):
            cli._handle_resume_command("/resume session_b")

        assert cli.session_id == "session_b"
        assert cli.conversation_history == messages_b
        # The stale A display history must have been replaced by B's.
        assert cli._resume_display_history == messages_b
        assert "from session A" not in [
            m.get("content", "") for m in cli._resume_display_history
        ]
        display_mock.assert_called_once_with()


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



class TestResumeDisplaySanitization:
    """Stored history replayed by /resume must not carry raw terminal
    escapes or control chars (openai/codex#31494 bug class)."""

    def _capture_display(self, cli_obj):
        buf = StringIO()
        cli_obj.console.file = buf
        cli_obj._display_resumed_history()
        return buf.getvalue()

    def test_escape_sequences_stripped_from_user_and_assistant(self):
        cli = _make_cli()
        cli.conversation_history = [
            {"role": "user", "content": "hi \x1b[2J\x1b]0;pwned\x07 there"},
            {"role": "assistant", "content": "ok \x9b31m fine\x07"},
        ]
        output = self._capture_display(cli)
        # Rich adds its own SGR styling escapes when force_terminal is on;
        # what must NOT survive are the injected non-SGR sequences.
        assert "\x1b[2J" not in output
        assert "\x1b]0;pwned" not in output
        assert "\x9b" not in output
        assert "\x07" not in output
        assert "hi" in output and "there" in output
        assert "fine" in output
