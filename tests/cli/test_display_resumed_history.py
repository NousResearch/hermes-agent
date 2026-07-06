"""Tests for _display_resumed_history() handling of empty messages.

Regression coverage for issue #59265 — IndexError when a resumed session
contains an empty (zero-length) message.
"""

import os
import sys
from io import StringIO
from unittest.mock import patch

import cli as cli_mod

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
        patch.dict(cli_mod.__dict__, {"CLI_CONFIG": _clean_config}),
    ):
        return HermesCLI(**kwargs)


# ── Fixtures ────────────────────────────────────────────────────────


def _empty_user_history():
    """History with a single empty-string user message (the crashing case)."""
    return [
        {"role": "user", "content": ""},
    ]


def _whitespace_user_history():
    """History with a whitespace-only user message."""
    return [
        {"role": "user", "content": "   \n\t  \n"},
    ]


def _empty_assistant_history():
    """History with an empty assistant message."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": ""},
    ]


def _all_empty_history():
    """History where every visible message is empty."""
    return [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": ""},
    ]


def _mixed_empty_history():
    """History mixing empty and non-empty messages."""
    return [
        {"role": "user", "content": "Real question"},
        {"role": "assistant", "content": ""},     # empty — should not crash
        {"role": "user", "content": "Another real question"},
        {"role": "assistant", "content": "Real answer"},
        {"role": "user", "content": ""},           # empty — should not crash
    ]


def _normal_history():
    """Baseline non-empty history used to ensure nothing regresses."""
    return [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level language."},
    ]


# ── Tests for empty-message handling in _display_resumed_history ───


class TestEmptyMessageHandling:
    """Regression tests for #59265: empty messages must not crash."""

    def _capture_display(self, cli_obj):
        """Run _display_resumed_history and capture the Rich console output."""
        buf = StringIO()
        cli_obj.console.file = buf
        # The function under test should never raise IndexError again.
        cli_obj._display_resumed_history()
        return buf.getvalue()

    def test_empty_user_message_does_not_crash(self):
        """Single empty user message must not raise IndexError (#59265)."""
        cli = _make_cli()
        cli.conversation_history = _empty_user_history()
        # Should not raise
        output = self._capture_display(cli)

        # The panel header should still be rendered.
        assert "Previous Conversation" in output
        # The user role label should be present.
        assert "You:" in output

    def test_whitespace_only_user_message_does_not_crash(self):
        """A whitespace-only user message must not crash either."""
        cli = _make_cli()
        cli.conversation_history = _whitespace_user_history()
        # Should not raise
        output = self._capture_display(cli)

        assert "Previous Conversation" in output
        assert "You:" in output

    def test_empty_assistant_message_does_not_crash(self):
        """Empty assistant message must not raise IndexError."""
        cli = _make_cli()
        cli.conversation_history = _empty_assistant_history()
        # Should not raise
        output = self._capture_display(cli)

        # Panel + user + (handled) assistant all rendered.
        assert "Previous Conversation" in output
        assert "You:" in output
        assert "Hello" in output

    def test_all_empty_messages_still_shows_session_metadata(self):
        """Even when every message is empty, the panel (with title) is shown
        so the user gets session metadata context (issue #59265: session
        metadata should still be visible)."""
        cli = _make_cli()
        cli.conversation_history = _all_empty_history()
        # Should not raise
        output = self._capture_display(cli)

        # The panel header is the session metadata anchor — must still appear.
        assert "Previous Conversation" in output

    def test_mixed_empty_and_nonempty_messages(self):
        """Non-empty messages render normally; empty messages don't crash
        the whole resume and the panel still appears."""
        cli = _make_cli()
        cli.conversation_history = _mixed_empty_history()
        # Should not raise
        output = self._capture_display(cli)

        # Non-empty content must be present
        assert "Real question" in output
        assert "Real answer" in output
        assert "Another real question" in output
        # Panel title still rendered
        assert "Previous Conversation" in output

    def test_normal_history_still_works(self):
        """Sanity: the fix must not regress the non-empty case."""
        cli = _make_cli()
        cli.conversation_history = _normal_history()
        output = self._capture_display(cli)

        assert "What is Python?" in output
        assert "Python is a high-level language." in output
        assert "Previous Conversation" in output

    def test_empty_message_logs_at_debug(self):
        """Empty-message encounters should emit a DEBUG log entry so they're
        not silently swallowed (per the issue's hardening guidance)."""
        import logging

        cli = _make_cli()
        cli.conversation_history = _mixed_empty_history()

        # Capture DEBUG log output for the mixin module's logger.
        from hermes_cli import cli_agent_setup_mixin as mixin_mod
        logger_name = mixin_mod.__name__

        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        old_level = logging.getLogger(logger_name).level
        logging.getLogger(logger_name).setLevel(logging.DEBUG)
        logging.getLogger(logger_name).addHandler(handler)
        try:
            try:
                cli._display_resumed_history()
            except Exception:
                # Even if the test environment chokes, the log should still
                # have been written. We don't care about the underlying output
                # here — only that the DEBUG line was emitted.
                pass
            log_output = buf.getvalue()
        finally:
            logging.getLogger(logger_name).removeHandler(handler)
            logging.getLogger(logger_name).setLevel(old_level)

        assert "_display_resumed_history" in log_output
        assert "empty" in log_output.lower()
