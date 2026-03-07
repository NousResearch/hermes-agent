"""Headless CLI integration tests.

Drives HermesCLI through scripted interactions without a terminal,
validating the full pipeline: commands, chat, think blocks, paste,
tool_call stripping, and response rendering.

Run:
    python -m pytest tests/test_headless_cli.py -v
"""

import io
import os
import re
import base64
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_run_result(content, tool_calls=None):
    """Build a dict matching AIAgent.run_conversation() return shape."""
    messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": content},
    ]
    if tool_calls:
        messages[-1]["tool_calls"] = tool_calls
    return {
        "final_response": content,
        "messages": messages,
        "api_calls": 1,
        "completed": True,
        "partial": False,
        "interrupted": False,
    }


@pytest.fixture
def cli_instance():
    """Create a HermesCLI with a mocked agent, bypassing prompt_toolkit."""
    # Patch heavy deps before import
    with patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test-key-fake",
        "HERMES_QUIET": "1",
    }):
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        # Minimal init — skip __init__ which starts prompt_toolkit
        cli.model = "test/mock-model"
        cli.api_key = "test-key-fake"
        cli.base_url = "http://localhost:9999/v1"
        cli.provider = "test"
        cli.api_mode = "chat_completions"
        cli.max_turns = 10
        cli.enabled_toolsets = None
        cli.verbose = False
        cli.compact = False
        cli.show_thinking = False
        cli.conversation_history = []
        cli.session_id = "test-session"
        cli._attached_images = []
        cli._image_counter = 0
        cli._app = None
        cli._clarify_state = False
        cli._clarify_freetext = False
        cli._interrupt_queue = MagicMock()
        cli._interrupt_queue.empty.return_value = True
        cli._pending_input = MagicMock()
        cli._resumed = False
        cli._session_db = None
        cli.requested_provider = "test"
        cli._provider_source = None
        cli._explicit_api_key = "test-key-fake"
        cli._explicit_base_url = "http://localhost:9999/v1"
        cli.tool_progress_mode = "off"
        cli.console = MagicMock()
        cli._background_jobs = []
        cli._next_job_id = 1
        cli._background_requested = False

        # Mock agent
        cli.agent = MagicMock()
        cli.agent.run_conversation = MagicMock()
        cli.agent.session_prompt_tokens = 0
        cli.agent.session_completion_tokens = 0
        cli.agent.session_total_tokens = 0
        cli.agent.session_api_calls = 0
        cli.agent._interrupt_requested = False
        cli.agent.interrupt = MagicMock()

        # Stub methods that touch real I/O
        cli._ensure_runtime_credentials = MagicMock(return_value=True)
        cli._init_agent = MagicMock(return_value=True)

        yield cli


def _capture_chat(cli, message, response_content):
    """Run cli.chat() with a mocked response and capture stdout."""
    cli.agent.run_conversation.return_value = _make_run_result(response_content)

    buf = io.StringIO()
    # cli.chat() uses both print() and _cprint() (prompt_toolkit).
    # Patch _cprint to go to our buffer too.
    from cli import _cprint as _real_cprint

    captured_lines = []

    def _capture_cprint(text):
        # Strip ANSI for assertions
        clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
        captured_lines.append(clean)

    with patch('cli._cprint', side_effect=_capture_cprint):
        with patch('sys.stdout', new=buf):
            cli.chat(message)

    stdout_text = buf.getvalue()
    cprint_text = "\n".join(captured_lines)
    return stdout_text + "\n" + cprint_text


def _strip_ansi(text):
    """Remove ANSI escape sequences."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


# ── Test Scenarios ───────────────────────────────────────────────────────


class TestBasicChat:
    """1. Basic greeting — send a message, get a response."""

    def test_hey_boo_gets_response(self, cli_instance):
        output = _capture_chat(cli_instance, "hey boo", "Hey there! How can I help?")
        assert "Hey there!" in output
        assert "How can I help?" in output

    def test_response_in_hermes_box(self, cli_instance):
        output = _capture_chat(cli_instance, "hello", "World")
        assert "Hermes" in output
        assert "World" in output

    def test_conversation_history_updated(self, cli_instance):
        _capture_chat(cli_instance, "test msg", "test response")
        # chat() replaces history with agent result messages
        assert len(cli_instance.conversation_history) > 0
        assert any(m.get("role") == "assistant" for m in cli_instance.conversation_history)

    def test_agent_called_with_message(self, cli_instance):
        _capture_chat(cli_instance, "hey boo", "response")
        cli_instance.agent.run_conversation.assert_called_once()
        call_kwargs = cli_instance.agent.run_conversation.call_args
        assert "hey boo" in str(call_kwargs)


class TestThinkBlocks:
    """2-5. Think block visibility toggling."""

    RESPONSE_WITH_THINK = "<think>I should greet them warmly</think>Hello! Nice to see you."

    def test_think_hidden_by_default(self, cli_instance):
        assert cli_instance.show_thinking is False
        output = _capture_chat(cli_instance, "hi", self.RESPONSE_WITH_THINK)
        assert "Hello! Nice to see you." in output
        assert "I should greet" not in output
        assert "<think>" not in output
        assert "</think>" not in output

    def test_thinkon_command(self, cli_instance):
        buf = io.StringIO()
        with patch('sys.stdout', new=buf):
            result = cli_instance.process_command("/thinkon")
        assert result is True
        assert cli_instance.show_thinking is True

    def test_think_visible_after_thinkon(self, cli_instance):
        cli_instance.process_command("/thinkon")
        assert cli_instance.show_thinking is True
        output = _capture_chat(cli_instance, "what do you think?", self.RESPONSE_WITH_THINK)
        # When thinking is on, the think content should appear (styled)
        assert "Hello! Nice to see you." in output
        # The think content should be rendered (either raw or styled)
        assert "greet" in output or "thinking" in output.lower()

    def test_thinkoff_command(self, cli_instance):
        cli_instance.show_thinking = True
        buf = io.StringIO()
        with patch('sys.stdout', new=buf):
            result = cli_instance.process_command("/thinkoff")
        assert result is True
        assert cli_instance.show_thinking is False

    def test_think_hidden_after_thinkoff(self, cli_instance):
        cli_instance.show_thinking = True
        cli_instance.process_command("/thinkoff")
        output = _capture_chat(
            cli_instance, "what aren't you thinking?",
            "<think>Deep internal reasoning about existence</think>I'm just here to help!"
        )
        assert "I'm just here to help!" in output
        assert "Deep internal reasoning" not in output
        assert "<think>" not in output

    def test_unclosed_think_stripped(self, cli_instance):
        output = _capture_chat(
            cli_instance, "test",
            "<think>This think block never closes and the model ran out of tokens"
        )
        assert "<think>" not in output
        assert "never closes" not in output

    def test_orphaned_close_think_stripped(self, cli_instance):
        output = _capture_chat(
            cli_instance, "test",
            "reasoning from separate field</think>\nActual answer here"
        )
        assert "</think>" not in output
        assert "Actual answer here" in output
        assert "reasoning from separate" not in output


class TestToolCallStripping:
    """6-7. <tool_call> tags stripped from display."""

    def test_complete_tool_call_stripped(self, cli_instance):
        response = 'Let me help.\n<tool_call>{"name": "execute_code", "arguments": {"code": "print(1)"}}</tool_call>\nDone.'
        output = _capture_chat(cli_instance, "run code", response)
        assert "<tool_call>" not in output
        assert "</tool_call>" not in output
        assert "execute_code" not in output

    def test_truncated_tool_call_stripped(self, cli_instance):
        response = 'Planning...\n<tool_call>{"name": "delegate_task", "arguments": {"goal": "test"'
        output = _capture_chat(cli_instance, "do task", response)
        assert "<tool_call>" not in output
        assert "delegate_task" not in output

    def test_text_around_tool_calls_preserved(self, cli_instance):
        response = 'Before tool.\n<tool_call>{"name": "x", "arguments": {}}</tool_call>\nAfter tool.'
        output = _capture_chat(cli_instance, "test", response)
        assert "Before tool." in output
        assert "After tool." in output
        assert "<tool_call>" not in output


class TestPasteCommand:
    """8-9. /paste command — clipboard image attachment."""

    def test_paste_no_image(self, cli_instance):
        buf = io.StringIO()
        with patch('cli._cprint') as mock_cprint:
            with patch('hermes_cli.clipboard.has_clipboard_image', return_value=False):
                with patch('sys.stdout', new=buf):
                    cli_instance.process_command("/paste")
        # Should print "no image found" message
        mock_cprint.assert_called()
        output = str(mock_cprint.call_args_list)
        assert "No image" in output or "no image" in output.lower() or "._." in output

    def test_paste_with_image(self, cli_instance):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Write a tiny valid PNG
            f.write(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
            tmp_path = f.name

        try:
            with patch('cli._cprint') as mock_cprint:
                with patch('hermes_cli.clipboard.has_clipboard_image', return_value=True):
                    with patch('hermes_cli.clipboard.save_clipboard_image', return_value=True):
                        buf = io.StringIO()
                        with patch('sys.stdout', new=buf):
                            cli_instance.process_command("/paste")
            output = str(mock_cprint.call_args_list)
            assert "attached" in output.lower() or "Image" in output
            assert len(cli_instance._attached_images) == 1
        finally:
            os.unlink(tmp_path)

    def test_paste_adds_to_attached(self, cli_instance):
        with patch('hermes_cli.clipboard.has_clipboard_image', return_value=True):
            with patch('hermes_cli.clipboard.save_clipboard_image', return_value=True):
                with patch('cli._cprint'):
                    with patch('sys.stdout', new=io.StringIO()):
                        cli_instance.process_command("/paste")
                        cli_instance.process_command("/paste")
        assert len(cli_instance._attached_images) == 2


class TestSlashCommands:
    """10-12. Various slash commands work correctly."""

    def test_help_command(self, cli_instance):
        buf = io.StringIO()
        with patch('sys.stdout', new=buf):
            result = cli_instance.process_command("/help")
        assert result is True
        output = buf.getvalue()
        assert "/help" in output or "help" in output.lower()

    def test_quit_returns_false(self, cli_instance):
        result = cli_instance.process_command("/quit")
        assert result is False

    def test_exit_returns_false(self, cli_instance):
        assert cli_instance.process_command("/exit") is False

    def test_q_returns_false(self, cli_instance):
        assert cli_instance.process_command("/q") is False

    def test_unknown_command(self, cli_instance):
        buf = io.StringIO()
        with patch('sys.stdout', new=buf):
            with patch('cli._cprint'):
                result = cli_instance.process_command("/nonexistent_cmd_xyz")
        assert result is True  # Unknown commands don't exit


class TestErrorHandling:
    """13-14. Error responses handled gracefully."""

    def test_failed_result_shows_error(self, cli_instance):
        cli_instance.agent.run_conversation.return_value = {
            "final_response": None,
            "messages": [],
            "api_calls": 1,
            "completed": False,
            "partial": False,
            "interrupted": False,
            "failed": True,
            "error": "Invalid model: test/nonexistent",
        }
        output = _capture_chat.__wrapped__(cli_instance, "test", None) if hasattr(_capture_chat, '__wrapped__') else ""
        # Manually run chat and capture
        buf = io.StringIO()
        captured = []
        with patch('cli._cprint', side_effect=lambda t: captured.append(re.sub(r'\x1b\[[0-9;]*m', '', t))):
            with patch('sys.stdout', new=buf):
                cli_instance.chat("test")
        output = buf.getvalue() + "\n".join(captured)
        assert "Error" in output or "error" in output.lower() or "Invalid model" in output

    def test_empty_response_handled(self, cli_instance):
        cli_instance.agent.run_conversation.return_value = _make_run_result("")
        buf = io.StringIO()
        captured = []
        with patch('cli._cprint', side_effect=lambda t: captured.append(t)):
            with patch('sys.stdout', new=buf):
                cli_instance.chat("hello")
        # Should not crash — empty response just produces no box


class TestResponseRendering:
    """15-17. Response box rendering."""

    def test_response_box_has_borders(self, cli_instance):
        output = _capture_chat(cli_instance, "test", "Hello world")
        assert "╭" in output
        assert "╰" in output
        assert "Hermes" in output

    def test_multiline_response(self, cli_instance):
        response = "Line 1\nLine 2\nLine 3"
        output = _capture_chat(cli_instance, "test", response)
        assert "Line 1" in output
        assert "Line 2" in output
        assert "Line 3" in output

    def test_special_chars_in_response(self, cli_instance):
        response = "Code: `x = 1` and <html> & \"quotes\""
        output = _capture_chat(cli_instance, "test", response)
        assert "x = 1" in output


class TestFullScenario:
    """18. End-to-end scenario walking through multiple interactions."""

    def test_full_interaction_flow(self, cli_instance):
        # Step 1: Greeting
        output = _capture_chat(cli_instance, "hey boo", "Hey! What's up?")
        assert "Hey!" in output

        # Step 2: Verify thinking is off by default
        assert cli_instance.show_thinking is False

        # Step 3: Turn on thinking
        with patch('sys.stdout', new=io.StringIO()):
            cli_instance.process_command("/thinkon")
        assert cli_instance.show_thinking is True

        # Step 4: Ask something — think blocks should be visible
        output = _capture_chat(
            cli_instance,
            "what do you think?",
            "<think>The user wants my opinion. Let me be thoughtful.</think>I think we should keep building!",
        )
        assert "I think we should keep building!" in output
        # Think content should be visible (styled or raw)
        assert "thoughtful" in output or "thinking" in output.lower()

        # Step 5: Turn off thinking
        with patch('sys.stdout', new=io.StringIO()):
            cli_instance.process_command("/thinkoff")
        assert cli_instance.show_thinking is False

        # Step 6: Ask again — think blocks should be hidden
        output = _capture_chat(
            cli_instance,
            "what aren't you thinking?",
            "<think>I'm secretly reasoning about quantum mechanics</think>Nothing special, just here to help!",
        )
        assert "Nothing special" in output
        assert "quantum mechanics" not in output
        assert "<think>" not in output

        # Step 7: Response with tool_call tags should be clean
        output = _capture_chat(
            cli_instance,
            "run a test",
            'Running now.\n<tool_call>{"name": "execute_code", "arguments": {"code": "print(42)"}}</tool_call>\nDone!',
        )
        assert "Running now." in output
        assert "Done!" in output
        assert "<tool_call>" not in output
        assert "execute_code" not in output

        # Step 8: Quit
        result = cli_instance.process_command("/quit")
        assert result is False


class TestContextCommand:
    """19-22. /context command tests."""

    def _run_context(self, cli_instance):
        """Helper to run /context and capture all output (print + console.print)."""
        buf = io.StringIO()
        console_captured = []
        cli_instance.console = MagicMock()
        cli_instance.console.print = MagicMock(
            side_effect=lambda t, **kw: console_captured.append(
                re.sub(r'\[.*?\]', '', str(t))
            )
        )
        with patch('cli._cprint', side_effect=lambda t: console_captured.append(t)):
            with patch('sys.stdout', new=buf):
                result = cli_instance.process_command("/context")
        output = buf.getvalue() + "\n".join(console_captured)
        return result, output

    def test_context_no_agent(self, cli_instance):
        """Context command works even without active agent."""
        cli_instance.model = "local/qwen3.5-9b"
        cli_instance.agent = None
        result, output = self._run_context(cli_instance)
        assert result is True
        assert "32,768" in output or "32768" in output

    def test_context_with_agent(self, cli_instance):
        """Context shows usage from compressor."""
        cli_instance.model = "local/qwen3.5-9b"
        compressor = MagicMock()
        compressor.context_length = 32768
        compressor.last_prompt_tokens = 8192
        cli_instance.agent.context_compressor = compressor
        result, output = self._run_context(cli_instance)
        assert "8,192" in output or "8192" in output
        assert "24,576" in output or "24576" in output

    def test_context_unsupported_model(self, cli_instance):
        """Context shows error for unsupported models."""
        cli_instance.model = "some/unknown-model"
        cli_instance.agent = None
        result, output = self._run_context(cli_instance)
        assert "not available" in output.lower() or "not supported" in output.lower() or "qwen" in output.lower()

    def test_context_bar_shows_percentage(self, cli_instance):
        """Context bar shows percentage used."""
        cli_instance.model = "local/qwen3.5-9b"
        compressor = MagicMock()
        compressor.context_length = 32768
        compressor.last_prompt_tokens = 16384  # 50%
        cli_instance.agent.context_compressor = compressor
        result, output = self._run_context(cli_instance)
        assert "50.0%" in output


class TestBackgroundJobs:
    """23-26. Background job tests."""

    def test_jobs_empty(self, cli_instance):
        """No background jobs initially."""
        buf = io.StringIO()
        with patch('cli._cprint'):
            with patch('sys.stdout', new=buf):
                result = cli_instance.process_command("/jobs")
        assert result is True
        output = buf.getvalue()
        assert "No background" in output or "no background" in output.lower()

    def test_jobs_command_registered(self, cli_instance):
        """The /jobs command doesn't crash."""
        buf = io.StringIO()
        with patch('cli._cprint'):
            with patch('sys.stdout', new=buf):
                result = cli_instance.process_command("/jobs")
        assert result is True

    def test_fg_no_jobs(self, cli_instance):
        """Foreground with no jobs shows helpful message."""
        buf = io.StringIO()
        with patch('cli._cprint'):
            with patch('sys.stdout', new=buf):
                result = cli_instance.process_command("/fg")
        assert result is True
        output = buf.getvalue()
        assert "No background" in output or "no background" in output.lower() or "not found" in output.lower()

    def test_fg_invalid_id(self, cli_instance):
        """Foreground with invalid ID shows error."""
        buf = io.StringIO()
        with patch('cli._cprint'):
            with patch('sys.stdout', new=buf):
                result = cli_instance.process_command("/fg abc")
        assert result is True
        output = buf.getvalue()
        assert "Invalid" in output or "invalid" in output.lower()

    def test_background_fields_exist(self, cli_instance):
        """Background job tracking fields exist on CLI instance."""
        assert hasattr(cli_instance, '_background_jobs')
        assert hasattr(cli_instance, '_next_job_id')
        assert isinstance(cli_instance._background_jobs, list)
