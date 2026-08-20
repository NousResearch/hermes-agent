"""Regression tests for classic-CLI mid-run /steer dispatch.

Background
----------
/steer sent while the agent is running used to be queued through
``self._pending_input`` alongside ordinary user input.  ``process_loop``
pulls from that queue and calls ``process_command()`` — but while the
agent is running, ``process_loop`` is blocked inside ``self.chat()``.
By the time the queued /steer was pulled, ``_agent_running`` had
already flipped back to False, so ``process_command()`` took the idle
fallback (``"No agent running; queued as next turn"``) and delivered
the steer as an ordinary next-turn message.

The fix dispatches /steer inline on the UI thread when the agent is
running — matching the existing pattern for /model — so the steer
reaches ``agent.steer()`` (thread-safe) without touching the queue.

These tests exercise the detector + inline dispatch without starting a
prompt_toolkit app.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


def _make_cli():
    """Create a HermesCLI instance with prompt_toolkit stubbed out."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI()


class TestSteerInlineDetector:
    """_should_handle_steer_command_inline gates the busy-path fast dispatch."""

    def test_detects_steer_when_agent_running(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_steer_command_inline("/steer focus on error handling") is True

    def test_ignores_steer_when_agent_idle(self):
        """Idle-path /steer should fall through to the normal process_loop
        dispatch so the queue-style fallback message is emitted."""
        cli = _make_cli()
        cli._agent_running = False
        assert cli._should_handle_steer_command_inline("/steer do something") is False

    def test_ignores_non_slash_input(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_steer_command_inline("steer without slash") is False
        assert cli._should_handle_steer_command_inline("") is False

    def test_ignores_other_slash_commands(self):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_steer_command_inline("/queue hello") is False
        assert cli._should_handle_steer_command_inline("/stop") is False
        assert cli._should_handle_steer_command_inline("/help") is False

    def test_ignores_steer_with_attached_images(self):
        """Image payloads take the normal path; steer doesn't accept images."""
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_steer_command_inline("/steer text", has_images=True) is False


class TestSteerBusyPathDispatch:
    """When the detector fires, process_command('/steer ...') must call
    agent.steer() directly rather than the idle-path fallback."""

    def test_process_command_routes_to_agent_steer(self):
        """With _agent_running=True and agent.steer present, /steer reaches
        agent.steer(payload), NOT _pending_input."""
        cli = _make_cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        cli.agent.steer = MagicMock(return_value=True)
        # Make sure the idle-path fallback would be observable if taken
        cli._pending_input = MagicMock()

        cli.process_command("/steer focus on errors")

        cli.agent.steer.assert_called_once_with("focus on errors")
        cli._pending_input.put.assert_not_called()

    def test_idle_path_queues_as_next_turn(self):
        """Control — when the agent is NOT running, /steer correctly falls
        back to next-turn queue semantics.  Demonstrates why the fix was
        needed: the queue path only works when you can actually drain it."""
        cli = _make_cli()
        cli._agent_running = False
        cli.agent = MagicMock()
        cli.agent.steer = MagicMock(return_value=True)
        cli._pending_input = MagicMock()

        cli.process_command("/steer would-be-next-turn")

        # Idle path does NOT call agent.steer
        cli.agent.steer.assert_not_called()
        # It puts the payload in the queue as a normal next-turn message
        cli._pending_input.put.assert_called_once_with("would-be-next-turn")


class _FakeBuffer:
    """Minimal stand-in for a prompt_toolkit buffer.

    Records reset() calls and the buffer text at the moment of each reset so a
    test can assert ordering relative to the steer dispatch.
    """

    def __init__(self, text: str = ""):
        self.text = text
        self.reset_calls: list[str] = []
        self.append_to_history_flags: list[bool] = []

    def reset(self, append_to_history: bool = False):
        # Capture what was in the buffer at reset time, then clear it.
        self.reset_calls.append(self.text)
        self.append_to_history_flags.append(append_to_history)
        self.text = ""


class TestInlineSteerBufferClear:
    """Regression tests for issue #34569.

    The submitted ``/steer <prompt>`` must never linger in the TUI input
    buffer, where the next Enter / turn-handoff could re-submit it.  The
    Enter handler routes the inline-steer case through
    ``_dispatch_inline_steer``, which must reset the buffer BEFORE running
    ``process_command`` so the clear is not skipped if the dispatch re-enters
    the event loop or raises.
    """

    def test_buffer_cleared_before_dispatch(self):
        cli = _make_cli()
        cli._agent_running = True
        cli.agent = MagicMock()
        cli.agent.steer = MagicMock(return_value=True)

        buf = _FakeBuffer("/steer focus on errors")
        observed = {}

        real_process = cli.process_command

        def _spy_process(cmd):
            # At dispatch time the buffer must already be empty.
            observed["text_at_dispatch"] = buf.text
            return real_process(cmd)

        with patch.object(cli, "process_command", side_effect=_spy_process):
            cli._dispatch_inline_steer(buf, "/steer focus on errors")

        # Buffer reset happened, captured the submitted text, and appended to history.
        assert buf.reset_calls == ["/steer focus on errors"]
        assert buf.append_to_history_flags == [True]
        # Reset happened strictly before process_command ran.
        assert observed["text_at_dispatch"] == ""
        # And the steer still reached the agent.
        cli.agent.steer.assert_called_once_with("focus on errors")
        # Final buffer state is empty — nothing left to accidentally re-submit.
        assert buf.text == ""

    def test_buffer_cleared_even_if_dispatch_raises(self):
        """If process_command raises, the buffer must STILL be clear.

        This is the core of #34569: a post-dispatch reset would be skipped on
        an exception, leaving the prompt in the box.  Reset-first guarantees
        the input-state contract holds regardless.
        """
        cli = _make_cli()
        cli._agent_running = True

        buf = _FakeBuffer("/steer do the thing")

        with patch.object(cli, "process_command", side_effect=RuntimeError("boom")):
            try:
                cli._dispatch_inline_steer(buf, "/steer do the thing")
            except RuntimeError:
                pass  # exception is allowed to propagate; the buffer must still be clear

        assert buf.reset_calls == ["/steer do the thing"]
        assert buf.text == ""


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__, "-v"])
