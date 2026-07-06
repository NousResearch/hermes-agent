"""Regression tests for the CLI interrupt-acknowledgement race.

Symptom (user report, July 2026): interrupting an active turn is
unreliable — the interrupt message is sometimes "vacuumed into the void".

Root cause: ``HermesCLI.chat()`` fires ``agent.interrupt(msg)`` from its
monitor loop, but only re-queued the message when the turn RESULT carried
``interrupted=True``. Two races defeat that:

  1. The agent thread passes its last ``_interrupt_requested`` check (or
     finishes entirely) just before the interrupt lands — the turn
     completes "normally", ``finalize_turn()`` never acknowledges the
     interrupt, and the user's message was silently dropped.
  2. Worse, when the interrupt lands *after* ``finalize_turn()``'s
     ``clear_interrupt()``, the stale ``_interrupt_requested`` flag
     survives on the agent and instantly aborts the NEXT turn at its
     first loop check.

The fix: when ``chat()`` consumed an ``interrupt_msg`` but the result
doesn't acknowledge the interrupt, re-queue the message as the next turn
and clear the stale agent flag (only when the agent thread has exited).
"""

from __future__ import annotations

import importlib
import queue
import sys
import time
from unittest.mock import MagicMock, patch


def _make_cli():
    """Build a HermesCLI with prompt_toolkit stubbed (same pattern as
    test_cli_interrupt_drain_regression.py)."""
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


class _StubAgent:
    """Agent whose turn completes WITHOUT acknowledging the interrupt."""

    def __init__(self, session_id, turn_seconds=0.5):
        self.session_id = session_id
        self.turn_seconds = turn_seconds
        self._interrupt_requested = False
        self._interrupt_message = None
        self._active_children = []
        self.interrupt_calls = []
        self.clear_calls = 0
        self.max_iterations = 90
        self.model = "test/model"
        self.platform = "cli"

    def run_conversation(self, **kwargs):
        # Simulate a turn that finishes normally — it never observed the
        # interrupt flag (raced past its last check).
        time.sleep(self.turn_seconds)
        return {
            "final_response": "turn finished normally",
            "messages": [
                {"role": "user", "content": "original"},
                {"role": "assistant", "content": "turn finished normally"},
            ],
            "api_calls": 1,
            "completed": True,
            # NOTE: no "interrupted" key — the race means finalize_turn
            # never saw the flag (or cleared it before it was re-set).
            "partial": True,  # skip auto-title thread in the test
            # Skip the Rich Panel rendering path (crashes under the
            # prompt_toolkit/skin mocks; irrelevant to this regression).
            "response_previewed": True,
        }

    def interrupt(self, message=None):
        self.interrupt_calls.append(message)
        self._interrupt_requested = True
        self._interrupt_message = message

    def clear_interrupt(self):
        self.clear_calls += 1
        self._interrupt_requested = False
        self._interrupt_message = None


def test_unacknowledged_interrupt_message_is_requeued_not_dropped():
    cli = _make_cli()
    agent = _StubAgent(cli.session_id)
    cli.agent = agent

    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._interrupt_queue.put("urgent new message")

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat("original")

    # The interrupt fired against the agent...
    assert agent.interrupt_calls == ["urgent new message"]
    # ...the turn result never acknowledged it, so the message must be
    # re-queued as the next turn instead of dropped.
    queued = []
    while not cli._pending_input.empty():
        queued.append(cli._pending_input.get_nowait())
    assert any("urgent new message" in str(q) for q in queued), (
        f"interrupt message was dropped; pending_input={queued!r}"
    )
    # ...and the stale flag must be cleared so the NEXT turn doesn't
    # instantly self-abort at its first _interrupt_requested check.
    assert agent._interrupt_requested is False
    assert agent.clear_calls >= 1


def test_acknowledged_interrupt_still_requeues_message():
    """The pre-existing path (result carries interrupted=True) still works."""
    cli = _make_cli()

    class _AckAgent(_StubAgent):
        def run_conversation(self, **kwargs):
            # Wait until the monitor loop delivers the interrupt.
            for _ in range(100):
                if self._interrupt_requested:
                    break
                time.sleep(0.05)
            return {
                "final_response": "partial work",
                "messages": [{"role": "assistant", "content": "partial work"}],
                "api_calls": 1,
                "completed": False,
                "interrupted": True,
                "interrupt_message": self._interrupt_message,
                "partial": True,
            }

    agent = _AckAgent(cli.session_id)
    cli.agent = agent
    cli._interrupt_queue = queue.Queue()
    cli._pending_input = queue.Queue()
    cli._interrupt_queue.put("redirect please")

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": cli._active_agent_route_signature,
             "model": None, "runtime": None, "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        cli.chat("original")

    queued = []
    while not cli._pending_input.empty():
        queued.append(cli._pending_input.get_nowait())
    assert any("redirect please" in str(q) for q in queued)
    assert cli._last_turn_interrupted is True
