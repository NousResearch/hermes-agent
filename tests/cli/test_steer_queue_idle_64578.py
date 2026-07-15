"""Regression tests for #64578 — /steer and /queue should be no-ops with
clear notice when agent is idle, not silent demotions.

Background: ``HermesCLI.process_command`` in ``cli.py`` handles ``/queue`` and
``/steer`` by always pushing the payload to ``self._pending_input``. When
the agent is idle (not thinking/responding), this silently demotes the
steer/queue semantics to "next-turn message" with no clear notice. The
user believes they nudged the (already-completed) run.

Fix: when ``self._agent_running`` is False, print a clear "Agent is idle"
notice and early-return. The user's command is not delivered as a
next-turn message — they have to send a normal message instead.

This test mocks the relevant HermesCLI state and asserts:
1. The notice is printed
2. ``_pending_input.put`` is NOT called
3. ``process_command`` returns without dispatching
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock

import pytest


def _make_handler_with_agent_running(running: bool):
    """Build a HermesCLI with ``_agent_running = running`` and mocked
    dependencies. Returns the handler instance.
    """
    from cli import HermesCLI

    handler = HermesCLI.__new__(HermesCLI)  # bypass __init__
    handler._agent_running = running
    handler.agent = MagicMock()
    handler.agent.steer = MagicMock(return_value=True)
    handler._pending_input = MagicMock()
    return handler


def _strip_ansi(s: str) -> str:
    """_cprint emits ANSI codes; strip them for substring matching."""
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def test_steer_when_idle_prints_notice():
    """When agent is idle, /steer should print 'Agent is idle' notice
    and NOT call _pending_input.put."""
    handler = _make_handler_with_agent_running(running=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        handler.process_command("/steer some steering text")
    output = _strip_ansi(buf.getvalue())
    assert "Agent is idle" in output or "idle" in output.lower(), (
        f"#64578 regression: /steer when idle should print 'Agent is idle' notice, "
        f"got: {output!r}"
    )


def test_steer_when_idle_does_not_call_pending_input():
    """The /steer handler must NOT push to _pending_input when idle.
    The bug was the silent demotion via _pending_input.put."""
    handler = _make_handler_with_agent_running(running=False)
    handler._pending_input.put.reset_mock()
    handler.process_command("/steer some steering text")
    handler._pending_input.put.assert_not_called(), (
        "#64578 regression: /steer when idle called _pending_input.put, "
        "silently demoting to a next-turn message."
    )


def test_queue_when_idle_prints_notice():
    """When agent is idle, /queue should print 'Agent is idle' notice."""
    handler = _make_handler_with_agent_running(running=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        handler.process_command("/queue some queue text")
    output = _strip_ansi(buf.getvalue())
    assert "Agent is idle" in output or "idle" in output.lower(), (
        f"#64578 regression: /queue when idle should print 'Agent is idle' notice, "
        f"got: {output!r}"
    )


def test_queue_when_idle_does_not_call_pending_input():
    """The /queue handler must NOT push to _pending_input when idle."""
    handler = _make_handler_with_agent_running(running=False)
    handler._pending_input.put.reset_mock()
    handler.process_command("/queue some queue text")
    handler._pending_input.put.assert_not_called(), (
        "#64578 regression: /queue when idle called _pending_input.put, "
        "silently demoting to a next-turn message."
    )


def test_steer_when_running_calls_agent_steer():
    """Sanity: when the agent is running, /steer should call agent.steer()
    (the proper mid-run inject path), NOT print an idle notice."""
    handler = _make_handler_with_agent_running(running=True)
    handler._pending_input.put.reset_mock()
    handler.agent.steer.reset_mock()
    buf = io.StringIO()
    with redirect_stdout(buf):
        handler.process_command("/steer some steering text")
    # When running, the agent.steer() path should fire
    assert handler.agent.steer.called, (
        "When agent is running, /steer should call agent.steer(). "
        f"agent.steer.call_count={handler.agent.steer.call_count}, "
        f"stdout: {_strip_ansi(buf.getvalue())!r}"
    )