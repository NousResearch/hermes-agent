"""Tests for the structured stream-drop log + clearer single-line status.

Regression coverage for the change that:

1. Removed ``logger.setLevel(ERROR)`` on tools/run_agent/etc. in quiet mode.
   It was clobbering the root logger's file handlers (agent.log/errors.log)
   because Python checks logger-level before handler propagation, so
   subagent stream-drop diagnostics were never written to disk.
2. Replaced the two ``⚠️ Connection dropped …`` + ``🔄 Reconnected …``
   ``_emit_status`` calls with a single ``_emit_stream_drop`` helper that:
   - Always writes a structured WARNING to ``agent.log``.
   - Always emits exactly ONE user-visible status line per drop (no
     follow-up "Reconnected" line) that names the provider and error
     class so multi-provider sessions can attribute it cleanly.
   - Subagent lines get the ``[subagent-N]`` ``log_prefix`` automatically
     via ``_emit_status`` → ``_vprint``.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

import run_agent
from run_agent import AIAgent


def _make_agent() -> AIAgent:
    return AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_quiet_mode_does_not_clobber_runagent_logger_level():
    """quiet_mode must not raise the ``run_agent`` logger above WARNING.

    Setting ``setLevel(ERROR)`` on the logger filters records *before* root
    logger handlers (agent.log/errors.log) ever see them.  Stream-drop
    diagnostics must reach the file handlers regardless of console verbosity.
    """
    _ = _make_agent()
    for name in ("run_agent", "tools", "trajectory_compressor", "cron", "hermes_cli"):
        logger = logging.getLogger(name)
        assert logger.getEffectiveLevel() <= logging.WARNING, (
            f"{name} logger blocked at level {logger.getEffectiveLevel()} — "
            f"file handlers will lose WARNING records"
        )


def test_log_stream_retry_writes_structured_warning(caplog):
    agent = _make_agent()
    agent._delegate_depth = 1
    agent._subagent_id = "sa-7-cafef00d"
    agent.provider = "openrouter"

    err = ConnectionError("Network connection lost.")
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent._log_stream_retry(
            kind="drop mid tool-call",
            error=err,
            attempt=2,
            max_attempts=3,
            mid_tool_call=True,
        )

    matching = [r for r in caplog.records if "Stream drop mid tool-call" in r.getMessage()]
    assert matching, f"no stream-drop WARNING captured; got {[r.getMessage() for r in caplog.records]}"
    msg = matching[0].getMessage()
    assert "subagent_id=sa-7-cafef00d" in msg
    assert "depth=1" in msg
    assert "provider=openrouter" in msg
    assert "base_url=https://openrouter.ai/api/v1" in msg
    assert "error_type=ConnectionError" in msg


@pytest.mark.parametrize("depth", [0, 1])
def test_emit_stream_drop_emits_status_line(depth):
    """Both top-level and subagent paths emit exactly one status line.

    Subagent lines get the ``[subagent-N]`` log_prefix via the parent's
    ``_vprint`` plumbing — this test only checks that ``_emit_status`` is
    called once with the right content.  No "Reconnected" follow-up.
    """
    agent = _make_agent()
    agent._delegate_depth = depth
    if depth > 0:
        agent._subagent_id = "sa-2-cafe"
    agent.provider = "openrouter"

    with patch.object(agent, "_emit_status") as mock_emit:
        agent._emit_stream_drop(
            error=ConnectionError("boom"),
            attempt=2,
            max_attempts=3,
            mid_tool_call=True,
        )

    assert mock_emit.call_count == 1, (
        f"expected exactly one _emit_status call (no Reconnected follow-up), "
        f"got {mock_emit.call_count}"
    )
    msg = mock_emit.call_args.args[0]
    assert "openrouter" in msg, f"provider name missing from status: {msg}"
    assert "stream drop" in msg
    assert "ConnectionError" in msg
    assert "retry 2/3" in msg
    # Single line — no separate "Reconnected" message.  But the line itself
    # should mention reconnecting so the user knows we're recovering.
    assert "reconnect" in msg.lower()


@pytest.mark.parametrize("mid_tool_call", [True, False])
def test_emit_stream_drop_always_writes_to_log(caplog, mid_tool_call):
    """Both subagent and top-level drops produce a WARNING in agent.log."""
    agent = _make_agent()
    agent._delegate_depth = 1 if mid_tool_call else 0
    agent.provider = "openrouter"
    if mid_tool_call:
        agent._subagent_id = "sa-99-feed"

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent._emit_stream_drop(
            error=TimeoutError("read timeout"),
            attempt=2,
            max_attempts=3,
            mid_tool_call=mid_tool_call,
        )

    found = [r for r in caplog.records if r.getMessage().startswith("Stream drop")]
    assert found, "expected at least one Stream drop WARNING record"
    msg = found[0].getMessage()
    assert "error_type=TimeoutError" in msg
    assert "provider=openrouter" in msg


def test_emit_stream_drop_provider_named_when_multi_provider():
    """The user-visible line must name the provider so multi-provider
    sessions can tell which subagent dropped (the original two-line message
    only said 'provider', forcing a log dive)."""
    agent = _make_agent()
    agent._delegate_depth = 1
    agent._subagent_id = "sa-1"
    agent.provider = "anthropic"

    with patch.object(agent, "_emit_status") as mock_emit:
        agent._emit_stream_drop(
            error=ConnectionError("x"),
            attempt=3,
            max_attempts=3,
            mid_tool_call=False,
        )

    msg = mock_emit.call_args.args[0]
    assert "anthropic" in msg
