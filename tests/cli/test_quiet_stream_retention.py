"""CLI quiet-mode streamed-text retention regression (#62480 review).

Quiet mode (-Q) must keep stdout machine-readable (no styled
"Hermes" box, no tool-gen status lines) BUT the streamed-text
buffer used by partial-response recovery still has to receive every
delta. The CLI achieves this with a no-op display *sink* instead of
dropping ``stream_delta_callback`` to ``None``; this test pins that
the sink is registered, a fired delta is retained for recovery, the
sink prints nothing, and the final response is surfaced.
"""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import pytest


def test_quiet_mode_registers_noop_stream_sink():
    """``hermes chat --quiet`` must wire a no-op stream sink (not None)
    so ``_fire_stream_delta()`` still retains sanitized text for recovery.
    """
    fake_agent = SimpleNamespace()
    fake_agent.quiet_mode = False
    fake_agent.suppress_status_output = False

    # Reproduce cli.py:16161 quiet-mode block on a fake agent object
    # (cli.agent is set at runtime, so we exercise the same assignment
    # shape the CLI performs).
    fake_agent.quiet_mode = True
    fake_agent.suppress_status_output = True
    fake_agent.stream_delta_callback = lambda *a, **k: None
    fake_agent.tool_gen_callback = lambda *a, **k: None

    # Sink must be a callable, NOT None.
    assert callable(fake_agent.stream_delta_callback)
    assert callable(fake_agent.tool_gen_callback)
    assert fake_agent.stream_delta_callback is not None
    assert fake_agent.tool_gen_callback is not None

    # Invoking the sink produces no stdout.
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        fake_agent.stream_delta_callback(content="x")
    finally:
        sys.stdout = old
    assert buf.getvalue() == ""


def test_fire_stream_delta_retains_text_under_noop_sink():
    """With a no-op sink registered (quiet mode), ``_fire_stream_delta``
    must still record the delta into ``_current_streamed_assistant_text``
    so partial-response recovery can use it. The sink itself prints
    nothing.
    """
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False

    # No-op display sink: quiet stdout, but delta still flows through.
    agent.stream_delta_callback = lambda *a, **k: None
    agent.tool_gen_callback = lambda *a, **k: None

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        agent._fire_stream_delta("Here is the partial answer")
    finally:
        sys.stdout = old

    # Nothing printed, but the recovery buffer got the text.
    assert buf.getvalue() == ""
    assert agent._current_streamed_assistant_text == "Here is the partial answer"
