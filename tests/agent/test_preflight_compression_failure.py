"""A failed preflight compaction must not take the turn down with it.

``_compress_context`` re-raises every failure after releasing the session
lock — a summary-model "Request timed out" on a large transcript included.
Both preflight call sites in ``build_turn_context`` (the over-threshold pass
loop and the engine-driven sub-threshold maintenance pass) called it bare, so
that exception escaped the turn setup entirely. A session big enough to fail
compaction then took the process down on every attempt to serve it.

Preflight compaction is an optimization, not a precondition: a failure has to
degrade to "no compaction this turn" and let the loop's own error handling,
with its retry budget, deal with the oversized request.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from tests.agent.test_turn_context import _FakeAgent, _build


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _pressured_compressor():
    """Over-threshold stub that opens the preflight threshold path."""
    return types.SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        threshold_tokens=1,
        context_length=100_000,
        last_prompt_tokens=0,
        should_compress=lambda _tokens=None: True,
        should_compress_info=lambda _tokens=None: (True, None),
        should_defer_preflight_to_real_usage=lambda _t: False,
        get_active_compression_failure_cooldown=lambda: None,
    )


def _make_agent(compressor):
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent.context_compressor = compressor
    agent._emit_status = MagicMock()
    return agent


_HISTORY = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "older"}]


def test_threshold_pass_failure_does_not_escape_the_turn():
    """The reported crash: the summary model times out mid-compaction."""
    agent = _make_agent(_pressured_compressor())
    calls = []

    def _timing_out_compress(_messages, _system_message, **_kwargs):
        calls.append(1)
        raise TimeoutError("Request timed out.")

    agent._compress_context = _timing_out_compress

    ctx = _build(agent, conversation_history=list(_HISTORY))

    # The turn was built rather than destroyed, the failure armed the blocker,
    # and the loop stopped instead of retrying a compaction that just failed.
    assert ctx.preflight_compression_blocked is True
    assert calls == [1]


def test_threshold_pass_failure_keeps_the_transcript():
    """Degrading must not cost the turn its messages."""
    agent = _make_agent(_pressured_compressor())

    def _failing_compress(_messages, _system_message, **_kwargs):
        raise RuntimeError("aux provider exploded")

    agent._compress_context = _failing_compress

    ctx = _build(agent, conversation_history=list(_HISTORY))

    roles = [m.get("role") for m in ctx.messages]
    assert "user" in roles


def test_interrupts_still_propagate():
    """Only Exception is absorbed — cancellation must keep unwinding."""
    agent = _make_agent(_pressured_compressor())

    def _interrupted_compress(_messages, _system_message, **_kwargs):
        raise KeyboardInterrupt()

    agent._compress_context = _interrupted_compress

    with pytest.raises(KeyboardInterrupt):
        _build(agent, conversation_history=list(_HISTORY))
