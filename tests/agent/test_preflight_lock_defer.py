"""Preflight lock-defer must not arm the insufficient-progress blocker.

Companion to ``tests/run_agent/test_compression_lock_defer.py`` — pins the
``build_turn_context`` preflight loop's handling of a lock-contended
compression no-op (#69870 lock-skip signal, consumer salvaged from #49874):

* the pass stops the preflight loop (the lock winner is already shrinking
  the session) WITHOUT setting ``preflight_compression_blocked``, so the
  loop's provider-proven error handlers keep their full retry budget;
* a genuine no-progress no-op (flag unset) still arms the blocker exactly
  as before;
* a MagicMock-style truthy junk flag value does NOT take the defer branch
  (type-pin rule).
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
    """Over-threshold compressor stub that opens the preflight threshold path."""
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


def _make_agent():
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent.context_compressor = _pressured_compressor()
    agent._emit_status = MagicMock()
    return agent


_HISTORY = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "older"}]


def test_preflight_lock_skip_does_not_set_blocked_flag():
    agent = _make_agent()
    calls = []

    def _lock_skip_compress(messages, _system_message, **_kwargs):
        calls.append(1)
        agent._compression_skipped_due_to_lock = "pid=1:tid=2:agent=aa:nonce=bb"
        return messages, "SYSTEM"

    agent._compress_context = _lock_skip_compress

    ctx = _build(agent, conversation_history=list(_HISTORY))

    # Exactly one pass: the defer stops the loop without arming the blocker.
    assert calls == [1]
    assert ctx.preflight_compression_blocked is False


def test_preflight_lock_skip_true_unconfirmed_holder_also_defers():
    agent = _make_agent()
    calls = []

    def _lock_skip_compress(messages, _system_message, **_kwargs):
        calls.append(1)
        agent._compression_skipped_due_to_lock = True
        return messages, "SYSTEM"

    agent._compress_context = _lock_skip_compress

    ctx = _build(agent, conversation_history=list(_HISTORY))

    assert calls == [1]
    assert ctx.preflight_compression_blocked is False


def test_preflight_plain_noop_still_arms_blocker():
    """Control: flag unset → unchanged pre-fix behavior (blocker armed)."""
    agent = _make_agent()

    def _noop_compress(messages, _system_message, **_kwargs):
        agent._compression_skipped_due_to_lock = None
        return messages, "SYSTEM"

    agent._compress_context = _noop_compress

    ctx = _build(agent, conversation_history=list(_HISTORY))

    assert ctx.preflight_compression_blocked is True


def test_preflight_magicmock_flag_value_is_not_a_defer():
    """Type-pin: truthy junk (MagicMock auto-attribute shape) must not be
    treated as lock contention — the blocker arms as for a plain no-op."""
    agent = _make_agent()

    def _junk_flag_compress(messages, _system_message, **_kwargs):
        agent._compression_skipped_due_to_lock = MagicMock()
        return messages, "SYSTEM"

    agent._compress_context = _junk_flag_compress

    ctx = _build(agent, conversation_history=list(_HISTORY))

    assert ctx.preflight_compression_blocked is True
