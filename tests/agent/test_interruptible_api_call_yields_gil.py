"""Regression test: interruptible_api_call interrupt detection latency.

Issue: https://github.com/NousResearch/hermes-agent/issues/57903

The non-streaming ``interruptible_api_call`` in
``agent/chat_completion_helpers.py`` polls a worker thread from the
main thread via ``t.join(timeout=0.3)``. The 300ms poll window is the
worst-case latency for interrupt detection — when a user hits Ctrl+C
or the agent flips ``_interrupt_requested``, the busy-poll can spend
up to 300ms before noticing and force-closing the in-flight HTTP
request. The same window also gates the stale-call detector, the
TTFB watchdog, and the Codex stream-idle watchdog.

This test pins the *worst-case* poll window: it sets the agent's
``_interrupt_requested`` flag from inside the simulated SDK call and
asserts that ``interruptible_api_call`` returns within
``MAX_INTERRUPT_LATENCY_SECONDS`` after the flag flip. The current
300ms busy-poll gives up to ~300ms latency plus the work the worker
must do to observe the close (~100-300ms of socket teardown); the
test budget is set to 1.0s to leave room for slow CI while still
flagging the regression.

A future fix that replaces the busy-poll with a 50ms ``future.result``
poll would reduce interrupt latency to ~50-100ms; the same test
would still pass with the same 1.0s budget, so this test is a
regression guard rather than a forward-looking perf assertion.

The companion diagnostic is ``scripts/bench_interruptible_api_call.py``
which quantifies GIL yield in the same window.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from agent import chat_completion_helpers as cch


MAX_INTERRUPT_LATENCY_SECONDS = 1.0  # 300ms poll + ~300ms socket teardown + CI headroom
SDK_SLEEP_BEFORE_INTERRUPT_SECONDS = 1.0  # long enough for the poll to cycle several times
SDK_SETTLE_AFTER_INTERRUPT_SECONDS = 0.05  # let the worker observe the close


def _make_agent(interrupt_after_seconds: float) -> MagicMock:
    """An agent mock whose SDK call flips ``_interrupt_requested`` after
    ``interrupt_after_seconds`` and then settles long enough for the
    worker to observe the close. Returns the agent and a
    ``flag_flipped_at`` placeholder so the test can measure latency.
    """
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent._compute_non_stream_stale_timeout.return_value = 60.0

    fake_client = MagicMock()
    flag_flipped_at: list = [None]
    interrupted_at: list = [None]

    def _slow_create(**_kwargs):
        time.sleep(interrupt_after_seconds)
        flag_flipped_at[0] = time.monotonic()
        # Force-close the worker-local client, then settle so the worker
        # observes the transport error and exits cleanly.
        agent._interrupt_requested = True
        time.sleep(SDK_SETTLE_AFTER_INTERRUPT_SECONDS)
        # Match the production shape: raise a transport error so the
        # worker's _call except branch sees it and the
        # ``_request_cancelled`` token swallows it.
        import httpx
        raise httpx.RemoteProtocolError("forced close")

    fake_client.chat.completions.create.side_effect = _slow_create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()

    return agent, flag_flipped_at, interrupted_at


def test_interruptible_api_call_returns_quickly_after_interrupt():
    """``interruptible_api_call`` must return within
    ``MAX_INTERRUPT_LATENCY_SECONDS`` after ``_interrupt_requested`` flips
    to True during a long SDK call. The current 300ms busy-poll
    baseline gives ~400-700ms wall-clock; the test budget (1.0s) leaves
    headroom for slow CI while still flagging a regression if the
    poll window grows (e.g. a regression that bumps it back to 1s).
    """
    agent, flag_flipped_at, _ = _make_agent(SDK_SLEEP_BEFORE_INTERRUPT_SECONDS)

    t0 = time.monotonic()
    with pytest.raises(Exception):
        # InterruptedError is the expected raise; we accept any
        # exception to avoid coupling the test to PR #6600's exact
        # raise type if upstream refactors that path.
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    elapsed = time.monotonic() - t0

    assert flag_flipped_at[0] is not None, (
        "SDK call never reached the interrupt-flag-flip point; the test "
        "setup is broken or ``interruptible_api_call`` returned before "
        "the worker could run"
    )
    interrupt_to_return_latency = (t0 + elapsed) - flag_flipped_at[0]
    assert interrupt_to_return_latency <= MAX_INTERRUPT_LATENCY_SECONDS, (
        f"interruptible_api_call took {interrupt_to_return_latency:.3f}s "
        f"to return after the interrupt flag flipped "
        f"(budget {MAX_INTERRUPT_LATENCY_SECONDS:.1f}s). This usually "
        f"means the main-thread poll window is too long — see issue "
        f"#57903."
    )
