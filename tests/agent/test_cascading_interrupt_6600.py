"""Regression guard for the cascading-interrupt hang (PR #6600).

Original diagnosis and fix by Kristian Vastveit (@kristianvast) in PR #6600,
against the then-inline ``_interruptible_api_call`` /
``_interruptible_streaming_api_call`` methods in run_agent.py. Those methods
have since been extracted into ``agent/chat_completion_helpers.py``, so the
fix is reapplied there and these tests target the extracted functions.

The bug: when ``agent.interrupt()`` fires during an active LLM call, the main
poll loop force-closes the worker-local httpx client to stop token generation.
That raises a transport error (RemoteProtocolError) on the worker — the
EXPECTED consequence of our own close, not a network bug. The streaming retry
loop misclassified it as a transient connection error and retried, each doomed
retry stalling for the full stream-stale timeout (up to 300s). Because the
gateway caches AIAgent instances per session, the stale worker outlived the
turn and raced the next turn's request — the root of the multi-minute
cascading-interrupt hang.

The fix: a request-local ``_request_cancelled`` token set by the poll loop
right before the force-close. The worker's exception handler checks it and
exits cleanly (no retry, no fallback, no "reconnecting" status) instead of
treating the forced error as transient.
"""
import threading
import time
import types
from unittest.mock import MagicMock

import httpx
import pytest

from agent import chat_completion_helpers as cch


class _FakeInterruptError(Exception):
    """Stand-in for the transport error a force-close raises on the worker."""


def _make_agent():
    """A MagicMock agent wired with just enough surface for the helpers."""
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent.verbose_logging = False
    # _compute_non_stream_stale_timeout / streaming setup helpers return
    # benign values; the real call path is mocked per-test.
    agent._compute_non_stream_stale_timeout.return_value = 5.0
    return agent


def test_non_streaming_cancel_does_not_surface_network_error():
    """A force-close during a non-streaming call must raise InterruptedError,
    not the swallowed transport error."""
    agent = _make_agent()

    create_calls = {"n": 0}
    fake_client = MagicMock()

    def _create(**kwargs):
        create_calls["n"] += 1
        # Simulate the main thread firing an interrupt mid-call, then the
        # force-close raising a transport error on this worker.
        agent._interrupt_requested = True
        time.sleep(0.3)  # let the poll loop observe the interrupt + force-close
        raise httpx.RemoteProtocolError("peer closed connection")

    fake_client.chat.completions.create.side_effect = _create
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()

    t0 = time.time()
    with pytest.raises(InterruptedError):
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    elapsed = time.time() - t0

    # The forced RemoteProtocolError must NOT surface as the raised error.
    assert create_calls["n"] == 1
    assert elapsed < 10.0, f"interrupt took {elapsed:.1f}s — should be near-instant (guarding the 30s+ hang)"


def test_normal_transient_error_still_raises_when_not_cancelled():
    """Regression guard: a real transport error with NO interrupt must still
    surface to the caller (so the outer retry loop can recover)."""
    agent = _make_agent()
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = httpx.RemoteProtocolError(
        "genuine network drop"
    )
    agent._create_request_openai_client.return_value = fake_client
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()
    agent._interrupt_requested = False

    with pytest.raises(httpx.RemoteProtocolError):
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})


def test_request_cancelled_token_is_request_local():
    """The cancellation token must be created per call, not shared on the
    agent — a stale worker from a previous turn must not see the next turn's
    interrupt flag flip back to False and mistake its own forced error for a
    network bug. We assert the helper reads agent._interrupt_requested at the
    force-close site (request-local token set there), by confirming two
    independent calls don't share cancellation state."""
    agent = _make_agent()

    # First call: interrupted.
    fake_client_1 = MagicMock()

    def _create_1(**kwargs):
        agent._interrupt_requested = True
        time.sleep(0.3)
        raise httpx.RemoteProtocolError("forced close turn A")

    fake_client_1.chat.completions.create.side_effect = _create_1
    agent._create_request_openai_client.return_value = fake_client_1
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()

    with pytest.raises(InterruptedError):
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})

    # Second call: NOT interrupted (turn boundary cleared the flag). A genuine
    # error must still surface — the previous call's cancellation must not leak.
    agent._interrupt_requested = False
    fake_client_2 = MagicMock()
    fake_client_2.chat.completions.create.side_effect = httpx.RemoteProtocolError(
        "genuine drop turn B"
    )
    agent._create_request_openai_client.return_value = fake_client_2

    with pytest.raises(httpx.RemoteProtocolError):
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})


# ---------------------------------------------------------------------------
# #67142: direct-Anthropic stale/interrupt watchdog must abort the request-local
# client from the poll (stranger) thread and NEVER close/rebuild the shared
# _anthropic_client — closing it there released a live TLS FD that the kernel
# recycled into a SQLite handle, writing a TLS record over a DB header.
# ---------------------------------------------------------------------------


def _make_anthropic_agent():
    agent = _make_agent()
    agent.api_mode = "anthropic_messages"
    return agent


def _wait_for_mock_call(mock, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mock.called:
            return
        time.sleep(0.02)
    raise AssertionError(f"{mock!r} was not called within {timeout}s")


def test_anthropic_non_streaming_stale_aborts_request_client_not_shared():
    """Stale non-streaming Anthropic call: the poll thread aborts the
    request-local client's socket; the shared client is never closed/rebuilt,
    and the worker still unblocks and closes its own client (no #28161 hang)."""
    agent = _make_anthropic_agent()
    agent._compute_non_stream_stale_timeout.return_value = 0.05
    agent._codex_silent_hang_hint = MagicMock(return_value=None)

    request_client = MagicMock()
    agent._create_request_anthropic_client = MagicMock(return_value=request_client)
    agent._abort_request_anthropic_client = MagicMock()
    agent._close_request_anthropic_client = MagicMock()

    def _create(_api_kwargs, *, client):
        assert client is request_client
        # Outlive the 0.05s stale timeout AND the worker join (2.0s) so the
        # stale detector surfaces its TimeoutError.
        time.sleep(2.5)
        return object()

    agent._anthropic_messages_create = MagicMock(side_effect=_create)

    with pytest.raises(TimeoutError):
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})

    # Shared client untouched from the poll thread.
    agent._anthropic_client.close.assert_not_called()
    agent._rebuild_anthropic_client.assert_not_called()
    # Poll (stranger) thread aborts the request-local client's socket only.
    agent._abort_request_anthropic_client.assert_called_once_with(
        request_client, reason="stale_call_kill"
    )
    # Worker unblocks and closes its own request client from its own thread.
    _wait_for_mock_call(agent._close_request_anthropic_client)


def test_anthropic_non_streaming_interrupt_aborts_request_client_not_shared():
    """Interrupted non-streaming Anthropic call: near-instant InterruptedError,
    request-local client aborted from the poll thread, shared client untouched."""
    agent = _make_anthropic_agent()

    request_client = MagicMock()
    agent._create_request_anthropic_client = MagicMock(return_value=request_client)
    agent._abort_request_anthropic_client = MagicMock()
    agent._close_request_anthropic_client = MagicMock()

    def _create(_api_kwargs, *, client):
        assert client is request_client
        agent._interrupt_requested = True
        time.sleep(1.0)
        raise httpx.RemoteProtocolError("forced close would have happened")

    agent._anthropic_messages_create = MagicMock(side_effect=_create)

    t0 = time.time()
    with pytest.raises(InterruptedError):
        cch.interruptible_api_call(agent, {"model": "x", "messages": []})
    elapsed = time.time() - t0

    assert elapsed < 3.0, f"interrupt took {elapsed:.1f}s — should be near-instant"
    agent._anthropic_client.close.assert_not_called()
    agent._rebuild_anthropic_client.assert_not_called()
    agent._abort_request_anthropic_client.assert_called_once_with(
        request_client, reason="interrupt_abort"
    )
