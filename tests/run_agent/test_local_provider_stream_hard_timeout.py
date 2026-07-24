"""Regression test for the local-provider streaming deadlock on worker exit.

Bug: when the model endpoint is a local provider (LM Studio / oMLX / Ollama /
llama-cpp), ``_interruptible_streaming_api_call`` sets the stale-stream
timeout to ``float("inf")`` so that slow prefill on large contexts is never
killed. But that also removes the only mechanism that breaks a *half-dead*
connection — one where the server has gone quiet (no terminator, socket still
open). The ``_call`` thread blocks forever in ``httpcore ... read()``, the
outer ``while t.is_alive(): t.join(0.3)`` loop waits on it forever, and the
worker process never exits. With ``PARALLEL=1`` on a local GPU each stuck
worker holds an inference slot, producing ``+N queued`` on the server.

Confirmed in production via py-spy: a worker whose kanban task was already
``done`` sat in ``S`` (sleeping) for 20+ minutes with:
    Thread-2: join() at chat_completion_helpers.py:2487
    Thread-35: read() at httpcore/_backends/sync.py:128

Fix: an absolute hard ceiling (``HERMES_STREAM_HARD_TIMEOUT``, default 600s)
that force-closes the client even when the stale detector is disabled (inf),
breaking the deadlock so the worker can exit. Healthy prefill keeps delivering
chunks and refreshes the timer, so it is never killed.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest


class _DeadStream:
    """An SSE stream that accepts the connection but never yields a chunk —
    mimics a local server that goes quiet without closing the socket.

    A real socket ``read()`` would block forever; the hard-ceiling fix
    force-closes the underlying client, which severs that socket and makes
    ``read()`` raise. We emulate that here: ``__next__`` short-polls a
    ``closed`` event that the fix sets (via the patched force-close hook).
    Once set, the stream ends (severed connection → StopIteration)."""

    def __init__(self, closed_event: threading.Event):
        self._closed = closed_event

    def __iter__(self):
        return self

    def __next__(self):
        # Short poll so that once the force-close fires, the next iteration
        # promptly unwinds the _call thread (mirrors read() raising on a
        # severed socket). Hard cap keeps a buggy build from hanging the test.
        for _ in range(200):  # ~20s absolute cap
            if self._closed.wait(timeout=0.1):
                raise StopIteration
        raise StopIteration


def _make_local_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="http://localhost:1234/v1",   # local provider → stale timeout = inf
        model="qwen/qwen3-coder-next",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


@patch("run_agent.AIAgent._replace_primary_openai_client")
@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_local_provider_dead_stream_is_force_closed_by_hard_ceiling(
    mock_close, mock_create, mock_replace, monkeypatch
):
    """A local-provider stream that never delivers a chunk must be force-closed
    by the hard ceiling instead of hanging the worker forever."""
    # Small hard ceiling so the test is fast.
    monkeypatch.setenv("HERMES_STREAM_HARD_TIMEOUT", "1.0")

    closed_event = threading.Event()
    # The hard-ceiling fix force-closes via the request-client teardown and a
    # primary-client replacement. Either firing means the deadlock was broken;
    # both set the event so the dead stream unwinds (mirrors a severed socket).
    mock_close.side_effect = lambda *a, **k: closed_event.set()
    mock_replace.side_effect = lambda *a, **k: closed_event.set()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _DeadStream(closed_event)
    mock_create.return_value = mock_client

    agent = _make_local_agent()

    done = threading.Event()
    result_box = {}

    def _run():
        try:
            result_box["ret"] = agent._interruptible_streaming_api_call({})
        except BaseException as exc:  # the function may raise on the killed stream
            result_box["exc"] = exc
        finally:
            done.set()

    runner = threading.Thread(target=_run, daemon=True)
    runner.start()

    # The hard ceiling is 1.0s; allow generous slack. Without the fix this
    # NEVER completes (worker hangs forever) → the test times out / fails.
    finished = done.wait(timeout=8.0)

    assert finished, (
        "interruptible_streaming_api_call hung on a dead local-provider stream "
        "— the hard ceiling did not force-close the connection. This is the "
        "worker-never-exits deadlock."
    )
    # The fix breaks the deadlock by force-closing / replacing the client.
    assert closed_event.is_set(), (
        "expected the hard ceiling to force-close (or replace) the client"
    )
