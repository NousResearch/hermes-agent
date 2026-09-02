"""Regression test — and exhaustive stress test — for the fix that makes
CodexAppServerClient.close() fail every in-flight request() call
immediately instead of leaving it to ride out its own per-call timeout.

Before the fix, close() never touched self._pending. A thread blocked in
request() waiting on a JSON-RPC reply had no way to know the transport it
depended on had just been torn down — it kept waiting on its own
`timeout` argument (up to 30s by default) even though the subprocess was
already dead. This is the concrete mechanism behind the observed "codex
route hangs after session expiry" race: gateway/run.py's
`_session_expiry_watcher` can call `AIAgent.close()` ->
`codex_session.close()` while a turn is mid-flight (e.g. blocked in
`turn/start`), and the blocked caller thread would just sit there.

Fix: close() now pops everything out of self._pending and delivers a
synthetic JSON-RPC error to each queue, under the same lock discipline
_read_stdout uses to dispatch real replies — so a real reply landing at
the same instant still wins cleanly instead of being dropped or
corrupting the queue.

Everything here builds a bare CodexAppServerClient via object.__new__
with a mocked subprocess (no real process, no real pipes), so it stays
cheap in both time and memory even across the exhaustive/soak variant.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import Mock

import pytest

from agent.transports.codex_app_server import CodexAppServerClient, _Pending


def _bare_client() -> CodexAppServerClient:
    """A CodexAppServerClient with real _pending/_pending_lock/_send
    plumbing but a mocked, already-inert subprocess — no real OS process
    involved, so this is pure in-memory queue logic."""
    client = object.__new__(CodexAppServerClient)
    client._closed = False
    client._pending = {}
    client._pending_lock = threading.Lock()
    client._next_id = 1
    proc = Mock()
    proc.stdin = None
    proc.terminate = Mock()
    proc.kill = Mock()
    proc.wait = Mock(return_value=0)
    client._proc = proc
    # request() calls self._send(); stub it so it's a pure no-op instead
    # of trying to write to a real (nonexistent) pipe.
    client._send = Mock()
    return client


class TestClosePendingRequestsCancelledImmediately:
    """Proves the fix: a thread blocked in request() gets unblocked the
    instant close() runs, instead of riding out its own timeout."""

    def test_blocked_request_fails_fast_on_close(self):
        client = _bare_client()

        outcome: dict[str, object] = {}

        def _blocked_request():
            start = time.monotonic()
            try:
                outcome["result"] = client.request(
                    "turn/start", {"threadId": "t1"}, timeout=10.0
                )
            except Exception as exc:  # noqa: BLE001 - capturing for assertion
                outcome["error"] = exc
            outcome["elapsed"] = time.monotonic() - start

        t = threading.Thread(target=_blocked_request, daemon=True)
        t.start()

        # Let the request actually register in _pending before closing.
        deadline = time.monotonic() + 2.0
        while not client._pending and time.monotonic() < deadline:
            time.sleep(0.001)
        assert client._pending, "request() never registered in _pending"

        close_started = time.monotonic()
        client.close()
        close_elapsed = time.monotonic() - close_started

        t.join(timeout=5)

        assert close_elapsed < 1.0, "close() itself should return quickly"
        assert not t.is_alive(), "blocked request() thread never returned"
        assert "error" in outcome, "request() should fail once the client closes"
        # The fix: it fails immediately, not after riding out the full
        # 10s timeout it was given.
        assert outcome["elapsed"] < 1.0, (
            f"request() took {outcome['elapsed']:.2f}s to fail — it should "
            f"have been cancelled the instant close() ran"
        )
        assert client._pending == {}, "close() must drain _pending"

    def test_real_reply_racing_close_does_not_raise(self):
        """If the reader thread delivers a real reply into the pending
        queue at (almost) the same instant close() tries to cancel it,
        close() must not raise — it just accepts that the real reply won
        the race."""
        client = _bare_client()
        q: queue.Queue = queue.Queue(maxsize=1)
        client._pending[7] = _Pending(queue=q, method="turn/start")
        q.put_nowait({"id": 7, "result": {"turn": {"id": "turn-1"}}})  # reader won

        client.close()  # must not raise queue.Full

        assert client._pending == {}
        # The real reply is still sitting there for whoever called request().
        assert q.get_nowait() == {"id": 7, "result": {"turn": {"id": "turn-1"}}}

    def test_close_with_no_pending_requests_is_a_noop(self):
        client = _bare_client()
        client.close()  # must not raise on an empty _pending
        assert client._pending == {}


class TestClosePendingRequestsExhaustive:
    """Exhaustion / soak test: repeats the blocked-request-vs-close race
    hundreds of times with jittered timing to make sure the fix holds up
    under repeated pressure, not just one lucky pass. Sequential — never
    more than 2 live threads at once — so memory stays flat regardless of
    iteration count."""

    ITERATIONS = 300

    def test_no_slow_unblock_across_many_races(self):
        import random

        rng = random.Random(4242)
        failures: list[tuple[int, str]] = []

        for i in range(self.ITERATIONS):
            client = _bare_client()
            outcome: dict[str, object] = {}

            def _blocked_request(client=client, outcome=outcome):
                start = time.monotonic()
                try:
                    outcome["result"] = client.request(
                        "turn/start", {"threadId": "t1"}, timeout=10.0
                    )
                except Exception as exc:  # noqa: BLE001
                    outcome["error"] = exc
                outcome["elapsed"] = time.monotonic() - start

            t = threading.Thread(target=_blocked_request, daemon=True)
            t.start()

            # Jittered pre-close delay so close() lands at varying points
            # relative to request() registering itself in _pending.
            time.sleep(rng.uniform(0.0, 0.01))

            client.close()
            t.join(timeout=5)

            if t.is_alive():
                failures.append((i, "request thread never returned"))
            elif "error" not in outcome:
                failures.append((i, "request() did not fail on close()"))
            elif outcome.get("elapsed", 99) >= 1.0:
                failures.append(
                    (i, f"slow unblock: {outcome['elapsed']:.2f}s")
                )

        assert not failures, (
            f"{len(failures)}/{self.ITERATIONS} iterations hit the race "
            f"(showing up to 5): {failures[:5]}"
        )
