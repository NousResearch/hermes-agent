"""Regression test for upstream issue #132048.

Interrupting a Relay-managed synchronous plain-generator stream while its
``next()`` is executing on a worker thread must not turn the interrupt into a
secondary ``ValueError: generator already executing`` raised from
``ManagedLlmStream.close()``.

Bounded real-signal comparison spanning stream creation and consumption
(upstream reporter's test, adapted verbatim).
"""

import signal
import threading
import time

import pytest

from agent import relay_llm
from tests.agent.test_relay_llm import relay_turn  # noqa: F401


def test_signal_during_blocked_provider(relay_turn):
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    workers, close_errors = [], []

    def provider(request):
        try:
            workers.append(threading.current_thread())
            entered.set()
            assert release.wait(5), "fixture release failed"
            yield {"choices": [{"index": 0, "delta": {"content": "done"}}]}
        finally:
            exited.set()

    def interrupt(signum, frame):
        assert entered.is_set(), "provider never started"
        raise KeyboardInterrupt("interrupt during provider read")

    previous = signal.signal(signal.SIGALRM, interrupt)
    timer = threading.Timer(1.5, release.set)
    timer.start()
    stream = None
    start = time.monotonic()
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        with pytest.raises(KeyboardInterrupt):
            stream = relay_llm.stream(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                provider,
                session_id="session-1",
                name="custom",
                model_name="test",
                metadata={"api_mode": "chat_completions"},
                finalizer=lambda: {},
            )
            next(stream)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        release.set()
        timer.join(3)
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:
                close_errors.append(repr(exc))

    assert time.monotonic() - start < 4
    assert not close_errors, close_errors
    assert exited.wait(3), "provider did not finish"
    for worker in workers:
        if worker is not threading.current_thread():
            worker.join(3)
    assert all(
        worker is threading.current_thread() or not worker.is_alive() for worker in workers
    )
    if stream is not None:
        assert stream._loop is None
        assert stream._runtime_lease is None


def test_close_provider_resources_waits_out_busy_generator():
    """Unmanaged sibling seam: closing resources while a worker is inside next().

    ``_close_provider_resources`` must wait out the "generator already executing"
    window and deliver GeneratorExit (running the provider's ``finally``) instead
    of stashing a ValueError and leaving cleanup to the garbage collector.
    """
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()

    def provider(request):
        try:
            entered.set()
            assert release.wait(5), "fixture release failed"
            yield {"chunk": 1}
        finally:
            exited.set()

    stream = relay_llm.ManagedLlmStream(
        {"model": "test", "messages": []},
        provider,
        session_id="unmanaged-close-race",
        name="custom",
        model_name="test",
        finalizer=lambda: {},
    )
    worker = threading.Thread(target=lambda: next(stream, None), daemon=True)
    worker.start()
    assert entered.wait(3), "provider never started"
    assert stream._loop is None  # unmanaged path: resources close via _close_provider_resources
    release_timer = threading.Timer(0.3, release.set)
    release_timer.start()
    try:
        start = time.monotonic()
        stream._close_provider_resources()
        elapsed = time.monotonic() - start
    finally:
        release.set()
        release_timer.join(3)
        worker.join(3)
    assert exited.wait(3), "provider finally did not run during bounded close"
    assert elapsed < 2.0
    assert not worker.is_alive()
    stream.close()  # idempotent second close must not resurrect the stashed error

