"""Interrupting a managed stream must not close a generator its worker is executing.

The off-loop provider read runs in a to_thread worker; cancelling the managed
stream does not stop a started read, so closing the raw generator while the
worker is still inside next() raises "generator already executing" (#132048).
"""
import signal
import threading
import time

import pytest

from agent import relay_llm
from tests.agent.test_relay_llm import relay_turn


def _blocking_provider(entered, release, exited=None):
    def provider(_request):
        try:
            entered.set()
            assert release.wait(5), "fixture release failed"
            yield {"choices": [{"index": 0, "delta": {"content": "done"}}]}
        finally:
            if exited is not None:
                exited.set()

    return provider


def _managed_stream(provider):
    return relay_llm.stream(
        {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        provider,
        session_id="session-1",
        name="custom",
        model_name="test",
        metadata={"api_mode": "chat_completions"},
        finalizer=lambda: {},
    )


def test_close_drains_blocked_provider_read_before_closing_generator(relay_turn):
    """close() while the off-loop worker is inside next() waits for the read to
    land, then closes the generator instead of raising ValueError on it."""
    entered, release, exited = (threading.Event() for _ in range(3))
    stream = _managed_stream(_blocking_provider(entered, release, exited))
    assert entered.wait(3), "provider never started"
    timer = threading.Timer(0.2, release.set)
    timer.start()
    try:
        stream.close()
    finally:
        release.set()
        timer.join(3)
    assert exited.wait(3), "generator never received close"
    assert stream._loop is None
    assert stream._runtime_lease is None


def test_close_abandons_raw_close_when_provider_read_never_lands(relay_turn, monkeypatch):
    """A provider read that stays blocked past the drain bound must not hang or
    raise: the raw close is abandoned and the lease is still released."""
    monkeypatch.setattr(relay_llm, "_PROVIDER_READ_DRAIN_TIMEOUT", 0.2)
    monkeypatch.setattr(relay_llm, "_ACLOSE_TIMEOUT", 1.0)
    entered, release = (threading.Event() for _ in range(2))
    stream = _managed_stream(_blocking_provider(entered, release))
    assert entered.wait(3), "provider never started"
    start = time.monotonic()
    try:
        stream.close()
    finally:
        release.set()
    assert time.monotonic() - start < 5
    assert stream._runtime_lease is None


def test_interrupted_consumption_does_not_surface_generator_close_error(relay_turn):
    """A real SIGALRM interrupt while next() is blocked in the worker must still
    clean up without surfacing "generator already executing" from close()."""
    entered, release, exited = (threading.Event() for _ in range(3))
    close_errors = []

    def interrupt(_signum, _frame):
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
            stream = _managed_stream(_blocking_provider(entered, release, exited))
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
    assert exited.wait(3), "generator never received close"
    if stream is not None:
        assert stream._loop is None
        assert stream._runtime_lease is None
