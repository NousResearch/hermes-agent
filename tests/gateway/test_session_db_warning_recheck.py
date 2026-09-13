"""#108031: the startup state.db warning must not be broadcast once the store has healed."""
from __future__ import annotations

import asyncio
import threading

import pytest

import gateway.run as gateway_run
import hermes_state
import hermes_state_registry
from gateway.run import _SESSION_DB_UNPINNED
from gateway.session_db_recovery import RecoverableHandleCache


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _runner_with_startup_failure(monkeypatch, clock: _Clock, *, heals: bool):
    """A runner whose priming open failed with a lock; the store heals (or not) on the next open."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_init_error = "database is locked"
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()
    cache = RecoverableHandleCache(
        handles=runner._session_db_handles, lock=runner._session_db_handles_lock,
        clock=clock, initial_retry_delay=1,
    )
    runner._session_db_handle_cache = cache
    calls: list[str] = []

    def acquire(db_path=None):
        calls.append("open")
        if len(calls) == 1 or not heals:  # the priming open always fails: that is the startup lock
            raise RuntimeError("database is locked")
        return object()

    monkeypatch.setattr(hermes_state_registry, "acquire", acquire)
    monkeypatch.setattr(hermes_state, "AsyncSessionDB", lambda db: ("async", db))
    monkeypatch.setattr(runner, "session_store", None, raising=False)
    # Record the startup failure the way __init__ does (one failed open in the cache).
    with pytest.raises(RuntimeError):
        runner._open_session_db_for_active_scope(raise_on_error=True)
    clock.now = 5.0  # past the backoff: the pre-broadcast re-check is allowed to open
    sent: list[str] = []
    monkeypatch.setattr(runner, "_home_channel_transports", lambda: [("telegram", {}, "home", object())])

    async def _capture(_platform, _home, _transport, message, _fmt):
        sent.append(message)

    monkeypatch.setattr(runner, "_send_home_channel_message", _capture)
    return runner, sent


def test_lock_that_cleared_before_connect_is_not_broadcast(monkeypatch):
    clock = _Clock()
    runner, sent = _runner_with_startup_failure(monkeypatch, clock, heals=True)
    asyncio.run(runner._send_session_db_warning_notifications())
    assert sent == []
    assert runner._session_db_init_error is None


def test_still_unavailable_store_is_still_broadcast(monkeypatch):
    clock = _Clock()
    runner, sent = _runner_with_startup_failure(monkeypatch, clock, heals=False)
    asyncio.run(runner._send_session_db_warning_notifications())
    assert len(sent) == 1 and "Session database unavailable" in sent[0]
