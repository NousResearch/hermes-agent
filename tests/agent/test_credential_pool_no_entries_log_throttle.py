"""Regression: the credential-pool "no available entries" INFO line must be
throttled so an empty/exhausted pool cannot storm the shared rotating log.

Selection runs on a hot path (every model call plus auxiliary tasks). Before
the throttle, an empty/exhausted pool logged this line on *every* select(),
which on Windows storms concurrent-log-handler's cross-process lock
("Cannot acquire lock after 20 attempts"), stalls the asyncio event loop, and
fails the Desktop backend readiness handshake ("Timed out connecting to Hermes
backend after 15000ms"). See #58265 for the same fix class on another message.
"""

from __future__ import annotations

import logging

from agent.credential_pool import (
    NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS,
    CredentialPool,
    PooledCredential,
)

_NO_ENTRIES_MSG = "credential pool: no available entries (all exhausted or empty)"


class _FakeClock:
    """Deterministic monotonic clock driven by the test."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _no_entries_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == _NO_ENTRIES_MSG]


def _make_entry(entry_id: str) -> PooledCredential:
    return PooledCredential(
        provider="test",
        id=entry_id,
        label=entry_id,
        auth_type="api_key",
        source="manual",
        access_token=f"tok-{entry_id}",
        priority=0,
    )


def test_empty_pool_logs_once_within_throttle_window(monkeypatch, caplog):
    clock = _FakeClock()
    monkeypatch.setattr("agent.credential_pool.time.monotonic", clock)

    pool = CredentialPool("test", [])

    with caplog.at_level(logging.INFO, logger="agent.credential_pool"):
        for _ in range(50):
            clock.now += 0.1  # tighter than the throttle window
            assert pool.select() is None

    # 50 selections, well inside one window -> exactly one log line.
    assert len(_no_entries_records(caplog)) == 1


def test_logs_again_after_throttle_window_elapses(monkeypatch, caplog):
    clock = _FakeClock()
    monkeypatch.setattr("agent.credential_pool.time.monotonic", clock)

    pool = CredentialPool("test", [])

    with caplog.at_level(logging.INFO, logger="agent.credential_pool"):
        assert pool.select() is None  # log #1
        clock.now += NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS + 1
        assert pool.select() is None  # window elapsed -> log #2

    assert len(_no_entries_records(caplog)) == 2


def test_successful_selection_rearms_throttle(monkeypatch, caplog):
    """A recover -> re-exhaust transition must log immediately, even inside the
    window opened by the previous empty stretch (observability of the flip)."""
    clock = _FakeClock()
    monkeypatch.setattr("agent.credential_pool.time.monotonic", clock)

    pool = CredentialPool("test", [])

    with caplog.at_level(logging.INFO, logger="agent.credential_pool"):
        assert pool.select() is None  # log #1, throttle armed
        clock.now += 1
        assert pool.select() is None  # within window -> no log

        # Pool recovers: a successful selection re-arms the throttle.
        pool._entries = [_make_entry("a")]
        clock.now += 1
        assert pool.select() is not None

        # Pool empties again shortly after (< throttle window since log #1).
        pool._entries = []
        clock.now += 1
        assert pool.select() is None  # re-armed -> log #2 immediately

    assert len(_no_entries_records(caplog)) == 2
