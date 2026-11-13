"""Earliest reset selection, durable cold starts and session affinity for reset_first."""
from dataclasses import replace
from datetime import UTC, datetime
import json
import time

import pytest

from agent import account_usage_cache as cache
from agent import credential_pool_usage as routing
from agent.account_usage import AccountUsageSnapshot
from agent.credential_pool import CredentialPool, PooledCredential
from hermes_cli.auth import write_credential_pool
from hermes_constants import get_hermes_home


def _window(used, remaining, period=604800):
    return {"used_percent": used, "reset_at": time.time() + remaining, "limit_window_seconds": period}


def _snapshot(*windows):
    return AccountUsageSnapshot(provider="openai-codex", source="usage_api", fetched_at=datetime.now(UTC),
                                raw={"rate_limit": dict(zip(("primary_window", "secondary_window"), windows))})


def _pool(monkeypatch, snapshots):
    (get_hermes_home() / "config.yaml").write_text("credential_pool_strategies:\n  openai-codex: reset_first\n")
    rows = [PooledCredential(provider="openai-codex", id=str(i), label=f"account{i}", source="manual",
                            auth_type="api_key", access_token=f"fixture-{i}", priority=i)
            for i in range(len(snapshots))]
    write_credential_pool("openai-codex", [e.to_dict() for e in rows])
    for entry, snapshot in zip(rows, snapshots):
        if snapshot is not None:
            routing.remember_usage(cache._identity_id_for("openai-codex", entry), snapshot)
    monkeypatch.setattr(cache, "refresh_account_usage_entries_async", lambda requests: [])
    return CredentialPool("openai-codex", rows)


def test_first_selection_uses_disk_readings_and_retains_session(monkeypatch):
    pool = _pool(monkeypatch, [_snapshot(_window(100, 3600)), _snapshot(_window(20, 3600))])
    cache._snapshots.clear()  # A new CLI has no in-memory readings.
    assert pool.select().id == "1"
    assert pool.peek().id == "1"
    # A changed usage reading never moves an already bound session.
    routing.remember_usage(cache._identity_id_for("openai-codex", pool.entries()[1]), _snapshot(_window(100, 3600)))
    assert pool.peek().id == "1"
    before = sum(e.request_count for e in pool.entries())
    assert pool.reclaim("1").id == "1"
    assert sum(e.request_count for e in pool.entries()) == before


def test_earliest_reset_wins_even_with_less_headroom(monkeypatch):
    pool = _pool(monkeypatch, [_snapshot(_window(10, 86400)), _snapshot(_window(99, 3600))])
    assert [pool.select().id for _ in range(8)] == ["1"] * 8


def test_earliest_window_counts_but_every_window_needs_headroom(monkeypatch):
    pool = _pool(monkeypatch, [
        _snapshot(_window(10, 9000, 18000), _window(20, 302400)),
        _snapshot(_window(99, 3600, 18000), _window(50, 302400)),
        _snapshot(_window(20, 1, 18000), _window(100, 302400)),
    ])
    assert [pool.select().id for _ in range(4)] == ["1"] * 4


def test_equal_resets_rotate_in_pool_order(monkeypatch):
    window = _window(20, 3600)
    pool = _pool(monkeypatch, [_snapshot(window), _snapshot(window), _snapshot(window)])
    assert [pool.select().id for _ in range(6)] == ["0", "1", "2", "0", "1", "2"]


def test_known_headroom_takes_precedence_over_unknown(monkeypatch):
    pool = _pool(monkeypatch, [None, _snapshot(_window(99, 3600))])
    assert [pool.select().id for _ in range(4)] == ["1"] * 4


def test_missing_usage_round_robins_without_spending_known_exhausted(monkeypatch):
    pool = _pool(monkeypatch, [_snapshot(_window(100, 3600)), None, None])
    assert [pool.select().id for _ in range(4)] == ["1", "2", "1", "2"]


@pytest.mark.parametrize("change", [
    {"fetched_at": time.time() - 901}, {"fetched_at": float("nan")},
    {"windows": [_window(20, -1)]}, {"windows": [_window(20, 3600, 0)]},
    {"windows": [_window(float("nan"), 3600)]}, {"windows": [_window(-1, 3600)]},
    {"windows": [None]}, {"windows": []}, {"windows": [_window(10**400, 3600)]},
])
def test_invalid_or_stale_reading_is_unknown(change):
    reading = {"fetched_at": time.time(), "windows": [_window(20, 3600)]}
    reading.update(change)
    assert routing._next_reset(reading, time.time()) is None


def test_cooldown_is_not_overridden_by_good_usage(monkeypatch):
    pool = _pool(monkeypatch, [_snapshot(_window(0, 100)), _snapshot(_window(50, 3600))])
    pool._entries[0] = replace(pool._entries[0], last_status="exhausted", last_status_at=time.time(),
                               last_error_code=429, last_error_reset_at=time.time() + 3600)
    monkeypatch.setattr(pool, "_codex_quota_restored_upstream", lambda entry: False)
    assert pool.select().id == "1"


def test_disk_cache_contains_only_numeric_fields(monkeypatch):
    pool = _pool(monkeypatch, [_snapshot(_window(20, 3600)), None])
    entry = pool.entries()[0]
    identity = cache._identity_id_for(pool.provider, entry)
    snapshot = replace(_snapshot(_window(20, 3600)), raw={
        "rate_limit": {"primary_window": {**_window(20, 3600), "secret": "never-save"}},
        "email": "private@example.com", "token": "never-save"})
    routing.remember_usage(identity, snapshot)
    payload = json.loads(routing._path(identity).read_text())
    assert set(payload) == {"fetched_at", "windows"}
    assert set(payload["windows"][0]) == {"used_percent", "reset_at", "limit_window_seconds"}


def test_cold_refresh_is_read_only_and_outside_pool_lock(monkeypatch):
    from agent import account_usage

    refresh = cache.refresh_account_usage_entries_async
    pool = _pool(monkeypatch, [None, None])
    monkeypatch.setattr(cache, "refresh_account_usage_entries_async", refresh)
    cache._last_try.clear()
    cache._inflight.clear()
    seen = []

    def fetch(provider, **kwargs):
        assert kwargs["read_only"] is True
        assert pool._lock.acquire(blocking=False)
        pool._lock.release()
        seen.append(kwargs["api_key"])
        return _snapshot(_window(100 if kwargs["api_key"] == "fixture-0" else 20, 3600))

    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch)
    assert pool.select().id == "1"
    assert seen == ["fixture-0", "fixture-1"]
    assert pool.select().id == "1"
    assert len(seen) == 2  # The persistent cache throttles later selections too.


def test_cold_wait_is_bounded(monkeypatch):
    class Worker:
        def join(self, timeout):
            assert 0 <= timeout <= routing._COLD_WAIT_SECONDS

    pool = _pool(monkeypatch, [None, None])
    monkeypatch.setattr(cache, "refresh_account_usage_entries_async", lambda requests: [Worker()])
    assert pool.select() is not None


def test_elapsed_short_window_does_not_hide_exhausted_active_week(monkeypatch):
    pool = _pool(monkeypatch, [
        _snapshot(_window(20, -1, 18000), _window(100, 302400)), None, None,
    ])
    assert [pool.select().id for _ in range(4)] == ["1", "2", "1", "2"]


def test_runtime_resolution_uses_persisted_usage_on_first_cli_binding(monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _pool(monkeypatch, [_snapshot(_window(100, 3600)), _snapshot(_window(20, 3600))])
    cache._snapshots.clear()
    runtime = resolve_runtime_provider(requested="openai-codex", target_model="gpt-5.5")
    assert runtime["api_key"] == "fixture-1"
    assert runtime["credential_pool"].peek().id == "1"
