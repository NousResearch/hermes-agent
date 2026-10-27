"""Durable pool administration and selection invariants."""
import time
from dataclasses import replace

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from hermes_cli.auth import read_credential_pool, write_credential_pool


def _pool(provider="openrouter", *, exhausted=False):
    rows = [PooledCredential(
        provider=provider, id=f"row{i}", label=f"account{i}", source="manual",
        auth_type="api_key", access_token=f"fixture-{i}", priority=i,
        last_status="exhausted" if exhausted else None,
        last_status_at=time.time() if exhausted else None,
        last_error_code=429 if exhausted else None,
        last_error_reset_at=time.time() + 3600 if exhausted else None,
    ) for i in range(2)]
    write_credential_pool(provider, [e.to_dict() for e in rows])
    return CredentialPool(provider, rows)


def test_target_reset_preserves_sibling_cooldown():
    pool = _pool(exhausted=True)
    before = read_credential_pool(pool.provider)
    assert pool.reset_status("missing") is None
    assert read_credential_pool(pool.provider) == before
    assert pool.reset_status("row1").last_status is None
    after = {e["id"]: e for e in read_credential_pool(pool.provider)}
    assert after["row0"] == before[0]
    assert after["row1"].get("last_error_reset_at") is None
    assert pool.reset_statuses() == 1
    assert all(e.get("last_status") is None for e in read_credential_pool(pool.provider))


@pytest.mark.parametrize("strategy", ["fill_first", "round_robin", "random", "least_used"])
def test_selection_counts_only_returned_selections(strategy):
    pool = _pool()
    pool._strategy = strategy
    selected = [pool.select().id for _ in range(2)]
    assert {e.id: e.request_count for e in pool.entries()} == {
        e.id: selected.count(e.id) for e in pool.entries()}
    pool.reset_status("row0")  # existing persistence boundary, not a selection
    assert sum(e.get("request_count", 0) for e in read_credential_pool(pool.provider)) == 2
    pool._current_id = None
    pool.try_refresh_matching()  # API key is not refreshable; lookup must not count
    assert sum(e.request_count for e in pool.entries()) == 2
    assert pool.peek() is not None
    assert sum(e.request_count for e in pool.entries()) == 2


def test_priority_persists_contiguous_order_without_clearing_cooldown():
    pool = _pool(exhausted=True)
    before = {e.id: e.last_error_reset_at for e in pool.entries()}
    assert pool.move_entry("row1", -5).priority == 0
    assert [e["id"] for e in read_credential_pool(pool.provider)] == ["row1", "row0"]
    assert pool.move_entry("row1", 99).priority == 1
    assert [(e.id, e.priority) for e in pool.entries()] == [("row0", 0), ("row1", 1)]
    assert {e.id: e.last_error_reset_at for e in pool.entries()} == before
    snapshot = read_credential_pool(pool.provider)
    assert pool.move_entry("missing", 0) is None
    assert read_credential_pool(pool.provider) == snapshot


def test_priority_honors_anthropic_manual_first():
    pool = _pool("anthropic")
    pool._entries[1] = replace(pool._entries[1], source="env:ANTHROPIC_API_KEY")
    assert pool.move_entry("row1", 0).priority == 1
    assert [e.id for e in pool.entries()] == ["row0", "row1"]
