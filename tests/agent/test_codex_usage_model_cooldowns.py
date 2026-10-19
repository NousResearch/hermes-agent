"""Account quota fallback is not an inference request for a benched model."""

from dataclasses import replace
import time

import pytest

from agent import account_usage, credential_pool
from agent.credential_pool import CredentialPool, PooledCredential, STATUS_DEAD, STATUS_EXHAUSTED


MODEL = "unavailable-model"
BASE_URL = "https://chatgpt.com/backend-api/codex"


@pytest.fixture
def pool(monkeypatch):
    entry = PooledCredential(
        provider="openai-codex", id="seat", label="seat", auth_type="api_key",
        priority=0, source="manual", access_token="pooled-test-token", base_url=BASE_URL,
        model_cooldowns={MODEL: time.time() + 365 * 86400},
    )
    monkeypatch.setattr(credential_pool, "get_pool_strategy", lambda provider: "fill_first")
    pool = CredentialPool("openai-codex", [entry])
    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)
    return pool


def test_quota_fallback_uses_model_benched_credential(monkeypatch, pool):
    """An invalid singleton must not hide a valid pool token behind an unrelated entitlement."""
    def invalid_singleton(**kwargs):
        raise account_usage.AuthError(
            "Invalid refresh token", provider="openai-codex", code="invalid_refresh_token",
            relogin_required=True,
        )

    monkeypatch.setattr(account_usage, "resolve_codex_runtime_credentials", invalid_singleton)
    calls = []

    def get_json(url, headers, *, timeout):
        calls.append((url, headers))
        return {"plan_type": "pro", "rate_limit": {
            "primary_window": {"used_percent": 26, "limit_window_seconds": 604800},
        }}

    monkeypatch.setattr(account_usage, "_get_json", get_json)
    before = pool.entries()
    assert pool.select() is None
    snapshot = account_usage.fetch_account_usage("openai-codex")
    assert snapshot is not None
    assert [(w.label, w.used_percent) for w in snapshot.windows] == [("Weekly", 26.0)]
    assert len(calls) == 1
    assert calls[0][0] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0][1]["Authorization"] == "Bearer pooled-test-token"
    assert pool.entries() == before
    assert pool.current() is None
    assert pool.select(model=MODEL) is None
    assert pool.select() is None
    assert pool.select(model="available-model") is not None


@pytest.mark.parametrize("status", [STATUS_DEAD, STATUS_EXHAUSTED])
def test_usage_selection_preserves_credential_wide_blocks(pool, status, monkeypatch):
    entry = replace(pool.entries()[0], last_status=status, last_status_at=time.time(),
                    last_error_code=401, last_error_reset_at=time.time() + 3600)
    pool._entries = [entry]
    monkeypatch.setattr(pool, "_refresh_entry", lambda *a, **kw: pytest.fail("blocked token refreshed"))
    assert pool.select_for_usage() is None
    assert pool.entries() == [entry]


def test_usage_selection_refreshes_oauth_outside_pool_lock(pool, monkeypatch):
    entry = replace(pool.entries()[0], auth_type="oauth", refresh_token="test-refresh")
    pool._entries = [entry]
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda e: e.access_token == entry.access_token)
    refreshed = replace(entry, access_token="fresh-test-token")
    calls = []

    def refresh(e, *, force):
        assert not pool._lock._is_owned()
        calls.append((e.id, force))
        pool._entries = [refreshed]
        return refreshed

    monkeypatch.setattr(pool, "_refresh_entry", refresh)
    assert pool.select_for_usage() == refreshed
    assert calls == [(entry.id, False)]
    assert pool.entries()[0].model_cooldowns == entry.model_cooldowns
    assert pool.current() is None


def test_usage_selection_skips_failed_refresh(pool, monkeypatch):
    entry = replace(pool.entries()[0], auth_type="oauth", refresh_token="test-refresh")
    pool._entries = [entry]
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda e: True)
    monkeypatch.setattr(pool, "_refresh_entry", lambda *a, **kw: None)
    assert pool.select_for_usage() is None


def test_usage_selection_does_not_rotate_or_count_inference(pool):
    from agent.credential_pool import STRATEGY_ROUND_ROBIN

    first = pool.entries()[0]
    pool._entries.append(replace(first, id="other-seat", priority=1, access_token="other-test-token"))
    pool._strategy = STRATEGY_ROUND_ROBIN
    before = pool.entries()
    assert pool.select_for_usage() == first
    assert pool.select_for_usage() == first
    assert pool.entries() == before
    assert pool.current() is None
