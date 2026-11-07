"""Quota evidence is bounded, account-scoped, isolated and never a capacity promise."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent import quota_routing as quota

URL = "https://chatgpt.com/backend-api/codex"


def snapshot(used=100, *, now=None, **kw):
    now = now or datetime.now(UTC)
    return AccountUsageSnapshot("openai-codex", "usage_api", now,
        windows=(AccountUsageWindow("Weekly", used, now + timedelta(hours=1)),), **kw)


@pytest.fixture(autouse=True)
def clear_cache():
    quota._cache.clear()
    yield
    quota._cache.clear()


@pytest.mark.parametrize("used,state", [(100, "exhausted"), (110, "exhausted"), (99.9, "available"),
    (0, "available"), (None, "unknown"), (float("nan"), "unknown"), (float("inf"), "unknown"),
    (True, "unknown"), (-1, "unknown")])
def test_capacity_is_not_snapshot_information_availability(used, state):
    s = snapshot(used)
    assert s.available  # the UI has information even when capacity is exhausted
    assert quota.evaluate_snapshot(s).state == state


def test_any_exhausted_account_window_blocks_even_if_another_is_unknown():
    s = snapshot()
    s = replace(s, windows=s.windows + (AccountUsageWindow("Session", None),))
    assert quota.evaluate_snapshot(s).state == "exhausted"


def test_model_and_ambiguous_plugin_windows_do_not_exhaust_account():
    s = replace(snapshot(), windows=(AccountUsageWindow("Opus", 100, scope="model"),))
    assert quota.evaluate_snapshot(s).state == "unknown"
    s = replace(snapshot(), provider="claude-subscription", source="plugin")
    assert quota.evaluate_snapshot(s).state == "unknown"


@pytest.mark.parametrize("change", ["stale", "future", "reset", "naive"])
def test_stale_future_and_reset_observations_are_unknown(change):
    now = datetime.now(UTC)
    s = snapshot(now=now)
    if change == "stale":
        s = replace(s, fetched_at=now - timedelta(seconds=61))
    elif change == "future":
        s = replace(s, fetched_at=now + timedelta(seconds=1))
    elif change == "naive":
        s = replace(s, fetched_at=now.replace(tzinfo=None))
    else:
        s = replace(s, windows=(AccountUsageWindow("Weekly", 100, now),))
    assert quota.evaluate_snapshot(s, now=now).state == "unknown"


def test_shared_account_models_and_concurrent_children_fetch_once(monkeypatch):
    fetch = Mock(return_value=snapshot())
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fetch)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: quota.account_quota("openai-codex", URL, "synthetic-token"), range(24)))
    assert all(v.state == "exhausted" for v in results)
    fetch.assert_called_once_with("openai-codex", base_url=URL, api_key="synthetic-token",
                                 read_only=True, identity_id=quota.quota_group("openai-codex", URL, "synthetic-token")[-1])
    assert "synthetic-token" not in repr(quota._cache)


def test_native_identity_groups_rotated_tokens_not_accounts(monkeypatch):
    monkeypatch.setattr("agent.credential_pool._codex_principal_identity", lambda token: ("workspace", "member"))
    assert quota.quota_group("openai-codex", URL, "token-a") == quota.quota_group("openai-codex", URL + "/", "token-b")
    assert quota.quota_group("openai-codex", URL, "token-a") != quota.quota_group("openai-codex", "https://other.example", "token-a")


def test_different_credentials_and_profiles_do_not_share_unknown_identity(monkeypatch, tmp_path):
    fetch = Mock(return_value=snapshot())
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fetch)
    quota.account_quota("openai-codex", URL, "one")
    quota.account_quota("openai-codex", URL, "two")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "second-profile"))
    quota.account_quota("openai-codex", URL, "one")
    assert fetch.call_count == 3


def test_failure_replaces_old_exhaustion_and_is_negatively_cached(monkeypatch):
    fetch = Mock(side_effect=[snapshot(), None])
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fetch)
    assert quota.account_quota("openai-codex", URL, "one").state == "exhausted"
    quota.invalidate_quota("openai-codex", URL, "one")
    assert quota.account_quota("openai-codex", URL, "one").state == "fetch_failed"
    assert quota.account_quota("openai-codex", URL, "one").state == "fetch_failed"
    assert fetch.call_count == 2


def test_cache_refreshes_on_ttl_and_reset_crossing(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(quota.time, "monotonic", lambda: clock[0])
    now = datetime.now(UTC)
    almost_reset = replace(snapshot(), windows=(AccountUsageWindow("Session", 100, now + timedelta(seconds=2)),))
    fetch = Mock(side_effect=[almost_reset, snapshot(10), snapshot(20)])
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fetch)
    assert quota.account_quota("openai-codex", URL, "one").state == "exhausted"
    clock[0] += 3
    assert quota.account_quota("openai-codex", URL, "one").state == "available"
    clock[0] += 61
    assert quota.account_quota("openai-codex", URL, "one").state == "available"
    assert fetch.call_count == 3


def test_exception_is_secret_free_and_cache_is_size_bounded(monkeypatch, caplog):
    fetch = Mock(side_effect=RuntimeError("secret-token-must-not-be-logged"))
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fetch)
    monkeypatch.setattr(quota, "MAX_CACHE_ENTRIES", 3)
    for key in ("a", "b", "c", "d"):
        assert quota.account_quota("openai-codex", URL, key).state == "fetch_failed"
    assert len(quota._cache) == 3
    assert "secret-token-must-not-be-logged" not in caplog.text


def test_unbound_or_unsupported_providers_never_resolve_credentials(monkeypatch):
    fetch = Mock()
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fetch)
    assert quota.account_quota("openai-codex").state == "unknown"
    assert quota.account_quota("anthropic", "", "test-key").state == "unknown"
    assert quota.account_quota("openai-codex", "", lambda: "key").state == "unknown"
    fetch.assert_not_called()
