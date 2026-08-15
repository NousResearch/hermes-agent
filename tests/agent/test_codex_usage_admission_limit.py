import concurrent.futures
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)


def _snapshot(*windows: AccountUsageWindow) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=tuple(windows),
    )


def _entry(
    credential_id: str = "a1b2c3",
    *,
    token: str = "reserved-token",
    priority: int = 0,
) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=credential_id,
        label="codex-reserved",
        auth_type=AUTH_TYPE_API_KEY,
        priority=priority,
        source="manual",
        access_token=token,
        base_url="https://chatgpt.com/backend-api/codex",
    )


def _select_id(pool: CredentialPool) -> str:
    selected = pool.select()
    assert selected is not None
    return selected.id


@pytest.fixture(autouse=True)
def _clear_usage_admission_cache():
    import agent.credential_pool as credential_pool

    with credential_pool._USAGE_ADMISSION_CACHE_LOCK:
        credential_pool._USAGE_ADMISSION_CACHE.clear()
        credential_pool._USAGE_ADMISSION_INFLIGHT.clear()
    yield
    with credential_pool._USAGE_ADMISSION_CACHE_LOCK:
        credential_pool._USAGE_ADMISSION_CACHE.clear()
        credential_pool._USAGE_ADMISSION_INFLIGHT.clear()


def test_policy_window_uses_longest_duration_not_primary_secondary_label():
    from agent.credential_pool import _select_codex_policy_window

    seven_day_primary = AccountUsageWindow(
        label="Session",
        used_percent=79,
        limit_window_seconds=604_800,
    )
    five_hour_secondary = AccountUsageWindow(
        label="Weekly",
        used_percent=95,
        limit_window_seconds=18_000,
    )

    selected = _select_codex_policy_window(
        _snapshot(seven_day_primary, five_hour_secondary)
    )

    assert selected is seven_day_primary


def test_policy_window_falls_back_to_farthest_reset():
    from agent.credential_pool import _select_codex_policy_window

    now = datetime.now(timezone.utc)
    earlier = AccountUsageWindow(
        label="Primary",
        used_percent=95,
        reset_at=now + timedelta(hours=2),
    )
    later = AccountUsageWindow(
        label="Secondary",
        used_percent=79,
        reset_at=now + timedelta(days=5),
    )

    selected = _select_codex_policy_window(_snapshot(earlier, later))

    assert selected is later


def test_policy_window_without_duration_or_reset_fails_open():
    from agent.credential_pool import _select_codex_policy_window

    unknown = AccountUsageWindow(label="Unknown", used_percent=95)

    assert _select_codex_policy_window(_snapshot(unknown)) is None


def test_policy_window_mixed_durations_fail_open():
    from agent.credential_pool import _select_codex_policy_window

    now = datetime.now(timezone.utc)
    shorter_with_duration = AccountUsageWindow(
        label="Short",
        used_percent=95,
        limit_window_seconds=18_000,
        reset_at=now + timedelta(hours=2),
    )
    longer_without_duration = AccountUsageWindow(
        label="Long",
        used_percent=79,
        reset_at=now + timedelta(days=5),
    )

    assert (
        _select_codex_policy_window(
            _snapshot(shorter_with_duration, longer_without_duration)
        )
        is None
    )


@pytest.mark.parametrize("malformed_duration", ["604800", 0, -1, float("nan")])
def test_policy_window_malformed_duration_with_complete_resets_fails_open(
    malformed_duration,
):
    from agent.credential_pool import _select_codex_policy_window

    now = datetime.now(timezone.utc)
    valid = AccountUsageWindow(
        label="Valid",
        used_percent=79,
        limit_window_seconds=18_000,
        reset_at=now + timedelta(hours=2),
    )
    malformed = AccountUsageWindow(
        label="Malformed",
        used_percent=95,
        limit_window_seconds=malformed_duration,
        reset_at=now + timedelta(days=5),
    )

    assert _select_codex_policy_window(_snapshot(valid, malformed)) is None


def test_policy_window_mixed_durations_without_complete_resets_fails_open():
    from agent.credential_pool import _select_codex_policy_window

    shorter_with_duration = AccountUsageWindow(
        label="Short",
        used_percent=95,
        limit_window_seconds=18_000,
        reset_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
    )
    unknown_longer = AccountUsageWindow(label="Unknown", used_percent=79)

    assert (
        _select_codex_policy_window(_snapshot(shorter_with_duration, unknown_longer))
        is None
    )


def test_policy_window_equal_longest_duration_tie_fails_open():
    from agent.credential_pool import _select_codex_policy_window

    first = AccountUsageWindow(
        label="First",
        used_percent=90,
        limit_window_seconds=604_800,
    )
    second = AccountUsageWindow(
        label="Second",
        used_percent=10,
        limit_window_seconds=604_800,
    )

    assert _select_codex_policy_window(_snapshot(first, second)) is None


def test_policy_window_past_reset_fallback_fails_open():
    from agent.credential_pool import _select_codex_policy_window

    stale = AccountUsageWindow(
        label="Stale",
        used_percent=90,
        reset_at=datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )

    assert _select_codex_policy_window(_snapshot(stale)) is None


def test_policy_window_duration_with_past_reset_fails_open():
    from agent.credential_pool import _select_codex_policy_window

    stale = AccountUsageWindow(
        label="Stale weekly window",
        used_percent=90,
        limit_window_seconds=604_800,
        reset_at=datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )

    assert _select_codex_policy_window(_snapshot(stale)) is None


def test_policy_window_huge_duration_fails_open_without_raising():
    from agent.credential_pool import _select_codex_policy_window

    malformed = AccountUsageWindow(
        label="Malformed",
        used_percent=90,
        limit_window_seconds=10**400,
    )

    assert _select_codex_policy_window(_snapshot(malformed)) is None


@pytest.mark.parametrize("used_percent", ["80", " 80 ", "8e1"])
def test_policy_window_rejects_string_percentages(used_percent):
    from agent.credential_pool import _select_codex_policy_window

    malformed = AccountUsageWindow(
        label="Weekly",
        used_percent=used_percent,
        limit_window_seconds=604_800,
    )

    assert _select_codex_policy_window(_snapshot(malformed)) is None


def test_select_denies_credential_at_exact_threshold(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    calls = []
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: (
            calls.append((provider, kwargs))
            or _snapshot(
                AccountUsageWindow(
                    label="Primary",
                    used_percent=80,
                    limit_window_seconds=604_800,
                )
            )
        ),
    )

    selected = CredentialPool("openai-codex", [_entry()]).select()

    assert selected is None
    assert calls == [
        (
            "openai-codex",
            {
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "reserved-token",
                "timeout": 3.0,
            },
        )
    ]


def test_specific_lease_refuses_credential_at_threshold(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        ),
    )
    pool = CredentialPool("openai-codex", [_entry()])

    assert pool.acquire_lease("a1b2c3") is None


def test_failure_rotation_refreshes_limit_before_choosing_replacement(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        ),
    )
    pool = CredentialPool(
        "openai-codex",
        [
            _entry("failed", token="failed-token", priority=0),
            _entry(priority=1),
        ],
    )
    pool._current_id = "failed"

    replacement = pool.mark_exhausted_and_rotate(
        status_code=429,
        credential_id="failed",
    )

    assert replacement is None


def test_auxiliary_fresh_pool_refresh_targets_failed_token_and_applies_admission(
    monkeypatch,
):
    from dataclasses import replace

    import agent.auxiliary_client as auxiliary_client
    import agent.credential_pool as credential_pool

    failed = PooledCredential(
        provider="openai-codex",
        id="configured-entry",
        label="configured",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual:device_code",
        access_token="failed-old-token",
        refresh_token="refresh-token",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    sibling = _entry("sibling-entry", token="sibling-token", priority=1)
    pool = CredentialPool("openai-codex", [failed, sibling])
    refreshed_ids = []

    def _refresh_entry(entry, *, force=False, **_kw):
        assert force is True
        refreshed_ids.append(entry.id)
        refreshed = replace(entry, access_token="failed-fresh-token")
        pool._replace_entry(entry, refreshed)
        return refreshed

    pool._refresh_entry = _refresh_entry
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda _provider: {"configured-entry": 80.0},
    )

    def _usage(_provider, *, api_key, **_kwargs):
        used = 90 if api_key == "failed-fresh-token" else 10
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=used,
                limit_window_seconds=604_800,
                reset_at=datetime.now(timezone.utc) + timedelta(days=7),
            )
        )

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", _usage)
    monkeypatch.setattr(auxiliary_client, "load_pool", lambda _provider: pool)
    evicted = []
    monkeypatch.setattr(
        auxiliary_client,
        "_evict_cached_clients",
        lambda provider: evicted.append(provider),
    )

    auth_error = Exception("unauthorized")
    auth_error.status_code = 401

    assert auxiliary_client._recover_provider_pool(
        "openai-codex",
        auth_error,
        failed_api_key="failed-old-token",
    )
    assert refreshed_ids == ["configured-entry"]
    assert pool.current().id == "sibling-entry"
    assert pool.entries()[0].last_status is None
    assert evicted == ["openai-codex"]


def test_forced_refresh_rechecks_admission_for_new_token(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    probes = []

    def fetch_usage(provider, **kwargs):
        probes.append(kwargs["api_key"])
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch_usage)
    stale = _entry(token="stale-token")
    refreshed = _entry(token="fresh-token")
    pool = CredentialPool("openai-codex", [stale])
    monkeypatch.setattr(pool, "_refresh_entry", lambda entry, force, **_kw: refreshed)

    entry, denied_id = pool.try_refresh_matching_with_admission_status(
        credential_id="a1b2c3"
    )

    assert entry is None
    assert denied_id == "a1b2c3"
    assert probes == ["fresh-token"]


def test_primary_auth_recovery_rotates_without_exhausting_policy_denied_refresh():
    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.error_classifier import FailoverReason

    replacement = _entry("next-id", token="next-token", priority=1)

    class Pool:
        provider = "openai-codex"

        def try_refresh_matching_with_admission_status(self, **kwargs):
            assert kwargs == {
                "api_key_hint": "stale-token",
                "credential_id": "a1b2c3",
            }
            return None, "a1b2c3"

        def try_refresh_matching(self, **kwargs):
            raise AssertionError("recovery must request admission status")

        def select(self, **kwargs):
            return replacement

        def mark_exhausted_and_rotate(self, **kwargs):
            raise AssertionError("policy denial must not mutate exhaustion state")

    swapped = []
    agent = SimpleNamespace(
        _credential_pool=Pool(),
        _credential_pool_entry_id="a1b2c3",
        _auth_pool_refresh_counts={},
        provider="openai-codex",
        api_key="stale-token",
        _is_entitlement_failure=lambda context, status: False,
        _swap_credential=swapped.append,
    )

    recovered, retried = recover_with_credential_pool(
        agent,
        status_code=401,
        has_retried_429=False,
        classified_reason=FailoverReason.auth,
        error_context={},
    )

    assert (recovered, retried) == (True, False)
    assert swapped == [replacement]


def test_successful_usage_probe_is_cached_for_five_minutes(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    calls = []
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: (
            calls.append(kwargs)
            or _snapshot(
                AccountUsageWindow(
                    label="Weekly",
                    used_percent=79,
                    limit_window_seconds=604_800,
                )
            )
        ),
    )
    pool = CredentialPool("openai-codex", [_entry()])

    assert _select_id(pool) == "a1b2c3"
    assert _select_id(pool) == "a1b2c3"
    assert len(calls) == 1


def test_failed_probe_fails_open_then_retries_after_thirty_seconds(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    clock = [0.0]
    calls = []
    monkeypatch.setattr(credential_pool.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )

    def fake_fetch(provider, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return None
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)
    pool = CredentialPool("openai-codex", [_entry()])

    assert _select_id(pool) == "a1b2c3"
    clock[0] = 29.0
    assert _select_id(pool) == "a1b2c3"
    assert len(calls) == 1
    clock[0] = 30.0
    assert pool.select() is None
    assert len(calls) == 2


def test_cache_expiry_rechecks_and_readmits_after_reset(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    clock = [0.0]
    used_values = iter((80, 0))
    monkeypatch.setattr(credential_pool.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=next(used_values),
                limit_window_seconds=604_800,
            )
        ),
    )
    pool = CredentialPool("openai-codex", [_entry()])

    assert pool.select() is None
    clock[0] = 300.0
    assert _select_id(pool) == "a1b2c3"


def test_token_change_invalidates_cached_decision(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    calls = []
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )

    def fake_fetch(provider, **kwargs):
        calls.append(kwargs["api_key"])
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80 if kwargs["api_key"] == "old-token" else 0,
                limit_window_seconds=604_800,
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    assert CredentialPool("openai-codex", [_entry(token="old-token")]).select() is None
    selected = CredentialPool("openai-codex", [_entry(token="new-token")]).select()

    assert selected is not None
    assert selected.id == "a1b2c3"
    assert calls == ["old-token", "new-token"]


def test_stale_token_probe_does_not_delete_fresh_token_cache(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=10,
                limit_window_seconds=604_800,
            )
        ),
    )
    fresh_entry = _entry(token="token-after-refresh")
    stale_entry = _entry(token="token-before-refresh")
    fresh_key = credential_pool._usage_admission_cache_key(fresh_entry)

    assert CredentialPool("openai-codex", [fresh_entry]).select() is not None
    assert CredentialPool("openai-codex", [stale_entry]).select() is not None

    with credential_pool._USAGE_ADMISSION_CACHE_LOCK:
        assert fresh_key in credential_pool._USAGE_ADMISSION_CACHE


def test_usage_probe_runs_outside_pool_lock(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    pool = CredentialPool("openai-codex", [_entry()])

    def fake_fetch(provider, **kwargs):
        assert not getattr(pool._lock, "_is_owned")()
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=79,
                limit_window_seconds=604_800,
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    assert _select_id(pool) == "a1b2c3"


def test_usage_cache_never_contains_raw_access_token(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    secret_token = "secret-vicky-access-token"
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=79,
                limit_window_seconds=604_800,
            )
        ),
    )

    CredentialPool("openai-codex", [_entry(token=secret_token)]).select()

    with credential_pool._USAGE_ADMISSION_CACHE_LOCK:
        assert secret_token not in repr(credential_pool._USAGE_ADMISSION_CACHE)


def test_automatic_lease_skips_denied_credential(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        ),
    )
    pool = CredentialPool(
        "openai-codex",
        [
            _entry(priority=0),
            _entry("other", token="other-token", priority=1),
        ],
    )

    assert pool.acquire_lease() == "other"


def test_select_rechecks_usage_after_oauth_token_refresh(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    calls = []
    entry = _entry(token="old-token")
    entry.auth_type = AUTH_TYPE_OAUTH
    pool = CredentialPool("openai-codex", [entry])
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )

    def fake_fetch(provider, **kwargs):
        calls.append(kwargs["api_key"])
        if kwargs["api_key"] == "old-token":
            return None
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)
    monkeypatch.setattr(
        pool,
        "_entry_needs_refresh",
        lambda candidate: candidate.access_token == "old-token",
    )

    def fake_refresh(pending):
        entry.access_token = "new-token"

    monkeypatch.setattr(pool, "_refresh_pending_entries", fake_refresh)

    assert pool.select() is None
    assert calls == ["old-token", "new-token"]


def test_lease_rechecks_usage_after_oauth_token_refresh(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    calls = []
    entry = _entry(token="old-token")
    entry.auth_type = AUTH_TYPE_OAUTH
    pool = CredentialPool("openai-codex", [entry])
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )

    def fake_fetch(provider, **kwargs):
        calls.append(kwargs["api_key"])
        if kwargs["api_key"] == "old-token":
            return None
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)
    monkeypatch.setattr(
        pool,
        "_entry_needs_refresh",
        lambda candidate: candidate.access_token == "old-token",
    )

    def fake_refresh(pending):
        entry.access_token = "new-token"

    monkeypatch.setattr(pool, "_refresh_pending_entries", fake_refresh)

    assert pool.acquire_lease() is None
    assert calls == ["old-token", "new-token"]


def test_unmatched_configured_credential_warns_once(monkeypatch, caplog):
    import agent.credential_pool as credential_pool

    credential_pool._WARNED_UNMATCHED_USAGE_LIMITS.clear()
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"missing-id": 80.0},
    )
    pool = CredentialPool("openai-codex", [_entry("other")])

    pool.select()
    pool.select()

    messages = [
        record.getMessage()
        for record in caplog.records
        if "missing-id" in record.getMessage()
    ]
    assert len(messages) == 1


def test_threshold_change_reuses_fresh_usage_but_recomputes_admission(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    threshold = [80.0]
    calls = []
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": threshold[0]},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: (
            calls.append(kwargs)
            or _snapshot(
                AccountUsageWindow(
                    label="Weekly",
                    used_percent=85,
                    limit_window_seconds=604_800,
                )
            )
        ),
    )
    pool = CredentialPool("openai-codex", [_entry()])

    assert pool.select() is None
    threshold[0] = 90.0
    assert _select_id(pool) == "a1b2c3"
    assert len(calls) == 1


def test_explicit_token_recomputes_cached_telemetry_after_threshold_change(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    threshold = [80.0]
    calls = []
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": threshold[0]},
    )
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda provider: pool,
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: (
            calls.append(kwargs)
            or _snapshot(
                AccountUsageWindow(
                    label="Weekly",
                    used_percent=85,
                    limit_window_seconds=604_800,
                )
            )
        ),
    )
    pool = CredentialPool("openai-codex", [_entry()])

    assert pool.select() is None
    threshold[0] = 90.0
    assert credential_pool.usage_admission_policy_denied("openai-codex") is False
    assert (
        credential_pool.usage_admission_credential_denied(
            "openai-codex", "reserved-token"
        )
        is False
    )
    assert len(calls) == 1


def test_inflight_waiter_applies_its_current_threshold(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    threshold = [90.0]
    probe_started = threading.Event()
    release_probe = threading.Event()
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": threshold[0]},
    )

    def fetch_usage(provider, **kwargs):
        probe_started.set()
        assert release_probe.wait(timeout=2)
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=85,
                limit_window_seconds=604_800,
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch_usage)
    creator_pool = CredentialPool("openai-codex", [_entry()])
    waiter_pool = CredentialPool("openai-codex", [_entry()])
    results = {}

    creator = threading.Thread(
        target=lambda: results.setdefault("creator", creator_pool.select())
    )
    creator.start()
    assert probe_started.wait(timeout=2)
    threshold[0] = 80.0
    waiter = threading.Thread(
        target=lambda: results.setdefault("waiter", waiter_pool.select())
    )
    waiter.start()
    release_probe.set()
    creator.join(timeout=2)
    waiter.join(timeout=2)

    assert not creator.is_alive()
    assert not waiter.is_alive()
    assert results["creator"] is not None
    assert results["waiter"] is None


def test_completed_probe_callback_clears_inflight_without_reentrant_lock(
    monkeypatch,
):
    import agent.credential_pool as credential_pool

    class RejectReentrantLock:
        def __init__(self):
            self._lock = threading.Lock()
            self._owner = None

        def __enter__(self):
            owner = threading.get_ident()
            if self._owner == owner:
                raise RuntimeError("reentrant cache lock acquisition")
            self._lock.acquire()
            self._owner = owner
            return self

        def __exit__(self, exc_type, exc, tb):
            self._owner = None
            self._lock.release()

    class CompletedExecutor:
        def submit(self, fn, *args, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(None)
            return future

    pool = CredentialPool("openai-codex", [_entry()])
    cache_key = credential_pool._usage_admission_cache_key(pool.entries()[0])
    with monkeypatch.context() as patcher:
        patcher.setattr(
            credential_pool,
            "get_pool_usage_limits",
            lambda provider: {"a1b2c3": 80.0},
        )
        patcher.setattr(
            credential_pool,
            "_USAGE_ADMISSION_CACHE_LOCK",
            RejectReentrantLock(),
        )
        patcher.setattr(
            credential_pool,
            "_USAGE_ADMISSION_PROBE_EXECUTOR",
            CompletedExecutor(),
        )
        pool._refresh_usage_admission_cache()

    assert cache_key not in credential_pool._USAGE_ADMISSION_INFLIGHT


def test_selection_reads_usage_limit_config_once_outside_entry_filter(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    config_calls = []
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: config_calls.append(provider) or {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=79,
                limit_window_seconds=604_800,
            )
        ),
    )

    assert _select_id(CredentialPool("openai-codex", [_entry()])) == "a1b2c3"
    assert config_calls == ["openai-codex"]


def test_runtime_resolver_does_not_fall_back_to_singleton_after_pool_denial(
    monkeypatch,
):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool
    import hermes_cli.runtime_provider as runtime_provider
    from hermes_cli.auth import AuthError

    singleton_calls = []
    pool = CredentialPool("openai-codex", [_entry()])
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        ),
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_provider",
        lambda *args, **kwargs: "openai-codex",
    )
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {})
    monkeypatch.setattr(
        runtime_provider,
        "_resolve_explicit_runtime",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    monkeypatch.setattr(
        runtime_provider,
        "resolve_codex_runtime_credentials",
        lambda: (
            singleton_calls.append(True)
            or {
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "reserved-token",
            }
        ),
    )

    with pytest.raises(AuthError):
        runtime_provider.resolve_runtime_provider(requested="openai-codex")
    assert singleton_calls == []


def test_auxiliary_codex_does_not_fall_back_after_pool_denial(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    singleton_calls = []
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_read_codex_singleton_token",
        lambda: singleton_calls.append(True) or "reserved-token",
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_policy_denied",
        lambda provider: True,
        raising=False,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_create_openai_client",
        lambda **kwargs: object(),
    )

    client, model = auxiliary_client._build_codex_client("gpt-5.6-codex")

    assert client is None
    assert model is None
    assert singleton_calls == []


def test_runtime_resolver_keeps_legacy_singleton_fallback_without_policy_denial(
    monkeypatch,
):
    import hermes_cli.runtime_provider as runtime_provider

    class EmptyPool:
        def has_credentials(self):
            return True

        def select(self, **kwargs):
            return None

    singleton_calls = []
    monkeypatch.setattr(
        runtime_provider,
        "resolve_provider",
        lambda *args, **kwargs: "openai-codex",
    )
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {})
    monkeypatch.setattr(
        runtime_provider,
        "_resolve_explicit_runtime",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: EmptyPool())
    monkeypatch.setattr(
        runtime_provider,
        "usage_admission_policy_denied",
        lambda provider: False,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_codex_runtime_credentials",
        lambda: (
            singleton_calls.append(True)
            or {
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "legacy-singleton-token",
            }
        ),
    )

    runtime = runtime_provider.resolve_runtime_provider(requested="openai-codex")

    assert runtime["api_key"] == "legacy-singleton-token"
    assert singleton_calls == [True]


def test_auxiliary_codex_keeps_legacy_singleton_fallback_without_policy_denial(
    monkeypatch,
):
    import agent.auxiliary_client as auxiliary_client

    singleton_calls = []
    fake_client = SimpleNamespace(
        api_key="legacy-singleton-token",
        base_url="https://chatgpt.com/backend-api/codex",
        close=lambda: None,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_policy_denied",
        lambda provider: False,
        raising=False,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_read_codex_singleton_token",
        lambda: singleton_calls.append(True) or "legacy-singleton-token",
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_create_openai_client",
        lambda **kwargs: fake_client,
    )

    client, model = auxiliary_client._build_codex_client("gpt-5.6-codex")

    assert client is not None
    assert model == "gpt-5.6-codex"
    assert singleton_calls == [True]


def test_raw_codex_token_does_not_fall_back_after_pool_denial(monkeypatch):
    import agent.auxiliary_client as auxiliary_client
    import hermes_cli.auth as auth

    singleton_calls = []
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_policy_denied",
        lambda provider: True,
    )
    monkeypatch.setattr(
        auth,
        "_read_codex_tokens",
        lambda: (
            singleton_calls.append(True)
            or {"tokens": {"access_token": "legacy-singleton-token"}}
        ),
    )

    # Main split the pure singleton reader from the pool-aware resolver; the policy gate lives in the latter.
    token, _base_url = auxiliary_client._resolve_codex_credential_and_base("gpt-5.6-codex")
    assert token is None
    assert singleton_calls == []


def test_cached_auxiliary_codex_client_is_rechecked_after_policy_denial(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    cache_key = ("openai-codex", "codex-policy-test")
    cached_client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        auxiliary_client,
        "_client_cache_key",
        lambda *args, **kwargs: cache_key,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
        raising=False,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_policy_denied",
        lambda provider: True,
    )
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
        auxiliary_client._client_cache[cache_key] = (
            cached_client,
            "gpt-5.6-codex",
            None,
        )
    try:
        client, model = auxiliary_client._get_cached_client(
            "openai-codex",
            "gpt-5.6-codex",
        )
        assert client is None
        assert model is None
        assert cache_key not in auxiliary_client._client_cache
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_cached_auxiliary_explicit_codex_client_rejects_denied_token(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    cache_key = ("openai-codex", "explicit-policy-test")
    cached_client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        auxiliary_client,
        "_client_cache_key",
        lambda *args, **kwargs: cache_key,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_credential_denied",
        lambda provider, credential: credential == "denied-explicit-token",
    )
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
        auxiliary_client._client_cache[cache_key] = (
            cached_client,
            "gpt-5.6-codex",
            None,
        )
    try:
        client, model = auxiliary_client._get_cached_client(
            "openai-codex",
            "gpt-5.6-codex",
            api_key="denied-explicit-token",
        )
        assert client is None
        assert model is None
        assert cache_key not in auxiliary_client._client_cache
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_cached_auxiliary_codex_uses_freshly_selected_eligible_entry(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    denied_key = ("openai-codex", "A")
    eligible_key = ("openai-codex", "B")
    denied_client = SimpleNamespace(marker="DENIED_A", close=lambda: None)
    eligible_client = SimpleNamespace(marker="ELIGIBLE_B", close=lambda: None)
    eligible_entry = SimpleNamespace(
        id="B",
        runtime_api_key="eligible-token",
        access_token="eligible-token",
    )

    def cache_key(*args, **kwargs):
        hint = kwargs.get("pool_hint_override")
        return eligible_key if hint == "openai-codex:B" else denied_key

    monkeypatch.setattr(auxiliary_client, "_client_cache_key", cache_key)
    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"A": 80.0},
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, eligible_entry),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_policy_denied",
        lambda provider: True,
    )
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
        auxiliary_client._client_cache[denied_key] = (
            denied_client,
            "gpt-5.6-codex",
            None,
        )
        auxiliary_client._client_cache[eligible_key] = (
            eligible_client,
            "gpt-5.6-codex",
            None,
        )
    try:
        client, _model = auxiliary_client._get_cached_client(
            "openai-codex",
            "gpt-5.6-codex",
        )
        assert client is eligible_client
        assert client.marker == "ELIGIBLE_B"
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_auxiliary_codex_build_uses_same_freshly_selected_entry(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    selected = SimpleNamespace(
        id="B",
        runtime_api_key="token-B",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
        access_token="token-B",
    )
    built_with = []
    built_client = SimpleNamespace(marker="BUILT_WITH_B", close=lambda: None)

    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"A": 80.0},
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, selected),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_policy_denied",
        lambda provider: True,
    )

    def build(model, *, access_token=None, base_url=None):
        built_with.append((access_token, base_url))
        return built_client, model

    monkeypatch.setattr(auxiliary_client, "_build_codex_client", build)
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
    try:
        client, _model = auxiliary_client._get_cached_client(
            "openai-codex",
            "gpt-5.6-codex",
        )
        assert client is built_client
        assert built_with == [("token-B", "https://chatgpt.com/backend-api/codex")]
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_auto_auxiliary_uses_freshly_selected_codex_entry(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    selected = SimpleNamespace(
        id="B",
        runtime_api_key="token-B",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
        access_token="token-B",
    )
    calls = []
    built_client = SimpleNamespace(marker="BUILT_WITH_B", close=lambda: None)
    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"A": 80.0},
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, selected),
    )

    def resolve(provider, model, async_mode=False, **kwargs):
        calls.append((
            provider,
            kwargs.get("explicit_api_key"),
            kwargs.get("explicit_base_url"),
        ))
        return built_client, model

    monkeypatch.setattr(auxiliary_client, "resolve_provider_client", resolve)
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
    try:
        client, _model = auxiliary_client._get_cached_client(
            "auto",
            "gpt-5.6-codex",
            main_runtime={
                "provider": "openai-codex",
                "model": "gpt-5.6-codex",
                "api_key": "denied-token-A",
                "credential_pool_entry_id": "A",
                "base_url": "https://chatgpt.com/backend-api/codex",
            },
        )
        assert client is built_client
        assert calls == [
            (
                "openai-codex",
                "token-B",
                "https://chatgpt.com/backend-api/codex",
            )
        ]
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_auxiliary_cache_rebinds_after_same_id_codex_token_refresh(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    current = [
        SimpleNamespace(
            id="a1b2c3",
            runtime_api_key="token-old",
            runtime_base_url="https://old.example/codex",
            access_token="token-old",
        )
    ]
    built = []
    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, current[0]),
    )

    def resolve(provider, model, async_mode=False, **kwargs):
        client = SimpleNamespace(
            api_key=kwargs.get("explicit_api_key"),
            base_url=kwargs.get("explicit_base_url"),
            close=lambda: None,
        )
        built.append(client)
        return client, model

    monkeypatch.setattr(auxiliary_client, "resolve_provider_client", resolve)
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
    try:
        first, _model = auxiliary_client._get_cached_client(
            "openai-codex", "gpt-5.6-codex"
        )
        current[0] = SimpleNamespace(
            id="a1b2c3",
            runtime_api_key="token-new",
            runtime_base_url="https://new.example/codex",
            access_token="token-new",
        )
        second, _model = auxiliary_client._get_cached_client(
            "openai-codex", "gpt-5.6-codex"
        )

        assert first is not None
        assert second is not None
        assert first is not second
        assert second.api_key == "token-new"
        assert second.base_url == "https://new.example/codex"
        assert len(built) == 2
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_policy_denial_evicts_auto_clients_tagged_as_codex():
    import agent.auxiliary_client as auxiliary_client

    # Main scopes cache keys by profile home (key[0]) and evicts only the calling profile's entries.
    auto_key = (auxiliary_client.hermes_home_key(), "auto", "cached-codex")
    cached_client = SimpleNamespace(
        _hermes_aux_effective_provider="openai-codex",
        close=lambda: None,
    )
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
        auxiliary_client._client_cache[auto_key] = (
            cached_client,
            "gpt-5.6-codex",
            None,
        )
    try:
        auxiliary_client._evict_cached_clients("openai-codex")
        assert auto_key not in auxiliary_client._client_cache
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_auto_auxiliary_continues_to_fallback_after_codex_policy_denial(
    monkeypatch,
):
    import agent.auxiliary_client as auxiliary_client

    fallback_client = SimpleNamespace(
        _hermes_aux_effective_provider="openrouter",
        close=lambda: None,
    )
    calls = []
    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider, model=None: (True, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_policy_denied",
        lambda provider: True,
    )
    monkeypatch.setattr(
        auxiliary_client, "_evict_cached_clients", lambda provider: None
    )

    def resolve(provider, model, async_mode=False, **kwargs):
        calls.append((provider, model))
        return fallback_client, "fallback-model"

    monkeypatch.setattr(auxiliary_client, "resolve_provider_client", resolve)
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
    try:
        client, model = auxiliary_client._get_cached_client(
            "auto",
            "gpt-5.6-codex",
            main_runtime={
                "provider": "openai-codex",
                "model": "gpt-5.6-codex",
            },
        )

        assert client is fallback_client
        assert model == "gpt-5.6-codex"
        assert calls == [("auto", "gpt-5.6-codex")]
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_non_codex_auxiliary_pool_entry_never_inherits_codex_base_url(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    selected = SimpleNamespace(
        id="deepseek-entry",
        provider="deepseek",
        runtime_api_key="legacy-pool-token",
        runtime_base_url=None,
        access_token="legacy-pool-token",
    )
    calls = []
    monkeypatch.setattr(auxiliary_client, "get_pool_usage_limits", lambda provider: {})
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(auxiliary_client, "pool_for_aux_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        runtime_provider, "choose_pool_key", lambda *args, **kwargs: SimpleNamespace(api_key=selected.runtime_api_key)
    )

    def resolve(provider, model, async_mode=False, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(close=lambda: None), model

    monkeypatch.setattr(auxiliary_client, "resolve_provider_client", resolve)
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
    try:
        client, _model = auxiliary_client._get_cached_client(
            "deepseek", "deepseek-chat"
        )
        assert client is not None
        assert calls[0]["explicit_api_key"] == "legacy-pool-token"
        assert calls[0]["explicit_base_url"] is None
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_multiple_cold_admission_probes_run_concurrently(monkeypatch):
    import threading

    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    barrier = threading.Barrier(2)
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"A": 80.0, "B": 80.0},
    )

    def fetch(provider, **kwargs):
        barrier.wait(timeout=1.0)
        used = 80 if kwargs["api_key"] == "token-A" else 10
        return _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=used,
                limit_window_seconds=604_800,
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch)
    pool = CredentialPool(
        "openai-codex",
        [
            _entry("A", token="token-A", priority=0),
            _entry("B", token="token-B", priority=1),
        ],
    )

    selected = pool.select()

    assert selected is not None
    assert selected.id == "B"


@pytest.mark.parametrize("raw_codex", [False, True])
def test_resolver_rejects_explicit_policy_denied_codex_token(monkeypatch, raw_codex):
    import agent.account_usage as account_usage
    import agent.auxiliary_client as auxiliary_client
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        ),
    )
    assert CredentialPool("openai-codex", [_entry()]).select() is None
    assert (
        credential_pool.usage_admission_credential_denied(
            "openai-codex", "other-eligible-token"
        )
        is False
    )

    created = []
    monkeypatch.setattr(
        auxiliary_client,
        "_create_openai_client",
        lambda **kwargs: created.append(kwargs) or SimpleNamespace(**kwargs),
    )

    client, model = auxiliary_client.resolve_provider_client(
        "openai-codex",
        "gpt-5.6-codex",
        raw_codex=raw_codex,
        explicit_api_key="reserved-token",
        explicit_base_url="https://chatgpt.com/backend-api/codex",
    )

    assert client is None
    assert model is None
    assert created == []


def test_cold_explicit_matching_pool_token_is_probed_before_admission(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool

    pool = CredentialPool("openai-codex", [_entry()])
    calls = []
    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: (
            calls.append((provider, kwargs))
            or _snapshot(
                AccountUsageWindow(
                    label="Weekly",
                    used_percent=80,
                    limit_window_seconds=604_800,
                )
            )
        ),
    )

    assert (
        credential_pool.usage_admission_credential_denied(
            "openai-codex", "reserved-token"
        )
        is True
    )
    assert (
        credential_pool.usage_admission_credential_denied(
            "openai-codex", "unrelated-explicit-token"
        )
        is False
    )
    assert len(calls) == 1


def test_denied_cache_decision_expires_at_quota_reset(monkeypatch):
    import agent.credential_pool as credential_pool

    now = [1_800_000_000.0]
    monkeypatch.setattr(credential_pool.time, "time", lambda: now[0])
    decision = credential_pool._UsageAdmissionDecision(
        checked_at=10.0,
        probe_succeeded=True,
        used_percent=80.0,
        reset_at=now[0] + 5.0,
    )

    assert credential_pool._usage_admission_decision_is_fresh(decision, 10.0)
    now[0] += 5.0
    assert not credential_pool._usage_admission_decision_is_fresh(decision, 10.0)


def test_main_runtime_rejects_explicit_policy_denied_codex_token(monkeypatch):
    import agent.account_usage as account_usage
    import agent.credential_pool as credential_pool
    import hermes_cli.runtime_provider as runtime_provider
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(
        credential_pool,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            AccountUsageWindow(
                label="Weekly",
                used_percent=80,
                limit_window_seconds=604_800,
            )
        ),
    )
    assert CredentialPool("openai-codex", [_entry()]).select() is None

    with pytest.raises(AuthError) as exc_info:
        runtime_provider._resolve_explicit_runtime(
            provider="openai-codex",
            requested_provider="openai-codex",
            model_cfg={},
            explicit_api_key="reserved-token",
        )
    assert exc_info.value.code == "codex_credential_usage_policy_denied"

    allowed = runtime_provider._resolve_explicit_runtime(
        provider="openai-codex",
        requested_provider="openai-codex",
        model_cfg={},
        explicit_api_key="other-eligible-token",
    )
    assert allowed is not None
    assert allowed["api_key"] == "other-eligible-token"


def test_auxiliary_cache_hit_rejects_explicit_policy_denied_codex_token(
    monkeypatch,
):
    import agent.auxiliary_client as auxiliary_client

    cached_client = SimpleNamespace(
        _hermes_aux_effective_provider="openai-codex",
        close=lambda: None,
    )
    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        auxiliary_client,
        "usage_admission_credential_denied",
        lambda provider, token: (
            provider == "openai-codex" and token == "reserved-token"
        ),
    )
    cache_key = auxiliary_client._client_cache_key(
        "openai-codex",
        async_mode=False,
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="reserved-token",
        model="gpt-5.6-codex",
        pool_hint_override="",
    )
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache[cache_key] = (
            cached_client,
            "gpt-5.6-codex",
            None,
        )
    try:
        client, model = auxiliary_client._get_cached_client(
            "openai-codex",
            "gpt-5.6-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key="reserved-token",
        )
        assert client is None
        assert model is None
        with auxiliary_client._client_cache_lock:
            assert cache_key not in auxiliary_client._client_cache
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()


def test_auto_runtime_falls_through_after_codex_pool_policy_denial(monkeypatch):
    import hermes_cli.runtime_provider as runtime_provider

    class DeniedPool:
        def has_credentials(self):
            return True

        def select(self, **kwargs):
            return None

    singleton_calls = []
    monkeypatch.setattr(
        runtime_provider,
        "resolve_provider",
        lambda *args, **kwargs: "openai-codex",
    )
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {})
    monkeypatch.setattr(
        runtime_provider,
        "_resolve_explicit_runtime",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        runtime_provider,
        "load_pool",
        lambda provider: DeniedPool(),
    )
    monkeypatch.setattr(
        runtime_provider,
        "usage_admission_policy_denied",
        lambda provider: True,
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_codex_runtime_credentials",
        lambda: singleton_calls.append(True),
    )
    monkeypatch.setattr(
        runtime_provider,
        "_resolve_openrouter_runtime",
        lambda **kwargs: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "unrelated-fallback-token",
            "source": "test",
            "requested_provider": "auto",
        },
    )

    runtime = runtime_provider.resolve_runtime_provider(requested="auto")

    assert runtime["provider"] == "openrouter"
    assert runtime["api_key"] == "unrelated-fallback-token"
    assert singleton_calls == []


def test_explicit_unrelated_codex_token_does_not_relabel_as_pool_entry(monkeypatch):
    import agent.auxiliary_client as auxiliary_client

    selected_calls = []
    cache_hints = []
    built_with = []
    built_client = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        auxiliary_client,
        "get_pool_usage_limits",
        lambda provider: {"a1b2c3": 80.0},
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_select_pool_entry",
        lambda provider: selected_calls.append(provider) or (True, _entry()),
    )
    original_cache_key = auxiliary_client._client_cache_key

    def cache_key(*args, **kwargs):
        cache_hints.append(kwargs.get("pool_hint_override"))
        return original_cache_key(*args, **kwargs)

    monkeypatch.setattr(auxiliary_client, "_client_cache_key", cache_key)

    def build(model, *, access_token=None, base_url=None):
        built_with.append((access_token, base_url))
        return built_client, model

    monkeypatch.setattr(auxiliary_client, "_build_codex_client", build)
    with auxiliary_client._client_cache_lock:
        auxiliary_client._client_cache.clear()
    try:
        client, _model = auxiliary_client._get_cached_client(
            "openai-codex",
            "gpt-5.6-codex",
            api_key="unrelated-explicit-token",
        )
        assert client is built_client
        assert selected_calls == []
        assert cache_hints == [""]
        assert built_with == [("unrelated-explicit-token", None)]
    finally:
        with auxiliary_client._client_cache_lock:
            auxiliary_client._client_cache.clear()
