from dataclasses import replace

from agent import account_usage
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow


def _snapshot(provider="anthropic", *, unavailable_reason=None):
    return AccountUsageSnapshot(
        provider=provider,
        source="test",
        fetched_at=account_usage._utc_now(),
        windows=(AccountUsageWindow(label="Session", used_percent=25),),
        unavailable_reason=unavailable_reason,
    )


def test_fetch_pool_account_usage_reads_each_pool_entry_and_marks_active(monkeypatch):
    entries = [
        {"id": "a", "label": "work", "access_token": "sk-ant-oat-work", "source": "manual"},
        {"id": "b", "label": "backup", "access_token": "sk-ant-oat-backup", "source": "manual"},
    ]
    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", lambda provider: entries)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "sk-ant-oat-backup")
    calls = []

    def fake_fetch(provider, *, api_key=None, base_url=None):
        calls.append((provider, api_key))
        return _snapshot()

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    readings = account_usage.fetch_pool_account_usage("anthropic")

    assert [reading.label for reading in readings] == ["work", "backup"]
    assert [reading.active for reading in readings] == [False, True]
    assert [key for _, key in calls] == ["sk-ant-oat-work", "sk-ant-oat-backup"]
    assert all(reading.fingerprint.startswith("sha256:") for reading in readings)


def test_pool_429_only_marks_that_account_unavailable(monkeypatch):
    entries = [
        {"id": "a", "label": "work", "access_token": "token-a", "source": "manual"},
        {"id": "b", "label": "backup", "access_token": "token-b", "source": "manual"},
    ]
    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", lambda provider: entries)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "token-b")

    def fake_fetch(provider, *, api_key=None, base_url=None):
        if api_key == "token-a":
            return _snapshot(unavailable_reason="Usage endpoint returned HTTP 429. Rate limited; wait before retrying.")
        return _snapshot()

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)
    readings = account_usage.fetch_pool_account_usage("anthropic")

    assert readings[0].snapshot.available is False
    assert readings[1].snapshot.available is True


def test_empty_pool_preserves_single_account_shape(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", lambda provider: [])
    snapshot = _snapshot()
    monkeypatch.setattr(account_usage, "fetch_account_usage", lambda provider: snapshot)

    readings = account_usage.fetch_pool_account_usage("anthropic")

    assert len(readings) == 1
    assert readings[0].label == "default"
    assert readings[0].active is True
    assert readings[0].snapshot is snapshot


def test_pool_usage_reuses_snapshot_cache_per_credential(monkeypatch):
    entries = [{"id": "a", "label": "work", "access_token": "token-success", "source": "manual"}]
    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", lambda provider: entries)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "token-success")
    calls = []
    snapshot = _snapshot()
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, *, api_key=None, base_url=None: calls.append(api_key) or snapshot,
    )

    account_usage.fetch_pool_account_usage("anthropic")
    account_usage.fetch_pool_account_usage("anthropic")

    assert calls == ["token-success"]


def test_pool_usage_reuses_unavailable_snapshot_cache_per_credential(monkeypatch):
    entries = [{"id": "a", "label": "work", "access_token": "token-unavailable", "source": "manual"}]
    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", lambda provider: entries)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "token-unavailable")
    calls = []
    snapshot = _snapshot(unavailable_reason="Usage endpoint returned HTTP 429.")
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, *, api_key=None, base_url=None: calls.append(api_key) or snapshot,
    )

    first = account_usage.fetch_pool_account_usage("anthropic")
    second = account_usage.fetch_pool_account_usage("anthropic")

    assert calls == ["token-unavailable"]
    assert first[0].snapshot.available is False
    assert second[0].snapshot is snapshot


def test_pool_usage_caches_none_as_unavailable_snapshot(monkeypatch):
    entries = [{"id": "a", "label": "work", "access_token": "token-none", "source": "manual"}]
    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", lambda provider: entries)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "token-none")
    calls = []
    monkeypatch.setattr(
        account_usage,
        "fetch_account_usage",
        lambda provider, *, api_key=None, base_url=None: calls.append(api_key) or None,
    )

    first = account_usage.fetch_pool_account_usage("anthropic")
    second = account_usage.fetch_pool_account_usage("anthropic")

    assert calls == ["token-none"]
    assert first[0].snapshot.available is False
    assert first[0].snapshot.unavailable_reason == "Usage lookup unavailable."
    assert second[0].snapshot is first[0].snapshot


def test_pool_entries_sharing_stale_stored_fingerprint_do_not_share_cache(monkeypatch):
    """Two entries with the same *stored* secret_fingerprint but different live
    tokens must each get their own usage lookup (cache keyed on the live
    token, not the stored metadata)."""
    account_usage._POOL_USAGE_CACHE.clear()
    entries = [
        {"id": "a", "label": "acct-a", "access_token": "tok-a", "secret_fingerprint": "sha256:stale", "last_status": "ok"},
        {"id": "b", "label": "acct-b", "access_token": "tok-b", "secret_fingerprint": "sha256:stale", "last_status": "ok"},
    ]
    monkeypatch.setattr("hermes_cli.auth.read_credential_pool", lambda provider: entries)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "tok-a")
    seen = []

    def fake_fetch(provider, api_key=None, **kwargs):
        seen.append(api_key)
        return account_usage.AccountUsageSnapshot(
            provider=provider, source="t", fetched_at=account_usage._utc_now(),
            unavailable_reason=f"from:{api_key}",
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)
    readings = account_usage.fetch_pool_account_usage("anthropic")

    assert seen == ["tok-a", "tok-b"]
    assert [r.snapshot.unavailable_reason for r in readings] == ["from:tok-a", "from:tok-b"]
    assert readings[0].fingerprint != readings[1].fingerprint
    assert [r.active for r in readings] == [True, False]
