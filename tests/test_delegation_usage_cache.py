"""Tests for the secret-free cached provider usage behind delegation routing.

Every fetch is mocked. These tests assert the cache never persists a token,
account/project identity, header, raw error, raw provider response, or a
credential path, and that route selection never blocks on the network.
"""

import json
import os
import stat
from datetime import datetime, timedelta, timezone

import pytest

from agent import delegation_usage_cache as duc
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow


def _snapshot(provider="openai-codex", used=40.0, fetched_at=None):
    return AccountUsageSnapshot(
        provider=provider,
        source="usage_api",
        fetched_at=fetched_at or datetime.now(timezone.utc),
        plan="Pro",
        windows=(
            AccountUsageWindow(
                label="Session",
                used_percent=used,
                reset_at=datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc),
                detail="resets soon",
            ),
        ),
    )


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(duc, "_cache_path", lambda: tmp_path / "usage.json")
    duc.reset_refresh_state()
    return duc


class TestProjection:
    def test_projection_keeps_only_normalized_fields(self, cache):
        record = cache.project_snapshot(_snapshot())
        assert set(record) == {
            "provider",
            "fetched_at",
            "source",
            "windows",
        }
        window = record["windows"][0]
        assert set(window) == {"label", "used_percent", "remaining_percent", "reset_at"}
        assert window["used_percent"] == 40.0
        assert window["remaining_percent"] == 60.0
        assert window["label"] == "Session"

    def test_projection_excludes_secret_and_identity_fields(self, cache):
        snapshot = AccountUsageSnapshot(
            provider="google-antigravity",
            source="quota_summary",
            fetched_at=datetime.now(timezone.utc),
            plan="Ultra",
            windows=(
                AccountUsageWindow(
                    label="Gemini Models",
                    used_percent=10.0,
                    detail="account bob@example.com project my-gcp-project",
                ),
            ),
            details=("token=ya29.SECRET", "/home/u/.hermes/creds.json"),
            unavailable_reason="HTTP 403 from https://x/y?key=SECRET",
        )
        blob = json.dumps(cache.project_snapshot(snapshot))
        for leaked in (
            "SECRET",
            "ya29",
            "bob@example.com",
            "my-gcp-project",
            "creds.json",
            "403",
            "Authorization",
            "Ultra",
        ):
            assert leaked not in blob, f"projection leaked {leaked!r}"

    def test_projection_drops_unrecognized_source_and_window_label(self, cache):
        snapshot = AccountUsageSnapshot(
            provider="openai-codex",
            source="account alice@example.com",
            fetched_at=datetime.now(timezone.utc),
            windows=(
                AccountUsageWindow(
                    label="alice@example.com private account",
                    used_percent=25.0,
                ),
            ),
        )
        record = cache.project_snapshot(snapshot)
        assert record["source"] == ""
        assert record["windows"][0]["label"] == ""
        assert record["windows"][0]["remaining_percent"] == 75.0
        assert "alice@example.com" not in json.dumps(record)

    @pytest.mark.parametrize("value", [10**1000, float("inf"), float("-inf"), float("nan")])
    def test_projection_rejects_non_finite_or_overflowing_percent(self, cache, value):
        record = cache.project_snapshot(_snapshot(used=value))
        assert record["windows"][0]["used_percent"] is None
        assert record["windows"][0]["remaining_percent"] is None

    def test_unavailable_snapshot_projects_no_windows(self, cache):
        snapshot = AccountUsageSnapshot(
            provider="openai-codex",
            source="usage_api",
            fetched_at=datetime.now(timezone.utc),
            unavailable_reason="boom",
        )
        record = cache.project_snapshot(snapshot)
        assert record["windows"] == []

    def test_remaining_percent_is_worst_window(self, cache):
        snapshot = AccountUsageSnapshot(
            provider="openai-codex",
            source="usage_api",
            fetched_at=datetime.now(timezone.utc),
            windows=(
                AccountUsageWindow(label="Session", used_percent=10.0),
                AccountUsageWindow(label="Weekly", used_percent=95.0),
            ),
        )
        cache.store_snapshot(snapshot)
        entry = cache.read_provider_usage("openai-codex", ttl_seconds=300, stale_seconds=1800)
        assert entry.remaining_percent == 5.0

    def test_window_prefixes_narrow_the_reading(self, cache):
        snapshot = AccountUsageSnapshot(
            provider="google-antigravity",
            source="quota_summary",
            fetched_at=datetime.now(timezone.utc),
            windows=(
                AccountUsageWindow(label="Gemini Models (5h)", used_percent=20.0),
                AccountUsageWindow(label="Claude and GPT models (5h)", used_percent=99.0),
            ),
        )
        cache.store_snapshot(snapshot)
        entry = cache.read_provider_usage(
            "google-antigravity",
            ttl_seconds=300,
            stale_seconds=1800,
            window_prefixes=("Gemini Models",),
        )
        assert entry.remaining_percent == 80.0

    def test_routes_on_same_provider_keep_independent_usage_pools(self, cache):
        snapshot = AccountUsageSnapshot(
            provider="google-antigravity",
            source="quota_summary",
            fetched_at=datetime.now(timezone.utc),
            windows=(
                AccountUsageWindow(label="Gemini Models (5h)", used_percent=20.0),
                AccountUsageWindow(
                    label="Claude and GPT models (5h)", used_percent=99.0
                ),
            ),
        )
        cache.store_snapshot(snapshot)

        class _Route:
            provider = "google-antigravity"

            def __init__(self, route_id, prefix):
                self.id = route_id
                self.usage_window_prefixes = (prefix,)

        view = cache.build_route_usage_view(
            [
                _Route("gemini", "Gemini Models"),
                _Route("third-party", "Claude and GPT models"),
            ],
            ttl_seconds=300,
            stale_seconds=1800,
            refresh=False,
        )
        assert view.entries["gemini"].remaining_percent == 80.0
        assert view.entries["third-party"].remaining_percent == 1.0

    def test_route_usage_refresh_is_deduplicated_per_provider(self, cache, monkeypatch):
        class _Route:
            provider = "google-antigravity"
            usage_window_prefixes = ()

            def __init__(self, route_id):
                self.id = route_id

        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))
        cache.build_route_usage_view(
            [_Route("one"), _Route("two")],
            ttl_seconds=300,
            stale_seconds=1800,
            refresh=True,
        )
        assert scheduled == ["google-antigravity"]


class TestAtomicWrite:
    def test_cache_file_is_0600(self, cache, tmp_path):
        cache.store_snapshot(_snapshot())
        path = tmp_path / "usage.json"
        assert path.exists()
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    def test_write_leaves_no_temp_files(self, cache, tmp_path):
        cache.store_snapshot(_snapshot())
        cache.store_snapshot(_snapshot(provider="google-antigravity"))
        leftovers = [
            p.name
            for p in tmp_path.iterdir()
            if p.name.startswith(".tmp_") or p.name.endswith(".tmp")
        ]
        assert leftovers == []

    def test_store_merges_providers(self, cache):
        cache.store_snapshot(_snapshot(provider="openai-codex"))
        cache.store_snapshot(_snapshot(provider="google-antigravity"))
        raw = cache.read_raw()
        assert set(raw["providers"]) == {"openai-codex", "google-antigravity"}

    def test_corrupt_cache_is_treated_as_missing(self, cache, tmp_path):
        (tmp_path / "usage.json").write_text("{not json")
        entry = cache.read_provider_usage("openai-codex", ttl_seconds=300, stale_seconds=1800)
        assert entry.freshness == "unknown"


class TestFreshness:
    def _store_at(self, cache, age_seconds, provider="openai-codex"):
        fetched = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
        cache.store_snapshot(_snapshot(provider=provider, fetched_at=fetched))

    def test_fresh_within_ttl(self, cache):
        self._store_at(cache, 10)
        entry = cache.read_provider_usage("openai-codex", ttl_seconds=300, stale_seconds=1800)
        assert entry.freshness == "fresh"
        assert entry.remaining_percent == 60.0
        assert 0 <= entry.age_seconds < 60

    def test_stale_between_ttl_and_stale_window(self, cache):
        self._store_at(cache, 900)
        entry = cache.read_provider_usage("openai-codex", ttl_seconds=300, stale_seconds=1800)
        assert entry.freshness == "stale"
        assert entry.remaining_percent == 60.0

    def test_expired_beyond_stale_window_is_unknown(self, cache):
        self._store_at(cache, 5000)
        entry = cache.read_provider_usage("openai-codex", ttl_seconds=300, stale_seconds=1800)
        assert entry.freshness == "unknown"
        assert entry.remaining_percent is None

    def test_missing_provider_is_unknown(self, cache):
        entry = cache.read_provider_usage("openai-codex", ttl_seconds=300, stale_seconds=1800)
        assert entry.freshness == "unknown"
        assert entry.remaining_percent is None


class TestRefreshScheduling:
    def _fake_fetch(self, calls):
        def _fetch(provider):
            calls.append(provider)
            return _snapshot(provider=provider)

        return _fetch

    def test_fresh_cache_triggers_zero_fetches(self, cache, monkeypatch):
        cache.store_snapshot(_snapshot())
        calls = []
        monkeypatch.setattr(cache, "_fetch_account_usage", self._fake_fetch(calls))
        view = cache.build_usage_view(
            ["openai-codex"], ttl_seconds=300, stale_seconds=1800, refresh=True
        )
        assert calls == []
        assert view.entries["openai-codex"].freshness == "fresh"

    def test_stale_cache_is_used_and_schedules_one_refresh(self, cache, monkeypatch):
        fetched = datetime.now(timezone.utc) - timedelta(seconds=900)
        cache.store_snapshot(_snapshot(fetched_at=fetched))
        calls = []
        monkeypatch.setattr(cache, "_fetch_account_usage", self._fake_fetch(calls))
        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))

        view = cache.build_usage_view(
            ["openai-codex"], ttl_seconds=300, stale_seconds=1800, refresh=True
        )
        assert view.entries["openai-codex"].freshness == "stale"
        assert view.entries["openai-codex"].remaining_percent == 60.0
        assert scheduled == ["openai-codex"]
        assert calls == []  # never fetched inline

    def test_expired_schedules_refresh_and_reports_unknown(self, cache, monkeypatch):
        fetched = datetime.now(timezone.utc) - timedelta(seconds=9000)
        cache.store_snapshot(_snapshot(fetched_at=fetched))
        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))
        view = cache.build_usage_view(
            ["openai-codex"], ttl_seconds=300, stale_seconds=1800, refresh=True
        )
        assert view.entries["openai-codex"].freshness == "unknown"
        assert scheduled == ["openai-codex"]

    def test_missing_cache_schedules_refresh(self, cache, monkeypatch):
        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))
        view = cache.build_usage_view(
            ["google-antigravity"], ttl_seconds=300, stale_seconds=1800, refresh=True
        )
        assert view.entries["google-antigravity"].freshness == "unknown"
        assert scheduled == ["google-antigravity"]

    def test_fresh_unavailable_snapshot_is_negative_cached_until_ttl(self, cache, monkeypatch):
        cache.store_snapshot(
            AccountUsageSnapshot(
                provider="openai-codex",
                source="usage_api",
                fetched_at=datetime.now(timezone.utc),
                unavailable_reason="not supported for this account",
            )
        )
        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))
        view = cache.build_usage_view(
            ["openai-codex"], ttl_seconds=300, stale_seconds=1800, refresh=True
        )
        assert view.entries["openai-codex"].freshness == "unknown"
        assert view.entries["openai-codex"].age_seconds is not None
        assert scheduled == []

    def test_refresh_is_deduplicated_per_provider(self, cache, monkeypatch):
        calls = []
        monkeypatch.setattr(cache, "_fetch_account_usage", self._fake_fetch(calls))
        started = []
        monkeypatch.setattr(cache, "_start_thread", lambda fn: started.append(fn))

        cache.schedule_refresh("openai-codex")
        cache.schedule_refresh("openai-codex")
        cache.schedule_refresh("openai-codex")
        assert len(started) == 1

        started[0]()  # run the worker inline
        assert calls == ["openai-codex"]

        cache.schedule_refresh("openai-codex")
        assert len(started) == 2

    def test_refresh_worker_persists_only_projection(self, cache, monkeypatch):
        def _fetch(provider):
            return AccountUsageSnapshot(
                provider=provider,
                source="usage_api",
                fetched_at=datetime.now(timezone.utc),
                plan="Team",
                windows=(AccountUsageWindow(label="Session", used_percent=25.0),),
                details=("api_key=sk-SECRET",),
            )

        monkeypatch.setattr(cache, "_fetch_account_usage", _fetch)
        cache.refresh_provider_now("openai-codex")
        blob = json.dumps(cache.read_raw())
        assert "SECRET" not in blob and "Team" not in blob
        entry = cache.read_provider_usage("openai-codex", ttl_seconds=300, stale_seconds=1800)
        assert entry.remaining_percent == 75.0

    def test_refresh_failure_does_not_raise_or_corrupt(self, cache, monkeypatch):
        def _boom(provider):
            raise RuntimeError("network down: https://api?key=SECRET")

        monkeypatch.setattr(cache, "_fetch_account_usage", _boom)
        cache.refresh_provider_now("openai-codex")  # must not raise
        assert "SECRET" not in json.dumps(cache.read_raw())

    def test_failed_missing_refresh_is_negative_cached_until_ttl(
        self, cache, monkeypatch
    ):
        monkeypatch.setattr(cache, "_fetch_account_usage", lambda _provider: None)
        cache.refresh_provider_now("openai-codex")

        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))
        view = cache.build_usage_view(
            ["openai-codex"], ttl_seconds=300, stale_seconds=1800, refresh=True
        )

        assert view.entries["openai-codex"].freshness == "unknown"
        assert view.entries["openai-codex"].age_seconds is not None
        assert scheduled == []
        blob = json.dumps(cache.read_raw())
        assert "refresh_attempted_at" in blob

    def test_failed_stale_refresh_preserves_numbers_and_backs_off(
        self, cache, monkeypatch
    ):
        fetched = datetime.now(timezone.utc) - timedelta(seconds=900)
        cache.store_snapshot(_snapshot(fetched_at=fetched))

        def _boom(_provider):
            raise RuntimeError("identity@example.com?token=SECRET")

        monkeypatch.setattr(cache, "_fetch_account_usage", _boom)
        cache.refresh_provider_now("openai-codex")

        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))
        view = cache.build_usage_view(
            ["openai-codex"], ttl_seconds=300, stale_seconds=1800, refresh=True
        )

        entry = view.entries["openai-codex"]
        assert entry.freshness == "stale"
        assert entry.remaining_percent == 60.0
        assert scheduled == []
        blob = json.dumps(cache.read_raw())
        assert "SECRET" not in blob and "identity@example.com" not in blob

    def test_refresh_disabled_makes_no_calls(self, cache, monkeypatch):
        scheduled = []
        monkeypatch.setattr(cache, "_spawn_refresh", lambda p: scheduled.append(p))
        cache.build_usage_view(
            ["openai-codex"], ttl_seconds=300, stale_seconds=1800, refresh=False
        )
        assert scheduled == []


class TestCacheLocation:
    def test_cache_lives_under_active_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        duc.reset_refresh_state()
        path = duc._cache_path()
        assert str(tmp_path) in str(path)
        assert path.name.endswith(".json")
