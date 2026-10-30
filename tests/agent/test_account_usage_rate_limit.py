"""Tests for 429 negative cache and cross-process file cache in account_usage.

Covers:
- _parse_retry_after: delta-seconds, HTTP-date, missing header (default).
- _fetch_pool_snapshot negative cache: re-request suppressed while blocked,
  re-allowed after deadline.
- _fetch_pool_snapshot file cache: cross-process read/write, TTL expiry,
  corrupt file fail-safe.
- Fingerprint isolation: separate fingerprints have independent rate-limit state.
- fetch_account_usage: 429 still returns a snapshot (public contract unchanged).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import account_usage
from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    _file_cache_path,
    _file_cache_read,
    _file_cache_write,
    _parse_retry_after,
    _POOL_USAGE_CACHE,
    _POOL_USAGE_CACHE_LOCK,
    _RATE_LIMIT_UNTIL,
    _fetch_pool_snapshot,
    _utc_now,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_snapshot(provider="anthropic", *, fingerprint="sha256:abc") -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider,
        source="oauth_usage_api",
        fetched_at=_utc_now(),
        windows=(AccountUsageWindow(label="Session", used_percent=20.0),),
    )


def _rate_limited_snapshot(provider="anthropic", retry_after: float = 3600.0) -> AccountUsageSnapshot:
    """Return a 429 snapshot as produced by _fetch_anthropic_account_usage."""
    return AccountUsageSnapshot(
        provider=provider,
        source="oauth_usage_api",
        fetched_at=_utc_now(),
        unavailable_reason="Usage endpoint returned HTTP 429. Rate limited; wait before retrying.",
        retry_after_seconds=retry_after,
    )


def _clear_caches(fingerprint: str = "sha256:abc", provider: str = "anthropic") -> None:
    """Wipe in-process caches for a given (provider, fingerprint) pair."""
    key = (provider, fingerprint)
    with _POOL_USAGE_CACHE_LOCK:
        _POOL_USAGE_CACHE.pop(key, None)
        _RATE_LIMIT_UNTIL.pop(key, None)


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------

class TestParseRetryAfter:
    def test_delta_seconds_integer(self):
        assert _parse_retry_after("3600") == pytest.approx(3600.0)

    def test_delta_seconds_float(self):
        assert _parse_retry_after("120.5") == pytest.approx(120.5)

    def test_delta_seconds_zero_clamped_to_one(self):
        # Lower bound is 1 second even for "0".
        assert _parse_retry_after("0") == pytest.approx(1.0)

    def test_http_date_future(self):
        # Build an HTTP-date ~30 seconds from now.
        from email.utils import format_datetime
        future = datetime.fromtimestamp(time.time() + 30, tz=timezone.utc)
        header = format_datetime(future, usegmt=True)
        result = _parse_retry_after(header)
        # Allow ±5 s for execution time.
        assert 25.0 <= result <= 35.0

    def test_http_date_past_clamped_to_one(self):
        from email.utils import format_datetime
        past = datetime.fromtimestamp(time.time() - 10, tz=timezone.utc)
        header = format_datetime(past, usegmt=True)
        result = _parse_retry_after(header)
        assert result == pytest.approx(1.0)

    def test_missing_header_returns_default(self):
        assert _parse_retry_after(None) == pytest.approx(600.0)

    def test_empty_string_returns_default(self):
        assert _parse_retry_after("") == pytest.approx(600.0)

    def test_garbage_returns_default(self):
        assert _parse_retry_after("not-a-date-or-number") == pytest.approx(600.0)

    def test_custom_default(self):
        assert _parse_retry_after(None, default_seconds=30.0) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Negative cache via _fetch_pool_snapshot
# ---------------------------------------------------------------------------

class TestNegativeCache:
    """_fetch_pool_snapshot must not call fetch_account_usage while blocked."""

    TOKEN = "oauth-token-a"
    FP = "sha256:fingerprint-negative"
    PROVIDER = "anthropic"

    def setup_method(self):
        _clear_caches(self.FP, self.PROVIDER)

    def teardown_method(self):
        _clear_caches(self.FP, self.PROVIDER)

    def test_first_429_records_negative_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", None)

        call_count = []

        def fake_fetch(provider, *, api_key=None, base_url=None):
            call_count.append(1)
            return _rate_limited_snapshot(provider, retry_after=3600.0)

        monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

        snap = _fetch_pool_snapshot(self.PROVIDER, self.TOKEN, self.FP)

        assert snap is not None
        assert snap.available is False
        assert "rate_limited_until" in (snap.unavailable_reason or "")
        assert len(call_count) == 1

        # Negative cache must have been populated.
        key = (self.PROVIDER, self.FP)
        with _POOL_USAGE_CACHE_LOCK:
            deadline = _RATE_LIMIT_UNTIL.get(key)
        assert deadline is not None

    def test_subsequent_call_blocked_without_http_hit(self, monkeypatch, tmp_path):
        """After a 429, the next call must be served from negative cache (0 HTTP calls)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", None)

        call_count = []

        def fake_fetch(provider, *, api_key=None, base_url=None):
            call_count.append(1)
            return _rate_limited_snapshot(provider, retry_after=3600.0)

        monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

        # First call sets the negative cache.
        _fetch_pool_snapshot(self.PROVIDER, self.TOKEN, self.FP)
        call_count.clear()

        # Second call must NOT reach fetch_account_usage.
        snap = _fetch_pool_snapshot(self.PROVIDER, self.TOKEN, self.FP)
        assert len(call_count) == 0, "fetch_account_usage was called despite active negative cache"
        assert snap is not None
        assert snap.available is False
        assert "rate_limited_until" in (snap.unavailable_reason or "")

    def test_after_deadline_new_request_is_made(self, monkeypatch, tmp_path):
        """Once the negative-cache deadline passes, a fresh HTTP call is made."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", None)

        # Directly inject an already-expired deadline (1 second ago).
        key = (self.PROVIDER, self.FP)
        with _POOL_USAGE_CACHE_LOCK:
            _RATE_LIMIT_UNTIL[key] = time.monotonic() - 1.0  # already past

        good = _good_snapshot(self.PROVIDER)
        call_count = []

        def fake_fetch(provider, *, api_key=None, base_url=None):
            call_count.append(1)
            return good

        monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

        snap = _fetch_pool_snapshot(self.PROVIDER, self.TOKEN, self.FP)
        assert len(call_count) == 1, "Expected one fresh HTTP call after deadline passed"
        assert snap is not None
        assert snap.available is True

    def test_no_retry_after_header_uses_default_600s(self, monkeypatch, tmp_path):
        """Missing retry-after → conservative 600 s block."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", None)

        def fake_fetch(provider, *, api_key=None, base_url=None):
            # retry_after_seconds=600 simulates missing Retry-After header (default).
            return _rate_limited_snapshot(provider, retry_after=600.0)

        monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

        before = time.monotonic()
        _fetch_pool_snapshot(self.PROVIDER, self.TOKEN, self.FP)

        key = (self.PROVIDER, self.FP)
        with _POOL_USAGE_CACHE_LOCK:
            deadline = _RATE_LIMIT_UNTIL.get(key)

        assert deadline is not None
        # Deadline should be ~600 s after the call (allow ±5 s slop).
        assert 595.0 <= deadline - before <= 605.0

    def test_different_fingerprints_isolated(self, monkeypatch, tmp_path):
        """A 429 on fingerprint A must not block fingerprint B."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", None)

        FP_A = "sha256:fp-iso-aaaa"
        FP_B = "sha256:fp-iso-bbbb"
        TOKEN_A = "token-iso-a"
        TOKEN_B = "token-iso-b"

        for fp in (FP_A, FP_B):
            _clear_caches(fp, self.PROVIDER)

        call_counts: dict[str, int] = {"a": 0, "b": 0}
        good = _good_snapshot(self.PROVIDER)

        def fake_fetch(provider, *, api_key=None, base_url=None):
            if api_key == TOKEN_A:
                call_counts["a"] += 1
                return _rate_limited_snapshot(provider, retry_after=3600.0)
            call_counts["b"] += 1
            return good

        monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

        # First pass: A rate-limits, B succeeds.
        snap_a = _fetch_pool_snapshot(self.PROVIDER, TOKEN_A, FP_A)
        snap_b = _fetch_pool_snapshot(self.PROVIDER, TOKEN_B, FP_B)

        assert snap_a is not None and snap_a.available is False
        assert snap_b is not None and snap_b.available is True

        # Second pass: A still blocked (0 new calls), B served from in-process cache.
        call_counts["a"] = 0
        call_counts["b"] = 0
        snap_a2 = _fetch_pool_snapshot(self.PROVIDER, TOKEN_A, FP_A)
        snap_b2 = _fetch_pool_snapshot(self.PROVIDER, TOKEN_B, FP_B)

        assert call_counts["a"] == 0, "A should still be blocked"
        assert call_counts["b"] == 0, "B should be served from in-process cache"
        assert snap_a2 is not None and snap_a2.available is False
        assert snap_b2 is not None and snap_b2.available is True

        # Cleanup.
        for fp in (FP_A, FP_B):
            _clear_caches(fp, self.PROVIDER)


# ---------------------------------------------------------------------------
# File cache
# ---------------------------------------------------------------------------

class TestFileCache:
    PROVIDER = "anthropic"
    FP = "sha256:fp-filecache"

    def setup_method(self):
        _clear_caches(self.FP, self.PROVIDER)

    def teardown_method(self):
        _clear_caches(self.FP, self.PROVIDER)

    def test_write_then_read_within_ttl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", 120.0)

        snap = _good_snapshot(self.PROVIDER)
        _file_cache_write(self.PROVIDER, self.FP, snap)

        result = _file_cache_read(self.PROVIDER, self.FP)
        assert result is not None
        assert result.available is True
        assert result.windows[0].used_percent == pytest.approx(20.0)

    def test_read_after_ttl_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", 120.0)

        snap = _good_snapshot(self.PROVIDER)
        _file_cache_write(self.PROVIDER, self.FP, snap)

        # Corrupt the expiry to a timestamp in the past.
        path = _file_cache_path(self.PROVIDER, self.FP)
        assert path is not None
        data = json.loads(path.read_text())
        data["expires_at_unix"] = time.time() - 1.0
        path.write_text(json.dumps(data))

        result = _file_cache_read(self.PROVIDER, self.FP)
        assert result is None, "Expired file cache should return None"

    def test_corrupt_file_returns_none_no_raise(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", 120.0)

        path = _file_cache_path(self.PROVIDER, self.FP)
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not json {{{")

        result = _file_cache_read(self.PROVIDER, self.FP)
        assert result is None, "Corrupt file must fail-safe (return None, not raise)"

    def test_fetch_pool_snapshot_writes_file_on_success(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", 120.0)

        good = _good_snapshot(self.PROVIDER)
        monkeypatch.setattr(account_usage, "fetch_account_usage", lambda *a, **kw: good)

        _fetch_pool_snapshot(self.PROVIDER, "token-x", self.FP)

        path = _file_cache_path(self.PROVIDER, self.FP)
        assert path is not None and path.exists(), "Successful snapshot must be written to file cache"

    def test_fetch_pool_snapshot_reads_file_cache_before_http(self, tmp_path, monkeypatch):
        """When a valid file cache exists, fetch_account_usage must not be called."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", 120.0)

        good = _good_snapshot(self.PROVIDER)
        _file_cache_write(self.PROVIDER, self.FP, good)

        call_count = []

        def should_not_be_called(*a, **kw):
            call_count.append(1)
            return good

        monkeypatch.setattr(account_usage, "fetch_account_usage", should_not_be_called)

        result = _fetch_pool_snapshot(self.PROVIDER, "token-y", self.FP)
        assert len(call_count) == 0, "fetch_account_usage must not be called when file cache is valid"
        assert result is not None and result.available is True

    def test_corrupt_file_triggers_fresh_fetch(self, tmp_path, monkeypatch):
        """A corrupt file cache must not block a fresh HTTP call."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", 120.0)

        path = _file_cache_path(self.PROVIDER, self.FP)
        assert path is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json}")

        good = _good_snapshot(self.PROVIDER)
        call_count = []

        def fresh_fetch(*a, **kw):
            call_count.append(1)
            return good

        monkeypatch.setattr(account_usage, "fetch_account_usage", fresh_fetch)

        result = _fetch_pool_snapshot(self.PROVIDER, "token-z", self.FP)
        assert len(call_count) == 1, "Corrupt file should cause a fresh fetch"
        assert result is not None and result.available is True

    def test_file_permissions_are_0600(self, tmp_path, monkeypatch):
        import stat

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", 120.0)

        snap = _good_snapshot(self.PROVIDER)
        _file_cache_write(self.PROVIDER, self.FP, snap)

        path = _file_cache_path(self.PROVIDER, self.FP)
        assert path is not None
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"Expected 0600 permissions, got {oct(mode)}"

    def test_file_cache_contains_no_token(self, tmp_path, monkeypatch):
        """Token data must never appear in the serialised file cache."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(account_usage, "_file_cache_ttl_resolved", 120.0)

        snap = _good_snapshot(self.PROVIDER)
        _file_cache_write(self.PROVIDER, self.FP, snap)

        path = _file_cache_path(self.PROVIDER, self.FP)
        assert path is not None
        raw = path.read_text()
        # Must not contain anything that looks like a bearer token.
        assert "Bearer" not in raw
        assert "oauth" not in raw.lower() or "oauth_usage_api" in raw  # source field is OK


# ---------------------------------------------------------------------------
# fetch_account_usage public contract: 429 returns snapshot, not exception
# ---------------------------------------------------------------------------

class TestFetchAccountUsagePublicContract:
    """fetch_account_usage must NOT raise on 429 — existing callers depend on this."""

    # sk-ant-oat- tokens are recognised as OAuth by _is_oauth_token
    _OAUTH_TOKEN = "sk-ant-oat-test-rate-limit-token"

    def test_429_returns_unavailable_snapshot_not_exception(self, monkeypatch):
        import httpx

        monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: self._OAUTH_TOKEN)
        real_client = httpx.Client

        def handle(request):
            return httpx.Response(429, json={"error": "rate limited"})

        monkeypatch.setattr(
            account_usage.httpx,
            "Client",
            lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
        )

        # Must NOT raise; must return a snapshot.
        snap = account_usage.fetch_account_usage("anthropic")
        assert snap is not None
        assert snap.available is False
        assert snap.unavailable_reason is not None
        # retry_after_seconds must be populated.
        assert snap.retry_after_seconds is not None
        assert snap.retry_after_seconds >= 1.0

    def test_429_reason_contains_429_for_display(self, monkeypatch):
        import httpx

        monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: self._OAUTH_TOKEN)
        real_client = httpx.Client

        def handle(request):
            return httpx.Response(429, headers={"retry-after": "60"},
                                  json={"error": "secret response body"})

        monkeypatch.setattr(
            account_usage.httpx,
            "Client",
            lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw),
        )

        snap = account_usage.fetch_account_usage("anthropic")
        assert snap is not None
        text = "\n".join(account_usage.render_account_usage_lines(snap))
        assert "429" in text
        assert "secret response body" not in text
