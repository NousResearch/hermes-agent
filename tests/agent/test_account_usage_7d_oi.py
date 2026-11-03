"""Tests for 7d_oi utilization display in the Hermes usage meter.

Covers:
- record_unified_7d_oi_headers() caches the 7d_oi-utilization header value
- get_unified_7d_oi_cache() returns the cached value (or None when absent/expired)
- _fetch_anthropic_account_usage() picks up seven_day_oi from the OAuth API payload
- _fetch_anthropic_account_usage() falls back to the header cache when the API omits it
- extract_api_error_context() calls record_unified_7d_oi_headers() on 429 responses
- record_unified_7d_oi_headers() persists the cache to a JSON file on disk
- get_unified_7d_oi_cache() lazy-loads the backing file on first access (restart simulation)
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.account_usage as account_usage
from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    get_unified_7d_oi_cache,
    record_unified_7d_oi_headers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeOAuthResponse:
    """Mimics httpx.Response for the /api/oauth/usage endpoint."""

    status_code = 200
    headers: dict = {}

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeOAuthClient:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        return _FakeOAuthResponse(self._payload)


@pytest.fixture(autouse=True)
def _reset_7d_oi_cache(tmp_path, monkeypatch):
    """Clear the 7d_oi in-memory cache and redirect the file cache to a
    temporary directory before/after each test for full isolation.

    Setting ``_UNIFIED_7D_OI_CACHE_LOADED = True`` prevents the lazy-file-load
    from running inside tests that directly seed ``_UNIFIED_7D_OI_CACHE``
    (they simulate pre-existing in-memory state, not a cold start).  Individual
    file-persistence tests explicitly reset this flag to False to exercise the
    cold-start path.
    """
    account_usage._UNIFIED_7D_OI_CACHE.clear()
    account_usage._UNIFIED_7D_OI_CACHE_LOADED = True  # skip file-load by default
    monkeypatch.setattr(
        account_usage,
        "_UNIFIED_7D_OI_CACHE_FILE",
        tmp_path / "7d_oi_rl_cache.json",
    )
    yield
    account_usage._UNIFIED_7D_OI_CACHE.clear()
    account_usage._UNIFIED_7D_OI_CACHE_LOADED = False  # restore to cold-start state


# ---------------------------------------------------------------------------
# record_unified_7d_oi_headers
# ---------------------------------------------------------------------------


def test_record_headers_stores_utilization():
    headers = {
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.85",
        "anthropic-ratelimit-unified-7d_oi-status": "allowed",
        "anthropic-ratelimit-unified-7d_oi-reset": "1900000000",
    }
    record_unified_7d_oi_headers(headers)
    cached = get_unified_7d_oi_cache()
    assert cached is not None
    assert cached["utilization"] == pytest.approx(0.85)
    assert cached["status"] == "allowed"
    assert cached["reset_at"] == pytest.approx(1900000000.0)


def test_record_headers_noop_when_header_absent():
    # Headers without the utilization key → cache stays empty
    record_unified_7d_oi_headers({"other-header": "value"})
    assert get_unified_7d_oi_cache() is None


def test_record_headers_noop_on_bad_value():
    record_unified_7d_oi_headers({"anthropic-ratelimit-unified-7d_oi-utilization": "not-a-float"})
    assert get_unified_7d_oi_cache() is None


def test_record_headers_case_insensitive():
    # HTTP headers are case-insensitive; our parsing should lower-case them.
    headers = {"Anthropic-Ratelimit-Unified-7d_oi-Utilization": "0.42"}
    record_unified_7d_oi_headers(headers)
    cached = get_unified_7d_oi_cache()
    assert cached is not None
    assert cached["utilization"] == pytest.approx(0.42)


def test_get_cache_returns_none_when_expired(monkeypatch):
    account_usage._UNIFIED_7D_OI_CACHE["_global"] = {
        "utilization": 0.5, "captured_at": time.time() - 4000.0
    }
    # TTL is 3600 s; 4000 s ago → expired
    assert get_unified_7d_oi_cache() is None


def test_get_cache_returns_value_when_fresh():
    account_usage._UNIFIED_7D_OI_CACHE["_global"] = {
        "utilization": 0.9, "reset_at": None, "status": "rejected", "captured_at": time.time()
    }
    cached = get_unified_7d_oi_cache()
    assert cached is not None
    assert cached["utilization"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# _fetch_anthropic_account_usage — OAuth API includes seven_day_oi
# ---------------------------------------------------------------------------


def _make_oauth_payload(**windows):
    """Build a minimal /api/oauth/usage payload."""
    payload: dict = {}
    for key, utilization in windows.items():
        payload[key] = {"utilization": utilization, "resets_at": None}
    return payload


def test_fetch_anthropic_usage_shows_seven_day_oi_from_api(monkeypatch):
    """When the OAuth API returns seven_day_oi, the snapshot includes it."""
    payload = _make_oauth_payload(
        five_hour=20.0,
        seven_day=40.0,
        seven_day_oi=75.0,  # ← the key we added to the mapping
    )
    monkeypatch.setattr(account_usage, "_is_oauth_token", lambda t: True)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "sk-ant-oauth-token")
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeOAuthClient(payload),
    )

    snap = account_usage._fetch_anthropic_account_usage()

    assert snap is not None
    labels = [w.label for w in snap.windows]
    assert "Output-input week" in labels, f"Expected 'Output-input week' in {labels}"
    oi_win = next(w for w in snap.windows if w.label == "Output-input week")
    assert oi_win.used_percent == pytest.approx(75.0)


def test_fetch_anthropic_usage_falls_back_to_header_cache(monkeypatch):
    """When the OAuth API omits seven_day_oi, the header cache supplements it."""
    # No seven_day_oi in the API response
    payload = _make_oauth_payload(five_hour=10.0, seven_day=30.0)
    monkeypatch.setattr(account_usage, "_is_oauth_token", lambda t: True)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "sk-ant-oauth-token")
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeOAuthClient(payload),
    )

    # Seed the cache as if a 429 was seen earlier (utilization 0-1 fraction)
    account_usage._UNIFIED_7D_OI_CACHE["_global"] = {
        "utilization": 0.95,
        "reset_at": 1900000000.0,
        "status": "rejected",
        "captured_at": time.time(),
    }

    snap = account_usage._fetch_anthropic_account_usage()

    assert snap is not None
    labels = [w.label for w in snap.windows]
    assert "Output-input week" in labels, f"Expected 'Output-input week' in {labels}"
    oi_win = next(w for w in snap.windows if w.label == "Output-input week")
    # Cache fraction 0.95 → 95 percent
    assert oi_win.used_percent == pytest.approx(95.0)
    expected_reset = datetime.fromtimestamp(1900000000.0, tz=timezone.utc)
    assert oi_win.reset_at == expected_reset


def test_fetch_anthropic_api_takes_precedence_over_cache(monkeypatch):
    """API-returned seven_day_oi wins over cached header value."""
    payload = _make_oauth_payload(seven_day_oi=50.0)
    monkeypatch.setattr(account_usage, "_is_oauth_token", lambda t: True)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "sk-ant-oauth-token")
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeOAuthClient(payload),
    )
    # Conflicting cache value
    account_usage._UNIFIED_7D_OI_CACHE["_global"] = {
        "utilization": 0.99, "reset_at": None, "status": None, "captured_at": time.time()
    }

    snap = account_usage._fetch_anthropic_account_usage()
    assert snap is not None
    oi_wins = [w for w in snap.windows if w.label == "Output-input week"]
    assert len(oi_wins) == 1, "Exactly one Output-input week window expected"
    # API returned 50 % → should NOT be overridden by cache's 99 %
    assert oi_wins[0].used_percent == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# extract_api_error_context records 7d_oi headers on 429
# ---------------------------------------------------------------------------


def test_extract_api_error_context_caches_7d_oi_on_429():
    """extract_api_error_context() calls record_unified_7d_oi_headers() when present."""
    from agent.agent_runtime_helpers import extract_api_error_context

    mock_headers = {
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.88",
        "anthropic-ratelimit-unified-7d_oi-status": "allowed",
        "anthropic-ratelimit-unified-7d_oi-reset": "1950000000",
        "anthropic-ratelimit-unified-reset": "1950000000",
    }
    mock_response = SimpleNamespace(headers=mock_headers)
    mock_error = Exception("rate limited")
    mock_error.response = mock_response  # type: ignore[attr-defined]
    mock_error.body = None  # type: ignore[attr-defined]

    extract_api_error_context(mock_error)

    cached = get_unified_7d_oi_cache()
    assert cached is not None, "7d_oi headers should have been cached"
    assert cached["utilization"] == pytest.approx(0.88)
    assert cached["status"] == "allowed"
    assert cached["reset_at"] == pytest.approx(1950000000.0)


def test_extract_api_error_context_no_cache_without_7d_oi_header():
    """extract_api_error_context() must NOT write to cache when 7d_oi header absent."""
    from agent.agent_runtime_helpers import extract_api_error_context

    mock_headers = {
        "anthropic-ratelimit-unified-reset": "1950000000",
        "retry-after": "3600",
    }
    mock_response = SimpleNamespace(headers=mock_headers)
    mock_error = Exception("rate limited")
    mock_error.response = mock_response  # type: ignore[attr-defined]
    mock_error.body = None  # type: ignore[attr-defined]

    extract_api_error_context(mock_error)

    assert get_unified_7d_oi_cache() is None


# ---------------------------------------------------------------------------
# File-based persistent cache
# ---------------------------------------------------------------------------


def test_record_headers_persists_to_file(tmp_path):
    """record_unified_7d_oi_headers() writes the cache to the backing JSON file."""
    cache_file = tmp_path / "7d_oi_rl_cache.json"
    account_usage._UNIFIED_7D_OI_CACHE_FILE = cache_file

    headers = {
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.73",
        "anthropic-ratelimit-unified-7d_oi-status": "allowed",
        "anthropic-ratelimit-unified-7d_oi-reset": "1900000000",
    }
    record_unified_7d_oi_headers(headers)

    assert cache_file.exists(), "Cache file must be created after record call"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    # New format: top-level key is the fingerprint ("_global" when none passed)
    entry = data.get("_global", data)  # tolerate both new and old callers in assertions
    assert entry["utilization"] == pytest.approx(0.73)
    assert entry["status"] == "allowed"
    assert entry["reset_at"] == pytest.approx(1900000000.0)
    assert "captured_at" in entry


def test_get_cache_loads_from_file_on_first_access(tmp_path):
    """get_unified_7d_oi_cache() lazy-loads the backing file on first call."""
    cache_file = tmp_path / "7d_oi_rl_cache.json"
    # Write a fresh-looking cache entry directly to the file (simulating data
    # that was recorded in a previous process run) in the new fingerprint-keyed
    # format.
    payload = {
        "_global": {
            "utilization": 0.61,
            "reset_at": 1900000000.0,
            "status": "allowed",
            "captured_at": time.time(),  # fresh → must NOT be considered expired
        }
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    # Redirect the module to use our tmp file and simulate a cold start.
    account_usage._UNIFIED_7D_OI_CACHE_FILE = cache_file
    account_usage._UNIFIED_7D_OI_CACHE_LOADED = False

    cached = get_unified_7d_oi_cache()
    assert cached is not None, "Cache must be loaded from file"
    assert cached["utilization"] == pytest.approx(0.61)
    assert cached["status"] == "allowed"


def test_get_cache_returns_none_when_file_entry_expired(tmp_path):
    """get_unified_7d_oi_cache() ignores a stale file entry (TTL expired)."""
    cache_file = tmp_path / "7d_oi_rl_cache.json"
    stale_payload = {
        "_global": {
            "utilization": 0.99,
            "reset_at": None,
            "status": "rejected",
            "captured_at": time.time() - 4000.0,  # 4000 s ago → beyond 3600 s TTL
        }
    }
    cache_file.write_text(json.dumps(stale_payload), encoding="utf-8")

    account_usage._UNIFIED_7D_OI_CACHE_FILE = cache_file
    account_usage._UNIFIED_7D_OI_CACHE_LOADED = False

    assert get_unified_7d_oi_cache() is None, "Expired file entry must return None"


def test_file_persistence_survives_restart_cycle(tmp_path):
    """Full round-trip: write via record_unified_7d_oi_headers, then load into a
    fresh in-memory state (simulating a process restart)."""
    cache_file = tmp_path / "7d_oi_rl_cache.json"
    account_usage._UNIFIED_7D_OI_CACHE_FILE = cache_file

    # First 'process': record headers → persists to file.
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "0.55"}
    )
    assert cache_file.exists()

    # Simulate restart: wipe in-memory state, keep file intact.
    account_usage._UNIFIED_7D_OI_CACHE.clear()
    account_usage._UNIFIED_7D_OI_CACHE_LOADED = False

    # Second 'process': get_unified_7d_oi_cache must restore from file.
    cached = get_unified_7d_oi_cache()
    assert cached is not None, "Cache must be restored from file after restart"
    assert cached["utilization"] == pytest.approx(0.55)


def test_file_io_failure_does_not_break_record(tmp_path):
    """A bad cache file path must not raise — record_unified_7d_oi_headers
    should complete silently even when the file write fails."""
    # Point to a location that cannot be created (root-owned dir on macOS/Linux).
    account_usage._UNIFIED_7D_OI_CACHE_FILE = Path("/root/no_permission/7d_oi.json")

    # Must not raise despite the unwritable path.
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "0.3"}
    )

    # In-memory cache must still be populated.
    cached = get_unified_7d_oi_cache()
    assert cached is not None
    assert cached["utilization"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Bug 1: Lazy-load race condition — record sets CACHE_LOADED inside the lock
# ---------------------------------------------------------------------------


def test_record_headers_sets_cache_loaded_flag():
    """record_unified_7d_oi_headers() must set _UNIFIED_7D_OI_CACHE_LOADED=True
    inside the lock so a concurrent get_unified_7d_oi_cache() cannot see
    LOADED=False and trigger a file reload that overwrites new values."""
    account_usage._UNIFIED_7D_OI_CACHE_LOADED = False  # simulate cold start
    # record should flip the flag even without a get call first
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "0.70"}
    )
    assert account_usage._UNIFIED_7D_OI_CACHE_LOADED is True, (
        "_UNIFIED_7D_OI_CACHE_LOADED must be True after record_unified_7d_oi_headers"
    )
    # And the value must be readable without triggering another file load
    cached = get_unified_7d_oi_cache()
    assert cached is not None
    assert cached["utilization"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Bug 2: Cross-credential cache isolation
# ---------------------------------------------------------------------------


def test_cross_credential_cache_isolation():
    """Two different credential fingerprints must not share cache entries."""
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "0.20"},
        credential_fingerprint="sha256:aaa",
    )
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "0.80"},
        credential_fingerprint="sha256:bbb",
    )

    cached_a = get_unified_7d_oi_cache("sha256:aaa")
    cached_b = get_unified_7d_oi_cache("sha256:bbb")
    cached_global = get_unified_7d_oi_cache()  # "_global" bucket — must be empty

    assert cached_a is not None and cached_a["utilization"] == pytest.approx(0.20)
    assert cached_b is not None and cached_b["utilization"] == pytest.approx(0.80)
    assert cached_global is None, "\"_global\" bucket must stay empty when fingerprints differ"


# ---------------------------------------------------------------------------
# Bug 3: utilization / reset_at input validation
# ---------------------------------------------------------------------------


def test_record_headers_ignores_nan_utilization():
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "nan"}
    )
    assert get_unified_7d_oi_cache() is None, "NaN utilization must not be cached"


def test_record_headers_ignores_inf_utilization():
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "inf"}
    )
    assert get_unified_7d_oi_cache() is None, "Inf utilization must not be cached"


def test_record_headers_ignores_out_of_range_utilization():
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "1.5"}
    )
    assert get_unified_7d_oi_cache() is None, "utilization > 1.0 must not be cached"


def test_record_headers_rejects_negative_reset_at():
    """A negative reset_at must be dropped; utilization is still cached."""
    record_unified_7d_oi_headers({
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.50",
        "anthropic-ratelimit-unified-7d_oi-reset": "-100",
    })
    cached = get_unified_7d_oi_cache()
    assert cached is not None, "Valid utilization must still be cached"
    assert cached["reset_at"] is None, "Negative reset_at must be dropped to None"


def test_record_headers_rejects_nan_reset_at():
    record_unified_7d_oi_headers({
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.50",
        "anthropic-ratelimit-unified-7d_oi-reset": "nan",
    })
    cached = get_unified_7d_oi_cache()
    assert cached is not None
    assert cached["reset_at"] is None, "NaN reset_at must be dropped to None"


# ---------------------------------------------------------------------------
# Bug 4: malformed captured_at must not raise
# ---------------------------------------------------------------------------




def test_get_cache_handles_malformed_captured_at():
    """get_unified_7d_oi_cache() must not raise on non-numeric captured_at;
    it should treat it as 0 (epoch) and return None because that falls outside
    the TTL window (time.time() - 0 >> 3600)."""
    account_usage._UNIFIED_7D_OI_CACHE["_global"] = {
        "utilization": 0.77,
        "reset_at": None,
        "status": None,
        "captured_at": "not-a-number",  # malformed
    }
    # Must not raise, must return None (treated as expired)
    result = get_unified_7d_oi_cache()
    assert result is None, "Malformed captured_at must be treated as expired (return None)"


# ---------------------------------------------------------------------------
# Codex review 2026-11-03: 9 additional tests
# ---------------------------------------------------------------------------


def test_concurrent_record_no_corruption():
    """Two threads simultaneously calling record_unified_7d_oi_headers() with
    different fingerprints must not corrupt each other's cache entries."""
    errors: list = []

    def writer_a() -> None:
        try:
            for _ in range(20):
                record_unified_7d_oi_headers(
                    {"anthropic-ratelimit-unified-7d_oi-utilization": "0.10"},
                    credential_fingerprint="sha256:concurrent-A",
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def writer_b() -> None:
        try:
            for _ in range(20):
                record_unified_7d_oi_headers(
                    {"anthropic-ratelimit-unified-7d_oi-utilization": "0.90"},
                    credential_fingerprint="sha256:concurrent-B",
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=writer_a)
    t2 = threading.Thread(target=writer_b)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, f"Threads raised exceptions: {errors}"
    cached_a = get_unified_7d_oi_cache("sha256:concurrent-A")
    cached_b = get_unified_7d_oi_cache("sha256:concurrent-B")
    assert cached_a is not None, "fingerprint-A entry must be present"
    assert cached_b is not None, "fingerprint-B entry must be present"
    assert cached_a["utilization"] == pytest.approx(0.10)
    assert cached_b["utilization"] == pytest.approx(0.90)


def test_atomic_save_exception_cleans_tmp(tmp_path, monkeypatch):
    """When os.replace() raises, _save_7d_oi_cache_to_file must delete the
    orphaned tmp file so the cache directory stays clean."""
    monkeypatch.setattr(
        account_usage,
        "_UNIFIED_7D_OI_CACHE_FILE",
        tmp_path / "7d_oi_rl_cache.json",
    )

    with patch("os.replace", side_effect=OSError("disk full")):
        record_unified_7d_oi_headers(
            {"anthropic-ratelimit-unified-7d_oi-utilization": "0.50"}
        )

    orphaned = list(tmp_path.glob("*.tmp"))
    assert not orphaned, f"Orphaned .tmp files must be removed: {orphaned}"
    # In-memory cache must still be populated despite the write failure.
    cached = get_unified_7d_oi_cache()
    assert cached is not None
    assert cached["utilization"] == pytest.approx(0.50)


def test_lazy_load_idempotent_concurrent(tmp_path, monkeypatch):
    """Two threads concurrently calling get_unified_7d_oi_cache() on first
    access (CACHE_LOADED=False) must trigger _load_7d_oi_cache_from_file()
    exactly once — the lock prevents a double-load."""
    monkeypatch.setattr(
        account_usage,
        "_UNIFIED_7D_OI_CACHE_FILE",
        tmp_path / "7d_oi_rl_cache.json",
    )
    account_usage._UNIFIED_7D_OI_CACHE_LOADED = False

    load_calls: list = []
    original_load = account_usage._load_7d_oi_cache_from_file

    def counting_load() -> None:
        load_calls.append(1)
        original_load()

    monkeypatch.setattr(account_usage, "_load_7d_oi_cache_from_file", counting_load)

    def reader() -> None:
        get_unified_7d_oi_cache()

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert len(load_calls) == 1, (
        f"_load_7d_oi_cache_from_file must be called exactly once; "
        f"called {len(load_calls)} time(s)"
    )


def test_record_sets_cache_loaded_true():
    """record_unified_7d_oi_headers() must set _UNIFIED_7D_OI_CACHE_LOADED=True
    inside the lock so a racing get() cannot trigger a stale file-reload."""
    account_usage._UNIFIED_7D_OI_CACHE_LOADED = False  # cold-start state
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "0.60"}
    )
    assert account_usage._UNIFIED_7D_OI_CACHE_LOADED is True, (
        "_UNIFIED_7D_OI_CACHE_LOADED must be True after record_unified_7d_oi_headers"
    )


def test_cross_credential_isolation():
    """A value recorded under fingerprint A must not appear when queried under
    fingerprint B.  The global ("_global") bucket must also remain empty.

    Note: the current implementation stores entries in a fingerprint-keyed dict
    (_UNIFIED_7D_OI_CACHE), so isolation is provided by default.  If future
    changes collapse this to a single global slot, this test will catch the
    regression.
    """
    # TODO: extend to multi-account pool validation once the pool meter UI
    #       gains per-credential display rows.
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "0.15"},
        credential_fingerprint="sha256:iso-A",
    )

    result_b = get_unified_7d_oi_cache("sha256:iso-B")
    result_global = get_unified_7d_oi_cache()  # "_global" bucket

    assert result_b is None, "fingerprint B must not see fingerprint A's value"
    assert result_global is None, '"_global" bucket must stay empty when a fingerprint was used'


def test_malformed_utilization_nan():
    """utilization='nan' must not be recorded in the cache."""
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "nan"}
    )
    assert get_unified_7d_oi_cache() is None, "NaN utilization must be rejected"


def test_malformed_utilization_over_1():
    """utilization='2.0' (> 1.0) must not be recorded in the cache."""
    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "2.0"}
    )
    assert get_unified_7d_oi_cache() is None, "utilization > 1.0 must be rejected"


def test_malformed_reset_negative():
    """reset_at='-999' must be stored as None; valid utilization must still be cached."""
    record_unified_7d_oi_headers({
        "anthropic-ratelimit-unified-7d_oi-utilization": "0.40",
        "anthropic-ratelimit-unified-7d_oi-reset": "-999",
    })
    cached = get_unified_7d_oi_cache()
    assert cached is not None, "Valid utilization must still be cached"
    assert cached["reset_at"] is None, "Negative reset_at must be coerced to None"
    assert cached["utilization"] == pytest.approx(0.40)


def test_stale_zero_update():
    """A new header with utilization=0.0 must overwrite an existing 0.5 entry
    and update captured_at.  Zero is a valid utilization value and must not be
    silently treated as falsy/empty."""
    old_ts = time.time() - 10.0  # 10 s ago — still within TTL
    account_usage._UNIFIED_7D_OI_CACHE["_global"] = {
        "utilization": 0.5,
        "reset_at": None,
        "status": None,
        "captured_at": old_ts,
    }

    record_unified_7d_oi_headers(
        {"anthropic-ratelimit-unified-7d_oi-utilization": "0.0"}
    )

    cached = get_unified_7d_oi_cache()
    assert cached is not None, "Cache must not be empty after recording 0.0"
    assert cached["utilization"] == pytest.approx(0.0), "0.0 must overwrite the previous 0.5"
    assert cached["captured_at"] > old_ts, "captured_at must be updated to the new timestamp"
