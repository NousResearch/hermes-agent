"""Tests for the quota warning engine (agent/quota_warnings.py).

Pure unit tests over plain dict configs — never touch load_config()
(that's Task A's surface). Monkeypatching targets
``agent.quota_warnings.*`` so the module's own namespace is the seam.
"""

import threading
from datetime import datetime, timedelta, timezone

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.quota_warnings import (
    QuotaThresholds,
    clear_quota_cache,
    fetch_quota_snapshot,
    fetch_quota_snapshot_bounded,
    get_quota_warnings,
    quota_thresholds,
    quota_warning_lines,
    startup_warning_lines,
)

_NOW = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)


def _window(label="Session", used_percent=None, reset_at=None, detail=None):
    return AccountUsageWindow(
        label=label,
        used_percent=used_percent,
        reset_at=reset_at,
        detail=detail,
    )


def _snapshot(windows=(), plan=None, unavailable_reason=None):
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=_NOW,
        title="Account limits",
        plan=plan,
        windows=tuple(windows),
        details=(),
        unavailable_reason=unavailable_reason,
    )


DT_80 = QuotaThresholds(80, 90, 95)


# ── quota_thresholds ──────────────────────────────────────────────────────


def test_quota_thresholds_parses_config_values():
    cfg = {
        "quota": {
            "warning_threshold": 75,
            "strong_threshold": 88,
            "critical_threshold": 92,
        }
    }
    t = quota_thresholds(cfg)
    assert t == QuotaThresholds(75, 88, 92)
    assert isinstance(t.warning, float)
    assert isinstance(t.strong, float)
    assert isinstance(t.critical, float)
    assert t.warning == 75.0
    assert t.strong == 88.0
    assert t.critical == 92.0


def test_quota_thresholds_missing_keys_use_defaults():
    t1 = quota_thresholds({})
    t2 = quota_thresholds({"quota": {}})
    t3 = quota_thresholds(None)
    for t in (t1, t2, t3):
        assert t == QuotaThresholds(80, 90, 95)
        assert t.warning == 80.0
        assert t.strong == 90.0
        assert t.critical == 95.0


def test_quota_thresholds_garbage_values_fall_back_to_defaults():
    cfg = {
        "quota": {
            "warning_threshold": "abc",
            "strong_threshold": None,
            "critical_threshold": "nope",
        }
    }
    t = quota_thresholds(cfg)
    assert t == QuotaThresholds(80, 90, 95)


def test_quota_thresholds_non_dict_quota_section_uses_defaults():
    assert quota_thresholds({"quota": "not-a-dict"}) == QuotaThresholds(80, 90, 95)
    assert quota_thresholds({"quota": None}) == QuotaThresholds(80, 90, 95)


def test_quota_thresholds_accepts_floaty_strings():
    cfg = {"quota": {"warning_threshold": "77.5"}}
    t = quota_thresholds(cfg)
    assert t.warning == 77.5


# ── get_quota_warnings: threshold boundaries ──────────────────────────────


def test_get_quota_warnings_none_snapshot_returns_empty():
    assert get_quota_warnings(None, thresholds=DT_80) == []


def test_get_quota_warnings_below_warning_returns_empty():
    snap = _snapshot(windows=[_window(used_percent=79.9)])
    assert get_quota_warnings(snap, thresholds=DT_80) == []


def test_get_quota_warnings_exact_boundary_warning():
    snap = _snapshot(windows=[_window(used_percent=80)])
    lines = get_quota_warnings(snap, thresholds=DT_80)
    assert lines == ["  ⚠ Quota warning: 80% used (threshold 80%)"]


def test_get_quota_warnings_exact_boundary_strong():
    snap = _snapshot(windows=[_window(used_percent=90)])
    lines = get_quota_warnings(snap, thresholds=DT_80)
    assert lines == ["  ⚠⚠ Strong quota warning: 90% used (threshold 90%)"]


def test_get_quota_warnings_exact_boundary_critical():
    snap = _snapshot(windows=[_window(used_percent=95)])
    lines = get_quota_warnings(snap, thresholds=DT_80)
    assert lines == ["  🚨 Critical quota warning: 95% used (threshold 95%)"]


def test_get_quota_warnings_just_below_strong_still_warning():
    # 80 <= pct < 90 → warning (90.0 would flip to strong)
    snap = _snapshot(windows=[_window(used_percent=89.9)])
    assert get_quota_warnings(snap, thresholds=DT_80) == [
        "  ⚠ Quota warning: 90% used (threshold 80%)",
    ]


def test_get_quota_warnings_max_window_semantics():
    # max across windows wins, single line for the highest level reached.
    assert get_quota_warnings(_snapshot(windows=[_window(used_percent=70), _window(used_percent=85)]), thresholds=DT_80) == [
        "  ⚠ Quota warning: 85% used (threshold 80%)",
    ]
    assert get_quota_warnings(_snapshot(windows=[_window(used_percent=70), _window(used_percent=92)]), thresholds=DT_80) == [
        "  ⚠⚠ Strong quota warning: 92% used (threshold 90%)",
    ]
    assert get_quota_warnings(_snapshot(windows=[_window(used_percent=70), _window(used_percent=98)]), thresholds=DT_80) == [
        "  🚨 Critical quota warning: 98% used (threshold 95%)",
    ]


def test_get_quota_warnings_skips_none_pct_windows():
    # All windows have None pct → no usable value → [].
    snap = _snapshot(windows=[_window(used_percent=None), _window(used_percent=None)])
    assert get_quota_warnings(snap, thresholds=DT_80) == []


def test_get_quota_warnings_mixed_none_and_real_takes_real():
    # None-pct windows are skipped; the real one drives the result.
    snap = _snapshot(windows=[_window(used_percent=None), _window(used_percent=85)])
    assert get_quota_warnings(snap, thresholds=DT_80) == [
        "  ⚠ Quota warning: 85% used (threshold 80%)",
    ]


def test_get_quota_warnings_empty_windows_returns_empty():
    assert get_quota_warnings(_snapshot(windows=()), thresholds=DT_80) == []


def test_get_quota_warnings_unavailable_snapshot_returns_empty():
    snap = _snapshot(
        windows=[_window(used_percent=99)],
        unavailable_reason="not authenticated",
    )
    assert get_quota_warnings(snap, thresholds=DT_80) == []


def test_get_quota_warnings_appends_reset_when_present(monkeypatch):
    monkeypatch.setattr("agent.account_usage._utc_now", lambda: _NOW)
    reset_at = _NOW + timedelta(hours=2, minutes=30)
    snap = _snapshot(windows=[_window(used_percent=85, reset_at=reset_at)])
    lines = get_quota_warnings(snap, thresholds=DT_80)
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("  ⚠ Quota warning: 85% used (threshold 80%)")
    assert " — resets in 2h 30m" in line


def test_get_quota_warnings_no_reset_when_absent():
    snap = _snapshot(windows=[_window(used_percent=85, reset_at=None)])
    lines = get_quota_warnings(snap, thresholds=DT_80)
    assert lines == ["  ⚠ Quota warning: 85% used (threshold 80%)"]


def test_get_quota_warnings_rounds_pct_to_int():
    snap = _snapshot(windows=[_window(used_percent=84.6)])
    lines = get_quota_warnings(snap, thresholds=DT_80)
    # 84.6 rounds to 85 for display, 84.6 >= 80 → warning
    assert lines == ["  ⚠ Quota warning: 85% used (threshold 80%)"]


def test_get_quota_warnings_custom_thresholds():
    thresholds = QuotaThresholds(70, 80, 90)
    snap = _snapshot(windows=[_window(used_percent=75)])
    lines = get_quota_warnings(snap, thresholds=thresholds)
    assert lines == ["  ⚠ Quota warning: 75% used (threshold 70%)"]


# ── quota_warning_lines (suppression-aware) ───────────────────────────────


def test_quota_warning_lines_suppressed_returns_empty():
    cfg = {"quota": {"suppress_warnings": True}}
    snap = _snapshot(windows=[_window(used_percent=98)])  # critical
    assert quota_warning_lines(snap, config=cfg) == []


def test_quota_warning_lines_not_suppressed_returns_warning():
    cfg = {"quota": {"suppress_warnings": False}}
    snap = _snapshot(windows=[_window(used_percent=98)])
    lines = quota_warning_lines(snap, config=cfg)
    assert lines == ["  🚨 Critical quota warning: 98% used (threshold 95%)"]


def test_quota_warning_lines_no_config_shows_warning():
    snap = _snapshot(windows=[_window(used_percent=85)])
    assert quota_warning_lines(snap) == ["  ⚠ Quota warning: 85% used (threshold 80%)"]


def test_quota_warning_lines_none_snapshot_returns_empty():
    assert quota_warning_lines(None) == []
    assert quota_warning_lines(None, config={"quota": {"suppress_warnings": True}}) == []


# ── startup_warning_lines (ignores suppression) ───────────────────────────


def test_startup_warning_lines_ignores_suppression():
    cfg = {"quota": {"suppress_warnings": True}}
    snap = _snapshot(windows=[_window(used_percent=98)])  # critical
    lines = startup_warning_lines(snap, config=cfg)
    assert lines == ["  🚨 Critical quota warning: 98% used (threshold 95%)"]


def test_startup_warning_lines_shows_warning_when_enabled():
    cfg = {"quota": {"suppress_warnings": False}}
    snap = _snapshot(windows=[_window(used_percent=85)])
    assert startup_warning_lines(snap, config=cfg) == [
        "  ⚠ Quota warning: 85% used (threshold 80%)",
    ]


def test_startup_warning_lines_none_snapshot_returns_empty():
    assert startup_warning_lines(None) == []


# ── fetch_quota_snapshot: TTL cache ────────────────────────────────────────


def _make_fake_snapshot(pct=85):
    return _snapshot(windows=[_window(used_percent=pct)])


def test_cache_within_max_age_calls_fetch_once(monkeypatch):
    calls = []

    def fake(provider, *, base_url=None, api_key=None):
        calls.append((provider, base_url))
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_account_usage", fake)
    clear_quota_cache()
    s1 = fetch_quota_snapshot("openai-codex")
    s2 = fetch_quota_snapshot("openai-codex")
    assert s1 is s2  # same cached object
    assert len(calls) == 1


def test_cache_refetches_when_max_age_zero(monkeypatch):
    calls = []

    def fake(provider, *, base_url=None, api_key=None):
        calls.append(provider)
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_account_usage", fake)
    clear_quota_cache()
    fetch_quota_snapshot("openai-codex", max_age=0.0)
    fetch_quota_snapshot("openai-codex", max_age=0.0)
    assert len(calls) == 2


def test_clear_quota_cache_forces_refetch(monkeypatch):
    calls = []

    def fake(provider, *, base_url=None, api_key=None):
        calls.append(provider)
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_account_usage", fake)
    clear_quota_cache()
    fetch_quota_snapshot("openai-codex")
    assert len(calls) == 1
    clear_quota_cache()
    fetch_quota_snapshot("openai-codex")
    assert len(calls) == 2


def test_cache_distinguishes_base_url(monkeypatch):
    calls = []

    def fake(provider, *, base_url=None, api_key=None):
        calls.append((provider, base_url))
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_account_usage", fake)
    clear_quota_cache()
    fetch_quota_snapshot("openai-codex", base_url="https://a")
    fetch_quota_snapshot("openai-codex", base_url="https://b")  # different key
    fetch_quota_snapshot("openai-codex", base_url="https://a")  # cached
    assert len(calls) == 2
    assert ("openai-codex", "https://a") in calls
    assert ("openai-codex", "https://b") in calls


def test_cache_distinguishes_api_key(monkeypatch):
    # Two credentials for the same provider/base_url must NOT share a cached
    # snapshot — a cross-account stale-data leak (cross-vendor review).
    calls = []

    def fake(provider, *, base_url=None, api_key=None):
        calls.append((provider, base_url, api_key))
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_account_usage", fake)
    clear_quota_cache()
    # First credential → 1 fetch.
    s1a = fetch_quota_snapshot("openai-codex", base_url="https://x", api_key="key-a")
    s1b = fetch_quota_snapshot("openai-codex", base_url="https://x", api_key="key-a")
    assert s1a is s1b  # same cached object for same key
    assert len(calls) == 1
    # Second credential → different key, must refetch (NOT a cache hit).
    s2 = fetch_quota_snapshot("openai-codex", base_url="https://x", api_key="key-b")
    assert s2 is not s1a
    assert len(calls) == 2
    # First credential still served from its own cache entry.
    s1c = fetch_quota_snapshot("openai-codex", base_url="https://x", api_key="key-a")
    assert s1c is s1a
    assert len(calls) == 2


def test_cache_does_not_cache_fetch_failures(monkeypatch):
    calls = []
    state = {"fail": True}

    def fake(provider, *, base_url=None, api_key=None):
        calls.append(provider)
        if state["fail"]:
            raise RuntimeError("network down")
        return _make_fake_snapshot(85)

    monkeypatch.setattr("agent.quota_warnings.fetch_account_usage", fake)
    clear_quota_cache()
    # First call: fetch raises → None, nothing cached.
    assert fetch_quota_snapshot("openai-codex", max_age=600.0) is None
    assert len(calls) == 1
    # Second call within TTL would normally hit cache, but failure wasn't
    # cached so it retries.
    assert fetch_quota_snapshot("openai-codex", max_age=600.0) is None
    assert len(calls) == 2
    # Now recovery — retries and caches the success.
    state["fail"] = False
    result = fetch_quota_snapshot("openai-codex", max_age=600.0)
    assert result is not None
    assert len(calls) == 3
    # Subsequent call within TTL hits the cache (no new fetch).
    fetch_quota_snapshot("openai-codex", max_age=600.0)
    assert len(calls) == 3


def test_cache_wraps_none_fetch_result_and_does_not_cache_none(monkeypatch):
    calls = []

    def fake(provider, *, base_url=None, api_key=None):
        calls.append(provider)
        return None  # unsupported provider / no creds

    monkeypatch.setattr("agent.quota_warnings.fetch_account_usage", fake)
    clear_quota_cache()
    assert fetch_quota_snapshot("anthropic", max_age=600.0) is None
    # None is not cached (same fail-open reasoning: retry next time).
    assert fetch_quota_snapshot("anthropic", max_age=600.0) is None
    assert len(calls) == 2


# ── Bool threshold rejection ────────────────────────────────────────────


def test_quota_thresholds_scalar_bool_falls_back_to_default():
    # bool would coerce to 1.0/0.0 via float(); it must fall back to default.
    t = quota_thresholds({"quota": {"warning_threshold": True}})
    assert t.warning == 80.0
    assert t.strong == 90.0
    assert t.critical == 95.0


def test_quota_thresholds_all_bool_falls_back_to_defaults():
    cfg = {
        "quota": {
            "warning_threshold": True,
            "strong_threshold": False,
            "critical_threshold": True,
        }
    }
    t = quota_thresholds(cfg)
    assert t == QuotaThresholds(80, 90, 95)


# ── Misordered / out-of-range thresholds ──────────────────────────────


def test_quota_warning_lines_misordered_thresholds_returns_empty():
    # warning > strong > critical is invalid ordering; no mislabeled line.
    cfg = {
        "quota": {
            "warning_threshold": 95,
            "strong_threshold": 90,
            "critical_threshold": 80,
        }
    }
    snap = _snapshot(windows=[_window(used_percent=85)])  # mislabeled strong/critical
    assert quota_warning_lines(snap, config=cfg) == []
    assert startup_warning_lines(snap, config=cfg) == []


def test_quota_warning_lines_out_of_range_warning_returns_empty():
    cfg = {"quota": {"warning_threshold": -5}}
    snap = _snapshot(windows=[_window(used_percent=85)])
    assert quota_warning_lines(snap, config=cfg) == []


def test_quota_warning_lines_out_of_range_critical_returns_empty():
    cfg = {"quota": {"critical_threshold": 150}}
    snap = _snapshot(windows=[_window(used_percent=85)])
    assert quota_warning_lines(snap, config=cfg) == []


def test_startup_warning_lines_out_of_range_critical_returns_empty():
    cfg = {"quota": {"critical_threshold": 150}}
    snap = _snapshot(windows=[_window(used_percent=98)])
    assert startup_warning_lines(snap, config=cfg) == []


# ── Strict suppress_warnings: only boolean True suppresses ──────────────


def test_quota_warning_lines_string_false_not_suppressed():
    cfg = {"quota": {"suppress_warnings": "false"}}
    snap = _snapshot(windows=[_window(used_percent=85)])
    assert quota_warning_lines(snap, config=cfg) == [
        "  ⚠ Quota warning: 85% used (threshold 80%)",
    ]


def test_quota_warning_lines_boolean_true_suppressed():
    cfg = {"quota": {"suppress_warnings": True}}
    snap = _snapshot(windows=[_window(used_percent=98)])
    assert quota_warning_lines(snap, config=cfg) == []


# ── fetch_quota_snapshot_bounded: in-flight future reuse ────────────────


def test_bounded_fetch_reuses_in_flight_future(monkeypatch):
    """A probe whose previous fetch for the same key is still running must
    reuse that future, not spawn a second executor/thread (review feedback:
    per-turn executors stacked one stuck worker per turn)."""
    calls = []
    gate = threading.Event()

    def slow_fetch(provider, *, base_url=None, api_key=None):
        calls.append((provider, base_url, api_key))
        gate.wait(timeout=5.0)
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_quota_snapshot", slow_fetch)
    clear_quota_cache()
    try:
        # First probe: fetch starts and blocks past the wait bound.
        s1 = fetch_quota_snapshot_bounded(
            "openai-codex", base_url="https://x", api_key="key-a", timeout=0.1
        )
        assert s1 is None  # timed out, fetch still in flight
        assert len(calls) == 1
        # Second probe while the first fetch is still stuck: must reuse the
        # in-flight future — no second fetch, no second thread.
        s2 = fetch_quota_snapshot_bounded(
            "openai-codex", base_url="https://x", api_key="key-a", timeout=0.1
        )
        assert s2 is None
        assert len(calls) == 1  # the reuse assertion
    finally:
        gate.set()  # unblock the worker so it exits (no thread leak)


def test_bounded_fetch_replaces_completed_future(monkeypatch):
    """A completed future must be replaced (and its idle worker shut down),
    so finished probes leave no lingering executor threads and the next probe
    fetches fresh rather than reusing a stale future forever."""
    calls = []

    def fast_fetch(provider, *, base_url=None, api_key=None):
        calls.append(provider)
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_quota_snapshot", fast_fetch)
    clear_quota_cache()
    s1 = fetch_quota_snapshot_bounded("openai-codex", timeout=5.0)
    s2 = fetch_quota_snapshot_bounded("openai-codex", timeout=5.0)
    assert s1 is not None
    assert s2 is not None
    assert len(calls) == 2  # second probe started a fresh fetch


def test_clear_quota_cache_shuts_down_in_flight(monkeypatch):
    """Session reset must shut down in-flight probes: the next probe starts
    fresh instead of reusing (or being blocked by) the pre-clear future."""
    calls = []
    gate = threading.Event()

    def slow_fetch(provider, *, base_url=None, api_key=None):
        calls.append(provider)
        gate.wait(timeout=5.0)
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_quota_snapshot", slow_fetch)
    clear_quota_cache()
    try:
        assert fetch_quota_snapshot_bounded("openai-codex", timeout=0.1) is None
        assert len(calls) == 1
        clear_quota_cache()
        assert fetch_quota_snapshot_bounded("openai-codex", timeout=0.1) is None
        assert len(calls) == 2  # fresh fetch after clear, dead entry not reused
    finally:
        gate.set()  # release both blocked workers


def test_bounded_fetch_per_key_isolation(monkeypatch):
    """Distinct credential triples must NOT share an in-flight future: a
    blocked fetch for one key must not prevent another key from fetching
    (per-key isolation — a global single worker would wedge all probes)."""
    calls = []
    gate = threading.Event()

    def slow_fetch(provider, *, base_url=None, api_key=None):
        calls.append((provider, api_key))
        gate.wait(timeout=5.0)
        return _make_fake_snapshot()

    monkeypatch.setattr("agent.quota_warnings.fetch_quota_snapshot", slow_fetch)
    clear_quota_cache()
    try:
        fetch_quota_snapshot_bounded("provider-a", api_key="k1", timeout=0.1)
        fetch_quota_snapshot_bounded("provider-b", api_key="k2", timeout=0.1)
        assert len(calls) == 2  # each key fetches independently
    finally:
        gate.set()

