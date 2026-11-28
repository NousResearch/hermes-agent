"""Kimi Code quota wire contracts (PR #74424, 17 October review)."""

from datetime import UTC, datetime

import httpx
import pytest

from agent import account_usage


_RESET = "2030-01-02T03:04:05Z"


def _fetch(monkeypatch, payload):
    calls = []

    def get(self, url, *, headers):
        calls.append((str(url), headers["Authorization"]))
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.Client, "get", get)
    snapshot = account_usage.fetch_account_usage(
        "kimi-coding", base_url="https://api.kimi.com/coding", api_key="synthetic-test-key",
    )
    assert snapshot is not None
    assert calls == [("https://api.kimi.com/coding/v1/usages", "Bearer synthetic-test-key")]
    return snapshot


@pytest.mark.parametrize("detail,expected", [
    ({"limit": "100", "remaining": "100"}, 0),
    ({"limit": 100, "remaining": 75}, 25),
    ({"limit": "100", "used": "0", "remaining": "95"}, 0),
    ({"limit": "100", "used": "130"}, 130),
])
def test_kimi_count_fallback_keeps_idle_windows(monkeypatch, detail, expected):
    detail = {**detail, "resetTime": _RESET}
    snapshot = _fetch(monkeypatch, {
        "usage": detail,
        "limits": [{"window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"}, "detail": detail}],
    })
    assert [(w.label, w.used_percent) for w in snapshot.windows] == [("Weekly", expected), ("5-hour", expected)]
    assert all(w.reset_at == datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC) for w in snapshot.windows)
    if expected == 0:
        assert any("5-hour: 100% remaining (0% used)" in line
                   for line in account_usage.render_account_usage_lines(snapshot))


@pytest.mark.parametrize("include_usage", [False, True])
@pytest.mark.parametrize("absent_pool", [None, "limit_7d", "limit_5h"])
@pytest.mark.parametrize("as_string", [False, True])
def test_kimi_ratio_pools_take_precedence_per_window(monkeypatch, include_usage, absent_pool, as_string):
    ratios = {"limit_7d": 0.0543, "limit_5h": 0, "limit_month_total": 1, "limit_month_code": 0.1256}
    pools = {key: {"used_ratio": str(value) if as_string else value, "reset_time": _RESET}
             for key, value in ratios.items() if key != absent_pool}
    payload = {
        "usages": pools,
        "limits": [{"window": {"duration": 5, "timeUnit": "TIME_UNIT_HOUR"},
                    "detail": {"limit": "100", "used": "25", "resetTime": "2031-01-01T00:00:00Z"}},
                   {"window": {"duration": 60, "timeUnit": "TIME_UNIT_MINUTE"},
                    "detail": {"limit": "100", "remaining": "100"}}],
    }
    if include_usage:
        payload["usage"] = {"limit": "100", "used": "5", "resetTime": "2031-01-01T00:00:00Z"}
    snapshot = _fetch(monkeypatch, payload)
    windows = {w.label: w for w in snapshot.windows}
    expected = {"Weekly": 5.43, "5-hour": 0, "Monthly total": 100, "Monthly coding": 12.56, "Hourly": 0}
    if absent_pool == "limit_7d":
        if include_usage:
            expected["Weekly"] = 5
        else:
            del expected["Weekly"]
    if absent_pool == "limit_5h":
        expected["5-hour"] = 25
    assert len(snapshot.windows) == len(expected)  # No rounded-count duplicate of a pool window.
    assert {label: w.used_percent for label, w in windows.items()} == pytest.approx(expected)
    for key, label in (("limit_7d", "Weekly"), ("limit_5h", "5-hour"),
                       ("limit_month_total", "Monthly total"), ("limit_month_code", "Monthly coding")):
        if key in pools:
            assert windows[label].reset_at == datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.mark.parametrize("pool", [
    None, [], {}, {"used_ratio": None}, {"used_ratio": True},
    {"used_ratio": "nan"}, {"used_ratio": "inf"}, {"used_ratio": "bad"},
    {"used_ratio": -0.1}, {"used_ratio": 1.01},
])
def test_kimi_invalid_reported_pool_does_not_fall_back_to_counts(monkeypatch, pool):
    snapshot = _fetch(monkeypatch, {
        "usages": {"limit_7d": pool, "limit_month_total": {"used_ratio": 0.5}},
        "usage": {"limit": 100, "used": 5},
    })
    assert [(w.label, w.used_percent) for w in snapshot.windows] == [("Monthly total", 50)]


@pytest.mark.parametrize("detail", [
    {}, {"limit": 0, "remaining": 0}, {"limit": True, "used": 0},
    {"limit": "inf", "used": 0}, {"limit": 100, "used": "nan"},
    {"limit": 100, "remaining": "inf"}, {"limit": 100, "used": False},
    {"limit": 100, "used": None, "remaining": 100},
    {"limit": 100, "remaining": 101}, {"limit": 100, "used": "invalid"},
])
def test_kimi_invalid_counts_do_not_invent_idle_quota(detail):
    assert account_usage._percent_from_counts(detail) is None


@pytest.mark.parametrize("duration,unit", [
    (10080, "TIME_UNIT_MINUTE"), (168, "TIME_UNIT_HOUR"),
    (7, "TIME_UNIT_DAY"), (1, "TIME_UNIT_WEEK"),
])
def test_kimi_weekly_ratio_suppresses_equivalent_count_window(monkeypatch, duration, unit):
    snapshot = _fetch(monkeypatch, {
        "usages": {"limit_7d": {"used_ratio": 0.0543}},
        "limits": [{"window": {"duration": duration, "timeUnit": unit},
                    "detail": {"limit": 100, "used": 5}}],
    })
    assert [(w.label, w.used_percent) for w in snapshot.windows] == [("Weekly", pytest.approx(5.43))]
