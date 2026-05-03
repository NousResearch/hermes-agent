from datetime import datetime, timedelta, timezone

from agent.codex_quota import (
    CodexQuotaResult,
    CodexQuotaWindow,
    classify_quota_usage,
    compute_elapsed_percent,
    fetch_codex_quota,
    format_codex_quota_full,
    format_remaining,
    format_window_metadata,
    parse_codex_quota_payload,
    render_quota_bar_text,
)


class _Response:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self._payload


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self.calls.append((url, headers or {}))
        return self.response


def test_compute_elapsed_percent_from_reset_window():
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    reset_at = now + timedelta(hours=2)
    assert compute_elapsed_percent(
        used_percent=22,
        limit_window_seconds=5 * 3600,
        reset_at=reset_at,
        now=now,
    ) == 60.0


def test_compute_elapsed_percent_falls_back_to_usage_when_timing_missing():
    assert compute_elapsed_percent(used_percent=37, limit_window_seconds=None, reset_at=None) == 37


def test_render_quota_bar_cursor_and_fill_are_separate():
    # 25% usage over a 10-cell bar fills 2 cells; 50% elapsed puts cursor at index 4.
    assert render_quota_bar_text(25, 50, width=10) == "━━━━|━━━━━"


def test_classify_quota_usage_thresholds_and_over_pace():
    assert classify_quota_usage(91, 90) == "error"
    assert classify_quota_usage(75, 80) == "warning"
    assert classify_quota_usage(61, 40) == "warning"
    assert classify_quota_usage(55, 50) == "accent"
    assert classify_quota_usage(30, 40) == "success"


def test_remaining_time_formatting():
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert format_remaining(now - timedelta(seconds=1), now=now) == "now"
    assert format_remaining(now + timedelta(minutes=42), now=now) == "42m"
    assert format_remaining(now + timedelta(hours=3, minutes=13), now=now) == "3h13m"
    assert format_remaining(now + timedelta(days=2, hours=6), now=now) == "2d6h"


def test_format_window_metadata_compact_removes_statusbar_waste():
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    window = CodexQuotaWindow(
        key="primary_window",
        label="Primary",
        used_percent=2.0,
        elapsed_percent=7,
        reset_at=now + timedelta(hours=4, minutes=39),
    )
    assert format_window_metadata(window, now=now) == "2% · 4h39m→2:39pm"
    assert format_window_metadata(window, now=now, compact=True) == "2%·4h39m→2:39pm"


def test_parse_codex_quota_payload_handles_primary_secondary_and_elapsed():
    now = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    result = parse_codex_quota_payload(
        {
            "plan_type": "team",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 22.4,
                    "limit_window_seconds": 18000,
                    "reset_at": int((now + timedelta(hours=3, minutes=13)).timestamp()),
                },
                "secondary_window": {
                    "used_percent": 46.8,
                    "limit_window_seconds": 604800,
                    "reset_at": int((now + timedelta(days=2, hours=6)).timestamp()),
                },
            },
        },
        now=now,
    )
    assert result.ok
    assert result.plan_type == "team"
    assert [w.key for w in result.windows] == ["primary_window", "secondary_window"]
    assert round(result.windows[0].used_percent) == 22
    assert result.windows[0].elapsed_percent > 0


def test_format_codex_quota_full_reports_missing_windows():
    result = CodexQuotaResult(ok=True, fetched_at=datetime.now(timezone.utc), windows=())
    assert format_codex_quota_full(result) == "codex quota unavailable"


def test_fetch_codex_quota_auth_expired(monkeypatch):
    monkeypatch.setattr(
        "agent.codex_quota.resolve_codex_runtime_credentials",
        lambda refresh_if_expiring=True: {"api_key": "token"},
    )
    monkeypatch.setattr("agent.codex_quota._read_codex_tokens", lambda: {"tokens": {}})
    monkeypatch.setattr("agent.codex_quota.httpx.Client", lambda timeout=8.0: _Client(_Response(status_code=403)))

    result = fetch_codex_quota()

    assert not result.ok
    assert result.error == "codex auth expired"
    assert result.status_code == 403


def test_fetch_codex_quota_missing_credentials(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr("agent.codex_quota.resolve_codex_runtime_credentials", _raise)

    result = fetch_codex_quota()

    assert not result.ok
    assert result.error == "no codex auth"


def test_fetch_codex_quota_malformed_response(monkeypatch):
    monkeypatch.setattr(
        "agent.codex_quota.resolve_codex_runtime_credentials",
        lambda refresh_if_expiring=True: {"api_key": "token"},
    )
    monkeypatch.setattr("agent.codex_quota._read_codex_tokens", lambda: {"tokens": {}})
    monkeypatch.setattr("agent.codex_quota.httpx.Client", lambda timeout=8.0: _Client(_Response(payload=[])))

    result = fetch_codex_quota()

    assert not result.ok
    assert result.error == "codex quota: malformed response"
