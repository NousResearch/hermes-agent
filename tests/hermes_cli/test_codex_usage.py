from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import httpx

from hermes_cli.codex_usage import (
    CodexUsageSnapshot,
    CodexUsageWindow,
    fetch_codex_usage_snapshot,
    format_codex_usage_report_lines,
    format_codex_usage_summary,
)


class _Response:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_fetch_codex_usage_snapshot_parses_windows_and_plan():
    def _get(_url, *, headers, timeout):
        assert headers["Authorization"] == "Bearer token"
        assert headers["User-Agent"] == "CodexBar"
        assert timeout == 5.0
        return _Response(
            200,
            {
                "rate_limit": {
                    "primary_window": {
                        "limit_window_seconds": 10_800,
                        "used_percent": 35.5,
                        "reset_at": 1_700_000_000,
                    },
                    "secondary_window": {
                        "limit_window_seconds": 604_800,
                        "used_percent": 75,
                        "reset_at": 1_700_500_000,
                    },
                },
                "plan_type": "Plus",
                "credits": {"balance": "12.5"},
            },
        )

    snapshot = fetch_codex_usage_snapshot("token", timeout=5.0, get=_get)

    assert snapshot.available is True
    assert snapshot.plan == "Plus ($12.50)"
    assert snapshot.windows == [
        CodexUsageWindow(label="3h", used_percent=35.5, reset_at_ms=1_700_000_000_000),
        CodexUsageWindow(label="Week", used_percent=75.0, reset_at_ms=1_700_500_000_000),
    ]


def test_fetch_codex_usage_snapshot_prefers_weekly_cadence_over_day_label():
    snapshot = fetch_codex_usage_snapshot(
        "token",
        get=lambda *_args, **_kwargs: _Response(
            200,
            {
                "rate_limit": {
                    "primary_window": {
                        "limit_window_seconds": 10_800,
                        "used_percent": 14,
                        "reset_at": 1_700_000_000,
                    },
                    "secondary_window": {
                        "limit_window_seconds": 86_400,
                        "used_percent": 20,
                        "reset_at": 1_700_000_000 + 5 * 24 * 60 * 60,
                    },
                }
            },
        ),
    )

    assert [window.label for window in snapshot.windows] == ["3h", "Week"]


def test_fetch_codex_usage_snapshot_returns_unavailable_on_http_error():
    snapshot = fetch_codex_usage_snapshot(
        "token",
        get=lambda *_args, **_kwargs: _Response(401, {"error": "unauthorized"}),
    )

    assert snapshot.available is False
    assert snapshot.error == "HTTP 401"
    assert snapshot.windows == []


def test_fetch_codex_usage_snapshot_returns_unavailable_on_network_error():
    def _get(*_args, **_kwargs):
        raise httpx.ConnectError("boom")

    snapshot = fetch_codex_usage_snapshot("token", get=_get)

    assert snapshot.available is False
    assert snapshot.windows == []
    assert snapshot.error == "boom"


def test_format_codex_usage_summary_includes_remaining_and_resets():
    snapshot = CodexUsageSnapshot(
        available=True,
        plan="Plus ($12.50)",
        windows=[
            CodexUsageWindow(label="3h", used_percent=35.5, reset_at_ms=10_000_000),
            CodexUsageWindow(label="Week", used_percent=75.0, reset_at_ms=100_000_000),
        ],
    )

    summary = format_codex_usage_summary(snapshot, now_ms=1_000_000, include_resets=True)

    assert summary == "3h 64% left ⏱2h 30m · Week 25% left ⏱1d 3h"


def test_format_codex_usage_report_lines_renders_plan_and_windows():
    snapshot = CodexUsageSnapshot(
        available=True,
        plan="Plus ($12.50)",
        windows=[
            CodexUsageWindow(label="3h", used_percent=35.5, reset_at_ms=10_000_000),
            CodexUsageWindow(label="Week", used_percent=75.0, reset_at_ms=100_000_000),
        ],
    )

    lines = format_codex_usage_report_lines(snapshot, now_ms=1_000_000)

    assert lines == [
        "Codex Account Usage:",
        "  Plan: Plus ($12.50)",
        "  3h: 64% left · resets 2h 30m",
        "  Week: 25% left · resets 1d 3h",
    ]
