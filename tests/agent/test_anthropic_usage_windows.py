"""Anthropic OAuth usage windows: utilization from /api/oauth/usage is already a percent.

The API returns utilization on a 0-100 scale (reporter cross-checked the weekly window
against the VS Code extension: 29-30%). Passing ``fraction=True`` into ``_usage_windows``
double-scales any value <= 1 (0.5 -> 50.0, 1.0 -> 100.0), so a nearly-unused session
shows as 100%. These tests pin pass-through semantics at the HTTP seam (``_get_json``
stubbed, no network) and do not assert anything about wire shape beyond the documented
utilization/ resets_at fields the parser already consumes.
"""

from agent import account_usage


def _stub_anthropic_fetch(monkeypatch, payload):
    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "sk-ant-oat01-fake")
    monkeypatch.setattr(account_usage, "_is_oauth_token", lambda token: True)
    monkeypatch.setattr(account_usage, "_get_json", lambda url, headers, *, timeout: payload)


def test_anthropic_utilization_at_or_below_one_is_not_rescaled(monkeypatch):
    """Gap: small utilization values must stay as returned (0.5% stays 0.5, 1.0 stays 1.0)."""
    _stub_anthropic_fetch(monkeypatch, {
        "five_hour": {"utilization": 0.5, "resets_at": "2026-10-15T20:00:00Z"},
        "seven_day": {"utilization": 1.0, "resets_at": "2026-10-20T00:00:00Z"},
    })

    snapshot = account_usage._fetch_anthropic_account_usage()

    assert snapshot is not None
    windows = {w.label: w for w in snapshot.windows}
    assert windows["Current session"].used_percent == 0.5
    assert windows["Current week"].used_percent == 1.0


def test_anthropic_utilization_above_one_passes_through_unchanged(monkeypatch):
    """Regression guard: values above 1 were never scaled and must stay untouched."""
    _stub_anthropic_fetch(monkeypatch, {
        "five_hour": {"utilization": 1.1, "resets_at": "2026-10-15T20:00:00Z"},
        "seven_day": {"utilization": 29.0, "resets_at": "2026-10-20T00:00:00Z"},
        "seven_day_opus": {"utilization": 42.5, "resets_at": "2026-10-20T00:00:00Z"},
    })

    snapshot = account_usage._fetch_anthropic_account_usage()

    assert snapshot is not None
    windows = {w.label: w for w in snapshot.windows}
    assert windows["Current session"].used_percent == 1.1
    assert windows["Current week"].used_percent == 29.0
    assert windows["Opus week"].used_percent == 42.5
    assert windows["Opus week"].scope == "model"
