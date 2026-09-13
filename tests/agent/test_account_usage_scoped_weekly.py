"""Anthropic OAuth usage: per-model weekly caps from ``limits[]`` become their own windows."""

from agent.account_usage import _anthropic_scoped_weekly_windows


def _limit(kind, percent, model=None, resets_at="2026-09-13T21:00:00+00:00"):
    scope = {"model": {"id": None, "display_name": model}, "surface": None} if model else None
    return {"kind": kind, "group": "weekly", "percent": percent, "severity": "normal",
            "resets_at": resets_at, "scope": scope, "is_active": bool(model)}


def test_scoped_weekly_limits_become_model_week_windows():
    payload = {"limits": [_limit("session", 36), _limit("weekly_all", 30), _limit("weekly_scoped", 57, model="Fable")]}
    windows = _anthropic_scoped_weekly_windows(payload)
    assert [(w.label, w.used_percent) for w in windows] == [("Fable week", 57.0)]
    assert windows[0].reset_at is not None and windows[0].reset_at.year == 2026


def test_unscoped_and_malformed_limit_entries_are_ignored():
    payload = {"limits": [_limit("weekly_scoped", 57), {"kind": "weekly_scoped"}, "junk", None,
                          {"kind": "weekly_scoped", "percent": "n/a", "scope": {"model": {"display_name": "X"}}}]}
    assert _anthropic_scoped_weekly_windows(payload) == []
    assert _anthropic_scoped_weekly_windows({}) == []
