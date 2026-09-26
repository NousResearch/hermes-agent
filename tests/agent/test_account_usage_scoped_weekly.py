"""Anthropic OAuth usage: account-wide windows from ``limits[]``, per-model weekly caps beside them."""

from agent.account_usage import _anthropic_account_windows, _anthropic_scoped_weekly_windows


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


def test_account_windows_read_percent_not_a_fraction():
    """A 1%-used window is 1%, never 100%.

    ``utilization`` and ``percent`` are the same whole-number percent. Reading a value <= 1 as a
    0-1 fraction turned the first percent of every fresh window into a false "limit reached".
    """
    payload = {"limits": [_limit("session", 1), _limit("weekly_all", 59)],
               "five_hour": {"utilization": 1.0}, "seven_day": {"utilization": 59.0}}
    assert [(w.label, w.used_percent) for w in _anthropic_account_windows(payload)] == [
        ("Current session", 1.0), ("Current week", 59.0)]
    # Same contract on the legacy-only shape, where utilization is the sole source.
    legacy_only = {"five_hour": {"utilization": 1.0}, "seven_day": {"utilization": 0.0}}
    assert [(w.label, w.used_percent) for w in _anthropic_account_windows(legacy_only)] == [
        ("Current session", 1.0), ("Current week", 0.0)]


def test_account_windows_prefer_limits_and_fall_back_per_window():
    """``limits[]`` wins where present; the legacy keys fill only the windows it omits."""
    payload = {"limits": [_limit("session", 2), _limit("weekly_scoped", 100, model="Fable")],
               "five_hour": {"utilization": 77.0}, "seven_day": {"utilization": 59.0},
               "seven_day_opus": {"utilization": 12.0}}
    assert [(w.label, w.used_percent) for w in _anthropic_account_windows(payload)] == [
        ("Current session", 2.0), ("Current week", 59.0), ("Opus week", 12.0)]


def test_scoped_limits_never_supply_an_account_window():
    """A model-scoped entry must not be mistaken for the account-wide window of the same kind."""
    payload = {"limits": [_limit("session", 100, model="Fable")], "five_hour": {"utilization": 3.0}}
    assert [(w.label, w.used_percent) for w in _anthropic_account_windows(payload)] == [("Current session", 3.0)]
