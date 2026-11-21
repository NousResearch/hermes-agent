"""Anthropic usage: the model-scoped weekly cap reported only in ``limits[]``."""

from __future__ import annotations

from agent import account_usage

_TOKEN = "sk-ant-oat01-test"


def _payload(**overrides):
    payload = {
        "five_hour": {"utilization": 0.0, "resets_at": None},
        "seven_day": {"utilization": 66.0, "resets_at": "2026-11-18T06:59:59+00:00"},
        "seven_day_opus": None,
        "seven_day_sonnet": None,
        "limits": [
            {"kind": "session", "group": "session", "percent": 0, "resets_at": None, "scope": None, "is_active": False},
            {"kind": "weekly_all", "group": "weekly", "percent": 66,
             "resets_at": "2026-11-18T06:59:59+00:00", "scope": None, "is_active": False},
            {"kind": "weekly_scoped", "group": "weekly", "percent": 100, "severity": "critical",
             "resets_at": "2026-11-18T06:59:59+00:00",
             "scope": {"model": {"id": None, "display_name": "Fable"}, "surface": None}, "is_active": True},
        ],
    }
    payload.update(overrides)
    return payload


def _fetch(monkeypatch, payload):
    monkeypatch.setattr(account_usage, "_get_json", lambda *a, **k: payload)
    return account_usage._fetch_anthropic_account_usage(api_key=_TOKEN)


def _by_label(snapshot):
    return {w.label: w for w in snapshot.windows}


def test_weekly_scoped_limit_becomes_model_window(monkeypatch):
    snap = _fetch(monkeypatch, _payload())
    win = _by_label(snap)["Fable week"]
    assert win.used_percent == 100.0
    assert win.scope == "model"
    assert win.reset_at is not None and win.reset_at.year == 2026
    assert win.detail == "limit active"
    # The account-wide windows are untouched.
    assert {"Current session", "Current week"} <= set(_by_label(snap))


def test_scoped_limit_without_model_name_gets_generic_label(monkeypatch):
    payload = _payload()
    payload["limits"][2]["scope"] = None
    snap = _fetch(monkeypatch, payload)
    assert _by_label(snap)["Model week"].used_percent == 100.0


def test_flat_model_keys_take_precedence_over_limits(monkeypatch):
    payload = _payload(seven_day_opus={"utilization": 40.0, "resets_at": "2026-11-18T06:59:59+00:00"})
    payload["limits"][2]["scope"]["model"]["display_name"] = "Opus"
    snap = _fetch(monkeypatch, payload)
    assert [w.used_percent for w in snap.windows if w.label == "Opus week"] == [40.0]


def test_flat_model_window_does_not_hide_other_scoped_limit(monkeypatch):
    payload = _payload(seven_day_opus={"utilization": 40.0, "resets_at": "2026-11-18T06:59:59+00:00"})
    snap = _fetch(monkeypatch, payload)
    model_windows = {w.label: w.used_percent for w in snap.windows if w.scope == "model"}
    assert model_windows == {"Opus week": 40.0, "Fable week": 100.0}


def test_missing_or_malformed_limits_add_nothing(monkeypatch):
    for limits in (None, [], [{"kind": "weekly_scoped", "percent": None}], ["junk"]):
        snap = _fetch(monkeypatch, _payload(limits=limits))
        assert not any(w.scope == "model" for w in snap.windows)
