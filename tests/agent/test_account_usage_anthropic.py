"""Anthropic OAuth usage API: money is in MINOR units and must match Claude Settings → Usage.

Regression: ``extra_usage.used_credits=1568`` / ``monthly_limit=4000`` (cents) was printed as
"Extra usage: 1568.00 / 4000.00 EUR" while the console showed €15.68 of €40 — a 100x inflated,
seemingly fast-growing figure.
"""
import pytest

import agent.account_usage as au


def _payload(**overrides):
    # Shape captured from the live endpoint (2026-10-27), trimmed to the parts we read.
    payload = {
        "five_hour": {"utilization": 100.0, "resets_at": "2026-10-27T04:50:00+00:00"},
        "seven_day": {"utilization": 25.0, "resets_at": "2026-11-02T00:00:00+00:00"},
        "seven_day_opus": None,
        "seven_day_sonnet": None,
        "extra_usage": {
            "is_enabled": True, "monthly_limit": 4000, "used_credits": 1568.0, "utilization": 39.2,
            "currency": "EUR", "decimal_places": 2, "spend_limit_reached": False,
        },
        "spend": {
            "used": {"amount_minor": 1568, "currency": "EUR", "exponent": 2},
            "limit": {"amount_minor": 4000, "currency": "EUR", "exponent": 2},
            "percent": 39, "enabled": True, "balance": None,
        },
    }
    payload.update(overrides)
    return payload


def _fetch(monkeypatch, payload):
    monkeypatch.setattr(au, "resolve_anthropic_token", lambda: "sk-ant-oat01-test")
    monkeypatch.setattr(au, "_is_oauth_token", lambda token: True)
    monkeypatch.setattr(au, "_get_json", lambda url, headers, timeout: payload)
    return au._fetch_anthropic_account_usage()


def test_spend_is_scaled_from_minor_units_to_console_values(monkeypatch):
    snap = _fetch(monkeypatch, _payload())
    assert snap.details[0] == "Monthly spend limit: 15.68 of 40.00 EUR used this month"
    # The purchased-credit balance is not returned (spend.balance: null) → say so, don't invent it.
    assert snap.details[1] == "Usage credits remaining: unavailable (not returned by the API)"
    assert not any("1568" in d or "4000" in d for d in snap.details)
    assert [(w.label, w.used_percent) for w in snap.windows] == [("Current session", 100.0), ("Current week", 25.0)]


def test_reported_spend_never_exceeds_the_limit_scale(monkeypatch):
    # Sanity invariant that would have caught the bug: spent/limit shown must agree with the API's own percent.
    snap = _fetch(monkeypatch, _payload())
    line = snap.details[0]
    used, limit = (float(x) for x in line.split(": ")[1].split(" EUR")[0].split(" of "))
    assert limit == 40.0
    assert round(used / limit * 100) == 39


def test_legacy_extra_usage_fallback_honours_decimal_places(monkeypatch):
    snap = _fetch(monkeypatch, _payload(spend=None))
    assert snap.details[0] == "Monthly spend limit: 15.68 of 40.00 EUR used this month"


def test_legacy_extra_usage_without_declared_scale_is_not_guessed(monkeypatch):
    extra = {"is_enabled": True, "monthly_limit": 4000, "used_credits": 1568.0, "currency": "EUR"}
    snap = _fetch(monkeypatch, _payload(spend=None, extra_usage=extra))
    assert snap.details[0] == "Monthly spend limit: unavailable"
    assert not any("1568" in d for d in snap.details)


def test_balance_is_shown_when_api_returns_it(monkeypatch):
    payload = _payload()
    payload["spend"]["balance"] = {"amount_minor": 528, "currency": "EUR", "exponent": 2}
    snap = _fetch(monkeypatch, payload)
    assert snap.details[1] == "Usage credits remaining: 5.28 EUR"


@pytest.mark.parametrize("payload_overrides", [
    {"spend": {"enabled": False}, "extra_usage": {"is_enabled": False}},
])
def test_disabled_usage_credits(monkeypatch, payload_overrides):
    snap = _fetch(monkeypatch, _payload(**payload_overrides))
    assert list(snap.details) == ["Usage credits: disabled"]
