"""Read-only Codex quota contract; fixtures never contact a provider."""

import json

import pytest

from agent import account_usage


def test_codex_parser_exports_only_sanitized_quota_and_drives_usage_lines(monkeypatch):
    from agent import account_usage_quota as quota

    payload = {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {"used_percent": 21, "reset_at": 1900000000},
            "secondary_window": {"used_percent": None, "reset_at": None},
            "unknown_window": {"used_percent": 98},
        },
        "rate_limit_reset_credits": {"available_count": 2},
        "credits": {"has_credits": True, "balance": 12.5},
        "access_token": "fixture-secret-must-not-escape",
        "email": "private@example.invalid",
        "account_id": "private-account",
    }
    result = quota.parse_codex_quota(payload, fetched_at=1800000000)
    wire = result.to_dict()
    assert wire["status"] == "ok" and wire["supported"] is True
    assert wire["fetched_at"] == 1800000000
    assert wire["plan"] == "Plus" and wire["banked_resets"] == 2
    assert wire["windows"] == [
        {"id": "primary_window", "label": "Session", "used_percent": 21.0, "reset_at": 1900000000},
        {"id": "secondary_window", "label": "Weekly", "used_percent": None, "reset_at": None},
    ]
    text = json.dumps(wire, allow_nan=False)
    assert not any(value in text for value in ("fixture-secret", "private", "unknown_window"))
    lines = account_usage.render_account_usage_lines(account_usage.codex_quota_snapshot(result))
    assert any("21% used" in line for line in lines)
    assert not any("Weekly:" in line for line in lines)  # /usage omits unknown usage.
    assert any("2 resets banked" in line for line in lines)
    assert "Credits balance: $12.50" in lines

    # Exercise the real HTTP decoding -> parser -> presentation path offline.
    import httpx
    from datetime import datetime, timezone

    client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    monkeypatch.setattr(account_usage.httpx, "Client", lambda **kw: client(transport=transport, **kw))
    monkeypatch.setattr(account_usage, "_utc_now", lambda: datetime.fromtimestamp(result.fetched_at, timezone.utc))
    snapshot = account_usage.fetch_account_usage("openai-codex", api_key="fixture-only-token")
    assert snapshot == account_usage.codex_quota_snapshot(result)
    assert "fixture-only-token" not in repr(snapshot)


@pytest.mark.parametrize("http_status, expected", [
    (401, "auth"), (403, "auth"), (429, "rate_limit"),
    (None, "network"), (500, "network"), (503, "network"),
    (404, "unsupported"), (400, "unsupported"), (302, "unsupported"),
])
def test_failures_are_typed_and_never_echo_provider_material(http_status, expected):
    from agent import account_usage_quota as quota

    result = quota.parse_codex_quota(
        {"message": "fixture-secret", "plan_type": "private@example.invalid"},
        fetched_at=1800000000, http_status=http_status,
    )
    assert isinstance(result, quota.CodexQuotaFailure)
    assert result.error == expected
    wire = result.to_dict()
    assert wire["status"] == "error"
    assert wire["supported"] is (expected != "unsupported")
    assert wire["fetched_at"] == 1800000000
    assert "windows" not in wire  # Failure is not an empty/zero quota success.
    assert "fixture-secret" not in json.dumps(wire)
    assert "private" not in repr(result)
    assert account_usage.codex_quota_snapshot(result) is None


@pytest.mark.parametrize("value, expected", [
    (None, None), (0, 0), (21.5, 21.5), (125, 125),
    (True, None), (False, None), (-1, None), (float("nan"), None),
    (float("inf"), None), (10**400, None), ("fixture-secret", None),
    ({"access_token": "fixture-secret"}, None), ([], None),
])
def test_missing_and_invalid_numbers_never_become_zero_or_leak(value, expected):
    from agent import account_usage_quota as quota

    payload = {
        "plan_type": "private@example.invalid",
        "rate_limit": {"primary_window": {"used_percent": value, "reset_at": value,
                                           "label": "fixture-secret"}},
        "rate_limit_reset_credits": {"available_count": value},
        "credits": {"has_credits": True, "balance": value},
    }
    result = quota.parse_codex_quota(payload, fetched_at=1800000000)
    assert isinstance(result, quota.CodexQuotaSuccess)
    assert result.plan is None
    assert result.windows[0].used_percent == expected
    integer = int(expected) if expected is not None and expected == int(expected) else None
    assert result.windows[0].reset_at == integer
    assert result.banked_resets == integer
    assert result.credits_balance == expected
    assert result.windows[1].used_percent is None  # Absent window is not zero.
    assert result.windows[1].reset_at is None
    text = json.dumps(result.to_dict(), allow_nan=False) + repr(result)
    assert "fixture-secret" not in text and "private" not in text


@pytest.mark.parametrize("payload", [None, [], "fixture-secret", {"rate_limit": []}])
def test_unsupported_response_shape_is_not_success_with_zero_usage(payload):
    from agent import account_usage_quota as quota

    result = quota.parse_codex_quota(payload, fetched_at=1800000000)
    assert isinstance(result, quota.CodexQuotaFailure)
    assert result.error == "unsupported"


@pytest.mark.parametrize("stamp", [True, -1, float("nan"), "fixture-secret", 1800000000000])
def test_observation_timestamp_must_be_epoch_seconds(stamp):
    from agent import account_usage_quota as quota

    with pytest.raises(ValueError, match="fetched_at must be epoch seconds"):
        quota.parse_codex_quota({}, fetched_at=stamp)


def test_resets_are_seconds_and_presentation_preserves_missing_and_unlimited():
    from agent import account_usage_quota as quota

    result = quota.parse_codex_quota({
        "rate_limit": {
            "primary_window": {"used_percent": 0, "reset_at": "2030-03-17T17:46:40Z"},
            "secondary_window": {"used_percent": 100, "reset_at": 1900000000000},
        },
        "credits": {"has_credits": True, "unlimited": True},
    }, fetched_at=1800000000)
    assert result.banked_resets is None and result.plan is None
    assert result.windows[0].reset_at == 1900000000
    assert result.windows[1].reset_at is None  # Milliseconds are not seconds.
    snapshot = account_usage.codex_quota_snapshot(result)
    assert snapshot.fetched_at.timestamp() == result.fetched_at
    assert snapshot.windows[0].reset_at.timestamp() == result.windows[0].reset_at
    assert "Credits balance: unlimited" in snapshot.details
    assert "Session: 100% remaining (0% used)" in account_usage.render_account_usage_lines(snapshot)[2]
