"""``/usage`` shows the active provider's block, then the Nous balance whenever Nous is signed in.

On a Claude session the Nous balance printed inside the Claude block with "/topup to add to your
balance", which read as a Claude balance. Hiding it lost a balance the user holds, so the Nous
section now always follows a signed-in profile's provider block: off a Nous route it has its own
heading, no top-up nudge, and is read with the stored token only (a /usage read never refreshes
the Nous credential the session is not running on).
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from agent.account_usage import NOUS_OTHER_ROUTE_HEADING, nous_usage_lines
from tests.hermes_cli.test_nous_account import _account_payload, _jwt

_PAYLOAD = _account_payload(allowed=True, subscription=None, subscription_credits=0, purchased_credits=7.75)


@pytest.fixture
def portal(monkeypatch):
    """A signed-in profile whose portal answers ``_PAYLOAD``; records each account fetch's token and any refresh."""
    seen = {"tokens": [], "refreshed": 0}
    state = {"access_token": _jwt({"sub": "u", "exp": int(time.time()) + 900}), "client_id": "hermes-cli",
             "portal_base_url": "https://portal.example.test",
             "expires_at": (datetime.now(UTC) + timedelta(minutes=15)).isoformat()}

    def refresh(**kw):
        seen["refreshed"] += 1
        return state["access_token"]

    def fetch(token, base=None):
        seen["tokens"].append(token)
        return _PAYLOAD
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: state if seen.get("signed_in", True) else {})
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", refresh)
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", fetch)
    monkeypatch.delenv("HERMES_DEV_CREDITS_FIXTURE", raising=False)
    return seen, state


def test_another_providers_session_gets_the_labelled_nous_balance_without_a_topup_nudge(portal):
    seen, state = portal
    lines = nous_usage_lines(nous_route=False)
    assert lines[0] == NOUS_OTHER_ROUTE_HEADING
    assert "Top-up credits: $7.75" in lines
    assert not any("/topup" in line or line.startswith("Top up:") for line in lines), lines
    assert seen == {"tokens": [state["access_token"]], "refreshed": 0}  # the stored token, never refreshed


def test_a_nous_session_keeps_the_balance_and_its_topup_lines(portal):
    lines = nous_usage_lines(nous_route=True)
    assert NOUS_OTHER_ROUTE_HEADING not in lines
    assert "Top-up credits: $7.75" in lines and "(or run /topup)" in lines


def test_a_profile_not_signed_into_nous_has_no_nous_section(portal):
    seen, _ = portal
    seen["signed_in"] = False
    assert nous_usage_lines(nous_route=False) == [] and nous_usage_lines(nous_route=True) == []
    assert seen["tokens"] == []


def test_an_expiring_stored_token_is_not_refreshed_for_another_providers_session(portal):
    seen, state = portal
    state["expires_at"] = (datetime.now(UTC) + timedelta(seconds=5)).isoformat()
    assert nous_usage_lines(nous_route=False) == []
    assert seen == {"tokens": [], "refreshed": 0}
