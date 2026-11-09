"""Classic CLI ``/usage`` on a Nous route reads the Nous portal account once.

The owner's ``session.account_usage`` already read the account for the Nous section, then the CLI's
two-bar balance block read it again (and a window plan a third time, for its fallback lines). The
owner now sends the dollar model with the section and the CLI only renders it; off the gateway the
CLI takes both from one read too.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tests.hermes_cli.test_cli_status_bar import _make_cli
from tests.hermes_cli.test_nous_account import _account_payload, _jwt

NOUS_URL = "https://inference-api.nousresearch.com/v1"
_CREDIT_PLAN = _account_payload(
    allowed=True, subscription_credits=14.0, purchased_credits=7.75,
    subscription={"plan": "Plus", "tier": 2, "monthly_charge": 20, "monthly_credits": 20.0,
                  "current_period_end": "2026-12-01T00:00:00.000Z", "credits_remaining": 14.0})


def _window_plan() -> dict:
    reset_ms = int(time.time() * 1000) + 3_600_000
    gauges = [{"id": "day", "scope": "all", "label": "Today", "used_bp": 1_200, "resets_at_ms": reset_ms}]
    return {**_CREDIT_PLAN, "subscription": None, "usage_window": {"gauges": gauges}}


@pytest.fixture
def portal(monkeypatch):
    """A profile signed into Nous; counts every portal account read."""
    from hermes_cli.nous_account import reset_nous_portal_account_info_cache
    reset_nous_portal_account_info_cache()
    seen = {"reads": 0, "payload": _CREDIT_PLAN}
    state = {"access_token": _jwt({"sub": "u", "exp": int(time.time()) + 900}), "client_id": "hermes-cli",
             "portal_base_url": "https://portal.example.test",
             "expires_at": (datetime.now(UTC) + timedelta(minutes=15)).isoformat()}

    def fetch(token, base=None):
        seen["reads"] += 1
        return seen["payload"]
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: state)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda **kw: state["access_token"])
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", fetch)
    monkeypatch.delenv("HERMES_DEV_CREDITS_FIXTURE", raising=False)
    yield seen
    reset_nous_portal_account_info_cache()


class _Owner:
    def __init__(self, answer):
        self.answer = answer

    def call(self, method, **params):
        assert method == "session.account_usage"
        return self.answer


def _nous_cli():
    cli_obj = _make_cli()
    cli_obj.session_id = "s"
    cli_obj.provider, cli_obj.base_url, cli_obj.api_key = "nous", NOUS_URL, None
    cli_obj.agent = SimpleNamespace(provider="nous", base_url=NOUS_URL, api_key="", session_api_calls=0)
    return cli_obj


@pytest.mark.asyncio
async def test_a_gateway_session_on_nous_reads_the_account_once(tmp_path, monkeypatch, capsys, portal):
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority

    async def answer(event):
        return "ok"
    authority = await _authority(tmp_path, monkeypatch, answer)
    agent = SimpleNamespace(provider="nous", base_url=NOUS_URL, api_key="", _credits_state=None)
    authority.runner._cached_agent_for = lambda route: agent
    connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
    try:
        await connection.dispatch({"id": 1, "method": "session.resume", "params": {"session_id": "s"}})
        reply = await connection.dispatch({"id": 2, "method": "session.account_usage", "params": {"session_id": "s"}})
    finally:
        await connection.close()
    assert reply["result"]["nous_route"] is True, reply
    from tui_gateway.contracts.registry import CANONICAL_METHODS
    CANONICAL_METHODS["session.account_usage"].result.model_validate(reply["result"])
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: _Owner(reply["result"]))
    _nous_cli()._show_usage()
    out = capsys.readouterr().out
    assert "Plan: Plus" in out and "$14.00 left of $20.00" in out and "/topup" in out, out
    assert portal["reads"] == 1


def test_a_local_window_plan_session_reads_the_account_once(monkeypatch, capsys, portal):
    portal["payload"] = _window_plan()
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: None)
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", lambda *a, **kw: None)
    _nous_cli()._show_usage()
    out = capsys.readouterr().out
    assert "Today" in out and "12% used" in out and "/topup" in out, out
    assert portal["reads"] == 1
