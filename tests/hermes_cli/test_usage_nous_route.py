"""Classic CLI ``/usage`` on a gateway session: the owner decides the route and sends the Nous section.

``GatewayAgent`` keeps its launch ``base_url`` while ``provider`` follows ``/model``, so a CLI launched on
Nous that moved to OpenRouter paired ``openrouter`` with the Nous endpoint and printed the Nous ``/topup``
nudge under OpenRouter. The owner's ``session.account_usage`` now answers for the route, puts the Nous
balance after the provider's block, and on a Nous route sends the dollar model from the same account read,
so ``/usage`` reads the Nous portal once.
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


class _Owner:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def call(self, method, **params):
        self.calls.append(method)
        return self.answer


def _cli(provider: str, base_url: str):
    cli_obj = _make_cli()
    cli_obj.session_id = "s"
    cli_obj.provider, cli_obj.base_url, cli_obj.api_key = provider, base_url, None
    cli_obj.agent = SimpleNamespace(provider=provider, base_url=base_url, api_key="", session_api_calls=0)
    return cli_obj


def test_the_owners_route_decides_and_the_nous_balance_follows_the_providers_block(monkeypatch, capsys):
    owner = _Owner({"lines": ["📈 OpenRouter limits"], "nous_lines": ["Nous Portal balance", "Top-up credits: $7.75"],
                    "nous_route": False})
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: owner)
    # After `/model ... --provider openrouter` on a Nous launch: provider moved, base_url did not.
    _cli("openrouter", NOUS_URL)._show_usage()
    out = capsys.readouterr().out
    assert owner.calls == ["session.account_usage"]
    assert out.index("📈 OpenRouter limits") < out.index("Nous Portal balance") < out.index("Top-up credits: $7.75")
    assert "/topup" not in out and "Plan:" not in out


@pytest.fixture
def portal(monkeypatch):
    """A profile signed into Nous; counts every portal account read."""
    from hermes_cli.nous_account import reset_nous_portal_account_info_cache
    reset_nous_portal_account_info_cache()
    seen = {"reads": 0}
    state = {"access_token": _jwt({"sub": "u", "exp": int(time.time()) + 900}), "client_id": "hermes-cli",
             "portal_base_url": "https://portal.example.test",
             "expires_at": (datetime.now(UTC) + timedelta(minutes=15)).isoformat()}

    def fetch(token, base=None):
        seen["reads"] += 1
        return _CREDIT_PLAN
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: state)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda **kw: state["access_token"])
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", fetch)
    monkeypatch.delenv("HERMES_DEV_CREDITS_FIXTURE", raising=False)
    yield seen
    reset_nous_portal_account_info_cache()


@pytest.mark.asyncio
async def test_usage_on_a_nous_session_reads_the_account_once(tmp_path, monkeypatch, capsys, portal):
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority
    from tui_gateway.contracts.registry import CANONICAL_METHODS

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
    CANONICAL_METHODS["session.account_usage"].result.model_validate(reply["result"])
    assert reply["result"]["nous_route"] is True, reply
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: _Owner(reply["result"]))
    _cli("nous", NOUS_URL)._show_usage()
    out = capsys.readouterr().out
    assert "Plan: Plus" in out and "$14.00 left of $20.00" in out and "/topup" in out, out
    assert portal["reads"] == 1
