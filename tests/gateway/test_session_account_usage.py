"""``session.account_usage``: the shared gateway's ``/usage`` plan-limits block for a session's route.

Ink's ``/usage`` on the shared gateway read only ``session.stats``, so a provider plugin's
``fetch_account_usage`` hook (Claude plan limits on DirectSDK) never reached the TUI while the classic
CLI showed it. The read runs the plugin hook through ``account_limit_lines``, the CLI's own path.
"""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import providers as _providers
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from providers.base import ProviderProfile


class _PlanProfile(ProviderProfile):
    def fetch_account_usage(self, *, base_url=None, api_key=None):
        return AccountUsageSnapshot(
            provider=self.name, source="plugin", fetched_at=datetime.now(UTC), title="Claude plan limits",
            windows=(AccountUsageWindow(label="Current session", used_percent=1.0),))


@pytest.fixture
def plan_provider(monkeypatch):
    _providers._discover_providers()
    monkeypatch.setattr(_providers, "_REGISTRY", dict(_providers._REGISTRY))
    monkeypatch.setattr(_providers, "_ALIASES", dict(_providers._ALIASES))
    monkeypatch.setattr(_providers, "_PROVIDER_LIST_CACHE", None)
    _providers.register_provider(_PlanProfile(name="plan-process", auth_type="external_process",
                                              base_url="process://plan-process"))


@pytest.mark.asyncio
async def test_account_usage_returns_the_plugin_plan_limits_for_the_session_agent(tmp_path, monkeypatch, plan_provider):
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_surface_admission import _authority
    from tui_gateway.contracts.registry import CANONICAL_METHODS

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    agent = SimpleNamespace(provider="plan-process", base_url="process://plan-process", api_key="")
    authority.runner._cached_agent_for = lambda route: agent
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        reply = await connection.dispatch({'id': 2, 'method': 'session.account_usage', 'params': {'session_id': 's'}})
        assert 'result' in reply, reply
        CANONICAL_METHODS['session.account_usage'].result.model_validate(reply['result'])
        lines = reply['result']['lines']
        assert lines[0] == "📈 Claude plan limits"
        assert "Current session: 99% remaining (1% used)" in lines
        assert reply["result"]["nous_route"] is False  # a process provider: no Nous balance or /topup CTA
    finally:
        await connection.close()
