"""``session.account_usage``: the shared gateway's ``/usage`` plan-limits block for a session's route.

Ink's ``/usage`` on the shared gateway read only ``session.stats``, so a provider's account limits (a
plugin's ``fetch_account_usage`` hook: Claude plan limits on DirectSDK) never reached the TUI while the
classic CLI showed them. Before the first turn the read also used the raw ``model.provider: auto``; it now
asks the resolver that turn calls.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow


@pytest.mark.asyncio
async def test_a_cold_session_reports_the_plan_limits_of_the_route_its_first_turn_resolves(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from tui_gateway.contracts.registry import CANONICAL_METHODS

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': 'auto', 'default': 'm'},
                                                  'platform_toolsets': {'cli': []}}))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-test')
    fetched = []

    def fetch(provider, **kw):
        fetched.append(provider)
        return AccountUsageSnapshot(provider=provider, source='plugin', fetched_at=datetime.now(UTC), title='Plan limits',
                                    windows=(AccountUsageWindow(label='Current session', used_percent=1.0),))
    monkeypatch.setattr('agent.account_usage.fetch_account_usage', fetch)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    try:
        created = await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'cold', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': []}})
        sid = created['result']['session_id']
        reply = await connection.dispatch({'id': 2, 'method': 'session.account_usage', 'params': {'session_id': sid}})
    finally:
        await connection.close()
        store.close_all_db_handles()
    CANONICAL_METHODS['session.account_usage'].result.model_validate(reply['result'])
    assert fetched == ['openrouter'], fetched  # never the raw ``auto``
    assert reply['result'] == {
        'lines': ['📈 Plan limits', 'Provider: openrouter', 'Current session: 99% remaining (1% used)'],
        'nous_lines': [], 'nous_route': False}, reply
