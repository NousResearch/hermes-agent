"""Behavior parity at lifecycle helpers extracted before the main merge."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize('provider,host,expected', [
    ('nous', 'https://welcome-api.nousresearch.com/v1', True),
    ('nous', 'https://inference-api.nousresearch.com/v1', False),
    ('custom', 'https://welcome-api.nousresearch.com/v1', False),
])
def test_readiness_reports_selected_free_route(monkeypatch, provider, host, expected):
    from hermes_cli.runtime_readiness import check_runtime_readiness
    monkeypatch.setattr('hermes_cli.main._has_any_provider_configured', lambda **kw: True)
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', lambda **kw: {
        'provider': provider, 'base_url': host, 'api_key': 'no-key-required', 'model': 'nous/welcome'})
    assert check_runtime_readiness()['free_tier'] is expected


def test_local_launch_bootstraps_before_provider_guard(monkeypatch):
    from hermes_cli.gateway_chat_startup import ensure_launch_provider
    events = []
    monkeypatch.delenv('HERMES_TUI_GATEWAY_URL', raising=False)
    monkeypatch.setattr('hermes_cli.free_tier_bootstrap.run_bootstrap',
                        lambda **kw: events.append(('boot', kw)))
    monkeypatch.setattr('hermes_cli.main._has_any_provider_configured',
                        lambda: events.append('inventory') or True)
    assert ensure_launch_provider(SimpleNamespace())
    assert events == [('boot', {'announce': False}), 'inventory']


@pytest.mark.asyncio
@pytest.mark.parametrize('owned', [False, True])
async def test_wisdom_and_identity_have_one_lifespan_owner(monkeypatch, owned):
    from fastapi import FastAPI
    from hermes_cli import web_server as web, web_server_app, free_tier_bootstrap
    from tui_gateway import methods_groups
    events = []
    boot = Mock()
    monkeypatch.setattr(free_tier_bootstrap, 'start_background_bootstrap', boot)
    monkeypatch.delenv('HERMES_DESKTOP', raising=False)
    for name in ('_warm_gateway_module', '_eager_reconcile_own_session_db'):
        monkeypatch.setattr(web, name, lambda: None)
    for name in ('start_hosted_room_service', 'stop_hosted_room_service'):
        monkeypatch.setattr(methods_groups, name, lambda **kw: None)
    for name in ('ensure_local_runtime', 'shutdown_local_runtime'):
        monkeypatch.setattr('hermes_cli.local_runtime.bootstrap.' + name, lambda *a: None)
    async def checker():
        events.append('started')
        try:
            await asyncio.Event().wait()
        finally:
            events.append('cancelled')
    monkeypatch.setattr(web, '_wisdom_checker_loop', checker)
    app = FastAPI()
    if owned:
        app.state.gateway_runner = object()
    async with web_server_app.app_lifespan(app):
        await asyncio.sleep(0)
        assert events == ([] if owned else ['started'])
        assert boot.call_count == (0 if owned else 1)
    await asyncio.sleep(0)
    assert events == ([] if owned else ['started', 'cancelled'])
