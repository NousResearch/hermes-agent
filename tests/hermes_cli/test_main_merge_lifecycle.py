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
    # Same field the legacy setup.runtime_check answers and Desktop reads.
    assert check_runtime_readiness()['free_tier_route'] is expected


def test_canonical_readiness_blames_the_pin_not_the_chain_tail(monkeypatch):
    """#124939 on the authority path: a chain that resolves only at its tail must not send
    onboarding after a provider the user never pinned."""
    from hermes_cli.runtime_readiness import check_runtime_readiness
    monkeypatch.setattr('hermes_cli.main._has_any_provider_configured', lambda **kw: True)
    monkeypatch.setattr('hermes_cli.config.load_config',
                        lambda: {'model': {'provider': 'zai', 'default': 'glm-5.3'}})
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider', lambda **kw: {
        'provider': 'openrouter', 'api_key': '', 'source': 'pool'})
    result = check_runtime_readiness()
    assert result == {'ok': False, 'provider': 'zai', 'model': 'glm-5.3', 'source': 'pool',
                      'error': 'No usable credentials found for zai.'}


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
