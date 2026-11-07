"""A confirmed model lock accepts a named custom provider served through the gateway runner.

Agents built by the gateway runner (the /v1/runs admission path) carry no ``_hermes_api_runtime``,
and a named custom provider such as ``ollama-launch`` resolves to the agent provider ``custom``.
The lock must compare resolved identities, and must not replace a failed turn's own error.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import gateway.run  # Initialize the launcher before the per-test home I/O guard.
from tests.gateway.test_api_server_runs import _make_adapter


def _agent(provider, model):
    return SimpleNamespace(provider=provider, model=model)  # no _hermes_api_runtime


def _meta(adapter, agent):
    return adapter._turn_runtime_metadata(
        agent, route={'provider': 'ollama-launch', 'model': 'qwen3.8:27b-mlx'},
        requested_runtime={'provider': 'ollama-launch', 'model': 'qwen3.8:27b-mlx'},
        route_source='raw_request', confirmed_runtime_lock=True)


def test_named_custom_provider_resolves_to_the_agents_runtime():
    adapter = _make_adapter()
    with patch.object(adapter, '_resolve_provider_runtime', return_value={'provider': 'custom'}):
        runtime = _meta(adapter, _agent('custom', 'qwen3.8:27b-mlx'))
    assert runtime['provider'] == 'custom' and runtime['model_lock'] == 'confirmed'


def test_a_fallback_runtime_still_fails_the_lock():
    adapter = _make_adapter()
    with patch.object(adapter, '_resolve_provider_runtime', return_value={'provider': 'custom'}):
        with pytest.raises(RuntimeError, match='runtime mismatch'):
            _meta(adapter, _agent('openai-codex', 'gpt-6.1-sol'))


def test_unresolvable_provider_falls_back_to_the_literal_slug():
    adapter = _make_adapter()
    with patch.object(adapter, '_resolve_provider_runtime', side_effect=RuntimeError('no catalog')):
        with pytest.raises(RuntimeError, match='runtime mismatch'):
            _meta(adapter, _agent('custom', 'qwen3.8:27b-mlx'))


def test_failed_turn_keeps_its_own_error_under_the_lock():
    from gateway import run_turn_runner
    settings = {'route': {}, 'requested_runtime': {}, 'route_source': 'raw_request'}
    api = {'settings': settings, 'adapter': SimpleNamespace(
        _turn_runtime_metadata=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('runtime mismatch')))}
    result = {'failed': True, 'error': 'connection refused', 'final_response': ''}
    with patch('gateway.session_api_turn.api_runtime_locked', return_value=api):
        run_turn_runner._settle_api_runtime_lock(_agent('custom', 'm'), result, {'usage': {}})
    assert result['error'] == 'connection refused'
