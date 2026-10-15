"""Construction and model switches must share metadata override precedence."""
from types import SimpleNamespace

import pytest

from agent.agent_init import _resolve_context_length
from agent.agent_runtime_helpers import _resolve_switch_context_length


@pytest.mark.parametrize('path', ['construction', 'switch'])
@pytest.mark.parametrize('override,pin,expected', [(500_000, None, 500_000), (None, None, 1_048_576), (500_000, 250_000, 250_000)])
def test_model_override_precedes_endpoint_default(monkeypatch, path, override, pin, expected):
    from agent import context_pin, models_dev
    import hermes_cli.config as config_module

    model = 'gpt-6.1-sol'
    url = 'https://proxy.example/v1'
    cfg = {
        'model': {'default': model, 'provider': 'openai-codex', 'base_url': url},
        'custom_providers': [{'name': 'proxy', 'base_url': url, 'context_length': 1_048_576}],
        'model_overrides': {'openai-codex': {model: {'context_window': override}}} if override else {},
    }
    if pin is not None:
        cfg['model']['context_length'] = pin
    monkeypatch.setattr(config_module, 'load_config', lambda: cfg)
    monkeypatch.setattr(models_dev, '_load_model_overrides', lambda: cfg['model_overrides'])
    monkeypatch.setattr(context_pin, 'warn_once_on_pin_disagreement', lambda *args: None)
    compressor = SimpleNamespace(_config_context_length=None)
    agent = SimpleNamespace(
        model=model, provider='openai-codex', base_url=url,
        context_compressor=compressor, request_overrides={},
        _ensure_lmstudio_runtime_loaded=lambda value: None,
        _lmstudio_load_was_unverified=lambda value: False,
        _effective_lmstudio_context_length=lambda explicit, runtime: explicit,
    )
    if path == 'construction':
        intent, _, effective, _ = _resolve_context_length(agent, cfg, url)
    else:
        _, effective = _resolve_switch_context_length(agent, {})
        intent = agent._config_context_length
        assert compressor._config_context_length == expected
    assert intent == expected
    assert effective == expected
