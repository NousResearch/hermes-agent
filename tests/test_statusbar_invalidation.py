"""Tests for status bar resolved model/context display and metadata tracking.

Portable — no hardcoded paths. Tests the real CLI snapshot via fixtures and the
real conversation_loop metadata helper via direct import.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import inspect

from agent.conversation_loop import _update_resolved_model_metadata
import agent.conversation_loop as _cl_mod


def _make_agent(**overrides):
    """Create a minimal agent mock for status bar tests."""
    def _none(*_a, **_k):
        return None

    a = SimpleNamespace(
        model=overrides.get('model', '@preset/hermes'),
        provider=overrides.get('provider', 'openrouter'),
        base_url=overrides.get('base_url', 'https://openrouter.ai/api/v1'),
        api_key=overrides.get('api_key', 'test-key-xyz'),
        _custom_providers=overrides.get('_custom_providers', None),
        _config_context_length=overrides.get('_config_context_length', 200000),
        _resolved_provider=overrides.get(
            '_resolved_provider', overrides.get('provider', 'openrouter')
        ),
        _resolved_model=overrides.get('_resolved_model', None),
        _resolved_context_length=overrides.get('_resolved_context_length', None),
        last_prompt_tokens=overrides.get('last_prompt_tokens', 0),
        compression_count=overrides.get('compression_count', 0),
        context_compressor=overrides.get(
            'context_compressor',
            SimpleNamespace(
                context_length=overrides.get('context_length', 200000),
                update_model=_none,
            ),
        ),
    )
    return a


def _call(agent, model_name):
    """Invoke the real metadata function with a mock response."""
    response = SimpleNamespace(model=model_name)
    _update_resolved_model_metadata(agent, response)


# --- Metadata helper tests ---


def test_response_without_model_preserves_state():
    agent = _make_agent(_resolved_model='model-a', _resolved_context_length=256000)
    _update_resolved_model_metadata(agent, SimpleNamespace())
    assert agent._resolved_model == 'model-a'
    assert agent._resolved_context_length == 256000


def test_same_model_does_not_re_resolve():
    agent = _make_agent(_resolved_model='model-a', _resolved_context_length=256000)
    with patch.object(_cl_mod, 'get_model_context_length') as m:
        _call(agent, 'model-a')
    m.assert_not_called()
    assert agent._resolved_context_length == 256000


def test_model_change_invalidates_and_recalculates():
    agent = _make_agent(_resolved_model='model-a', _resolved_context_length=256000)
    with patch.object(_cl_mod, 'get_model_context_length', return_value=1000000):
        _call(agent, 'model-b')
    assert agent._resolved_model == 'model-b'
    assert agent._resolved_context_length == 1000000


def test_five_successive_transitions_all_recalculate():
    agent = _make_agent()
    for i, ctx in enumerate([256000, 1000000, 32000, 8000, 512000], 1):
        with patch.object(_cl_mod, 'get_model_context_length', return_value=ctx):
            _call(agent, f'model-{i}')
        assert agent._resolved_context_length == ctx


def test_failed_resolution_does_not_keep_old_context():
    agent = _make_agent(_resolved_model='model-a', _resolved_context_length=256000)
    with patch.object(_cl_mod, 'get_model_context_length', side_effect=Exception('fail')):
        _call(agent, 'model-b')
    assert agent._resolved_context_length is None


def test_agent_model_not_overwritten():
    agent = _make_agent(model='@preset/hermes')
    with patch.object(_cl_mod, 'get_model_context_length', return_value=1000000):
        _call(agent, 'model-b')
    assert agent.model == '@preset/hermes'


def test_last_prompt_tokens_unchanged():
    agent = _make_agent(last_prompt_tokens=1234)
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000):
        _call(agent, 'model-a')
    assert agent.last_prompt_tokens == 1234


def test_compression_count_unchanged():
    agent = _make_agent(compression_count=5)
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000):
        _call(agent, 'model-a')
    assert agent.compression_count == 5


def test_compressor_update_model_not_called():
    """Ensure resolution does NOT trigger ContextCompressor.update_model()."""
    agent = _make_agent()
    mock = MagicMock()
    agent.context_compressor.update_model = mock
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000):
        _call(agent, 'model-a')
    mock.assert_not_called()


def test_base_url_passed_through():
    agent = _make_agent(base_url='https://example.com/v1')
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000) as m:
        _call(agent, 'model-a')
    assert m.call_args[1].get('base_url') == 'https://example.com/v1'


def test_api_key_passed_through():
    agent = _make_agent(api_key='test-key-xyz')
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000) as m:
        _call(agent, 'model-a')
    assert m.call_args[1].get('api_key') == 'test-key-xyz'


def test_provider_passed_through():
    agent = _make_agent(provider='openrouter')
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000) as m:
        _call(agent, 'model-a')
    assert m.call_args[1].get('provider') == 'openrouter'


def test_custom_providers_passed_through():
    custom = [SimpleNamespace(name='my-provider')]
    agent = _make_agent(_custom_providers=custom)
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000) as m:
        _call(agent, 'model-a')
    assert m.call_args[1].get('custom_providers') is custom


# --- Context override regression tests ---


def test_preset_context_not_applied_to_resolved_model():
    """Preset's _config_context_length must NOT be forwarded when response.model differs.

    Regression: @preset/hermes with _config_context_length=256000 must not
    override longcat-2.0's real context of 1000000.
    """
    agent = _make_agent(
        model='@preset/hermes',
        _config_context_length=256000,
    )
    with patch.object(_cl_mod, 'get_model_context_length', return_value=1000000) as m:
        _call(agent, 'longcat-2.0')
    # config_context_length must be None because resolved model != configured model
    assert m.call_args[1].get('config_context_length') is None
    assert agent._resolved_context_length == 1000000


def test_direct_model_preserves_configured_override():
    """When response.model matches agent.model, forward _config_context_length."""
    agent = _make_agent(
        model='modelo-direto',
        _config_context_length=256000,
    )
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000) as m:
        _call(agent, 'modelo-direto')
    # config_context_length must be 256000 because resolved model == configured model
    assert m.call_args[1].get('config_context_length') == 256000


def test_router_preset_then_real_model_then_another():
    """Preset → model A → model B: no preset context reused."""
    agent = _make_agent(
        model='@preset/hermes',
        _config_context_length=256000,
    )
    # First call: preset returns model-a
    with patch.object(_cl_mod, 'get_model_context_length', return_value=1000000) as m:
        _call(agent, 'model-a')
    assert m.call_args[1].get('config_context_length') is None
    assert agent._resolved_context_length == 1000000

    # Second call: model-a → model-b
    with patch.object(_cl_mod, 'get_model_context_length', return_value=500000) as m:
        _call(agent, 'model-b')
    assert m.call_args[1].get('config_context_length') is None
    assert agent._resolved_context_length == 500000


# --- Call site verification (portable, structural) ---


def test_callsite_in_conversation_loop():
    """Structural: the real source of conversation_loop calls our function."""
    src = inspect.getsource(_cl_mod.run_conversation)
    assert '_update_resolved_model_metadata(agent, response)' in src


def test_function_exists_at_module_level():
    """_update_resolved_model_metadata must be a module-level function."""
    assert hasattr(_cl_mod, '_update_resolved_model_metadata')
    assert inspect.isfunction(_cl_mod._update_resolved_model_metadata)


# --- CLI snapshot test (integrated helper -> snapshot) ---


def _make_cli_with_agent(agent):
    """Instantiate a HermesCLI-like object for snapshot testing.

    Uses the real _get_status_bar_snapshot from the installed CLI module
    without requiring full CLI initialization.
    """
    from cli import HermesCLI
    with patch.object(HermesCLI, '__init__', lambda self, *a, **k: None):
        cli = HermesCLI()
    cli.agent = agent
    cli.model = agent.model
    cli.session_start = __import__('datetime').datetime.now()
    cli._prompt_start_time = None
    cli._prompt_duration = 0.0
    cli._last_turn_finished_at = None
    cli._background_tasks = {}
    return cli


def test_snapshot_uses_resolved_model_over_agent_model():
    """_resolved_model has priority over agent.model in snapshot."""
    agent = _make_agent(model='@preset/hermes', _resolved_model='actual-model-xyz')
    agent._resolved_context_length = None
    cli = _make_cli_with_agent(agent)
    snap = cli._get_status_bar_snapshot()
    assert snap['model_name'] == 'actual-model-xyz'
    assert 'actual-model-xyz' in snap['model_short']


def test_snapshot_falls_back_to_agent_model_when_no_resolved():
    """Without _resolved_model, use agent.model."""
    agent = _make_agent(model='configured-model')
    agent._resolved_context_length = None
    cli = _make_cli_with_agent(agent)
    snap = cli._get_status_bar_snapshot()
    assert snap['model_name'] == 'configured-model'


def test_snapshot_falls_back_to_cli_model_when_no_agent():
    """Without agent model, use cli.model."""
    agent = _make_agent()
    agent.model = None
    agent._resolved_model = None
    agent._resolved_context_length = None
    cli = _make_cli_with_agent(agent)
    cli.model = 'cli-fallback-model'
    snap = cli._get_status_bar_snapshot()
    assert snap['model_name'] == 'cli-fallback-model'


def test_snapshot_falls_back_to_unknown_when_no_models():
    """Without any model, use 'unknown'."""
    agent = _make_agent(model=None, _resolved_model=None)
    agent._resolved_context_length = None
    cli = _make_cli_with_agent(agent)
    cli.model = None
    snap = cli._get_status_bar_snapshot()
    assert snap['model_name'] == 'unknown'


def test_snapshot_uses_resolved_context_length():
    """_resolved_context_length takes priority over compressor.context_length."""
    agent = _make_agent(_resolved_context_length=1000000)
    agent.context_compressor.context_length = 200000
    cli = _make_cli_with_agent(agent)
    snap = cli._get_status_bar_snapshot()
    assert snap['context_length'] == 1000000


def test_snapshot_falls_back_compressor_context_when_resolved_is_none():
    """When _resolved_context_length is None, use compressor.context_length."""
    agent = _make_agent()
    agent._resolved_context_length = None
    agent.context_compressor.context_length = 200000
    cli = _make_cli_with_agent(agent)
    snap = cli._get_status_bar_snapshot()
    assert snap['context_length'] == 200000


def test_snapshot_falls_back_compressor_context_when_resolved_is_zero():
    """When _resolved_context_length is 0, use compressor.context_length."""
    agent = _make_agent()
    agent._resolved_context_length = 0
    agent.context_compressor.context_length = 200000
    cli = _make_cli_with_agent(agent)
    snap = cli._get_status_bar_snapshot()
    assert snap['context_length'] == 200000


def test_snapshot_uses_resolved_context_for_percentage():
    """context_percent uses _resolved_context_length as denominator."""
    agent = _make_agent(_resolved_context_length=1000000)
    agent.context_compressor.last_prompt_tokens = 500000
    agent.context_compressor.context_length = 200000  # should be ignored
    cli = _make_cli_with_agent(agent)
    snap = cli._get_status_bar_snapshot()
    assert snap['context_length'] == 1000000
    assert snap['context_percent'] == 50


def test_snapshot_preserves_last_prompt_tokens_numerator():
    """last_prompt_tokens is the numerator and must be preserved."""
    agent = _make_agent(_resolved_context_length=1000000)
    agent.context_compressor.last_prompt_tokens = 750000
    cli = _make_cli_with_agent(agent)
    snap = cli._get_status_bar_snapshot()
    assert snap['context_tokens'] == 750000
    assert snap['context_percent'] == 75


def test_snapshot_preserves_compression_count():
    """compression_count must be preserved in snapshot."""
    agent = _make_agent()
    agent._resolved_context_length = None
    agent.context_compressor.compression_count = 7
    cli = _make_cli_with_agent(agent)
    snap = cli._get_status_bar_snapshot()
    assert snap['compressions'] == 7


def test_snapshot_change_a_to_b_reflected():
    """Model change A -> B is reflected in subsequent snapshot."""
    agent = _make_agent(model='@preset/hermes', _resolved_model='model-a',
                       _resolved_context_length=256000)
    cli = _make_cli_with_agent(agent)
    snap_a = cli._get_status_bar_snapshot()
    assert snap_a['model_name'] == 'model-a'
    assert snap_a['context_length'] == 256000

    # Simulate model change to B with new context
    agent._resolved_model = 'model-b'
    agent._resolved_context_length = 1000000
    snap_b = cli._get_status_bar_snapshot()
    assert snap_b['model_name'] == 'model-b'
    assert snap_b['context_length'] == 1000000


def test_snapshot_failed_resolution_shows_compressor_context():
    """When resolution fails, old context is NOT preserved; uses compressor fallback."""
    agent = _make_agent(_resolved_model='model-a', _resolved_context_length=256000)
    agent.context_compressor.context_length = 200000
    cli = _make_cli_with_agent(agent)
    snap_a = cli._get_status_bar_snapshot()
    assert snap_a['context_length'] == 256000

    # Resolution failed — _resolved_context_length is None
    agent._resolved_model = 'model-b'
    agent._resolved_context_length = None
    snap_b = cli._get_status_bar_snapshot()
    # Falls back to compressor.context_length
    assert snap_b['context_length'] == 200000


def test_snapshot_agent_model_not_overwritten_by_resolution():
    """agent.model must be the configured value after resolution, not the resolved one."""
    agent = _make_agent(model='@preset/hermes')
    agent._resolved_context_length = None
    cli = _make_cli_with_agent(agent)
    with patch.object(_cl_mod, 'get_model_context_length', return_value=1000000):
        _call(agent, 'longcat-2.0')
    snap = cli._get_status_bar_snapshot()
    assert agent.model == '@preset/hermes'
    assert snap['model_name'] == 'longcat-2.0'


def test_preset_256k_resolved_model_1m_shows_1m_in_snapshot():
    """Full integrated test: preset 256K → resolved model 1M shows 1M in status bar.

    Regression test: previously, config_context_length=256000 was forwarded
    to get_model_context_length() which returned it immediately, ignoring
    the real model's context window.
    """
    agent = _make_agent(
        model='@preset/hermes',
        _config_context_length=256000,
        context_length=256000,
    )
    cli = _make_cli_with_agent(agent)

    # Before: status bar shows preset's 256K
    snap_before = cli._get_status_bar_snapshot()
    assert snap_before['model_name'] == '@preset/hermes'
    assert snap_before['context_length'] == 256000

    # Resolve: API returns longcat-2.0 with 1M context
    with patch.object(_cl_mod, 'get_model_context_length', return_value=1000000):
        _call(agent, 'longcat-2.0')

    # After: status bar shows resolved model's 1M context
    snap_after = cli._get_status_bar_snapshot()
    assert snap_after['model_name'] == 'longcat-2.0'
    assert snap_after['context_length'] == 1000000
    assert agent.model == '@preset/hermes'


def test_direct_model_snapshot_shows_configured_override():
    """Direct model (no preset) with override shows configured context."""
    agent = _make_agent(
        model='claude-sonnet-4',
        _config_context_length=256000,
        context_length=200000,
    )
    cli = _make_cli_with_agent(agent)

    # Resolve: API returns same model
    with patch.object(_cl_mod, 'get_model_context_length', return_value=256000):
        _call(agent, 'claude-sonnet-4')

    snap = cli._get_status_bar_snapshot()
    assert snap['model_name'] == 'claude-sonnet-4'
    assert snap['context_length'] == 256000
