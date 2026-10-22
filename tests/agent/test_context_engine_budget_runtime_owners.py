"""Budget handoffs through the current runtime owners, using offline provider seams."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.agent.test_context_engine_compression_budget import BudgetEngine, make_agent


@pytest.mark.parametrize("accept", [True, False])
@pytest.mark.parametrize("overrides", [
    {"_compression_threshold_percent": float("nan")},
    {"_compression_threshold_percent": float("inf")},
    {"max_tokens": float("inf")},
    {"_compression_threshold_tokens_cap": float("inf")},
])
def test_switch_keeps_successful_new_runtime_when_optional_budget_cannot_derive(accept, overrides):
    from agent.agent_runtime_helpers import _snapshot_switch_state, _update_switch_compressor
    from agent.context_engine_budget import apply_context_engine_compression_budget
    from tests.agent.test_plugin_context_engine_init import _StubEngine

    class Engine(_StubEngine):
        def update_model(self, model, context_length, **runtime):
            super().update_model(model, context_length, **runtime)
            self.model = model
            self.base_url = runtime.get("base_url", "")
            self.provider = runtime.get("provider", "")

        def set_compression_budget(self, capacity, trigger, *, reason):
            if self.accept is True:
                self.private_budget = (capacity, trigger)
            return self.accept

    engine = Engine()
    engine.accept = True
    engine.update_model("old", 1_000_000, provider="openai", base_url="https://old.invalid")
    agent = make_agent(engine, model="old", provider="openai", max_tokens=None,
                       _compression_model_thresholds={})
    assert apply_context_engine_compression_budget(agent, 1_000_000, reason="init") is True
    agent._context_engine_aux_context_ceiling = 200_000
    private_before = engine.private_budget
    snapshot = _snapshot_switch_state(agent)
    agent.model = "new"
    agent.base_url = "https://new.invalid"
    engine.accept = accept
    for field, value in overrides.items():
        setattr(agent, field, value)
    with (patch("agent.model_metadata.get_model_context_length", return_value=128_000),
          patch("agent.conversation_compression.revalidate_compression_feasibility")):
        _update_switch_compressor(agent, [], None, snapshot)
    assert agent.model == engine.model == "new"
    assert agent.base_url == engine.base_url == "https://new.invalid"
    assert engine.provider == agent.provider == "openai"
    assert engine.context_length == 128_000
    assert engine.threshold_tokens == 96_000  # engine-owned update_model result
    assert engine.private_budget == private_before
    assert agent._context_engine_compression_budget_accepted is False
    assert agent._context_engine_budget_runtime == (agent.model, agent.provider, agent.base_url, agent.api_mode)
    assert agent._context_engine_aux_context_ceiling is None


@pytest.mark.parametrize("error", [ValueError, TypeError, OverflowError])
def test_runtime_update_model_errors_are_not_optional_budget_declines(error):
    from agent.context_engine_budget import update_runtime_context_compressor

    class BrokenEngine(BudgetEngine):
        def update_model(self, **runtime):
            raise error("actual update_model failure")

    agent = make_agent(BrokenEngine())
    with pytest.raises(error, match="actual update_model failure"):
        update_runtime_context_compressor(agent, 128_000, reason="switch", model="new")


def test_route_owners_preserve_auxiliary_ceiling_until_runtime_changes(monkeypatch):
    from agent import auxiliary_client, model_metadata
    from agent.chat_completion_helpers import _update_fallback_context_compressor
    from agent.turn_overflow import _adopt_provider_context_limit
    from agent.agent_runtime_helpers import _update_switch_compressor
    agent = make_agent(_buffer_vprint=lambda text: None)
    engine = agent.context_compressor
    monkeypatch.setattr(auxiliary_client, "_resolve_task_provider_model", lambda task: ("openai", None, None, None, None))
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", lambda *a, **k: (
        SimpleNamespace(base_url="https://aux.invalid", api_key=""), "aux"))
    def window(model, **kwargs):
        return 200_000 if model == "aux" else 1_000_000
    monkeypatch.setattr(model_metadata, "get_model_context_length", window)
    monkeypatch.setattr(model_metadata, "save_provider_context_length", lambda *a: None)
    agent._compression_feasibility_checked = True
    _update_fallback_context_compressor(agent)
    assert engine.budgets[-1] == (800_000, 200_000, "auxiliary_feasibility")
    _adopt_provider_context_limit(SimpleNamespace(agent=agent), "maximum context length is 900000 tokens", 1_000_000)
    assert engine.budgets[-1] == (700_000, 200_000, "provider_context_limit")
    engine.accept = False
    agent.provider = "openai"
    plugin_threshold = int(1_000_000 * engine.threshold_percent)
    _update_fallback_context_compressor(agent)
    assert agent._context_engine_compression_budget_accepted is False
    assert engine.threshold_tokens == plugin_threshold  # rejected policy belongs to update_model
    assert agent._context_engine_aux_context_ceiling is None
    engine.accept = True
    _update_switch_compressor(agent, [], None, {})
    assert engine.budgets[-2:] == [(800_000, 400_000, "model_switch"), (800_000, 200_000, "auxiliary_feasibility")]
    assert engine.threshold_tokens == engine.private_trigger == 200_000


@pytest.mark.parametrize("owner,window,reason,expected", [
    ("ollama", 65_536, "ollama_num_ctx", 55_705),
    ("growth", 131_072, "local_window_growth", 98_304),
    ("provider", 128_000, "provider_context_limit", 96_000),
    ("switch", 1_000_000, "model_switch", 500_000),
    ("restore", 1_000_000, "primary_runtime_restore", 500_000),
    ("codex", 1_000_000, "codex_runtime_context", 500_000),
])
def test_current_owner_delivers_effective_window(owner, window, reason, expected, monkeypatch):
    agent = make_agent(max_tokens=None, provider="openai", _buffer_vprint=lambda text: None,
                       _buffer_status=lambda text: None)
    engine = agent.context_compressor
    if owner == "ollama":
        from agent.agent_init_ollama import _clamp_compressor_to_ollama_num_ctx
        agent._ollama_num_ctx = window
        _clamp_compressor_to_ollama_num_ctx(agent)
    elif owner == "growth":
        from agent.turn_context_compaction import _apply_grown_window
        _apply_grown_window(agent, engine, window)
    elif owner == "provider":
        from agent.turn_overflow import _adopt_provider_context_limit
        from agent import model_metadata
        monkeypatch.setattr(model_metadata, "save_provider_context_length", lambda *a: None)
        assert _adopt_provider_context_limit(SimpleNamespace(agent=agent),
            "maximum context length is 128000 tokens", 1_000_000) == window
    elif owner == "switch":
        from agent.agent_runtime_helpers import _update_switch_compressor
        with (patch("agent.model_metadata.get_model_context_length", return_value=window),
              patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(None, None)),
              patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)),
              patch("agent.auxiliary_client._try_configured_fallback_for_unavailable_client", return_value=(None, None, ""))):
            _update_switch_compressor(agent, [], None, {})
    elif owner == "restore":
        from agent.route_binding import reinstall_runtime_snapshot
        rt = dict(model=agent.model, provider=agent.provider, use_prompt_caching=False,
                  compressor_model=agent.model, compressor_context_length=window,
                  compressor_base_url=agent.base_url, compressor_api_key="",
                  compressor_provider=agent.provider, compressor_api_mode=agent.api_mode)
        with (patch("agent.agent_runtime_helpers._apply_primary_runtime_fields"),
              patch("agent.agent_runtime_helpers._restore_runtime_capabilities"),
              patch("agent.agent_runtime_helpers._rebuild_primary_client"),
              patch("agent.chat_completion_helpers.rewrite_prompt_model_identity")):
            reinstall_runtime_snapshot(agent, rt)
    else:
        from agent.codex_runtime import _record_codex_app_server_usage
        agent.session_api_calls = 0
        agent.session_estimated_cost_usd = 0.0
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens",
                     "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
            setattr(agent, "session_" + name, 0)
        engine.update_from_response = lambda usage: None
        turn = SimpleNamespace(model_context_window=window, token_usage_last={"inputTokens": 100, "outputTokens": 10})
        with (patch("agent.codex_runtime._queue_token_counts"),
              patch("agent.usage_pricing.estimate_usage_cost", return_value=SimpleNamespace(
                  amount_usd=None, status="included", source="offline"))):
            _record_codex_app_server_usage(agent, turn)
    assert engine.context_length == window
    assert engine.budgets == [(window, expected, reason)]
    assert engine.threshold_tokens == engine.private_trigger == expected
