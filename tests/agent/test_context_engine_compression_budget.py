"""Host budget opt-in contract; all provider/client boundaries are offline."""
from types import SimpleNamespace

import pytest

from agent import conversation_compression as compression
from agent import context_engine_budget as budget


class BudgetEngine:
    def __init__(self):
        self.context_length = 1_000_000
        self.threshold_percent = 0.33
        self.threshold_tokens = 321_000
        self.accept = True
        self.budgets = []

    def set_compression_budget(self, capacity, trigger, *, reason):
        if isinstance(self.accept, Exception):
            raise self.accept
        if self.accept is True:
            self.budgets.append((capacity, trigger, reason))
            self.private_trigger = trigger
        return self.accept

    def update_model(self, model, context_length, **kwargs):
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)


def make_agent(engine=None, **overrides):
    values = dict(
        context_compressor=engine or BudgetEngine(), model="gpt-5.5", provider="openai-codex",
        base_url="https://offline.invalid/v1", api_mode="chat_completions", api_key="",
        max_tokens=200_000, _compression_threshold_percent=0.50,
        _compression_model_thresholds={"openai-codex:gpt-5.5": 0.85},
        _compression_threshold_tokens_cap=None,
        compression_enabled=True, _current_main_runtime=dict, _custom_providers=[],
        _aux_compression_context_length_config=None, _emit_diagnostic_status=lambda message: None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("window,reserve,cap,capacity,trigger", [
    (1_000_000, 200_000, 300_000, 800_000, 300_000),
    (1_000_000, 200_000, None, 800_000, 400_000),
    (131_072, None, None, 131_072, 98_304),
    (65_536, None, None, 65_536, 55_705),
    (65_536, 65_536, None, 65_536, 55_705),
])
def test_budget_policy_matches_builtin_without_constructing_clients(
    window, reserve, cap, capacity, trigger, monkeypatch,
):
    from agent.context_compressor import ContextCompressor
    compressor = ContextCompressor("offline", config_context_length=window, max_tokens=reserve,
                                   threshold_tokens_cap=cap, quiet_mode=True)
    expected = compressor.threshold_tokens
    def no_construction(*args, **kwargs):
        pytest.fail("pure policy must not construct a compressor or its clients")
    monkeypatch.setattr(ContextCompressor, "__init__", no_construction)
    agent = make_agent(max_tokens=reserve, _compression_model_thresholds={}, _compression_threshold_tokens_cap=cap)
    budget.apply_context_engine_compression_budget(agent, window, reason="policy")
    assert agent.context_compressor.budgets == [(capacity, trigger, "policy")]
    assert trigger == expected


@pytest.mark.parametrize("verdict", [False, None, 1, RuntimeError("declined")])
def test_auxiliary_delivery_can_relinquish_ownership_without_split_state(verdict, monkeypatch):
    agent = make_agent()
    budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="model_init")
    engine = agent.context_compressor
    engine.accept = verdict
    probe_aux(agent, monkeypatch)
    assert agent._context_engine_compression_budget_accepted is False
    assert engine.threshold_tokens == engine.private_trigger == 680_000
    assert agent._context_engine_aux_context_ceiling is None


def test_missing_hook_preserves_legacy_policy(monkeypatch):
    agent = make_agent(SimpleNamespace(threshold_tokens=680_000, threshold_percent=0.68, context_length=1_000_000))
    assert budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="model_init") is False
    probe_aux(agent, monkeypatch)
    assert agent.context_compressor.threshold_tokens == 680_000


def test_fallback_owner_refreshes_budget_and_gemini_reserve(monkeypatch):
    from agent.chat_completion_helpers import _update_fallback_context_compressor
    from agent import model_metadata
    monkeypatch.setattr(model_metadata, "get_model_context_length", lambda *a, **k: 1_000_000)
    agent = make_agent(max_tokens=None)
    for provider, reserve in (("gemini", 65_535), ("openai", 0), ("gemini", 65_535)):
        agent.provider = provider
        agent.base_url = "https://offline.invalid/v1"
        _update_fallback_context_compressor(agent)
        assert agent.context_compressor.budgets[-1] == (
            1_000_000 - reserve, int((1_000_000 - reserve) * 0.50), "fallback_model")
    agent.max_tokens = 123_456
    _update_fallback_context_compressor(agent)
    assert agent.context_compressor.budgets[-1][:2] == (876_544, 438_272)


def test_provider_scoped_configured_budget_updates_private_and_public_policy():
    agent = make_agent()
    assert budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="model_init") is True
    assert agent.context_compressor.budgets == [(800_000, 680_000, "model_init")]
    assert agent.context_compressor.threshold_tokens == agent.context_compressor.private_trigger == 680_000
    assert agent._context_engine_compression_budget_accepted is True


def probe_aux(agent, monkeypatch):
    from agent import auxiliary_client, model_metadata
    monkeypatch.setattr(auxiliary_client, "_resolve_task_provider_model", lambda task: ("openai", None, None, None, None))
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", lambda *a, **k: (SimpleNamespace(base_url="https://aux.invalid", api_key=""), "aux-model"))
    monkeypatch.setattr(model_metadata, "get_model_context_length", lambda *a, **k: 200_000)
    compression.check_compression_model_feasibility(agent)


def test_feasibility_budget_is_private_public_and_durable_for_same_runtime(monkeypatch):
    agent = make_agent()
    budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="model_init")
    probe_aux(agent, monkeypatch)
    engine = agent.context_compressor
    assert engine.threshold_tokens == engine.private_trigger == 200_000
    budget.apply_context_engine_compression_budget(agent, 900_000, reason="provider_limit")
    assert engine.budgets[-1] == (700_000, 200_000, "provider_limit")
    agent.model = "different-model"
    budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="model_switch")
    assert engine.budgets[-1] == (800_000, 400_000, "model_switch")


def test_acceptance_is_per_delivery_and_controls_later_feasibility(monkeypatch):
    agent = make_agent()
    engine = agent.context_compressor
    for verdict in (True, False, RuntimeError("declined"), 1, None, True):
        engine.accept = verdict
        engine.threshold_tokens = 680_000
        budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="runtime_change")
        assert agent._context_engine_compression_budget_accepted is (verdict is True)
        probe_aux(agent, monkeypatch)
        assert engine.threshold_tokens == (200_000 if verdict is True else 680_000)


def test_inherited_default_hook_declines_host_budget(monkeypatch):
    from agent.context_engine import ContextEngine
    class Legacy(ContextEngine):
        name = "legacy"
        def update_from_response(self, usage):
            pass
        def should_compress(self, prompt_tokens=None):
            return False
        def compress(self, messages, current_tokens=None):
            return messages
    engine = Legacy()
    assert engine.set_compression_budget(800_000, 400_000, reason="model_init") is False
    agent = make_agent(engine)
    engine.threshold_tokens = 680_000
    assert budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="model_init") is False
    probe_aux(agent, monkeypatch)
    assert engine.threshold_tokens == 680_000


@pytest.mark.parametrize("accept", [True, False])
@pytest.mark.parametrize("overrides", [
    {"_compression_threshold_percent": float("nan")},
    {"_compression_threshold_percent": float("inf")},
    {"_compression_threshold_percent": float("-inf")},
    {"_compression_threshold_percent": None},
    {"_compression_model_thresholds": {"offline": float("nan")}},
    {"max_tokens": float("inf")},
    {"_compression_threshold_tokens_cap": float("inf")},
])
def test_underivable_external_policy_declines_without_changing_engine_state(accept, overrides):
    agent = make_agent(model="offline", provider="openai", _compression_model_thresholds={})
    engine = agent.context_compressor
    assert budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="init") is True
    before = (engine.threshold_percent, engine.threshold_tokens, engine.private_trigger, list(engine.budgets))
    engine.accept = accept
    for field, value in overrides.items():
        setattr(agent, field, value)
    assert budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="invalid") is False
    assert agent._context_engine_compression_budget_accepted is False
    assert (engine.threshold_percent, engine.threshold_tokens, engine.private_trigger, engine.budgets) == before


@pytest.mark.parametrize("error", [ValueError, TypeError, OverflowError])
def test_auxiliary_derivation_failure_restores_previous_ceiling_and_budget(error, monkeypatch):
    agent = make_agent()
    engine = agent.context_compressor
    assert budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="init") is True
    assert budget.apply_auxiliary_context_ceiling(agent, 300_000) is True
    before = (engine.threshold_tokens, engine.private_trigger, list(engine.budgets))
    def fail_derivation(*args, **kwargs):
        assert agent._context_engine_aux_context_ceiling == 200_000
        raise error("underivable optional policy")
    monkeypatch.setattr(budget, "configured_compression_budget", fail_derivation)
    assert budget.apply_auxiliary_context_ceiling(agent, 200_000) is False
    assert agent._context_engine_aux_context_ceiling == 300_000
    assert agent._context_engine_compression_budget_accepted is False
    assert (engine.threshold_tokens, engine.private_trigger, engine.budgets) == before


def test_auxiliary_unexpected_derivation_error_still_restores_ceiling(monkeypatch):
    agent = make_agent()
    assert budget.apply_context_engine_compression_budget(agent, 1_000_000, reason="init") is True
    assert budget.apply_auxiliary_context_ceiling(agent, 300_000) is True
    engine = agent.context_compressor
    before = (engine.threshold_tokens, engine.private_trigger, list(engine.budgets))
    def fail_derivation(*args, **kwargs):
        raise RuntimeError("programming error, not a coercion decline")
    monkeypatch.setattr(budget, "configured_compression_budget", fail_derivation)
    with pytest.raises(RuntimeError, match="programming error"):
        budget.apply_auxiliary_context_ceiling(agent, 200_000)
    assert agent._context_engine_aux_context_ceiling == 300_000
    assert agent._context_engine_compression_budget_accepted is False
    assert (engine.threshold_tokens, engine.private_trigger, engine.budgets) == before
