"""Production AIAgent integration for route-scoped token budgets."""
import copy
from types import SimpleNamespace

import pytest

from agent import token_budget_policy
from run_agent import AIAgent


class _Compressor:
    def __init__(self):
        self.context_length = 872_000
        self.max_tokens = 128_000
        self.threshold_percent = 0.8
        self.threshold_tokens = 697_600
        self.summary_target_ratio = 0.20
        self.tail_token_budget = 139_520
        self.route_only_state = {"origin": "initial"}
        self.calls = []

    def update_model(self, *, max_tokens=None, **kwargs):
        self.calls.append({**kwargs, "max_tokens": max_tokens})
        self.context_length = kwargs["context_length"]
        self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens


def _policy_config():
    models = {}
    for model, canonical in token_budget_policy._CANONICAL_MODEL_BUDGETS.items():
        models[model] = {
            key: list(value) if key == "stages" else value
            for key, value in canonical.items()
        }
        # The new optional model schema requires deliberate per-model approval.
        # Keep this synthetic fixture valid; do not weaken production validation.
        if model in token_budget_policy._OPTIONAL_NEW_MODELS:
            models[model]["approved_stage"] = 272_000
    return {
        "token_budget_policy": {
            "enabled": True,
            "evidence_ttl_seconds": 3600,
            "safe_context_limit": 272_000,
            "approved_stage": 272_000,
            "providers": {"openai-codex": {"models": models}},
        }
    }


def _install_runtime(agent, compressor):
    agent.provider = "openai-codex"
    agent.model = "gpt-6-astra"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_mode = "codex_responses"
    agent.api_key = "test-only-opaque-token"
    agent._credential_pool = None
    agent._configured_max_tokens = 128_000
    agent.max_tokens = 128_000
    agent.context_compressor = compressor
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "compressor_context_length": compressor.context_length,
        "compressor_threshold_tokens": compressor.threshold_tokens,
    }


def _patch_policy(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: _policy_config()
    )
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )


def test_agent_initialization_applies_astra_budget_to_real_runtime_consumer(monkeypatch):
    _patch_policy(monkeypatch)
    compressor = _Compressor()

    def fake_init(agent, **_kwargs):
        _install_runtime(agent, compressor)

    monkeypatch.setattr("agent.agent_init.init_agent", fake_init)

    agent = AIAgent(
        model="gpt-6-astra",
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="test-only-opaque-token",
    )

    assert agent.max_tokens == 81_600
    assert compressor.context_length == 272_000
    assert compressor.threshold_tokens == 190_400
    assert compressor.max_tokens == 81_600
    assert agent._token_budget_status["runtime_context"] == 272_000
    assert agent._token_budget_status["runtime_soft_budget"] == 190_400
    assert agent._token_budget_status["request_output_cap"] is None
    # The policy must not overwrite the pre-policy baseline snapshot produced
    # by initialization; restore re-applies the policy after using this state.
    assert agent._primary_runtime["compressor_context_length"] == 872_000
    assert compressor.tail_token_budget == 38_080


def test_astra_policy_uses_provider_default_on_wire_without_touching_cache(monkeypatch):
    agent = object.__new__(AIAgent)
    agent._token_budget_status = {
        "output_cap_enforcement": "provider_default",
        "request_output_cap": None,
    }
    agent._ephemeral_max_output_tokens = None
    monkeypatch.setattr(
        "agent.chat_completion_helpers.build_api_kwargs",
        lambda *_args, **_kwargs: {
            "model": "gpt-6-astra",
            "max_output_tokens": 81_600,
            "prompt_cache_retention": "24h",
        },
    )

    wire = agent._build_api_kwargs([])

    assert "max_output_tokens" not in wire
    assert wire["prompt_cache_retention"] == "24h"


def test_astra_policy_keeps_explicit_ephemeral_output_cap_on_wire(monkeypatch):
    agent = object.__new__(AIAgent)
    agent._token_budget_status = {
        "output_cap_enforcement": "provider_default",
        "request_output_cap": None,
    }
    agent._ephemeral_max_output_tokens = 1_024
    monkeypatch.setattr(
        "agent.chat_completion_helpers.build_api_kwargs",
        lambda *_args, **_kwargs: {"max_output_tokens": 1_024},
    )

    assert agent._build_api_kwargs([])["max_output_tokens"] == 1_024


def test_fallback_and_primary_restore_reapply_runtime_budget(monkeypatch):
    _patch_policy(monkeypatch)
    compressor = _Compressor()
    agent = object.__new__(AIAgent)
    _install_runtime(agent, compressor)
    AIAgent._apply_runtime_token_budget(agent)

    def fake_fallback(runtime, _reason):
        runtime.provider = "anthropic"
        runtime.model = "claude-fallback"
        runtime.base_url = "https://api.anthropic.com"
        runtime.api_mode = "anthropic_messages"
        runtime.context_compressor.context_length = 200_000
        runtime.context_compressor.threshold_tokens = 160_000
        return True

    def fake_restore(runtime):
        runtime.provider = "openai-codex"
        runtime.model = "gpt-6-astra"
        runtime.base_url = "https://chatgpt.com/backend-api/codex"
        runtime.api_mode = "codex_responses"
        runtime.context_compressor.context_length = 872_000
        runtime.context_compressor.threshold_tokens = 697_600
        return True

    monkeypatch.setattr("agent.chat_completion_helpers.try_activate_fallback", fake_fallback)
    monkeypatch.setattr("agent.agent_runtime_helpers.restore_primary_runtime", fake_restore)

    assert AIAgent._try_activate_fallback(agent) is True
    assert agent.max_tokens == 128_000
    assert agent._token_budget_status is None
    assert compressor.context_length == 200_000

    assert AIAgent._restore_primary_runtime(agent) is True
    assert agent.max_tokens == 81_600
    assert compressor.context_length == 272_000
    assert compressor.threshold_tokens == 190_400
    assert agent._token_budget_status["runtime_output_reserve"] == 81_600


def test_policy_removal_restores_complete_same_route_compressor_baseline(monkeypatch):
    _patch_policy(monkeypatch)
    compressor = _Compressor()
    agent = object.__new__(AIAgent)
    _install_runtime(agent, compressor)
    baseline = copy.deepcopy(vars(compressor))

    AIAgent._apply_runtime_token_budget(agent)
    assert compressor.threshold_tokens == 190_400
    assert compressor.tail_token_budget == 38_080
    assert compressor.route_only_state == {"origin": "initial"}

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    assert AIAgent._apply_runtime_token_budget(agent) is None
    assert vars(compressor) == baseline
    assert agent.max_tokens == 128_000
    assert agent._token_budget_status is None


def test_hot_reload_legacy_three_models_fails_before_switch_then_migrates_and_removes(
    monkeypatch,
):
    valid = _policy_config()
    legacy = copy.deepcopy(valid)
    del legacy["token_budget_policy"]["providers"]["openai-codex"]["models"][
        "gpt-5.6-luna"
    ]
    current = {"config": legacy}
    calls = []
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: current["config"]
    )
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    compressor = _Compressor()
    agent = object.__new__(AIAgent)
    _install_runtime(agent, compressor)
    before_agent = copy.deepcopy(
        {name: value for name, value in vars(agent).items() if name != "context_compressor"}
    )
    before_compressor = copy.deepcopy(vars(compressor))

    def fake_switch(runtime, *_args):
        calls.append("switch")
        runtime.model = "gpt-5.6-sol"
        runtime.context_compressor.context_length = 1_000_000
        runtime.context_compressor.threshold_tokens = 800_000
        runtime.context_compressor.route_only_state = {"origin": "switch"}
        return "switched"

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch)
    with pytest.raises(token_budget_policy.TokenBudgetPolicyError, match="exactly"):
        agent.switch_model("gpt-5.6-sol", "openai-codex")
    assert calls == []
    assert (
        {name: value for name, value in vars(agent).items() if name != "context_compressor"}
        == before_agent
    )
    assert vars(compressor) == before_compressor

    current["config"] = valid
    assert agent.switch_model("gpt-5.6-sol", "openai-codex") == "switched"
    assert compressor.threshold_tokens == 217_600
    current["config"] = {}
    assert AIAgent._apply_runtime_token_budget(agent) is None
    assert compressor.context_length == 1_000_000
    assert compressor.threshold_tokens == 800_000
    assert compressor.route_only_state == {"origin": "switch"}


def test_invalid_policy_preflights_fallback_and_restore_without_mutation(monkeypatch):
    invalid = _policy_config()
    del invalid["token_budget_policy"]["providers"]["openai-codex"]["models"][
        "gpt-5.6-luna"
    ]
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: invalid)
    compressor = _Compressor()
    agent = object.__new__(AIAgent)
    _install_runtime(agent, compressor)
    before = copy.deepcopy(vars(compressor))
    calls = []
    monkeypatch.setattr(
        "agent.chat_completion_helpers.try_activate_fallback",
        lambda *_args: calls.append("fallback") or True,
    )
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.restore_primary_runtime",
        lambda *_args: calls.append("restore") or True,
    )

    with pytest.raises(token_budget_policy.TokenBudgetPolicyError, match="exactly"):
        agent._try_activate_fallback()
    with pytest.raises(token_budget_policy.TokenBudgetPolicyError, match="exactly"):
        agent._restore_primary_runtime()
    assert calls == []
    assert vars(compressor) == before


def test_policy_apply_failure_after_switch_rolls_back_full_runtime(monkeypatch):
    _patch_policy(monkeypatch)
    compressor = _Compressor()
    agent = object.__new__(AIAgent)
    _install_runtime(agent, compressor)
    agent.client = "primary-client"
    before_agent = copy.deepcopy(
        {name: value for name, value in vars(agent).items() if name != "context_compressor"}
    )
    before_compressor = copy.deepcopy(vars(compressor))

    class ReplacementClient:
        closed = False

        def close(self):
            self.closed = True

    replacement_client = ReplacementClient()

    def fake_switch(runtime, *_args):
        runtime.model = "gpt-5.6-sol"
        runtime.client = replacement_client
        runtime._primary_runtime = {"model": "gpt-5.6-sol"}
        runtime.context_compressor.context_length = 999_999
        runtime.context_compressor.route_only_state = {"origin": "mutated"}
        return "switched"

    def fail_apply(runtime, _config=None):
        runtime.max_tokens = 1
        runtime.context_compressor.threshold_tokens = 1
        raise RuntimeError("policy sync failed")

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fake_switch)
    monkeypatch.setattr(AIAgent, "_apply_runtime_token_budget", fail_apply)
    with pytest.raises(RuntimeError, match="policy sync failed"):
        agent.switch_model("gpt-5.6-sol", "openai-codex")
    assert (
        {name: value for name, value in vars(agent).items() if name != "context_compressor"}
        == before_agent
    )
    assert vars(compressor) == before_compressor
    assert replacement_client.closed is True


def test_policy_apply_failure_after_fallback_or_restore_rolls_back(monkeypatch):
    _patch_policy(monkeypatch)

    def make_agent():
        compressor = _Compressor()
        agent = object.__new__(AIAgent)
        _install_runtime(agent, compressor)
        agent.client = "primary-client"
        return agent, compressor

    def mutate(runtime, *_args):
        runtime.provider = "anthropic"
        runtime.model = "fallback-model"
        runtime.client = "fallback-client"
        runtime.context_compressor.context_length = 42
        runtime.context_compressor.route_only_state = {"origin": "mutated"}
        return True

    def fail_apply(runtime, _config=None):
        runtime.max_tokens = 1
        runtime.context_compressor.threshold_tokens = 1
        raise RuntimeError("policy sync failed")

    monkeypatch.setattr("agent.chat_completion_helpers.try_activate_fallback", mutate)
    monkeypatch.setattr("agent.agent_runtime_helpers.restore_primary_runtime", mutate)
    monkeypatch.setattr(AIAgent, "_apply_runtime_token_budget", fail_apply)

    for operation in ("fallback", "restore"):
        agent, compressor = make_agent()
        before_agent = copy.deepcopy(
            {
                name: value
                for name, value in vars(agent).items()
                if name != "context_compressor"
            }
        )
        before_compressor = copy.deepcopy(vars(compressor))
        with pytest.raises(RuntimeError, match="policy sync failed"):
            if operation == "fallback":
                agent._try_activate_fallback()
            else:
                agent._restore_primary_runtime()
        assert (
            {
                name: value
                for name, value in vars(agent).items()
                if name != "context_compressor"
            }
            == before_agent
        )
        assert vars(compressor) == before_compressor


@pytest.mark.parametrize("operation", ("switch", "fallback", "restore"))
def test_false_transition_result_restores_snapshot_and_closes_only_replacement(
    monkeypatch, operation
):
    """A False result is a failed transition, even after partial mutation."""
    _patch_policy(monkeypatch)
    compressor = _Compressor()
    agent = object.__new__(AIAgent)
    _install_runtime(agent, compressor)
    AIAgent._apply_runtime_token_budget(agent)

    class Client:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    primary_client = Client()
    replacement_client = Client()
    agent.client = primary_client
    before_route = (
        agent.provider,
        agent.model,
        agent.base_url,
        agent.api_mode,
    )
    before_compressor = copy.deepcopy(vars(compressor))
    before_policy = copy.deepcopy(agent._token_budget_status)

    def mutate_then_fail(runtime, *_args):
        runtime.provider = "anthropic"
        runtime.model = "claude-fallback"
        runtime.base_url = "https://api.anthropic.com"
        runtime.api_mode = "anthropic_messages"
        runtime.client = replacement_client
        runtime.context_compressor.context_length = 42
        runtime.context_compressor.threshold_tokens = 1
        runtime.context_compressor.route_only_state = {"origin": "failed-transition"}
        runtime._token_budget_status = {"route": "failed-transition"}
        return False

    if operation == "switch":
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate_then_fail)
        result = agent.switch_model("gpt-5.6-sol", "openai-codex")
    elif operation == "fallback":
        monkeypatch.setattr(
            "agent.chat_completion_helpers.try_activate_fallback", mutate_then_fail
        )
        result = agent._try_activate_fallback()
    else:
        monkeypatch.setattr(
            "agent.agent_runtime_helpers.restore_primary_runtime", mutate_then_fail
        )
        result = agent._restore_primary_runtime()

    assert result is False
    assert (agent.provider, agent.model, agent.base_url, agent.api_mode) == before_route
    assert agent.client is primary_client
    assert agent.context_compressor is compressor
    assert vars(compressor) == before_compressor
    assert agent._token_budget_status == before_policy
    assert primary_client.close_calls == 0
    assert replacement_client.close_calls == 1


@pytest.mark.parametrize("operation", ("switch", "fallback", "restore"))
def test_helper_exception_restores_snapshot_and_closes_only_replacement(
    monkeypatch, operation
):
    """Exception rollback uses the same ownership rules as a False result."""
    _patch_policy(monkeypatch)
    compressor = _Compressor()
    agent = object.__new__(AIAgent)
    _install_runtime(agent, compressor)

    class Client:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    primary_client = Client()
    replacement_client = Client()
    agent.client = primary_client
    before_route = (
        agent.provider,
        agent.model,
        agent.base_url,
        agent.api_mode,
    )
    before_compressor = copy.deepcopy(vars(compressor))

    def mutate_then_raise(runtime, *_args):
        runtime.provider = "anthropic"
        runtime.model = "claude-fallback"
        runtime.base_url = "https://api.anthropic.com"
        runtime.api_mode = "anthropic_messages"
        runtime.client = replacement_client
        runtime.context_compressor.context_length = 42
        raise RuntimeError("helper failed")

    if operation == "switch":
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate_then_raise)
        invoke = lambda: agent.switch_model("gpt-5.6-sol", "openai-codex")
    elif operation == "fallback":
        monkeypatch.setattr(
            "agent.chat_completion_helpers.try_activate_fallback", mutate_then_raise
        )
        invoke = agent._try_activate_fallback
    else:
        monkeypatch.setattr(
            "agent.agent_runtime_helpers.restore_primary_runtime", mutate_then_raise
        )
        invoke = agent._restore_primary_runtime

    with pytest.raises(RuntimeError, match="helper failed"):
        invoke()
    assert (agent.provider, agent.model, agent.base_url, agent.api_mode) == before_route
    assert agent.client is primary_client
    assert agent.context_compressor is compressor
    assert vars(compressor) == before_compressor
    assert primary_client.close_calls == 0
    assert replacement_client.close_calls == 1


@pytest.mark.parametrize("operation", ("switch", "fallback", "restore"))
def test_false_transition_rolls_back_credentials_and_policy_baselines_in_place(
    monkeypatch, operation
):
    """Credential/policy mutations are rollback-critical even when return is False."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    compressor = _Compressor()
    _install_runtime(agent, compressor)

    class Pool:
        def __init__(self):
            self.state = {"entry": "primary", "attempts": [0]}

    pool = Pool()
    baselines = {("primary",): {"cap": 128_000, "nested": ["original"]}}
    policy_config = {"token_budget_policy": {"enabled": True, "source": "original"}}
    agent._credential_pool = pool
    agent._credential_pool_entry_id = "primary-entry"
    agent._configured_max_tokens_captured = False
    agent._token_budget_route_baselines = baselines
    agent._token_budget_policy_config = policy_config
    before_baselines = copy.deepcopy(baselines)
    before_policy = copy.deepcopy(policy_config)

    def mutate_then_fail(runtime, *_args):
        runtime.api_key = "replacement-key"
        runtime._credential_pool = Pool()
        runtime._credential_pool_entry_id = "replacement-entry"
        runtime._configured_max_tokens_captured = True
        runtime._token_budget_route_baselines[("primary",)]["nested"].append("mutated")
        runtime._token_budget_policy_config["token_budget_policy"]["source"] = "mutated"
        pool.state["attempts"].append(1)
        return False

    if operation == "switch":
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate_then_fail)
        result = agent.switch_model("gpt-5.6-sol", "openai-codex")
    elif operation == "fallback":
        monkeypatch.setattr(
            "agent.chat_completion_helpers.try_activate_fallback", mutate_then_fail
        )
        result = agent._try_activate_fallback()
    else:
        monkeypatch.setattr(
            "agent.agent_runtime_helpers.restore_primary_runtime", mutate_then_fail
        )
        result = agent._restore_primary_runtime()

    assert result is False
    assert agent.api_key == "test-only-opaque-token"
    assert agent._credential_pool is pool
    assert agent._credential_pool_entry_id == "primary-entry"
    assert pool.state == {"entry": "primary", "attempts": [0]}
    assert agent._configured_max_tokens_captured is False
    assert agent._token_budget_route_baselines is baselines
    assert baselines == before_baselines
    assert agent._token_budget_policy_config is policy_config
    assert policy_config == before_policy


@pytest.mark.parametrize("operation", ("switch", "fallback", "restore"))
def test_false_without_critical_mutation_keeps_bookkeeping(monkeypatch, operation):
    """Ordinary False returns retain retry/cooldown bookkeeping by design."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    agent.retry_bookkeeping = []

    def record_noop(runtime, *_args):
        runtime.retry_bookkeeping.append("cooldown")
        return False

    if operation == "switch":
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", record_noop)
        result = agent.switch_model("gpt-5.6-sol", "openai-codex")
    elif operation == "fallback":
        monkeypatch.setattr(
            "agent.chat_completion_helpers.try_activate_fallback", record_noop
        )
        result = agent._try_activate_fallback()
    else:
        monkeypatch.setattr(
            "agent.agent_runtime_helpers.restore_primary_runtime", record_noop
        )
        result = agent._restore_primary_runtime()

    assert result is False
    assert agent.retry_bookkeeping == ["cooldown"]


@pytest.mark.parametrize("operation", ("switch", "fallback", "restore"))
def test_rollback_restores_mutated_original_client_without_closing_it(monkeypatch, operation):
    """Rollback repairs an existing transport in place and never retires it."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())

    class Client:
        def __init__(self):
            self.headers = {"route": "primary", "attempts": []}
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    primary_client = Client()
    agent.client = primary_client

    def mutate_then_fail(runtime, *_args):
        primary_client.headers["route"] = "replacement"
        primary_client.headers["attempts"].append("attempted")
        return False

    if operation == "switch":
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate_then_fail)
        result = agent.switch_model("gpt-5.6-sol", "openai-codex")
    elif operation == "fallback":
        monkeypatch.setattr(
            "agent.chat_completion_helpers.try_activate_fallback", mutate_then_fail
        )
        result = agent._try_activate_fallback()
    else:
        monkeypatch.setattr(
            "agent.agent_runtime_helpers.restore_primary_runtime", mutate_then_fail
        )
        result = agent._restore_primary_runtime()

    assert result is False
    assert agent.client is primary_client
    assert primary_client.headers == {"route": "primary", "attempts": []}
    assert primary_client.close_calls == 0


@pytest.mark.parametrize("operation", ("switch", "fallback", "restore"))
def test_exception_rollback_restores_slotted_compressor_state(monkeypatch, operation):
    """Slot-only compressors receive the same in-place rollback guarantee."""
    _patch_policy(monkeypatch)

    class SlottedCompressor:
        __slots__ = ("context_length", "threshold_tokens", "slot_state")

        def __init__(self):
            self.context_length = 872_000
            self.threshold_tokens = 697_600
            self.slot_state = {"origin": ["primary"]}

    agent = object.__new__(AIAgent)
    compressor = SlottedCompressor()
    _install_runtime(agent, compressor)

    def mutate_then_raise(runtime, *_args):
        runtime.context_compressor.context_length = 42
        runtime.context_compressor.threshold_tokens = 1
        runtime.context_compressor.slot_state["origin"].append("mutated")
        raise RuntimeError("slot compressor failed")

    if operation == "switch":
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate_then_raise)
        invoke = lambda: agent.switch_model("gpt-5.6-sol", "openai-codex")
    elif operation == "fallback":
        monkeypatch.setattr(
            "agent.chat_completion_helpers.try_activate_fallback", mutate_then_raise
        )
        invoke = agent._try_activate_fallback
    else:
        monkeypatch.setattr(
            "agent.agent_runtime_helpers.restore_primary_runtime", mutate_then_raise
        )
        invoke = agent._restore_primary_runtime

    with pytest.raises(RuntimeError, match="slot compressor failed"):
        invoke()
    assert agent.context_compressor is compressor
    assert compressor.context_length == 872_000
    assert compressor.threshold_tokens == 697_600
    assert compressor.slot_state == {"origin": ["primary"]}


def test_rollback_restores_slotted_client_and_survives_replacement_close_failure(
    monkeypatch, caplog
):
    """Slot-backed transports restore safely; failed replacement close is logged."""
    _patch_policy(monkeypatch)
    caplog.set_level("DEBUG")
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())

    class SlottedClient:
        __slots__ = ("route", "attempts")

        def __init__(self):
            self.route = "primary"
            self.attempts = []

    class BrokenReplacement:
        def close(self):
            raise RuntimeError("close failed")

    primary_client = SlottedClient()
    agent.client = primary_client

    def mutate_then_fail(runtime, *_args):
        primary_client.route = "replacement"
        primary_client.attempts.append("attempted")
        runtime.client = BrokenReplacement()
        return False

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate_then_fail)
    assert agent.switch_model("gpt-5.6-sol", "openai-codex") is False
    assert agent.client is primary_client
    assert primary_client.route == "primary"
    assert primary_client.attempts == []
    assert "failed to close rolled-back client" in caplog.text


@pytest.mark.parametrize("operation", ("switch", "fallback", "restore"))
def test_failed_transition_restores_prompt_reasoning_transport_and_fallback_markers(
    monkeypatch, operation
):
    """Every route-visible cache/marker is transactional, including aliases."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    shared_cache = {"nested": ["primary"]}
    agent._cached_system_prompt = "primary prompt"
    agent.reasoning_config = shared_cache
    agent._transport_cache = shared_cache
    agent._fallback_activated = False
    agent._provider_fallback_active = False
    agent._provider_fallback_route = None
    agent._fallback_index = 0
    agent._pending_fallback_notice = None

    def mutate_then_fail(runtime, *_args):
        runtime._cached_system_prompt = "fallback prompt"
        runtime.reasoning_config["nested"].append("failed")
        runtime._transport_cache = {"replacement": True}
        runtime._fallback_activated = True
        runtime._provider_fallback_active = True
        runtime._provider_fallback_route = ("anthropic", "claude-fallback")
        runtime._fallback_index = 2
        runtime._pending_fallback_notice = "trying fallback"
        return False

    if operation == "switch":
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate_then_fail)
        result = agent.switch_model("gpt-5.6-sol", "openai-codex")
    elif operation == "fallback":
        monkeypatch.setattr(
            "agent.chat_completion_helpers.try_activate_fallback", mutate_then_fail
        )
        result = agent._try_activate_fallback()
    else:
        monkeypatch.setattr(
            "agent.agent_runtime_helpers.restore_primary_runtime", mutate_then_fail
        )
        result = agent._restore_primary_runtime()

    assert result is False
    assert agent._cached_system_prompt == "primary prompt"
    assert agent.reasoning_config is shared_cache
    assert agent._transport_cache is shared_cache
    assert shared_cache == {"nested": ["primary"]}
    assert agent._fallback_activated is False
    assert agent._provider_fallback_active is False
    assert agent._provider_fallback_route is None
    assert agent._fallback_index == 0
    assert agent._pending_fallback_notice is None


def test_transition_fails_before_helper_for_opaque_mutable_policy_state(monkeypatch):
    """An uncopyable non-critical object fails closed rather than leaking state."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    calls = []

    class OpaqueMutable:
        __slots__ = ()

        @property
        def state(self):
            return opaque_state

    opaque_state = {"route": "primary"}
    agent.reasoning_config = {"opaque": OpaqueMutable()}

    def must_not_run(runtime, *_args):
        calls.append("switch")
        opaque_state["route"] = "failed"
        return False

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", must_not_run)
    with pytest.raises(RuntimeError, match="cannot snapshot"):
        agent.switch_model("gpt-5.6-sol", "openai-codex")
    assert calls == []
    assert opaque_state == {"route": "primary"}


def test_rollback_restores_private_inherited_slot_backing_property(monkeypatch):
    """Private slots must be mangled by their declaring MRO class."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())

    class BaseClient:
        __slots__ = ("__state",)

        def __init__(self):
            self.__state = {"route": "primary", "attempts": []}

        @property
        def state(self):
            return self.__state

    class Client(BaseClient):
        __slots__ = ("__child_state",)

        def __init__(self):
            super().__init__()
            self.__child_state = {"mode": "primary"}

        @property
        def child_state(self):
            return self.__child_state

    client = Client()
    agent.client = client

    def mutate_then_fail(runtime, *_args):
        client.state["route"] = "failed"
        client.state["attempts"].append("attempted")
        client.child_state["mode"] = "failed"
        return False

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate_then_fail)
    assert agent.switch_model("gpt-5.6-sol", "openai-codex") is False
    assert client.state == {"route": "primary", "attempts": []}
    assert client.child_state == {"mode": "primary"}


def test_policy_apply_failure_restores_transactional_prompt_cache_and_markers(monkeypatch):
    """Post-helper policy failures use the same complete graph rollback."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    shared_cache = {"messages": ["primary"]}
    agent._cached_system_prompt = "primary prompt"
    agent.reasoning_config = shared_cache
    agent._transport_cache = shared_cache
    agent._fallback_activated = False
    agent._provider_fallback_active = False
    agent._provider_fallback_route = None
    agent._fallback_index = 0
    agent._pending_fallback_notice = None

    def mutate(runtime, *_args):
        runtime._cached_system_prompt = "fallback prompt"
        runtime.reasoning_config["messages"].append("fallback")
        runtime._transport_cache = {"replacement": True}
        runtime._fallback_activated = True
        runtime._provider_fallback_active = True
        runtime._provider_fallback_route = ("anthropic", "claude-fallback")
        runtime._fallback_index = 1
        runtime._pending_fallback_notice = "fallback pending"
        return "switched"

    def fail_apply(*_args):
        raise RuntimeError("policy sync failed")

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate)
    monkeypatch.setattr(AIAgent, "_apply_runtime_token_budget", fail_apply)
    with pytest.raises(RuntimeError, match="policy sync failed"):
        agent.switch_model("gpt-5.6-sol", "openai-codex")
    assert agent._cached_system_prompt == "primary prompt"
    assert agent.reasoning_config is shared_cache
    assert agent._transport_cache is shared_cache
    assert shared_cache == {"messages": ["primary"]}
    assert agent._fallback_activated is False
    assert agent._provider_fallback_active is False
    assert agent._provider_fallback_route is None
    assert agent._fallback_index == 0
    assert agent._pending_fallback_notice is None


def test_policy_failure_restores_current_switch_side_effects(monkeypatch):
    """0.21.5 route metadata must roll back with the transport and compressor."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    capabilities = {"native_compaction": {"enabled": False}}
    overrides = {"extra_body": {"route": ["primary"]}}
    providers = [{"name": "primary"}]
    fallback_chain = [{"provider": "anthropic", "model": "claude-primary"}]
    agent.runtime_capabilities = capabilities
    agent.request_overrides = overrides
    agent._custom_providers = providers
    agent._fallback_chain = fallback_chain
    agent._fallback_model = fallback_chain[0]
    agent._credential_pool_revert_id = "primary-credential"
    agent._compression_feasibility_checked = True
    agent._last_feasibility_notice = "primary notice"
    agent._compression_warning = "primary warning"
    agent._consecutive_stale_streams = 2

    def mutate(runtime, *_args):
        runtime.runtime_capabilities["native_compaction"]["enabled"] = True
        runtime.request_overrides["extra_body"]["route"].append("fallback")
        runtime._custom_providers.append({"name": "fallback"})
        runtime._fallback_chain[:] = []
        runtime._fallback_model = None
        runtime._credential_pool_revert_id = None
        runtime._compression_feasibility_checked = False
        runtime._last_feasibility_notice = "fallback notice"
        runtime._compression_warning = "fallback warning"
        runtime._consecutive_stale_streams = 0
        return "switched"

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", mutate)
    monkeypatch.setattr(
        AIAgent,
        "_apply_runtime_token_budget",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("policy sync failed")),
    )

    with pytest.raises(RuntimeError, match="policy sync failed"):
        agent.switch_model("gpt-5.6-sol", "openai-codex")

    assert agent.runtime_capabilities is capabilities
    assert capabilities == {"native_compaction": {"enabled": False}}
    assert agent.request_overrides is overrides
    assert overrides == {"extra_body": {"route": ["primary"]}}
    assert agent._custom_providers is providers
    assert providers == [{"name": "primary"}]
    assert agent._fallback_chain is fallback_chain
    assert fallback_chain == [{"provider": "anthropic", "model": "claude-primary"}]
    assert agent._fallback_model is fallback_chain[0]
    assert agent._credential_pool_revert_id == "primary-credential"
    assert agent._compression_feasibility_checked is True
    assert agent._last_feasibility_notice == "primary notice"
    assert agent._compression_warning == "primary warning"
    assert agent._consecutive_stale_streams == 2
