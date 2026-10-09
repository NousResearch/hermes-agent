"""Production AIAgent integration for route-scoped token budgets."""
import copy
from datetime import datetime, timezone
import threading
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


def _owned_compressor_state(compressor):
    """Fields the agent explicitly owns across a route transaction."""
    names = (
        "context_length",
        "max_tokens",
        "threshold_percent",
        "threshold_tokens",
        "tail_token_budget",
        "summary_target_ratio",
    )
    return copy.deepcopy(
        {name: getattr(compressor, name) for name in names if hasattr(compressor, name)}
    )


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


def test_codex_backend_consumes_unsupported_ephemeral_output_cap(monkeypatch):
    """The real Responses builder must omit a cap Codex rejects, but record why."""
    from agent.transports.codex import ResponsesApiTransport

    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    agent._token_budget_status = None
    agent._ephemeral_max_output_tokens = 1_024
    agent._ephemeral_reasoning_off = False
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    agent.request_overrides = {}
    agent.service_tier = None
    agent.tools = []
    agent.session_id = None
    agent.platform = None
    agent._session_db = None
    agent._persist_disabled = False
    agent._codex_reasoning_replay_enabled = True
    agent.text_verbosity = None
    transport = ResponsesApiTransport()
    agent._get_transport = lambda: transport
    agent._prepare_messages_for_non_vision_model = lambda messages: messages
    agent._resolved_api_call_timeout = lambda: None
    monkeypatch.setattr(
        "agent.native_compaction.native_compaction_context_management",
        lambda *_args, **_kwargs: None,
    )

    wire = agent._build_api_kwargs([{"role": "user", "content": "continue"}])

    assert "max_output_tokens" not in wire
    assert agent._ephemeral_max_output_tokens is None
    assert agent._last_ephemeral_output_cap_disposition == "unsupported"


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
    before_compressor = _owned_compressor_state(compressor)

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
    assert _owned_compressor_state(compressor) == before_compressor

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
    before_compressor = _owned_compressor_state(compressor)

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
    assert _owned_compressor_state(compressor) == before_compressor
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
        before_compressor = _owned_compressor_state(compressor)
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
        assert _owned_compressor_state(compressor) == before_compressor


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
    before_compressor = _owned_compressor_state(compressor)
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
    assert _owned_compressor_state(compressor) == before_compressor
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
    before_compressor = _owned_compressor_state(compressor)

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
    assert _owned_compressor_state(compressor) == before_compressor
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
    assert pool.state == {"entry": "primary", "attempts": [0, 1]}
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
def test_rollback_treats_original_client_as_atomic_without_closing_it(monkeypatch, operation):
    """Rollback restores client identity without reflecting into SDK internals."""
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
    assert primary_client.headers == {"route": "replacement", "attempts": ["attempted"]}
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
    assert compressor.slot_state == {"origin": ["primary", "mutated"]}


def test_rollback_keeps_slotted_client_atomic_and_survives_replacement_close_failure(
    monkeypatch, caplog
):
    """Slot-backed transports stay atomic; failed replacement close is logged."""
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
    assert primary_client.route == "replacement"
    assert primary_client.attempts == ["attempted"]
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
    assert agent._fallback_index == 2
    assert agent._pending_fallback_notice is None


def test_sdk_like_client_is_atomic_and_does_not_mask_original_helper_error(monkeypatch):
    """SDK objects with classes, locks and callables are never reflected recursively."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    calls = []

    class SDKLikeClient:
        def __init__(self):
            self.factory = dict
            self.lock = threading.Lock()
            self.callback = lambda: None

    client = SDKLikeClient()
    agent.client = client

    def fail(runtime, *_args):
        calls.append("switch")
        raise ValueError("original helper error")

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", fail)
    with pytest.raises(ValueError, match="original helper error"):
        agent.switch_model("gpt-5.6-sol", "openai-codex")
    assert calls == ["switch"]
    assert agent.client is client


def test_rollback_does_not_reflect_private_inherited_client_slots(monkeypatch):
    """Private SDK slots remain opaque to the transaction snapshot."""
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
    assert client.state == {"route": "failed", "attempts": ["attempted"]}
    assert client.child_state == {"mode": "failed"}


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


def test_request_reapplies_policy_after_same_route_account_rotation(monkeypatch):
    """A promoted account must not lend its baseline/evidence to a rotated account."""
    config = _policy_config()
    config["token_budget_policy"]["approved_stage"] = 500_000
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)

    class Pool:
        entry_id = "account-a"

        def current(self):
            return SimpleNamespace(id=self.entry_id)

    pool = Pool()

    def evidence(**kwargs):
        if kwargs["account_key"] != "account-a":
            return None
        return token_budget_policy.ProviderContextEvidence(
            provider=kwargs["provider"],
            model=kwargs["model"],
            observed_context=500_000,
            observed_at=datetime.now(timezone.utc),
            source="codex_oauth_catalog",
            account_key=kwargs["account_key"],
            route_key="chatgpt.com/backend-api/codex",
        )

    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence", evidence
    )
    agent = object.__new__(AIAgent)
    compressor = _Compressor()
    _install_runtime(agent, compressor)
    agent._credential_pool = pool
    agent._credential_pool_entry_id = "account-a"
    AIAgent._apply_runtime_token_budget(agent)
    assert compressor.context_length == 500_000

    pool.entry_id = "account-b"
    agent._credential_pool_entry_id = "account-b"
    agent.api_key = "test-only-account-b-token"
    observed = {}

    def build(runtime, *_args, **_kwargs):
        observed.update(
            max_tokens=runtime.max_tokens,
            context_length=runtime.context_compressor.context_length,
        )
        return {"max_output_tokens": runtime.max_tokens}

    monkeypatch.setattr("agent.chat_completion_helpers.build_api_kwargs", build)
    agent._build_api_kwargs([])

    assert observed == {"max_tokens": 81_600, "context_length": 272_000}
    assert len(agent._token_budget_route_baselines) == 2
    account_keys = {route[-1] for route in agent._token_budget_route_baselines}
    assert account_keys == {"account-a", "account-b"}

    assert AIAgent._apply_runtime_token_budget(agent, {}) is None
    assert compressor.context_length == 872_000
    assert compressor.threshold_tokens == 697_600
    assert compressor.route_only_state == {"origin": "initial"}


def test_invalid_policy_preflights_before_agent_initialization(monkeypatch):
    """Malformed policy must fail before init allocates any runtime resources."""
    invalid = _policy_config()
    del invalid["token_budget_policy"]["providers"]["openai-codex"]["models"][
        "gpt-5.6-luna"
    ]
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: invalid)
    calls = []

    def fake_init(agent, **_kwargs):
        calls.append(agent)
        _install_runtime(agent, _Compressor())

    monkeypatch.setattr("agent.agent_init.init_agent", fake_init)

    with pytest.raises(token_budget_policy.TokenBudgetPolicyError, match="exactly"):
        AIAgent(model="gpt-6-astra", provider="openai-codex")
    assert calls == []


def test_initial_policy_apply_failure_rolls_back_and_closes_initialized_client(
    monkeypatch,
):
    """A compressor failure during init restores its baseline and retires resources."""
    _patch_policy(monkeypatch)
    holder = {}

    class Client:
        close_calls = 0

        def close(self):
            self.close_calls += 1

    class ExplodingCompressor(_Compressor):
        def update_model(self, *, max_tokens=None, **kwargs):
            super().update_model(max_tokens=max_tokens, **kwargs)
            self.route_only_state["origin"] = "failed-policy-apply"
            raise RuntimeError("compressor update failed")

    def fake_init(agent, **_kwargs):
        compressor = ExplodingCompressor()
        _install_runtime(agent, compressor)
        agent.client = Client()
        holder["agent"] = agent
        holder["client"] = agent.client
        holder["compressor"] = compressor
        holder["baseline"] = _owned_compressor_state(compressor)

    monkeypatch.setattr("agent.agent_init.init_agent", fake_init)

    with pytest.raises(RuntimeError, match="compressor update failed"):
        AIAgent(model="gpt-6-astra", provider="openai-codex")

    assert holder["agent"].max_tokens == 128_000
    assert _owned_compressor_state(holder["compressor"]) == holder["baseline"]
    assert holder["client"].close_calls == 1


@pytest.mark.parametrize("operation", ("fallback", "restore"))
def test_policy_off_false_transition_preserves_upstream_bookkeeping(
    monkeypatch, operation
):
    """Disabled policy is an integral fast-path, including legitimate False mutations."""
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    agent._fallback_index = 0

    def record(runtime, *_args, **_kwargs):
        runtime._fallback_index = 7
        runtime._credential_pool_entry_id = "upstream-bookkeeping"
        return False

    if operation == "fallback":
        monkeypatch.setattr(
            "agent.chat_completion_helpers.try_activate_fallback", record
        )
        result = agent._try_activate_fallback()
    else:
        monkeypatch.setattr(
            "agent.agent_runtime_helpers.restore_primary_runtime", record
        )
        result = agent._restore_primary_runtime()

    assert result is False
    assert agent._fallback_index == 7
    assert agent._credential_pool_entry_id == "upstream-bookkeeping"


def test_policy_failure_discards_staged_billing_route_write(monkeypatch):
    """The durable billing route is published only after policy commit."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())

    class SessionDB:
        def __init__(self):
            self.writes = []

        def update_session_billing_route(self, *args, **kwargs):
            self.writes.append((args, kwargs))

    session_db = SessionDB()
    agent._session_db = session_db
    agent.session_id = "session-1"

    def switch(runtime, *_args):
        runtime.provider = "anthropic"
        runtime.model = "claude-fallback"
        runtime.base_url = "https://api.anthropic.com"
        runtime.api_mode = "anthropic_messages"
        from agent.agent_runtime_helpers import _persist_switch_billing_route

        _persist_switch_billing_route(runtime)
        return "switched"

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", switch)
    monkeypatch.setattr(
        AIAgent,
        "_apply_runtime_token_budget",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("policy sync failed")),
    )

    with pytest.raises(RuntimeError, match="policy sync failed"):
        agent.switch_model("claude-fallback", "anthropic")
    assert session_db.writes == []


def test_policy_failure_discards_staged_fallback_notification(monkeypatch):
    """A failed policy commit must not publish a fallback that was rolled back."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    notifications = []
    agent._buffer_diagnostic_status = notifications.append

    def fallback(runtime, *_args):
        runtime.provider = "anthropic"
        runtime.model = "claude-fallback"
        from agent.chat_completion_helpers import _buffer_fallback_notice

        _buffer_fallback_notice(runtime, "fallback activated")
        return True

    monkeypatch.setattr(
        "agent.chat_completion_helpers.try_activate_fallback", fallback
    )
    monkeypatch.setattr(
        AIAgent,
        "_apply_runtime_token_budget",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("policy sync failed")),
    )

    with pytest.raises(RuntimeError, match="policy sync failed"):
        agent._try_activate_fallback()
    assert notifications == []


def test_request_build_failure_restores_consumed_one_shot_state(monkeypatch):
    """A builder exception cannot consume continuation output/reasoning state."""
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    wire_reasoning = {"enabled": True, "effort": "high"}
    agent._ephemeral_max_output_tokens = 32_768
    agent._ephemeral_reasoning_off = True
    agent._wire_reasoning_config = wire_reasoning

    def fail_after_consumption(runtime, *_args, **_kwargs):
        runtime._ephemeral_max_output_tokens = None
        runtime._ephemeral_reasoning_off = False
        runtime._wire_reasoning_config = {"enabled": False, "effort": "none"}
        raise RuntimeError("request build failed")

    monkeypatch.setattr(
        "agent.chat_completion_helpers.build_api_kwargs", fail_after_consumption
    )

    with pytest.raises(RuntimeError, match="request build failed"):
        agent._build_api_kwargs([])
    assert agent._ephemeral_max_output_tokens == 32_768
    assert agent._ephemeral_reasoning_off is True
    assert agent._wire_reasoning_config is wire_reasoning


@pytest.mark.parametrize("read_result", ("raise", "sentinel"))
def test_first_init_config_read_failure_fails_closed(monkeypatch, read_result):
    """An unreadable first config cannot silently disable the policy."""
    from hermes_cli.config_read_errors import FailedConfigRead

    failure = OSError("config unavailable")

    def load():
        if read_result == "raise":
            raise failure
        return FailedConfigRead({}, error=failure)

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", load)
    initialized = []
    monkeypatch.setattr(
        "agent.agent_init.init_agent", lambda *_args, **_kwargs: initialized.append(True)
    )

    with pytest.raises(token_budget_policy.TokenBudgetPolicyError, match="config"):
        AIAgent(model="gpt-6-astra", provider="openai-codex")
    assert initialized == []


def test_config_read_failure_uses_only_a_validated_last_known_good(monkeypatch):
    """Reload errors may reuse a policy only after a successful validated read."""
    current = {"value": _policy_config()}
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: current["value"]
    )
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    AIAgent._apply_runtime_token_budget(agent)

    current["value"] = None

    def fail_reload():
        raise OSError("reload failed")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fail_reload)
    loaded = agent._load_preflight_token_budget_config()

    assert loaded == _policy_config()


def test_route_identity_normalizes_equivalent_trailing_slash(monkeypatch):
    """Equivalent endpoint spelling shares one baseline and one removal path."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    compressor = _Compressor()
    _install_runtime(agent, compressor)
    AIAgent._apply_runtime_token_budget(agent)

    agent.base_url = "https://chatgpt.com/backend-api/codex/"
    AIAgent._apply_runtime_token_budget(agent)

    assert len(agent._token_budget_route_baselines) == 1
    AIAgent._apply_runtime_token_budget(agent, {})
    assert compressor.context_length == 872_000
    assert compressor.threshold_tokens == 697_600


def test_enabled_policy_preserves_actual_restore_false_bookkeeping(monkeypatch):
    """The real restore helper's exhausted-state reset survives policy rollback."""
    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    agent._fallback_activated = False
    agent._fallback_index = 4

    assert agent._restore_primary_runtime() is False
    assert agent._fallback_index == 0


def test_enabled_policy_preserves_actual_fallback_exhaustion_bookkeeping(monkeypatch):
    """Real fallback exhaustion retains index, unavailable set and cooldown state."""
    from agent.error_classifier import FailoverReason

    _patch_policy(monkeypatch)
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())
    agent._fallback_chain = [{"provider": "anthropic", "model": "claude-test"}]
    agent._fallback_index = 0
    agent._fallback_activated = False
    agent._unavailable_fallback_keys = set()
    agent._rate_limit_backoff_count = 0
    agent._rate_limited_until = 0
    monkeypatch.setattr(
        "agent.chat_completion_helpers._candidate_pool_exhausted",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
        lambda *_args, **_kwargs: "fixture unavailable",
    )

    assert agent._try_activate_fallback(FailoverReason.rate_limit) is False
    assert agent._fallback_index == 1
    assert len(agent._unavailable_fallback_keys) == 1
    assert agent._rate_limit_backoff_count == 1
    assert agent._rate_limited_until > 0


def test_init_cleanup_runs_once_even_when_restore_fails(monkeypatch):
    """Apply remains the surfaced error while every owned resource is retired once."""
    _patch_policy(monkeypatch)
    holder = {}

    class Closeable:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class MemoryManager:
        def __init__(self):
            self.shutdown_calls = 0
            self.end_calls = 0

        def on_session_end(self, *_args):
            self.end_calls += 1

        def shutdown_all(self):
            self.shutdown_calls += 1

    class Compressor(_Compressor):
        def __init__(self):
            super().__init__()
            self.end_calls = 0

        def on_session_end(self, *_args):
            self.end_calls += 1

    class SessionDB:
        pass

    def fake_init(agent, **_kwargs):
        compressor = Compressor()
        client = Closeable()
        transport = Closeable()
        codex_session = Closeable()
        memory = MemoryManager()
        session_db = SessionDB()
        _install_runtime(agent, compressor)
        agent.client = client
        agent._anthropic_client = client
        agent._transport_cache = {"client": client, "transport": transport}
        agent._codex_session = codex_session
        agent._memory_manager = memory
        agent._session_db = session_db
        agent._owns_session_db = True
        agent.session_id = "failed-init"
        holder.update(
            client=client,
            transport=transport,
            codex_session=codex_session,
            memory=memory,
            compressor=compressor,
            session_db=session_db,
        )

    released = []
    monkeypatch.setattr("agent.agent_init.init_agent", fake_init)
    monkeypatch.setattr(
        AIAgent,
        "_apply_runtime_token_budget",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("policy apply failed")),
    )
    monkeypatch.setattr(
        AIAgent,
        "_restore_token_budget_runtime",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("restore failed")),
    )
    monkeypatch.setattr(
        "hermes_state_registry.release_or_close", lambda resource: released.append(resource)
    )

    with pytest.raises(RuntimeError, match="policy apply failed"):
        AIAgent(model="gpt-6-astra", provider="openai-codex")

    assert holder["client"].close_calls == 1
    assert holder["transport"].close_calls == 1
    assert holder["codex_session"].close_calls == 1
    assert holder["memory"].shutdown_calls == 1
    assert holder["memory"].end_calls == 1
    assert holder["compressor"].end_calls == 1
    assert released == [holder["session_db"]]


def test_init_cleanup_does_not_close_injected_resources(monkeypatch):
    """Caller-owned memory and session DB survive a failed initialization."""
    _patch_policy(monkeypatch)

    class InjectedMemory:
        def __init__(self):
            self.shutdown_calls = 0

        def shutdown_all(self):
            self.shutdown_calls += 1

    class InjectedDB:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    memory = InjectedMemory()
    session_db = InjectedDB()

    def fake_init(agent, **kwargs):
        _install_runtime(agent, _Compressor())
        agent._memory_manager = kwargs["memory_manager"]
        agent._session_db = kwargs["session_db"]
        agent._owns_session_db = False

    monkeypatch.setattr("agent.agent_init.init_agent", fake_init)
    monkeypatch.setattr(
        AIAgent,
        "_apply_runtime_token_budget",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("policy apply failed")),
    )

    with pytest.raises(RuntimeError, match="policy apply failed"):
        AIAgent(
            model="gpt-6-astra",
            provider="openai-codex",
            memory_manager=memory,
            session_db=session_db,
        )

    assert memory.shutdown_calls == 0
    assert session_db.close_calls == 0


def test_policy_failure_discards_deferred_fallback_success_log(monkeypatch, caplog):
    """A rolled-back fallback cannot publish a success log during prepare."""
    from agent.chat_completion_helpers import _log_fallback_activated
    from agent.error_classifier import FailoverReason

    _patch_policy(monkeypatch)
    caplog.set_level("INFO")
    agent = object.__new__(AIAgent)
    _install_runtime(agent, _Compressor())

    def fallback(runtime, *_args):
        runtime.model = "claude-fallback"
        _log_fallback_activated(
            runtime,
            FailoverReason.rate_limit,
            "gpt-6-astra",
            "openai-codex",
            "claude-fallback",
            "anthropic",
        )
        return True

    monkeypatch.setattr("agent.chat_completion_helpers.try_activate_fallback", fallback)
    monkeypatch.setattr(
        AIAgent,
        "_apply_runtime_token_budget",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("policy sync failed")),
    )

    with pytest.raises(RuntimeError, match="policy sync failed"):
        agent._try_activate_fallback()
    assert "Fallback activated" not in caplog.text
