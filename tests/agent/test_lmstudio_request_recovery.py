"""LM Studio recovery verifies the next window before another request."""

from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from openai import NotFoundError

from agent.context_compressor import ContextCompressor
from agent.transports import get_transport
from agent.turn_retry_state import TurnRetryState
from hermes_cli import models, models_local
from hermes_cli import models_lmstudio_instances as instances
from run_agent import AIAgent

MODEL = "publisher/model"
BASE = "http://127.0.0.1:1234/v1"
ROOT = "http://127.0.0.1:1234"


@pytest.fixture(autouse=True)
def claim(monkeypatch):
    instances.remember_instance(ROOT, MODEL, "owned-id")
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", lambda *args, **kwargs: pytest.fail("recovery must not load a model"))
    yield
    instances.forget_instance(ROOT, MODEL)


def _transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


def _agent(pin=None):
    compressor = ContextCompressor(
        MODEL, config_context_length=65536, base_url=BASE, provider="lmstudio", api_mode="chat_completions",
    )
    compressor.update_model(MODEL, 65536, base_url=BASE, provider="lmstudio", api_mode="chat_completions")
    agent = SimpleNamespace(
        provider="lmstudio", model=MODEL, base_url=BASE, api_key="", api_mode="chat_completions",
        context_compressor=compressor, _config_context_length=pin,
        _compression_feasibility_checked=True,
        _effective_lmstudio_context_length=AIAgent._effective_lmstudio_context_length,
        _requested_output_cap_from_api_kwargs=AIAgent._requested_output_cap_from_api_kwargs,
        thinking_callback=None, _extract_api_error_context=lambda error: {},
        _invoke_api_request_error_hook=lambda **kwargs: None, _buffer_vprint=lambda text: None,
        _force_ascii_payload=False, _get_transport=_transport, _interruptible_api_call=lambda kwargs: None,
        max_iterations=2, suppress_status_output=True,
    )
    agent._primary_runtime = {
        "model": MODEL, "provider": "lmstudio", "base_url": BASE, "api_mode": "chat_completions",
        "compressor_context_length": 65536, "compressor_threshold_tokens": compressor.threshold_tokens,
    }
    agent._build_api_kwargs = lambda messages: _transport().build_kwargs(MODEL, messages, base_url=BASE)
    return agent


def _catalog(monkeypatch, context=4096):
    entry = {"key": MODEL, "loaded_instances": [{"id": "user-copy", "config": {"context_length": context}}]}
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **kwargs: [entry])
    return entry


def _error():
    response = httpx.Response(404, request=httpx.Request("POST", BASE + "/chat/completions"))
    return NotFoundError("instance missing", response=response, body={})


def _handle(agent, monkeypatch, retry):
    from agent import turn_api_error
    monkeypatch.setattr(turn_api_error, "recover_before_classification", lambda *args, **kwargs: (False, "system"))
    return turn_api_error.handle_api_error(
        agent, api_error=_error(), _retry=retry, thinking_spinner=None, messages=[], api_messages=[],
        api_kwargs={"model": "owned-id"}, system_message="system", active_system_prompt="system",
        conversation_history=[], approx_tokens=12000, retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=2, api_call_count=1,
        api_request_id="test-request", api_start_time=0, effective_task_id="test", turn_id="turn",
    )


def test_handler_restarts_preflight_with_smaller_budget(monkeypatch):
    _catalog(monkeypatch)
    agent = _agent()
    retry = TurnRetryState()
    verdict = _handle(agent, monkeypatch, retry)
    assert verdict.action == "break"
    assert retry.restart_with_rebuilt_messages
    assert agent.context_compressor.context_length == 4096
    assert agent.context_compressor.threshold_tokens <= 4096
    assert agent._primary_runtime["compressor_context_length"] == 4096
    assert agent._primary_runtime["compressor_threshold_tokens"] == agent.context_compressor.threshold_tokens
    assert not agent._compression_feasibility_checked
    assert verdict.retry_count == 0
    assert not instances.recover_stale_instance(agent, retry, 404, {"model": "owned-id"})


@pytest.mark.parametrize("pin,expected", [(2048, 2048), (8192, 4096), (None, 4096)])
def test_recovery_preserves_the_existing_pin_clamp(monkeypatch, pin, expected):
    _catalog(monkeypatch)
    agent = _agent(pin)
    assert instances.recover_stale_instance(agent, TurnRetryState(), 404, {"model": "owned-id"})
    assert agent.context_compressor.context_length == expected


@pytest.mark.parametrize("catalog", [None, [], [{"key": MODEL, "loaded_instances": []}]])
def test_unverified_recovery_does_not_arm_a_retry(monkeypatch, catalog):
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **kwargs: catalog)
    agent = _agent()
    retry = TurnRetryState()
    assert not instances.recover_stale_instance(agent, retry, 404, {"model": "owned-id"})
    assert not retry.restart_with_rebuilt_messages
    assert agent.context_compressor.context_length == 65536
    assert agent._compression_feasibility_checked


def test_unverified_handler_uses_normal_error_handling(monkeypatch):
    from agent import turn_api_error
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **kwargs: None)
    calls = []
    def normal_recovery(*args, **kwargs):
        calls.append(True)
        return True, False
    monkeypatch.setattr(turn_api_error, "recover_after_classification", normal_recovery)
    retry = TurnRetryState()
    assert _handle(_agent(), monkeypatch, retry).action == "continue"
    assert calls == [True]
    assert not retry.restart_with_rebuilt_messages


@pytest.mark.parametrize("context,eligible", [(8192, True), (None, False)])
def test_replacement_claim_survives_catalog_fetch(monkeypatch, context, eligible):
    def catalog(**kwargs):
        instances.remember_instance(ROOT, MODEL, "replacement-id")
        return [{"key": MODEL, "loaded_instances": [
            {"id": "user-copy", "config": {"context_length": 4096}},
            {"id": "replacement-id", "config": {"context_length": context}},
        ]}]
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", catalog)
    agent = _agent()
    assert instances.recover_stale_instance(agent, TurnRetryState(), 404, {"model": "owned-id"}) is eligible
    assert instances.owned_instance(ROOT, MODEL) == "replacement-id"
    if eligible:
        assert agent.context_compressor.context_length == 8192
        assert _transport().build_kwargs(MODEL, [], base_url=BASE)["model"] == "replacement-id"


def test_old_request_does_not_clear_a_replacement_claim(monkeypatch):
    instances.remember_instance(ROOT, MODEL, "replacement-id")
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **kwargs: pytest.fail("old errors must not probe a replacement"))
    assert not instances.recover_stale_instance(_agent(), TurnRetryState(), 404, {"model": "owned-id"})
    assert instances.owned_instance(ROOT, MODEL) == "replacement-id"


def test_recovery_does_not_update_another_primary_snapshot(monkeypatch):
    _catalog(monkeypatch)
    agent = _agent()
    agent._primary_runtime["model"] = "another/model"
    previous = deepcopy(agent._primary_runtime)
    assert instances.recover_stale_instance(agent, TurnRetryState(), 404, {"model": "owned-id"})
    assert agent._primary_runtime == previous


@pytest.mark.parametrize("catalog,expected", [(None, "owned-id"), ([], None), ([{"key": MODEL, "loaded_instances": []}], None)])
def test_ensure_distinguishes_failed_and_empty_catalogs(monkeypatch, catalog, expected):
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **kwargs: catalog)
    if catalog and expected is None:
        catalog[0]["loaded_instances"] = [{"id": "unknown", "config": {"context_length": None}}]
    assert models_local.ensure_lmstudio_model_loaded(MODEL, BASE, "", None) is None
    assert instances.owned_instance(ROOT, MODEL) == expected


def test_ensure_keeps_a_claim_replaced_during_the_catalog_fetch(monkeypatch):
    def catalog(**kwargs):
        instances.remember_instance(ROOT, MODEL, "replacement-id")
        return [{"key": MODEL, "loaded_instances": [{"id": "user-copy", "config": {"context_length": 4096}}]}]
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", catalog)
    assert models_local.ensure_lmstudio_model_loaded(MODEL, BASE, "", None) is None
    assert instances.owned_instance(ROOT, MODEL) == "replacement-id"


def test_conditional_clear_does_not_remove_a_replacement():
    instances.remember_instance(ROOT, MODEL, "replacement-id")
    assert not instances.forget_instance(ROOT, MODEL, expected_instance_id="owned-id")
    assert instances.owned_instance(ROOT, MODEL) == "replacement-id"


def test_load_refresh_keeps_a_replacement_claim(monkeypatch):
    validate = instances.verified_instance_context
    def replaced_before_validation(root, model, entry, **kwargs):
        instances.remember_instance(root, model, "replacement-id")
        return validate(root, model, entry, **kwargs)
    monkeypatch.setattr(instances, "verified_instance_context", replaced_before_validation)
    instances.remember_catalog_instance(ROOT, MODEL, "load-id", {"key": MODEL, "loaded_instances": []})
    assert instances.owned_instance(ROOT, MODEL) == "replacement-id"


def test_context_engine_threshold_is_saved(monkeypatch):
    _catalog(monkeypatch)
    agent = _agent()
    engine = SimpleNamespace(context_length=65536, threshold_tokens=32768)
    def update(**kwargs):
        engine.context_length = kwargs["context_length"]
        engine.threshold_tokens = engine.context_length // 2
    engine.update_model = update
    agent.context_compressor = engine
    assert instances.recover_stale_instance(agent, TurnRetryState(), 404, {"model": "owned-id"})
    assert engine.context_length == 4096
    assert agent._primary_runtime["compressor_context_length"] == 4096
    assert agent._primary_runtime["compressor_threshold_tokens"] == 2048


@pytest.mark.parametrize("restart_count,action", [(0, "continue"), (3, "break")])
def test_handler_restart_uses_the_existing_preflight_reset_and_cap(monkeypatch, restart_count, action):
    from agent.turn_iteration_prep import apply_retry_restarts
    _catalog(monkeypatch)
    agent = _agent()
    refunds = []
    agent.iteration_budget = SimpleNamespace(refund=lambda: refunds.append(True))
    retry = TurnRetryState()
    assert _handle(agent, monkeypatch, retry).action == "break"
    restart = apply_retry_restarts(
        agent, _retry=retry, response=None, interrupted=False, messages=[], conversation_history=[],
        user_message="work", api_kwargs={"model": "owned-id"}, current_turn_user_idx=0,
        final_response="", retry_count=0, max_retries=3, api_call_count=1,
        restart_count=restart_count, length_continue_retries=0,
        _preflight_compression_blocked=True, _turn_exit_reason="",
    )
    assert restart.action == action
    if action == "continue":
        assert not restart._preflight_compression_blocked
        assert refunds == [True]
    else:
        assert restart._turn_exit_reason == "rebuilt_restart_limit_exceeded"
        assert not refunds


def _summary(monkeypatch, agent, execute, messages):
    from agent import chat_completion_helpers as helpers
    monkeypatch.setattr(helpers, "_iteration_summary_api_messages", lambda agent, history: messages)
    monkeypatch.setattr(helpers, "_managed_summary_call", execute)
    return helpers.handle_max_iterations(agent, [{"role": "user", "content": "work"}], 2)


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop",
    )], usage=None)


def test_summary_404_rebuilds_and_keeps_empty_retry_independent(monkeypatch):
    _catalog(monkeypatch, 8192)
    calls = []
    def execute(agent, request_id, kwargs, callback, *, retry_count):
        calls.append((kwargs["model"], retry_count))
        if len(calls) == 1:
            raise _error()
        return _response("" if len(calls) == 2 else "summary")
    agent = _agent()
    assert _summary(monkeypatch, agent, execute, [{"role": "user", "content": "work"}]) == "summary"
    assert calls == [("owned-id", 0), (MODEL, 0), (MODEL, 1)]
    assert agent.context_compressor.context_length == 8192
    assert instances.owned_instance(ROOT, MODEL) is None


@pytest.mark.parametrize("reserve,long_input", [(8192, False), (None, True)])
def test_summary_does_not_send_payload_over_smaller_window(monkeypatch, reserve, long_input):
    _catalog(monkeypatch)
    messages = [{"role": "user", "content": "work " * (20000 if long_input else 1)}]
    original = deepcopy(messages)
    calls = []
    def execute(agent, request_id, kwargs, callback, *, retry_count):
        calls.append(kwargs["model"])
        raise _error()
    agent = _agent()
    agent._build_api_kwargs = lambda messages: _transport().build_kwargs(
        MODEL, messages, base_url=BASE, max_tokens=reserve, max_tokens_param_fn=lambda value: {"max_tokens": value},
    )
    result = _summary(monkeypatch, agent, execute, messages)
    assert "couldn't produce a summary" in result
    assert calls == ["owned-id"]
    assert messages == original
    assert agent.context_compressor.context_length == 4096


def test_summary_404_retry_is_bounded(monkeypatch):
    _catalog(monkeypatch)
    calls = []
    def execute(agent, request_id, kwargs, callback, *, retry_count):
        calls.append(kwargs["model"])
        raise _error()
    result = _summary(monkeypatch, _agent(), execute, [{"role": "user", "content": "work"}])
    assert "couldn't produce a summary" in result
    assert calls == ["owned-id", MODEL]


def test_summary_unknown_catalog_does_not_retry(monkeypatch):
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **kwargs: None)
    calls = []
    def execute(agent, request_id, kwargs, callback, *, retry_count):
        calls.append(kwargs["model"])
        raise _error()
    result = _summary(monkeypatch, _agent(), execute, [{"role": "user", "content": "work"}])
    assert "couldn't produce a summary" in result
    assert calls == ["owned-id"]


@pytest.mark.parametrize("extra", [
    {"max_tokens": 8192},
    {"max_completion_tokens": 8192},
    {"messages": [{"role": "user", "content": "work " * 20000}]},
    {"tools": [{"type": "function", "function": {"name": "large", "description": "work " * 20000}}]},
])
def test_summary_checks_the_final_sdk_body(monkeypatch, extra):
    _catalog(monkeypatch)
    agent = _agent()
    def build(messages):
        return {
            "model": instances.lmstudio_request_model(MODEL, BASE), "messages": messages,
            "max_tokens": 8, "max_completion_tokens": 8, "extra_body": extra,
        }
    agent._build_api_kwargs = build
    calls = []
    def execute(agent, request_id, kwargs, callback, *, retry_count):
        calls.append(kwargs["model"])
        raise _error()
    original = deepcopy(extra)
    result = _summary(monkeypatch, agent, execute, [{"role": "user", "content": "work"}])
    assert "couldn't produce a summary" in result
    assert calls == ["owned-id"]
    assert agent.context_compressor.context_length == 4096
    assert extra == original
