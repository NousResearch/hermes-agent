"""Owned LM Studio requests use the same instance as their context budget."""

from types import SimpleNamespace

import httpx
import pytest
from openai import NotFoundError

from agent.transports import get_transport
from agent.turn_retry_state import TurnRetryState
from hermes_cli import models, models_local
from hermes_cli.models_lmstudio_instances import forget_instance, owned_instance, remember_instance

MODEL = "publisher/model"
BASE_URL = "http://127.0.0.1:1234/v1"
ROOT = "http://127.0.0.1:1234"


@pytest.fixture
def claim():
    remember_instance(ROOT, MODEL, "owned-id")
    yield
    forget_instance(ROOT, MODEL)


def _transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


def test_transport_routes_owned_id_on_both_build_paths(claim):
    from providers.base import ProviderProfile
    transport = _transport()
    messages = [{"role": "user", "content": "hello"}]
    for profile in (None, ProviderProfile(name="lmstudio")):
        kwargs = transport.build_kwargs(MODEL, messages, None, base_url=BASE_URL, provider_profile=profile)
        assert kwargs["model"] == "owned-id"
        assert messages == [{"role": "user", "content": "hello"}]


def test_summary_attempts_share_owned_wire_id(claim, monkeypatch):
    from agent import chat_completion_helpers as helpers
    from agent.chat_completion_helpers_summary import chat_summary_attempt
    transport = _transport()
    calls = []

    def execute(agent, request_id, kwargs, callback, *, retry_count):
        calls.append(kwargs["model"])
        text = "" if retry_count == 0 else "summary"
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop"
        )], usage=None)

    agent = SimpleNamespace(
        provider="lmstudio", base_url=BASE_URL, api_mode="chat_completions", model=MODEL,
        _force_ascii_payload=False,
        _build_api_kwargs=lambda messages: transport.build_kwargs(MODEL, messages, base_url=BASE_URL),
        _get_transport=lambda: transport, _interruptible_api_call=lambda kwargs: None,
    )
    monkeypatch.setattr(helpers, "_managed_summary_call", execute)
    attempt = chat_summary_attempt(agent, [{"role": "user", "content": "work"}], "summary-test")
    assert attempt(0) == ""
    assert attempt(1) == "summary"
    assert calls == ["owned-id", "owned-id"]


def _catalog(monkeypatch, instances):
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **kwargs: [
        {"key": MODEL, "max_context_length": 131072, "loaded_instances": instances}
    ])
    monkeypatch.setattr(models, "_urlopen_model_catalog_request", lambda *args, **kwargs: pytest.fail("resident models must not be resized"))


def test_owned_context_does_not_use_larger_sibling(claim, monkeypatch):
    _catalog(monkeypatch, [
        {"id": "user-copy", "config": {"context_length": 65536}},
        {"id": "owned-id", "config": {"context_length": 8192}},
    ])
    result = models_local.ensure_lmstudio_model_loaded(MODEL, BASE_URL, "", None, return_load_result=True)
    assert result.context_length == 8192
    assert result.load_attempted is False
    assert _transport().build_kwargs(MODEL, [], base_url=BASE_URL)["model"] == "owned-id"


def test_stale_target_claim_falls_back_to_resident_context_and_model(claim, monkeypatch):
    _catalog(monkeypatch, [{"id": "user-copy", "config": {"context_length": 65536}}])
    result = models_local.ensure_lmstudio_model_loaded(MODEL, BASE_URL, "", None)
    assert result == 65536
    assert owned_instance(ROOT, MODEL) is None
    assert _transport().build_kwargs(MODEL, [], base_url=BASE_URL)["model"] == MODEL


def test_unverified_owned_context_clears_claim(claim, monkeypatch):
    _catalog(monkeypatch, [
        {"id": "owned-id", "config": {"context_length": True}},
        {"id": "user-copy", "config": {"context_length": 4096}},
    ])
    assert models_local.ensure_lmstudio_model_loaded(MODEL, BASE_URL, "", None) == 4096
    assert owned_instance(ROOT, MODEL) is None


def test_routed_404_rebuilds_once_with_catalog_model(claim, monkeypatch):
    from agent import turn_api_error
    from hermes_cli.models_lmstudio_instances import recover_stale_instance
    retry = TurnRetryState()
    notices = []
    from tests.agent.test_lmstudio_request_recovery import _agent
    _catalog(monkeypatch, [{"id": "user-copy", "config": {"context_length": 4096}}])
    runtime = _agent()
    agent = SimpleNamespace(
        provider="lmstudio", model=MODEL, base_url=BASE_URL, api_key="",
        api_mode="chat_completions", context_compressor=runtime.context_compressor,
        _effective_lmstudio_context_length=runtime._effective_lmstudio_context_length,
        thinking_callback=None, _extract_api_error_context=lambda error: {},
        _invoke_api_request_error_hook=lambda **kwargs: None, _buffer_vprint=notices.append,
    )
    response = httpx.Response(404, request=httpx.Request("POST", BASE_URL + "/chat/completions"))
    error = NotFoundError("instance missing", response=response, body={})
    monkeypatch.setattr(turn_api_error, "recover_before_classification", lambda *args, **kwargs: (False, "system"))
    verdict = turn_api_error.handle_api_error(
        agent, api_error=error, _retry=retry, thinking_spinner=None, messages=[], api_messages=[],
        api_kwargs={"model": "owned-id"}, system_message="system", active_system_prompt="system",
        conversation_history=[], approx_tokens=1, retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=2, api_call_count=1,
        api_request_id="request-test", api_start_time=0, effective_task_id="test", turn_id="turn",
    )
    assert verdict.action == "break"
    assert retry.restart_with_rebuilt_messages is True
    assert verdict.retry_count == 0
    assert retry.lmstudio_stale_instance_recovered is True
    assert owned_instance(ROOT, MODEL) is None
    assert _transport().build_kwargs(MODEL, [], base_url=BASE_URL)["model"] == MODEL
    assert not recover_stale_instance(agent, retry, 404, {"model": "owned-id"})
    assert len(notices) == 1


@pytest.mark.parametrize("status,wire_model", [(404, MODEL), (500, "owned-id"), (404, "unclaimed-id")])
def test_other_errors_preserve_claim(claim, status, wire_model):
    from hermes_cli.models_lmstudio_instances import recover_stale_instance
    agent = SimpleNamespace(model=MODEL, base_url=BASE_URL)
    assert not recover_stale_instance(agent, TurnRetryState(), status, {"model": wire_model})
    assert owned_instance(ROOT, MODEL) == "owned-id"
