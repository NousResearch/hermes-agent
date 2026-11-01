"""Local auxiliary routes are refused before any Relay or SDK request."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agent.auxiliary_client import (
    LocalAuxiliaryAdmissionRequired,
    _FallbackDestination,
    _call_fallback_candidate_async,
    _call_fallback_candidate_sync,
    _relay_async_completion,
    _relay_sync_stream,
    _require_nonlocal_auxiliary_route,
    _relay_sync_completion,
)


def test_local_auxiliary_completion_is_refused_before_provider_callback():
    provider_calls = []
    client = SimpleNamespace(base_url="http://127.0.0.1:11434/v1/")

    with pytest.raises(LocalAuxiliaryAdmissionRequired, match="no task/run-bound GPU admission"):
        _relay_sync_completion(
            client,
            {"model": "fixture-model", "messages": []},
            provider="ollama",
            create=lambda _request: provider_calls.append("called"),
        )

    assert provider_calls == []


def test_local_auxiliary_fallback_destination_is_refused_even_if_client_is_remote():
    remote_client = SimpleNamespace(base_url="https://remote.example/v1")

    with pytest.raises(LocalAuxiliaryAdmissionRequired, match="no task/run-bound GPU admission"):
        _require_nonlocal_auxiliary_route(
            remote_client,
            provider="openai",
            task="compression",
            base_url="http://10.0.0.8:11434/v1/",
        )


def test_local_auxiliary_stream_is_refused_before_provider_io():
    provider_calls = []
    client = SimpleNamespace(
        base_url="http://127.0.0.1:11434/v1/",
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: provider_calls.append("called"),
        )),
    )

    with pytest.raises(LocalAuxiliaryAdmissionRequired):
        _relay_sync_stream(
            client, {"model": "fixture-model", "messages": []}, provider="ollama",
        )

    assert provider_calls == []


def test_local_auxiliary_async_completion_is_refused_before_callback():
    provider_calls = []
    client = SimpleNamespace(
        base_url="http://127.0.0.1:11434/v1/",
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: provider_calls.append("called"),
        )),
    )

    async def run():
        with pytest.raises(LocalAuxiliaryAdmissionRequired):
            await _relay_async_completion(
                client, {"model": "fixture-model", "messages": []}, provider="ollama",
                create=lambda _request: provider_calls.append("called"),
            )

    asyncio.run(run())
    assert provider_calls == []


@pytest.mark.parametrize("async_mode", [False, True])
def test_local_auxiliary_fallback_candidate_is_refused_before_provider_io(monkeypatch, async_mode):
    import agent.auxiliary_client as aux

    provider_calls = []
    destination = _FallbackDestination(
        "ollama", "http://10.0.0.8:11434/v1/", "chat_completions", "fixture-model",
    )
    monkeypatch.setattr(
        aux, "_plan_fallback_candidate",
        lambda *_args, **_kwargs: (destination, {"model": "fixture-model", "messages": []}, lambda *_a: None),
    )
    client = SimpleNamespace(base_url="https://remote.example/v1")
    call_args = {
        "fb_client": client, "fb_model": "fixture-model", "fb_label": "local-fallback",
        "task": "compression", "messages": [], "temperature": None, "max_tokens": 8,
        "tools": None, "effective_timeout": 5.0, "effective_extra_body": {}, "reasoning_config": None,
    }

    if async_mode:
        async def run():
            with pytest.raises(LocalAuxiliaryAdmissionRequired):
                await _call_fallback_candidate_async(**call_args)
        asyncio.run(run())
    else:
        with pytest.raises(LocalAuxiliaryAdmissionRequired):
            _call_fallback_candidate_sync(**call_args)

    assert provider_calls == []


def test_local_auxiliary_admission_refusal_is_terminal_without_provider_retry_or_fallback(monkeypatch):
    """A policy denial is not a capacity error and must not enter the recovery ladder."""
    import agent.auxiliary_client as aux

    provider_calls = []
    fallback_calls = []
    client = SimpleNamespace(
        base_url="http://127.0.0.1:11434/v1/",
        chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **_kwargs: provider_calls.append("called"),
        )),
    )
    destination = SimpleNamespace(
        client=client,
        kwargs={"model": "fixture-model", "messages": []},
        request_provider="ollama",
        resolved_api_mode="chat_completions",
        base_info="http://127.0.0.1:11434/v1/",
        resolved_provider="ollama",
        resolved_model="fixture-model",
        resolved_base_url="http://127.0.0.1:11434/v1/",
        resolved_api_key="",
        final_model="fixture-model",
        timeout=5,
        main_runtime=None,
        route_info=None,
    )
    monkeypatch.setattr(
        aux, "_plan_aux_call",
        lambda *_args, **_kwargs: (destination, {"max_tokens": 8, "main_runtime": None}, {}),
    )

    def unexpected_fallback(*_args, **_kwargs):
        fallback_calls.append("called")
        raise AssertionError("admission refusal must not enter a fallback chain")

    for name in (
        "_try_configured_fallback_chain", "_try_main_fallback_chain",
        "_try_payment_fallback", "_try_main_agent_model_fallback",
    ):
        monkeypatch.setattr(aux, name, unexpected_fallback)

    with pytest.raises(LocalAuxiliaryAdmissionRequired, match="no task/run-bound GPU admission"):
        aux._call_llm_impl(task="compression", messages=[])

    assert provider_calls == []
    assert fallback_calls == []


def test_local_iteration_summary_is_refused_before_provider_io(monkeypatch):
    import agent.chat_completion_helpers as helpers
    from hermes_cli.middleware import LocalModelAdmissionBlockedError

    conversation_loop = ModuleType("agent.conversation_loop")
    conversation_loop._moa_client_consumes_prepared_request = lambda _client: False
    monkeypatch.setitem(sys.modules, "agent.conversation_loop", conversation_loop)

    provider_calls = []
    agent = SimpleNamespace(
        provider="ollama", base_url="http://127.0.0.1:11434/v1/", model="fixture-model",
        api_mode="chat_completions", client=SimpleNamespace(),
        _build_api_kwargs=lambda messages: {"model": "fixture-model", "messages": messages},
        _interruptible_api_call=lambda kwargs: helpers.interruptible_api_call(agent, kwargs),
    )
    monkeypatch.setattr(helpers, "sanitize_outbound_kwargs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        helpers, "_managed_summary_call",
        lambda _agent, _request_id, request, callback, *, retry_count: callback(request),
    )
    monkeypatch.setattr(helpers, "direct_api_call", lambda *_args, **_kwargs: provider_calls.append("called"))

    attempt = helpers._chat_summary_attempt(agent, [], "summary-request")
    with pytest.raises(LocalModelAdmissionBlockedError, match="missing_admission_identity"):
        attempt(0)

    assert provider_calls == []
