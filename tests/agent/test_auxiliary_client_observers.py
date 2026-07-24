import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import auxiliary_client
from agent.aux_accounting import reset_accounting_context, set_accounting_context


@pytest.fixture(autouse=True)
def _isolate_plugin_discovery(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)


def _response(content="ok", model="served-model", usage=None):
    message = SimpleNamespace(role="assistant", content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=usage,
    )


def _capture_hooks(monkeypatch):
    events = []
    monkeypatch.setattr(
        "hermes_cli.plugins.has_hook",
        lambda name: name in {
            "pre_api_request",
            "post_api_request",
            "api_request_error",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda name, **kwargs: events.append((name, kwargs)) or [],
    )
    return events


def _patch_sync_route(monkeypatch, client, provider="auto", model="served-model"):
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        lambda *args, **kwargs: (provider, model, None, None, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *args, **kwargs: (client, model),
    )


def test_auxiliary_success_pairs_ids_and_propagates_session(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    usage = SimpleNamespace(
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
    )
    client.chat.completions.create.return_value = _response(usage=usage)
    _patch_sync_route(monkeypatch, client)
    events = _capture_hooks(monkeypatch)
    session_db = MagicMock()

    token = set_accounting_context(session_db, "session-42")
    try:
        result = auxiliary_client.call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "title this"}],
        )
    finally:
        reset_accounting_context(token)

    assert result.choices[0].message.content == "ok"
    assert [name for name, _ in events] == [
        "pre_api_request",
        "post_api_request",
    ]
    pre = events[0][1]
    post = events[1][1]
    uuid.UUID(pre["auxiliary_call_id"])
    uuid.UUID(pre["api_request_id"])
    assert pre["auxiliary_call_id"] == post["auxiliary_call_id"]
    assert pre["api_request_id"] == post["api_request_id"]
    assert pre["request_kind"] == "auxiliary"
    assert pre["auxiliary_task"] == "title_generation"
    assert pre["attempt_index"] == 0
    assert pre["attempt_reason"] == "initial"
    assert pre["session_id"] == "session-42"
    assert pre["provider"] == "openrouter"
    assert pre["model"] == "served-model"
    assert pre["base_url"] == "https://openrouter.ai/api/v1"
    assert pre["api_mode"] == "chat_completions"
    session_db.record_auxiliary_usage.assert_called_once()


def test_auxiliary_retry_emits_error_before_next_pre(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.side_effect = [
        ConnectionError("connection reset by peer"),
        _response(),
    ]
    _patch_sync_route(monkeypatch, client)
    events = _capture_hooks(monkeypatch)
    monkeypatch.setattr(auxiliary_client.time, "sleep", lambda _: None)

    result = auxiliary_client.call_llm(
        task="session_search",
        messages=[{"role": "user", "content": "search"}],
    )

    assert result.choices[0].message.content == "ok"
    assert [name for name, _ in events] == [
        "pre_api_request",
        "api_request_error",
        "pre_api_request",
        "post_api_request",
    ]
    first_pre, first_error, second_pre, second_post = [event[1] for event in events]
    assert first_pre["api_request_id"] == first_error["api_request_id"]
    assert second_pre["api_request_id"] == second_post["api_request_id"]
    assert first_pre["api_request_id"] != second_pre["api_request_id"]
    assert {
        event["auxiliary_call_id"]
        for event in (first_pre, first_error, second_pre, second_post)
    } == {first_pre["auxiliary_call_id"]}
    assert first_pre["attempt_reason"] == "initial"
    assert second_pre["attempt_index"] == 1
    assert second_pre["attempt_reason"] == "retry:transient_transport"


def test_auxiliary_fallback_reports_actual_route(monkeypatch):
    payment_error = RuntimeError("Payment Required")
    payment_error.status_code = 402

    primary = MagicMock()
    primary.base_url = "https://integrate.api.nvidia.com/v1"
    primary.chat.completions.create.side_effect = payment_error

    fallback = MagicMock()
    fallback.base_url = "https://api.anthropic.com"
    fallback.chat.completions.create.return_value = _response(
        content="fallback",
        model="claude-sonnet",
    )

    _patch_sync_route(monkeypatch, primary, provider="nvidia", model="nim-model")
    monkeypatch.setattr(
        auxiliary_client,
        "_try_configured_fallback_chain",
        lambda *args, **kwargs: (
            fallback,
            "claude-sonnet",
            "fallback_chain[0](anthropic)",
        ),
    )
    events = _capture_hooks(monkeypatch)

    result = auxiliary_client.call_llm(
        task="compression",
        messages=[{"role": "user", "content": "summarize"}],
    )

    assert result.choices[0].message.content == "fallback"
    assert [name for name, _ in events] == [
        "pre_api_request",
        "api_request_error",
        "pre_api_request",
        "post_api_request",
    ]
    fallback_pre = events[2][1]
    assert fallback_pre["provider"] == "anthropic"
    assert fallback_pre["model"] == "claude-sonnet"
    assert fallback_pre["base_url"] == "https://api.anthropic.com"
    assert fallback_pre["api_mode"] == "anthropic_messages"
    assert fallback_pre["attempt_reason"] == "fallback:fallback_chain[0](anthropic)"


def test_auxiliary_accounting_records_each_provider_response_once(monkeypatch):
    usage = SimpleNamespace(
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
    )
    primary = MagicMock()
    primary.base_url = "https://integrate.api.nvidia.com/v1"
    primary.chat.completions.create.return_value = SimpleNamespace(
        choices=[],
        model="invalid-model",
        usage=usage,
    )
    fallback = MagicMock()
    fallback.base_url = "https://openrouter.ai/api/v1"
    fallback.chat.completions.create.return_value = _response(
        model="fallback-model",
        usage=usage,
    )
    _patch_sync_route(monkeypatch, primary, provider="nvidia", model="nim-model")
    monkeypatch.setattr(
        auxiliary_client,
        "_try_configured_fallback_chain",
        lambda *args, **kwargs: (fallback, "fallback-model", "openrouter"),
    )
    session_db = MagicMock()

    token = set_accounting_context(session_db, "session-42")
    try:
        result = auxiliary_client.call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "title"}],
        )
    finally:
        reset_accounting_context(token)

    assert result.model == "fallback-model"
    assert session_db.record_auxiliary_usage.call_count == 2
    assert [
        call.kwargs["model"]
        for call in session_db.record_auxiliary_usage.call_args_list
    ] == ["invalid-model", "fallback-model"]


def test_auxiliary_observers_are_fail_open(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _response()
    _patch_sync_route(monkeypatch, client)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _: True)
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        MagicMock(side_effect=RuntimeError("observer failed")),
    )

    result = auxiliary_client.call_llm(
        task="title_generation",
        messages=[{"role": "user", "content": "title"}],
    )

    assert result.choices[0].message.content == "ok"


def test_auxiliary_payload_serialization_is_fail_open(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _response()
    _patch_sync_route(monkeypatch, client)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _: True)
    monkeypatch.setattr(
        "agent.api_observer.api_request_payload_for_hook",
        MagicMock(side_effect=RuntimeError("serialization failed")),
    )

    result = auxiliary_client.call_llm(
        task="title_generation",
        messages=[{"role": "user", "content": "title"}],
    )

    assert result.choices[0].message.content == "ok"
    client.chat.completions.create.assert_called_once()


def test_auxiliary_response_serialization_failure_still_pairs(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _response()
    _patch_sync_route(monkeypatch, client)
    events = _capture_hooks(monkeypatch)
    monkeypatch.setattr(
        "agent.api_observer.api_response_payload_for_hook",
        MagicMock(side_effect=RuntimeError("serialization failed")),
    )

    result = auxiliary_client.call_llm(
        task="title_generation",
        messages=[{"role": "user", "content": "title"}],
    )

    assert result.choices[0].message.content == "ok"
    assert [name for name, _ in events] == [
        "pre_api_request",
        "post_api_request",
    ]
    assert events[1][1]["response"] == {"_serialization_error": True}


def test_auxiliary_no_listener_skips_payload_work(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _response()
    _patch_sync_route(monkeypatch, client)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _: False)
    invoke_hook = MagicMock()
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr(
        "agent.api_observer.api_request_payload_for_hook",
        MagicMock(side_effect=AssertionError("payload should not be built")),
    )

    result = auxiliary_client.call_llm(
        task="title_generation",
        messages=[{"role": "user", "content": "title"}],
    )

    assert result.choices[0].message.content == "ok"
    invoke_hook.assert_not_called()


def test_auxiliary_observers_discover_plugins_before_checking_hooks(monkeypatch):
    discover_plugins = MagicMock()
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", discover_plugins)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda _: False)

    assert auxiliary_client._auxiliary_observers_enabled("compression") is False
    discover_plugins.assert_called_once_with()


def test_moa_aggregator_uses_main_request_observers_only(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _response()
    _patch_sync_route(monkeypatch, client)
    events = _capture_hooks(monkeypatch)

    result = auxiliary_client.call_llm(
        task="moa_aggregator",
        messages=[{"role": "user", "content": "synthesize"}],
    )

    assert result.choices[0].message.content == "ok"
    assert events == []


def test_auxiliary_payloads_are_redacted(monkeypatch):
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    client = MagicMock()
    client.base_url = (
        f"https://user:password@openrouter.ai/api/v1?api_key={secret}"
    )
    client.chat.completions.create.return_value = _response(
        content=f"Authorization: Bearer {secret}",
    )
    _patch_sync_route(monkeypatch, client)
    events = _capture_hooks(monkeypatch)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

    auxiliary_client.call_llm(
        task="web_extract",
        messages=[
            {
                "role": "user",
                "content": f"Authorization: Bearer {secret}",
            }
        ],
        extra_body={"api_key": secret},
    )

    serialized = json.dumps(events)
    assert secret not in serialized
    assert "<redacted>" in serialized
    assert events[0][1]["base_url"] == (
        "https://user:***@openrouter.ai/api/v1?api_key=***"
    )


def test_auxiliary_payloads_are_bounded(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _response()
    _patch_sync_route(monkeypatch, client)
    events = _capture_hooks(monkeypatch)
    monkeypatch.setenv("HERMES_PLUGIN_PAYLOAD_MAX_CHARS", "1000")

    auxiliary_client.call_llm(
        task="web_extract",
        messages=[{"role": "user", "content": "x" * 20000}],
    )

    assert events[0][1]["request"]["_truncated"] is True


@pytest.mark.asyncio
async def test_async_auxiliary_attempt_uses_same_pairing_contract(monkeypatch):
    client = MagicMock()
    client.base_url = "https://api.githubcopilot.com"
    client.chat.completions.create = AsyncMock(return_value=_response(model="gpt-5"))
    monkeypatch.setattr(
        auxiliary_client,
        "_resolve_task_provider_model",
        lambda *args, **kwargs: ("auto", "gpt-5", None, None, None),
    )
    monkeypatch.setattr(
        auxiliary_client,
        "_get_cached_client",
        lambda *args, **kwargs: (client, "gpt-5"),
    )
    events = _capture_hooks(monkeypatch)

    result = await auxiliary_client.async_call_llm(
        task="skills_hub",
        messages=[{"role": "user", "content": "find skill"}],
    )

    assert result.choices[0].message.content == "ok"
    assert [name for name, _ in events] == [
        "pre_api_request",
        "post_api_request",
    ]
    assert events[0][1]["api_request_id"] == events[1][1]["api_request_id"]
    assert events[0][1]["provider"] == "copilot"


def test_auxiliary_session_context_propagates_to_worker_thread(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _response()
    _patch_sync_route(monkeypatch, client)
    events = _capture_hooks(monkeypatch)
    session_db = MagicMock()

    from tools.thread_context import propagate_context_to_thread

    token = set_accounting_context(session_db, "session-thread")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                propagate_context_to_thread(auxiliary_client.call_llm),
                task="moa_reference",
                messages=[{"role": "user", "content": "review"}],
            ).result()
    finally:
        reset_accounting_context(token)

    assert result.choices[0].message.content == "ok"
    assert [event[1]["session_id"] for event in events] == [
        "session-thread",
        "session-thread",
    ]
