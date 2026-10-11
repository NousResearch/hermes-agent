"""Auxiliary normalisation must preserve provider response facts."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from agent.auxiliary_client import _AnthropicCompletionsAdapter, _CodexCompletionsAdapter, _aggregate_chat_stream
from agent.usage_pricing import CanonicalUsage, normalize_usage
from agent.native_usage import NativeUsageError
from tests.agent.test_auxiliary_codex_completion_status import _adapter_response


@pytest.mark.parametrize("phase", ["analysis", "commentary"])
@pytest.mark.parametrize("streamed", [False, True])
def test_codex_refusal_metadata_excludes_intermediate_phases(phase, streamed):
    native = NS(
        status="completed", error=None, usage=None,
        output=[
            NS(type="message", status="completed", phase=phase,
               content=[NS(type="refusal", refusal="Intermediate refusal.")]),
            NS(type="message", status="completed", phase="final_answer",
               content=[NS(type="output_text", text="Final answer.")]),
        ],
    )

    response = _adapter_response(native, streamed=streamed)
    message = response.choices[0].message

    assert (message.content, message.refusal, message.reasoning) == (
        "Final answer.", None, "Intermediate refusal.",
    )


@pytest.mark.parametrize("async_mode", [False, True])
def test_transient_retry_preserves_selected_billing_route(async_mode):
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from agent.auxiliary_client import async_call_llm, call_llm

    native = NS(
        model=None, choices=[NS(message=NS(content="Answer."), finish_reason="stop")],
        usage=NS(prompt_tokens=100, completion_tokens=10, total_tokens=110),
    )
    create = (AsyncMock if async_mode else Mock)(side_effect=[httpx.ReadError("Connection reset"), native])
    client = NS(base_url="https://generativelanguage.googleapis.com/v1beta", chat=NS(completions=NS(create=create)))
    db = NS(record_auxiliary_usage=Mock())
    token = set_accounting_context(db, "session")
    try:
        with patch("agent.auxiliary_client._get_cached_client", return_value=(client, "gemini-2.5-flash")), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("gemini", "gemini-2.5-flash", None, "fixture", None),
        ), patch("agent.auxiliary_client.time.sleep"), patch("agent.auxiliary_client._transient_retry_count", return_value=1):
            kwargs = {"task": "web_extract", "messages": [{"role": "user", "content": "Answer."}]}
            response = asyncio.run(async_call_llm(**kwargs)) if async_mode else call_llm(**kwargs)
        assert (response.model, db.record_auxiliary_usage.call_args.kwargs) == (
            None,
            {"model": "gemini-2.5-flash", "billing_provider": "gemini", "billing_base_url": client.base_url,
             "input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 0, "cache_write_tokens": 0,
             "reasoning_tokens": 0, "estimated_cost_usd": 0.000021},
        )
    finally:
        reset_accounting_context(token)


@pytest.mark.parametrize("served_model", [None, "served-model"])
def test_native_auxiliary_accounting_uses_observed_or_selected_model(served_model):
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from agent.auxiliary_client import call_llm
    from agent.gemini_native_adapter import GeminiNativeClient

    native = {
        "candidates": [{"content": {"parts": [{"text": "Answer."}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
    }
    if served_model is not None:
        native["modelVersion"] = served_model
    db = NS(record_auxiliary_usage=Mock())
    token = set_accounting_context(db, "session")
    try:
        with GeminiNativeClient(
            api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=native))),
        ) as client, patch(
            "agent.auxiliary_client._get_cached_client", return_value=(client, "selected-model"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("gemini", "selected-model", None, "fixture", None),
        ), patch("agent.usage_pricing.estimate_usage_cost", return_value=NS(amount_usd=0.01)) as cost:
            response = call_llm(task="web_extract", messages=[{"role": "user", "content": "Answer."}])
        assert (response.model, cost.call_args.args[0], db.record_auxiliary_usage.call_args.kwargs) == (
            served_model, served_model or "selected-model",
            {"model": served_model or "selected-model", "billing_provider": "gemini", "billing_base_url": client.base_url,
             "input_tokens": 2, "output_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0,
             "reasoning_tokens": 0, "estimated_cost_usd": 0.01},
        )
    finally:
        reset_accounting_context(token)


@pytest.mark.parametrize("streamed", [False, True], ids=["response-object", "sse"])
@pytest.mark.parametrize("write_key", [None, "cache_write_tokens", "cache_creation_tokens"])
def test_codex_preserves_identity_refusal_and_usage(streamed, write_key):
    details = {"cached_tokens": 5}
    projected_details = {"cached_tokens": 5}
    if write_key is not None:
        details[write_key] = 2
        projected_details["cache_write_tokens"] = 2
    final = NS(
        id="resp_provider", model="served-model", status="completed", error=None,
        output=[NS(type="message", role="assistant", status="completed", phase="final_answer",
                   content=[NS(type="refusal", refusal="Cannot fulfil this request.")])],
        usage=NS(input_tokens=12, output_tokens=8, total_tokens=20,
                 input_tokens_details=NS(**details),
                 output_tokens_details=NS(reasoning_tokens=3)),
    )

    response = _adapter_response(final, streamed=streamed)
    choice, = response.choices

    assert (
        getattr(response, "id", None), response.model,
        choice.message.content, getattr(choice.message, "refusal", None), choice.finish_reason,
        vars(response.usage),
    ) == (
        "resp_provider", "served-model", "Cannot fulfil this request.",
        "Cannot fulfil this request.", "stop",
        {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
         "prompt_tokens_details": NS(**projected_details),
         "completion_tokens_details": NS(reasoning_tokens=3)},
    )
    assert normalize_usage(response.usage, provider="openai-codex", api_mode="codex_responses") == CanonicalUsage(
        input_tokens=7 if write_key is None else 5, output_tokens=8, cache_read_tokens=5,
        cache_write_tokens=0 if write_key is None else 2, reasoning_tokens=3,
    )


def test_anthropic_preserves_identity_and_all_prompt_tokens():
    native = NS(
        id="msg_provider", model="served-claude", stop_reason="end_turn",
        content=[NS(type="text", text="Answer.")],
        usage=NS(input_tokens=7, output_tokens=3,
                 cache_read_input_tokens=5, cache_creation_input_tokens=2),
    )
    with patch("agent.anthropic_adapter.create_anthropic_message", return_value=native):
        response = _AnthropicCompletionsAdapter(NS(), "requested-model").create(
            messages=[{"role": "user", "content": "Answer."}],
        )

    assert (getattr(response, "id", None), response.model, vars(response.usage)) == (
        "msg_provider", "served-claude",
        {"prompt_tokens": 14, "completion_tokens": 3, "total_tokens": 17,
         "cache_read_input_tokens": 5, "cache_creation_input_tokens": 2},
    )
    assert normalize_usage(response.usage, provider="anthropic") == CanonicalUsage(
        input_tokens=7, output_tokens=3, cache_read_tokens=5, cache_write_tokens=2,
    )


def test_anthropic_preserves_signed_reasoning_and_block_order():
    native = NS(
        id="msg_provider", model="served-claude", stop_reason="tool_use", usage=None,
        content=[NS(type="thinking", thinking="Consider.", signature="signature"),
                 NS(type="tool_use", id="call_provider", name="inspect", input={}),
                 NS(type="redacted_thinking", data="encrypted")],
    )
    with patch("agent.anthropic_adapter.create_anthropic_message", return_value=native):
        response = _AnthropicCompletionsAdapter(NS(), "requested-model").create(
            messages=[{"role": "user", "content": "Answer."}],
        )
    message = response.choices[0].message
    assert (
        getattr(message, "reasoning_details", None), getattr(message, "anthropic_content_blocks", None),
    ) == (
        [{"type": "thinking", "thinking": "Consider.", "signature": "signature"},
         {"type": "redacted_thinking", "data": "encrypted"}],
        [{"type": "thinking", "thinking": "Consider.", "signature": "signature"},
         {"type": "tool_use", "id": "call_provider", "name": "inspect", "input": {}},
         {"type": "redacted_thinking", "data": "encrypted"}],
    )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3, "input_tokens_details": {}}])
def test_codex_handles_absent_usage_and_optional_details(streamed, usage):
    native = NS(
        status="completed", error=None, usage=usage,
        output=[NS(type="message", status="completed", phase="final_answer",
                   content=[NS(type="output_text", text="Answer.")])],
    )
    response = _adapter_response(native, streamed=streamed)
    assert response.usage == (None if not usage else NS(
        prompt_tokens=2, completion_tokens=1, total_tokens=3,
        prompt_tokens_details=NS(cached_tokens=0),
    ))


def test_codex_preserves_refusal_deltas_without_done_items():
    class Stream:
        def __iter__(self):
            yield NS(type="response.refusal.delta", delta="Cannot ")
            yield NS(type="response.refusal.delta", delta="comply.")
            yield NS(type="response.completed", response=NS(
                id="resp_provider", model="served-model", status="completed", usage=None,
            ))

        def close(self):
            pass

    response = _CodexCompletionsAdapter(
        NS(base_url="", responses=NS(create=lambda **kwargs: Stream())), "requested-model",
    ).create(messages=[{"role": "user", "content": "Answer."}])
    message = response.choices[0].message
    assert (message.content, getattr(message, "refusal", None)) == ("Cannot comply.", "Cannot comply.")


@pytest.mark.parametrize("metadata_tail", [False, True])
def test_gemini_stream_preserves_identity_and_usage(metadata_tail):
    from agent.gemini_native_adapter import GeminiNativeClient

    metadata = {
        "responseId": "gemini_provider", "modelVersion": "served-gemini",
        "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
    }
    answer = {"candidates": [{"content": {"parts": [{"text": "Answer."}]}, "finishReason": "STOP"}]}
    events = [answer, metadata] if metadata_tail else [{**answer, **metadata}]
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    with GeminiNativeClient(
        api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body))),
    ) as client:
        chunks = client.chat.completions.create(
            model="requested-model", messages=[{"role": "user", "content": "Answer."}], stream=True,
        )
        response = _aggregate_chat_stream(chunks, model="requested-model")
    assert (response.id, response.model, response.choices[0].message.content, response.usage) == (
        "gemini_provider", "served-gemini", "Answer.",
        NS(prompt_tokens=2, completion_tokens=1, total_tokens=3,
           prompt_tokens_details=NS(cached_tokens=0), completion_tokens_details=NS(reasoning_tokens=0)),
    )


@pytest.mark.parametrize("metadata", [None, {}, {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0}])
def test_gemini_preserves_identity_and_usage_presence(metadata):
    from agent.gemini_native_adapter import translate_gemini_response

    native = {
        "responseId": "gemini_provider", "modelVersion": "served-gemini",
        "candidates": [{"content": {"parts": [{"text": "Answer."}]}, "finishReason": "STOP"}],
    }
    if metadata is not None:
        native["usageMetadata"] = metadata

    response = translate_gemini_response(native, "requested-model")

    assert (response.id, response.model, response.usage) == (
        "gemini_provider", "served-gemini",
        None if not metadata else NS(
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            prompt_tokens_details=NS(cached_tokens=0),
            completion_tokens_details=NS(reasoning_tokens=0),
        ),
    )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("metadata", [None, {}, {"inputTokens": 0, "outputTokens": 0}])
def test_bedrock_preserves_usage_presence(streamed, metadata):
    from agent.bedrock_adapter import normalize_converse_response, normalize_converse_stream_events

    if streamed:
        events = [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Answer."}}},
                  {"messageStop": {"stopReason": "end_turn"}}]
        if metadata is not None:
            events.append({"metadata": {"usage": metadata}})
        response = normalize_converse_stream_events({"stream": events})
    else:
        native = {"output": {"message": {"content": [{"text": "Answer."}]}}, "stopReason": "end_turn"}
        if metadata is not None:
            native["usage"] = metadata
        response = normalize_converse_response(native)

    assert response.usage == (
        None
        if not metadata
        else NS(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
    )


@pytest.mark.parametrize("provider", ["gemini", "codex", "anthropic"])
@pytest.mark.parametrize("counter", ["primary", "optional"])
@pytest.mark.parametrize("value", [True, False, "2", 1.5, -1])
@pytest.mark.parametrize("complete", [False, True])
def test_native_usage_rejects_malformed_counts(provider, counter, value, complete):
    from agent.auxiliary_anthropic_response import (
        _normalize_anthropic_auxiliary_response,
    )
    from agent.auxiliary_codex_response import _parse_codex_final_response
    from agent.gemini_native_adapter import translate_gemini_response

    if provider == "gemini":
        usage = {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3}
        usage["promptTokenCount" if counter == "primary" else "thoughtsTokenCount"] = (
            value
        )
        if not complete:
            usage.pop("candidatesTokenCount")
        field = "promptTokenCount" if counter == "primary" else "thoughtsTokenCount"
        gemini_native = {"usageMetadata": usage, "candidates": []}
        project = lambda: translate_gemini_response(gemini_native, "served-model")
    elif provider == "codex":
        usage = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        if counter == "primary":
            usage["input_tokens"] = value
        else:
            usage["input_tokens_details"] = {"cached_tokens": value}
        if not complete:
            usage.pop("output_tokens")
        field = (
            "input_tokens"
            if counter == "primary"
            else "input_tokens_details.cached_tokens"
        )
        native = NS(status="completed", output=[], usage=usage)
        project = lambda: _parse_codex_final_response(native)
    else:
        usage = {"input_tokens": 2, "output_tokens": 1}
        usage["input_tokens" if counter == "primary" else "cache_read_input_tokens"] = (
            value
        )
        if not complete:
            usage.pop("output_tokens")
        field = "input_tokens" if counter == "primary" else "cache_read_input_tokens"
        native = NS(content=[], stop_reason="end_turn", usage=NS(**usage))
        project = lambda: _normalize_anthropic_auxiliary_response(
            native, is_oauth=False
        )

    with pytest.raises(NativeUsageError, match="Invalid native usage") as error:
        project()
    assert (error.value.code, error.value.field) == ("invalid_usage", f"usage.{field}")


@pytest.mark.parametrize("provider", ["gemini", "codex", "anthropic"])
@pytest.mark.parametrize("primary", [None, "missing", 0])
def test_native_usage_requires_measured_primary_counts(provider, primary):
    from agent.auxiliary_anthropic_response import (
        _normalize_anthropic_auxiliary_response,
    )
    from agent.auxiliary_codex_response import _parse_codex_final_response
    from agent.gemini_native_adapter import translate_gemini_response

    if provider == "gemini":
        usage = {"candidatesTokenCount": 0, "thoughtsTokenCount": None}
        if primary != "missing":
            usage["promptTokenCount"] = primary
        response = translate_gemini_response(
            {
                "usageMetadata": usage,
                "candidates": [{"finishReason": "STOP"}],
            },
            "served-model",
        )
    elif provider == "codex":
        usage = {"output_tokens": 0, "input_tokens_details": {"cached_tokens": None}}
        if primary != "missing":
            usage["input_tokens"] = primary
        response = _parse_codex_final_response(
            NS(status="completed", output=[], usage=usage)
        )
    else:
        usage = {"output_tokens": 0, "cache_read_input_tokens": None}
        if primary != "missing":
            usage["input_tokens"] = primary
        response = _normalize_anthropic_auxiliary_response(
            NS(content=[], stop_reason="end_turn", usage=NS(**usage)),
            is_oauth=False,
        )

    if primary != 0:
        assert response.usage is None
        return
    assert normalize_usage(response.usage, provider=provider) == CanonicalUsage()
    assert response.usage is not None


def _openai_usage_http_response(usage, streamed):
    message = {
        "role": "assistant",
        "content": "Answer.",
        "refusal": "Refusal detail.",
        "tool_calls": [
            {
                "id": "call_provider",
                "type": "function",
                "index": 0,
                "function": {"name": "inspect", "arguments": "{}"},
            }
        ],
    }
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "logprobs": None,
        "delta" if streamed else "message": message,
    }
    native = {
        "id": "chat_provider",
        "model": "served-model",
        "created": 0,
        "object": "chat.completion.chunk" if streamed else "chat.completion",
        "choices": [choice],
        "usage": usage,
    }
    if streamed:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(native) + "\n\ndata: [DONE]\n\n",
        )
    return httpx.Response(200, json=native)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("value", [True, "2", 1.5, -1])
@pytest.mark.parametrize("counter", ["primary", "optional"])
def test_auxiliary_openai_rejects_raw_usage_before_sdk_coercion(
    async_mode, streamed, value, counter
):
    from agent.auxiliary_client import _create_openai_client, _to_async_client

    usage = {"prompt_tokens": value, "completion_tokens": 1, "total_tokens": 3}
    field = "usage.prompt_tokens"
    if counter == "optional":
        usage = {
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": value},
        }
        field = "usage.prompt_tokens_details.cached_tokens"
    transport = httpx.MockTransport(
        lambda request: _openai_usage_http_response(usage, streamed)
    )
    with _create_openai_client(
        api_key="fixture",
        base_url="https://native.example/v1",
        http_client=httpx.Client(transport=transport),
    ) as client:
        kwargs = {
            "model": "requested-model",
            "messages": [{"role": "user", "content": "Answer."}],
            "stream": streamed,
        }
        if not async_mode:
            with pytest.raises(NativeUsageError, match="Invalid native usage") as error:
                response = client.chat.completions.create(**kwargs)
                if streamed:
                    list(response)
            assert (error.value.code, error.value.field) == ("invalid_usage", field)
            return

        async def run():
            with patch(
                "agent.auxiliary_client._openai_http_client_kwargs",
                return_value={"http_client": httpx.AsyncClient(transport=transport)},
            ):
                async_client, _ = _to_async_client(client, "requested-model")
            async with async_client:
                with pytest.raises(
                    NativeUsageError, match="Invalid native usage"
                ) as error:
                    response = await async_client.chat.completions.create(**kwargs)
                    if streamed:
                        async for _ in response:
                            pass
                assert (error.value.code, error.value.field) == ("invalid_usage", field)

        asyncio.run(run())


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"completion_tokens": 0},
        {"prompt_tokens": 0, "completion_tokens": 0},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": None},
        {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 9},
    ],
)
def test_auxiliary_openai_keeps_unknown_zero_and_reported_totals(
    async_mode, streamed, usage
):
    from agent.auxiliary_client import _create_openai_client, _to_async_client

    transport = httpx.MockTransport(
        lambda request: _openai_usage_http_response(usage, streamed)
    )
    with _create_openai_client(
        api_key="fixture",
        base_url="https://native.example/v1",
        http_client=httpx.Client(transport=transport),
    ) as client:
        if not async_mode:
            response = client.with_options(max_retries=0).chat.completions.create(
                model="requested-model",
                messages=[],
                stream=streamed,
            )
            if streamed:
                (response,) = list(response)
        else:

            async def run():
                with patch(
                    "agent.auxiliary_client._openai_http_client_kwargs",
                    return_value={
                        "http_client": httpx.AsyncClient(transport=transport)
                    },
                ):
                    async_client, _ = _to_async_client(client, "requested-model")
                async with async_client:
                    response = await async_client.with_options(
                        max_retries=0
                    ).chat.completions.create(
                        model="requested-model",
                        messages=[],
                        stream=streamed,
                    )
                    if streamed:
                        (response,) = [chunk async for chunk in response]
                    return response

            response = asyncio.run(run())
        measured = (
            usage is not None
            and "prompt_tokens" in usage
            and "completion_tokens" in usage
        )
        expected = None
        if measured:
            expected = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage.get("total_tokens")
                if usage.get("total_tokens") is not None
                else usage["prompt_tokens"] + usage["completion_tokens"],
                "prompt_tokens_details": None,
                "completion_tokens_details": None,
            }
        assert (
            response.id,
            response.model,
            None if response.usage is None else response.usage.model_dump(),
        ) == (
            "chat_provider",
            "served-model",
            expected,
        )
        message = response.choices[0].delta if streamed else response.choices[0].message
        assert (
            message.role,
            message.content,
            message.refusal,
            [
                (call.id, call.type, call.function.model_dump())
                for call in message.tool_calls
            ],
        ) == (
            "assistant",
            "Answer.",
            "Refusal detail.",
            [("call_provider", "function", {"name": "inspect", "arguments": "{}"})],
        )


def _codex_usage_http_response(usage, streamed):
    native = {
        "id": "resp_provider",
        "model": "served-codex",
        "created_at": 0,
        "status": "completed",
        "output": [],
        "usage": usage,
    }
    if not streamed:
        return httpx.Response(200, json=native)
    event = {"type": "response.completed", "sequence_number": 0, "response": native}
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="data: " + json.dumps(event) + "\n\n",
    )


async def _compatible_responses_completion(native, async_mode):
    from agent.auxiliary_client import _create_openai_client
    from agent.native_usage_openai import UsageValidatedAsyncOpenAI

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=native))
    kwargs = {"api_key": "fixture", "base_url": "https://native.example/v1"}
    if async_mode:
        async with UsageValidatedAsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            raw = await client.chat.completions.with_raw_response.create(
                model="requested-model",
                messages=[],
            )
            return raw.parse()
    with _create_openai_client(
        **kwargs,
        http_client=httpx.Client(transport=transport),
    ) as client:
        raw = client.chat.completions.with_raw_response.create(
            model="requested-model",
            messages=[],
        )
        return raw.parse()


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("with_text", [False, True])
def test_compatible_responses_recovery_preserves_function_calls(
    async_mode, with_text, as_dict
):
    from agent.auxiliary_client import _validate_llm_response

    output = [
        {
            "type": "function_call",
            "call_id": "call",
            "name": "record",
            "arguments": "{}",
        }
    ]
    if with_text:
        output.insert(
            0,
            {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
        )
    native = {
        "id": "response-id",
        "object": "response",
        "status": "completed",
        "model": "native-model",
        "output": output,
    }
    sdk_response = asyncio.run(_compatible_responses_completion(native, async_mode))
    response = _validate_llm_response(
        sdk_response.model_dump() if as_dict else sdk_response, "web_extract"
    )
    (choice,) = response.choices
    assert (
        response.id,
        response.model,
        response.status,
        response.usage,
        choice.finish_reason,
        choice.message.content,
        [
            (call.id, call.type, call.function.name, call.function.arguments)
            for call in choice.message.tool_calls
        ],
    ) == (
        native["id"],
        native["model"],
        native["status"],
        None,
        "tool_calls",
        "answer" if with_text else None,
        [("call", "function", "record", "{}")],
    )


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("status", ["completed", "incomplete", "in_progress"])
@pytest.mark.parametrize("with_refusal", [False, True])
def test_compatible_responses_recovery_preserves_completion_facts(
    async_mode, status, with_refusal, text_only, as_dict
):
    from agent.auxiliary_client import _validate_llm_response

    part = (
        {"type": "refusal", "refusal": "Cannot comply."}
        if with_refusal
        else {"type": "output_text", "text": "answer"}
    )
    native = {
        "object": "response",
        "status": status,
        "incomplete_details": {"reason": "max_output_tokens"}
        if status == "incomplete"
        else None,
        "output": [{"type": "message", "content": [part]}],
        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 19},
    }
    if text_only:
        native["output"] = None
        native["output_text"] = "answer"
        with_refusal = False
    sdk_response = asyncio.run(_compatible_responses_completion(native, async_mode))
    response = _validate_llm_response(
        sdk_response.model_dump() if as_dict else sdk_response, "web_extract"
    )
    (choice,) = response.choices
    assert (
        response.id,
        response.model,
        response.status,
        response.incomplete_details,
        vars(response.usage),
        choice.finish_reason,
        choice.message.content,
        choice.message.refusal,
        choice.message.tool_calls,
    ) == (
        None,
        None,
        status,
        native["incomplete_details"],
        {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 19},
        "stop" if status == "completed" else "length",
        "Cannot comply." if with_refusal else "answer",
        "Cannot comply." if with_refusal else None,
        None,
    )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("value", [True, "2", 1.5, -1, 0, None])
def test_auxiliary_codex_checks_raw_usage_without_object_discriminator(streamed, value):
    from agent.auxiliary_client import _create_openai_client
    from agent.auxiliary_codex_response import _parse_codex_final_response

    usage = {"input_tokens": value, "output_tokens": 0, "total_tokens": 0}
    transport = httpx.MockTransport(
        lambda request: _codex_usage_http_response(usage, streamed)
    )
    with _create_openai_client(
        api_key="fixture",
        base_url="https://native.example/v1",
        http_client=httpx.Client(transport=transport),
    ) as client:

        def project():
            native = client.responses.create(
                model="requested-model", input="Answer.", stream=streamed
            )
            if streamed:
                (event,) = list(native)
                native = event.response
            return _parse_codex_final_response(native)

        if value is not None and (type(value) is not int or value < 0):
            with pytest.raises(NativeUsageError, match="Invalid native usage"):
                project()
            return
        response = project()
        assert (response.id, response.model, response.usage) == (
            "resp_provider",
            "served-codex",
            None
            if value is None
            else NS(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


def _anthropic_usage_http_response(usage, streamed):
    native = {
        "id": "msg_provider",
        "type": "message",
        "role": "assistant",
        "model": "served-claude",
        "content": [{"type": "text", "text": "Answer."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }
    if not streamed:
        return httpx.Response(200, json=native)
    events = [
        {
            "type": "message_start",
            "message": {**native, "content": [], "stop_reason": None},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "Answer."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": None
            if usage is None
            else {"output_tokens": usage.get("output_tokens")},
        },
        {"type": "message_stop"},
    ]
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        ),
    )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"output_tokens": 0},
        {"input_tokens": 0, "output_tokens": 0},
        {"input_tokens": 0},
        {"input_tokens": 0, "output_tokens": None},
        {"input_tokens": True, "output_tokens": 0},
        {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": "2"},
    ],
)
def test_auxiliary_anthropic_raw_usage_and_null_stream_provenance(streamed, usage):
    from agent.anthropic_adapter import _base_client_kwargs
    from agent.auxiliary_anthropic_response import (
        _normalize_anthropic_auxiliary_response,
    )
    from agent.auxiliary_client import _maybe_wrap_anthropic

    transport = httpx.MockTransport(
        lambda request: _anthropic_usage_http_response(usage, streamed)
    )
    with httpx.Client(transport=transport) as http_client:

        def client_kwargs(*args, **kwargs):
            route, options = _base_client_kwargs(*args, **kwargs)
            return route, {**options, "http_client": http_client}

        with patch(
            "agent.anthropic_adapter._base_client_kwargs", side_effect=client_kwargs
        ):
            wrapped = _maybe_wrap_anthropic(
                NS(), "requested-model", "fixture", "https://native.example/anthropic"
            )
        client = wrapped._real_client
        kwargs = {"model": "requested-model", "max_tokens": 8, "messages": []}

        def project():
            if streamed:
                return wrapped.chat.completions.create(**kwargs)
            native = client.messages.create(**kwargs)
            return _normalize_anthropic_auxiliary_response(native, is_oauth=False)

        if usage and (
            usage.get("input_tokens") is True or "cache_read_input_tokens" in usage
        ):
            with pytest.raises(NativeUsageError, match="Invalid native usage"):
                project()
            return
        response = project()
        assert (
            response.id,
            response.model,
            response.choices[0].message.content,
            response.usage,
        ) == (
            "msg_provider",
            "served-claude",
            "Answer.",
            None
            if not usage
            or usage.get("input_tokens") is None
            or usage.get("output_tokens") is None
            else NS(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("metadata", [False, 0, [], ""])
def test_gemini_validates_falsey_native_usage_containers(streamed, metadata):
    from agent.gemini_native_adapter import GeminiNativeClient

    native = {"usageMetadata": metadata, "candidates": []}
    response = (
        httpx.Response(200, text="data: " + json.dumps(native) + "\n\n")
        if streamed
        else httpx.Response(200, json=native)
    )
    with GeminiNativeClient(
        api_key="fixture",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: response),
        ),
    ) as client:
        with pytest.raises(NativeUsageError) as error:
            result = client.chat.completions.create(
                model="requested-model", messages=[], stream=streamed
            )
            if streamed:
                list(result)
        assert (error.value.code, error.value.field) == ("invalid_usage", "usage")
