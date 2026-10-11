"""Auxiliary normalisation must preserve provider response facts."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from agent.auxiliary_client import _AnthropicCompletionsAdapter, _CodexCompletionsAdapter, _aggregate_chat_stream
from agent.usage_pricing import CanonicalUsage, normalize_usage
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

    assert response.usage == (None if not metadata else NS(
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    ))
