"""Native usage validation must precede supported gateway envelope unwrapping."""

import asyncio
from contextlib import contextmanager
import json
from unittest.mock import patch

import httpx
import pytest

from agent.auxiliary_client import async_call_llm, call_llm
from agent.native_usage import NativeUsageError
from agent.native_usage_openai import UsageValidatedAsyncOpenAI, UsageValidatedOpenAI


def _completion(usage, streamed=False):
    message = {
        "role": "assistant",
        "content": "Answer.",
        "refusal": "Refusal detail.",
        "tool_calls": [
            {
                "id": "call_native",
                "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            }
        ],
    }
    return {
        "id": "chat_native",
        "object": "chat.completion.chunk" if streamed else "chat.completion",
        "model": "served-model",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "delta" if streamed else "message": message,
            }
        ],
        "usage": usage,
    }


def _wire_response(body, streamed):
    if streamed:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(body) + "\n\ndata: [DONE]\n\n",
        )
    return httpx.Response(200, json=body)


@contextmanager
def _route(client):
    with (
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "selected-model"),
        ),
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("openai", "selected-model", None, "fixture", None),
        ),
    ):
        yield


def _call(client, async_mode):
    kwargs = {
        "task": "session_search",
        "messages": [{"role": "user", "content": "Answer."}],
    }
    if async_mode:

        async def run():
            with _route(client):
                return await async_call_llm(**kwargs)

        return run()
    with _route(client):
        return call_llm(**kwargs)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("success", [True, None, "missing"])
@pytest.mark.parametrize("value", [True, "2", 2.0, -1])
@pytest.mark.parametrize("optional", [False, True])
def test_native_gateway_envelopes_reject_usage_before_sdk_coercion(
    async_mode, streamed, success, value, optional
):
    usage = {"prompt_tokens": value, "completion_tokens": 1, "total_tokens": 3}
    field = "usage.prompt_tokens"
    if optional:
        usage = {
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": value},
        }
        field = "usage.prompt_tokens_details.cached_tokens"
    body = {"data": _completion(usage, streamed)}
    if success != "missing":
        body["success"] = success
    transport = httpx.MockTransport(lambda request: _wire_response(body, streamed))
    with pytest.raises(NativeUsageError) as error:
        if not async_mode:
            with UsageValidatedOpenAI(
                api_key="fixture",
                max_retries=0,
                http_client=httpx.Client(transport=transport),
            ) as client:
                response = (
                    client.chat.completions.create(
                        model="selected-model", messages=[], stream=True
                    )
                    if streamed
                    else _call(client, False)
                )
                if streamed:
                    list(response)
        else:

            async def run():
                async with UsageValidatedAsyncOpenAI(
                    api_key="fixture",
                    max_retries=0,
                    http_client=httpx.AsyncClient(transport=transport),
                ) as client:
                    response = (
                        await client.chat.completions.create(
                            model="selected-model", messages=[], stream=True
                        )
                        if streamed
                        else await _call(client, True)
                    )
                    if streamed:
                        async for _ in response:
                            pass

            asyncio.run(run())
    assert (error.value.code, error.value.field) == ("invalid_usage", field)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("success", [True, None, "missing"])
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
def test_native_gateway_envelopes_keep_usage_and_response_facts(
    async_mode, success, usage
):
    body = {"data": _completion(usage)}
    if success != "missing":
        body["success"] = success
    transport = httpx.MockTransport(lambda request: _wire_response(body, False))
    if not async_mode:
        with UsageValidatedOpenAI(
            api_key="fixture",
            max_retries=0,
            http_client=httpx.Client(transport=transport),
        ) as client:
            result = _call(client, False)
    else:

        async def run():
            async with UsageValidatedAsyncOpenAI(
                api_key="fixture",
                max_retries=0,
                http_client=httpx.AsyncClient(transport=transport),
            ) as client:
                return await _call(client, True)

        result = asyncio.run(run())
    expected = _completion(None)
    expected["choices"][0]["message"]["annotations"] = None
    expected["choices"][0]["message"]["audio"] = None
    expected["choices"][0]["message"]["function_call"] = None
    expected.update(service_tier=None, system_fingerprint=None)
    if usage is not None and "prompt_tokens" in usage and "completion_tokens" in usage:
        expected["usage"] = {
            **usage,
            "total_tokens": usage.get("total_tokens")
            if usage.get("total_tokens") is not None
            else usage["prompt_tokens"] + usage["completion_tokens"],
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        }
    assert result.model_dump() == expected


@pytest.mark.parametrize("error_envelope", [False, True])
def test_native_gateway_envelopes_keep_response_and_error_precedence(error_envelope):
    nested = _completion({"prompt_tokens": True, "completion_tokens": 1})
    if error_envelope:
        nested["error"] = {"message": "Provider rejected the request."}
        body = {"success": False, "data": nested}
    else:
        body = {**_completion(None), "data": nested}
    transport = httpx.MockTransport(lambda request: _wire_response(body, False))
    with UsageValidatedOpenAI(
        api_key="fixture", max_retries=0, http_client=httpx.Client(transport=transport)
    ) as client:
        if error_envelope:
            with pytest.raises(RuntimeError, match="Provider rejected the request"):
                _call(client, False)
            return
        response = _call(client, False)
        assert (response.id, response.model, response.usage, response.data) == (
            "chat_native",
            "served-model",
            None,
            nested,
        )


def _json_call(body, async_mode):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    if not async_mode:
        with UsageValidatedOpenAI(
            api_key="fixture",
            max_retries=0,
            http_client=httpx.Client(transport=transport),
        ) as client:
            return _call(client, False)

    async def run():
        async with UsageValidatedAsyncOpenAI(
            api_key="fixture",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            return await _call(client, True)

    return asyncio.run(run())


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("finish", [None, "missing"])
def test_gateway_envelopes_preserve_unknown_finish_reason(async_mode, finish):
    completion = _completion(None)
    if finish == "missing":
        completion["choices"][0].pop("finish_reason")
    else:
        completion["choices"][0]["finish_reason"] = finish
    response = _json_call({"success": True, "data": completion}, async_mode)
    choice = response.choices[0]
    assert (
        response.id,
        response.model,
        choice.finish_reason,
        choice.message.content,
        choice.message.refusal,
        response.usage,
    ) == ("chat_native", "served-model", None, "Answer.", "Refusal detail.", None)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("identity", [None, "missing"])
def test_gateway_envelopes_preserve_absent_identity(async_mode, identity):
    completion = _completion(None)
    if identity == "missing":
        completion.pop("id")
    else:
        completion["id"] = identity
    response = _json_call({"success": True, "data": completion}, async_mode)
    assert (response.id, response.model, response.choices[0].finish_reason) == (
        None,
        "served-model",
        "stop",
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status_field", ["status", "status_code"])
def test_gateway_error_envelopes_preserve_authentication_facts(
    async_mode, status_field
):
    from agent.auxiliary_response_envelope import AuxiliaryEnvelopeError
    from agent.error_classifier import classify_api_error

    error = {
        "message": "Fixture authentication rejected.",
        status_field: 401,
        "type": "authentication_error",
        "code": "invalid_api_key",
    }
    completion = _completion({"prompt_tokens": True, "completion_tokens": 1})
    completion["error"] = error
    with pytest.raises(AuxiliaryEnvelopeError) as caught:
        _json_call({"success": False, "data": completion}, async_mode)
    exc = caught.value
    classified = classify_api_error(exc, provider="openai")
    assert (
        type(exc),
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(exc, "type", None),
        getattr(exc, "body", None),
        str(exc),
        classified.reason.value,
    ) == (
        AuxiliaryEnvelopeError,
        401,
        "invalid_api_key",
        "authentication_error",
        {"error": error},
        "Auxiliary session_search: provider returned error envelope: Fixture authentication rejected.",
        "auth",
    )
