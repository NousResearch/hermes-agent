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
