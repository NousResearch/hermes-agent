"""Offline real-SDK wire checks, including overrides made after kwargs construction."""
import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from tests.tools.test_azure_schema import PATTERN, get_pattern, tool
from tools.azure_schema import AzureToolSchemaError, sanitize_azure_tool_schemas


@pytest.mark.parametrize("reference", ["$ref", "$dynamicRef", "$recursiveRef"])
@pytest.mark.parametrize("encoded", [False, True])
def test_renamed_pattern_properties_with_references_fail_locally(reference, encoded):
    pointer = "#/patternProperties/%5Cp%7BZ%7D" if encoded else r"#/patternProperties/\p{Z}"
    params = {
        "patternProperties": {r"\p{Z}": {"type": "string", "minLength": 5}},
        "properties": {"x": {reference: pointer}},
    }
    original = {"type": "function", "name": "references", "parameters": params}
    with pytest.raises(AzureToolSchemaError, match="references with renamed pattern properties"):
        sanitize_azure_tool_schemas([original])
    assert params["properties"]["x"][reference] == pointer
    assert r"\p{Z}" in params["patternProperties"]


def _response(request, sent):
    body = json.loads(request.content)
    sent.append(body)
    if body.get("stream"):
        chunk = {
            "id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}],
        }
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")
    return httpx.Response(200, json={
        "id": "test", "object": "chat.completion", "created": 0, "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
    })


def _late_kwargs(bad):
    from agent.auxiliary_client import _build_call_kwargs
    kwargs = _build_call_kwargs(
        "azure-foundry", "test-model", [{"role": "user", "content": "test"}], tools=[tool()],
    )
    # Relay/plugin callbacks can replace tools after this builder returns.
    late_tools = [tool(r"\p{Unknown}" if bad else PATTERN)]
    kwargs["extra_body"] = {"tools": late_tools}
    return kwargs, late_tools


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("bad", [False, True])
def test_auxiliary_physical_send_preflights_late_overrides(async_mode, stream, bad):
    from agent.auxiliary_client import _acreate_with_progress, _create_with_progress_once
    sent = []
    transport = httpx.MockTransport(lambda request: _response(request, sent))
    kwargs, late_tools = _late_kwargs(bad)
    options = {"api_key": "test-key", "base_url": "https://private-proxy.test/v1", "max_retries": 0}

    async def async_send():
        async with AsyncOpenAI(**options, http_client=httpx.AsyncClient(transport=transport)) as client:
            client._hermes_aux_effective_provider = "azure-foundry"
            return await _acreate_with_progress(client, kwargs, force_stream=stream)

    def send():
        if async_mode:
            return asyncio.run(async_send())
        with OpenAI(**options, http_client=httpx.Client(transport=transport)) as client:
            client._hermes_aux_effective_provider = "azure-foundry"
            return _create_with_progress_once(client, kwargs, force_stream=stream)

    if bad:
        with pytest.raises(AzureToolSchemaError):
            send()
        assert not sent
    else:
        assert send().choices[0].message.content == "OK"
        assert len(sent) == 1
        assert get_pattern(sent[0]["tools"][0]) != PATTERN
        assert sent[0]["messages"] == kwargs["messages"]
        assert get_pattern(late_tools[0]) == PATTERN
