"""Auxiliary requests must preserve caller-required native controls."""

import asyncio
import json
import inspect
from types import SimpleNamespace as NS
from typing import Any, Callable, TypedDict

import httpx
import pytest

from agent import auxiliary_client as aux


TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect_item",
        "description": "Inspect an item.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}


class _CacheOptions(TypedDict):
    async_mode: bool
    base_url: str
    api_key: str
    model: str


@pytest.fixture
def native_route(monkeypatch):
    clients = []

    def build(wire, async_mode=False, scenario="normal"):
        original_get_cached = aux._get_cached_client
        requests: list[dict[str, Any]] = []
        dispatch_requirements = []
        closed_streams = []

        class ObservedBody(httpx.ByteStream):
            def close(self):
                closed_streams.append(wire)

            async def aclose(self):
                closed_streams.append(wire)

        cancellation = asyncio.CancelledError("fixture cancellation")
        model = "ft:gpt-4.1:fixture" if scenario.startswith("fine_tuned") else "gpt-4.1"

        def handle(request):
            from agent.auxiliary_request_requirements import required_output_cap

            dispatch_requirements.append(required_output_cap())
            requests.append(json.loads(request.content))
            if scenario == "cancel":
                raise cancellation
            if scenario.startswith("fallback") and len(requests) == 1:
                return httpx.Response(
                    429,
                    json={
                        "error": {
                            "message": "usage limit reached",
                            "type": "insufficient_quota",
                        }
                    },
                )
            if scenario == "auth" and len(requests) == 1:
                return httpx.Response(
                    401,
                    json={
                        "error": {
                            "message": "expired key",
                            "type": "authentication_error",
                        }
                    },
                )
            if scenario == "fallback_parameter" and len(requests) == 2:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": "unsupported_parameter: max_tokens",
                            "type": "invalid_request_error",
                            "code": "unsupported_parameter",
                        }
                    },
                )
            if scenario == "parameter" and len(requests) == 1:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": "unsupported_parameter: max_tokens",
                            "type": "invalid_request_error",
                            "code": "unsupported_parameter",
                        }
                    },
                )
            if scenario == "affordable" and len(requests) == 1:
                return httpx.Response(
                    402,
                    json={
                        "error": {
                            "message": "can only afford 800 tokens",
                            "type": "insufficient_quota",
                        }
                    },
                )
            if scenario == "sdk_retry" and len(requests) == 1:
                return httpx.Response(
                    500,
                    json={
                        "error": {
                            "message": "temporary failure",
                            "type": "server_error",
                        }
                    },
                )
            if wire == "messages":
                message = {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                }
                events: list[dict[str, Any]] = [
                    {"type": "message_start", "message": message},
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Answer."},
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    },
                    {"type": "message_stop"},
                ]
                return httpx.Response(
                    200,
                    text="".join(
                        "event: "
                        + event["type"]
                        + "\ndata: "
                        + json.dumps(event)
                        + "\n\n"
                        for event in events
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            if wire == "converse":
                return httpx.Response(
                    200,
                    json={
                        "output": {
                            "message": {
                                "role": "assistant",
                                "content": [{"text": "Answer."}],
                            }
                        },
                        "stopReason": "end_turn",
                        "usage": {
                            "inputTokens": 1,
                            "outputTokens": 1,
                            "totalTokens": 2,
                        },
                        "metrics": {"latencyMs": 1},
                    },
                )
            if wire == "gemini":
                payload = {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "Answer."}]},
                            "finishReason": "STOP",
                        }
                    ]
                }
                if "streamGenerateContent" in request.url.path:
                    return httpx.Response(
                        200,
                        text="data: " + json.dumps(payload) + "\n\n",
                        headers={"content-type": "text/event-stream"},
                    )
                return httpx.Response(200, json=payload)
            if wire == "responses":
                response = {
                    "id": "response",
                    "object": "response",
                    "status": "completed",
                    "model": model,
                    "output": [
                        {
                            "id": "item",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Answer.",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                }
                return httpx.Response(
                    200,
                    text="data: "
                    + json.dumps(
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": response["output"][0],
                        }
                    )
                    + "\n\ndata: "
                    + json.dumps({"type": "response.completed", "response": response})
                    + "\n\n",
                    headers={"content-type": "text/event-stream"},
                )
            if requests[-1].get("stream"):
                payload = {
                    "id": "chat",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Answer."},
                            "finish_reason": "stop",
                        }
                    ],
                }
                return httpx.Response(
                    200,
                    stream=ObservedBody(
                        ("data: " + json.dumps(payload) + "\n\n").encode()
                    ),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(
                200,
                json={
                    "id": "chat",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Answer."},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        transport = httpx.MockTransport(handle)
        if wire == "converse":
            import io
            boto3 = pytest.importorskip("boto3")
            pytest.importorskip("botocore")
            from botocore.awsrequest import AWSResponse
            from botocore.config import Config

            class ResponseBody:
                def __init__(self, content):
                    self.content = io.BytesIO(content)

                def stream(self, amt=1024):
                    while chunk := self.content.read(amt):
                        yield chunk

            sdk = boto3.client(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id="fixture",
                aws_secret_access_key="fixture",
                config=Config(retries={"max_attempts": 0}),
            )
            clients.append(sdk)

            def send(request):
                reply = handle(
                    httpx.Request(request.method, request.url, content=request.body)
                )
                return AWSResponse(
                    request.url,
                    reply.status_code,
                    dict(reply.headers),
                    ResponseBody(reply.content),
                )

            monkeypatch.setattr(sdk._endpoint.http_session, "send", send)
            monkeypatch.setattr(
                "agent.bedrock_adapter._get_bedrock_runtime_client", lambda region: sdk
            )
            model = "amazon.nova-pro-v1:0"
            provider = "bedrock"
            client = aux.BedrockAuxiliaryClient("us-east-1", model)
        elif wire == "messages":
            from agent.auxiliary_request_clients import anthropic_request_client_class

            client = anthropic_request_client_class()(
                api_key="fixture",
                base_url="https://api.anthropic.com",
                max_retries=0,
                http_client=httpx.Client(transport=transport),
            )
            clients.append(client)
            model = "claude-sonnet-4-5"
            client = aux.AnthropicAuxiliaryClient(
                client, model, "fixture", "https://api.anthropic.com"
            )
            provider = "anthropic"
        elif wire == "gemini":
            from agent.gemini_native_adapter import GeminiNativeClient

            client = GeminiNativeClient(
                api_key="fixture", http_client=httpx.Client(transport=transport)
            )
            model = "gemini-2.5-flash"
            provider = "gemini"
            clients.append(client)
        else:
            from agent.auxiliary_request_clients import _guarded_openai_client_classes

            client = _guarded_openai_client_classes()[0](
                api_key="fixture",
                base_url="https://unknown.example/v1"
                if scenario == "unknown"
                else "https://api.openai.com/v1",
                max_retries=1 if scenario == "sdk_retry" else 0,
                http_client=httpx.Client(transport=transport),
            )
            clients.append(client)
            provider = "openai"
            if wire == "responses":
                client = aux.CodexAuxiliaryClient(client, model)
        if async_mode:
            from agent.gemini_native_adapter import AsyncGeminiNativeClient
            from agent.auxiliary_request_clients import _guarded_openai_client_classes

            def async_chat(sync_client):
                result = _guarded_openai_client_classes()[1](
                    api_key=sync_client.api_key,
                    base_url=sync_client.base_url,
                    max_retries=sync_client.max_retries,
                    http_client=httpx.AsyncClient(transport=transport),
                )
                clients.append(result)
                return result

            async_factories: dict[str, Callable[[Any], Any]] = {
                "converse": aux.AsyncBedrockAuxiliaryClient,
                "gemini": AsyncGeminiNativeClient,
                "messages": aux.AsyncAnthropicAuxiliaryClient,
                "responses": aux.AsyncCodexAuxiliaryClient,
                "chat": async_chat,
            }
            client = async_factories[wire](client)
        monkeypatch.setattr(
            aux, "_get_cached_client", lambda *args, **kwargs: (client, model)
        )
        monkeypatch.setattr(
            aux,
            "_resolve_task_provider_model",
            lambda *args, **kwargs: (provider, model, None, "fixture", None),
        )
        monkeypatch.setattr(
            aux,
            "_refresh_provider_credentials",
            lambda *args, **kwargs: scenario == "auth",
        )
        monkeypatch.setattr(
            aux, "_recoverable_pool_provider", lambda *args, **kwargs: None
        )
        if scenario.startswith("fallback"):
            monkeypatch.setattr(
                aux,
                "_try_configured_fallback_chain",
                lambda *args, **kwargs: (client, model, "openai"),
            )
            monkeypatch.setattr(aux, "_to_async_client", lambda c, m, **kwargs: (c, m))
        return NS(
            requests=requests,
            dispatch_requirements=dispatch_requirements,
            closed_streams=closed_streams,
            cancellation=cancellation,
            client=client,
            original_get_cached=original_get_cached,
        )

    yield build
    for client in reversed(clients):
        pending = client.close()
        if inspect.isawaitable(pending):
            asyncio.run(pending)


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "wire,scenario,required,expected",
    [
        ("chat", "normal", True, None),
        ("chat", "parameter", True, "output_cap_missing"),
        ("chat", "parameter", False, None),
        ("chat", "affordable", True, None),
        ("chat", "sdk_retry", True, None),
        ("chat", "extra", True, "output_cap_exceeded"),
        ("chat", "dual_cap", True, "output_cap_exceeded"),
        ("responses", "normal", True, "output_cap_missing"),
        ("messages", "normal", True, None),
        ("messages", "thinking", True, "output_cap_exceeded"),
        ("gemini", "normal", True, None),
        ("gemini", "thinking", True, "output_cap_exceeded"),
        ("chat", "missing", True, "missing_value"),
        ("converse", "normal", True, None),
        ("chat", "auth", True, None),
        ("chat", "fallback", True, None),
        ("chat", "fallback_parameter", True, "output_cap_missing"),
        ("chat", "cancel", True, None),
        ("chat", "cache", True, None),
        ("gemini", "deferred", True, None),
        ("chat", "early_close", True, None),
    ],
)
def test_explicit_output_cap_is_preserved(
    native_route, wire, scenario, required, expected, async_mode, monkeypatch
):
    from agent.auxiliary_request_requirements import (
        AuxiliaryRequestRequirements,
        AuxiliaryRequestRequirementError,
    )

    route = native_route(wire, async_mode, scenario)
    requests = route.requests
    kwargs = {
        "task": "title",
        "messages": [{"role": "user", "content": "Answer."}],
        "max_tokens": None if scenario == "missing" else 1024,
        "requirements": AuxiliaryRequestRequirements(max_tokens=True)
        if required
        else None,
    }
    if scenario == "thinking":
        kwargs["reasoning_config"] = {"enabled": True, "effort": "medium"}
    if scenario == "extra":
        kwargs["extra_body"] = {"max_tokens": 2048}
    if scenario == "dual_cap":
        kwargs["extra_body"] = {"max_completion_tokens": 2048}

    if scenario == "cache":
        from agent.auxiliary_request_requirements import request_requirements_scope
        from agent.auxiliary_request_clients import openai_request_client_classes
        from openai import OpenAI

        key_kwargs: _CacheOptions = {
            "async_mode": False,
            "base_url": "https://api.openai.com/v1",
            "api_key": "fixture",
            "model": "gpt-4.1",
        }
        ordinary_key = aux._client_cache_key("openai", **key_kwargs)
        ordinary_classes = openai_request_client_classes()
        with request_requirements_scope(
            AuxiliaryRequestRequirements(max_tokens=True), None, 1024
        ):
            required_key = aux._client_cache_key("openai", **key_kwargs)
            required_classes = openai_request_client_classes()
        with request_requirements_scope(AuxiliaryRequestRequirements(), None, None):
            empty_key = aux._client_cache_key("openai", **key_kwargs)
            empty_classes = openai_request_client_classes()
        assert (
            ordinary_key == empty_key,
            ordinary_key != required_key,
            ordinary_classes[0] is OpenAI,
            empty_classes == ordinary_classes,
            required_classes[0] is not OpenAI,
            openai_request_client_classes() == ordinary_classes,
        ) == (True,) * 6
        aux.shutdown_cached_clients()
        built = []

        def resolve(*args, **kwargs):
            client = aux.OpenAI(
                api_key="fixture", base_url="https://api.openai.com/v1", max_retries=0
            )
            built.append(client)
            return client, "gpt-4.1"

        monkeypatch.setattr(aux, "resolve_provider_client", resolve)
        try:
            ordinary, _ = route.original_get_cached("openai", **key_kwargs)
            with request_requirements_scope(
                AuxiliaryRequestRequirements(max_tokens=True), None, 1024
            ):
                guarded, _ = route.original_get_cached("openai", **key_kwargs)
            repeated, _ = route.original_get_cached("openai", **key_kwargs)
            assert (
                type(ordinary) is OpenAI,
                ordinary is repeated,
                guarded is not ordinary,
                type(guarded) is required_classes[0],
                len(built),
            ) == (True, True, True, True, 2)
        finally:
            aux.shutdown_cached_clients()
        return

    def invoke():
        if scenario == "deferred" and not async_mode:
            response = aux.call_llm(**kwargs, stream=True)
            assert (requests, route.dispatch_requirements) == ([], [])
            from agent.auxiliary_request_requirements import (
                request_requirements_scope,
                required_output_cap,
            )

            with request_requirements_scope(
                AuxiliaryRequestRequirements(max_tokens=True), None, 7
            ):
                result = aux._aggregate_chat_stream(response, model="gemini-2.5-flash")
                restored = required_output_cap()
            assert (restored, required_output_cap()) == (7, None)
            return result
        return (
            asyncio.run(aux.async_call_llm(**kwargs))
            if async_mode
            else aux.call_llm(**kwargs)
        )

    if scenario == "early_close":
        if async_mode:
            with aux.aux_progress_hook(lambda: None):
                response = invoke()
            assert (response.choices[0].message.content, route.closed_streams) == (
                "Answer.",
                ["chat"],
            )
        else:
            stream = aux.call_llm(**kwargs, stream=True)
            assert route.closed_streams == []
            stream.close()
            assert route.closed_streams == ["chat"]
        return

    if scenario == "cancel":
        with pytest.raises(asyncio.CancelledError) as caught:
            invoke()
        assert (caught.value is route.cancellation, len(requests)) == (True, 1)
        return
    if expected is not None:
        with pytest.raises(AuxiliaryRequestRequirementError) as caught:
            invoke()
        assert (
            caught.value.requirement.value,
            caught.value.cause.value,
            len(requests),
        ) == (
            "max_tokens",
            expected,
            {"parameter": 1, "fallback_parameter": 2}.get(scenario, 0),
        )
        if scenario in {"parameter", "fallback_parameter"}:
            from openai import BadRequestError

            assert isinstance(caught.value.__cause__, BadRequestError)
        return
    if scenario == "affordable" and async_mode:
        from openai import APIStatusError

        with pytest.raises(APIStatusError) as caught:
            invoke()
        assert (caught.value.status_code, len(requests)) == (402, 1)
        return
    response = invoke()
    caps = []
    for body in requests:
        caps.append(
            {
                "chat": body.get("max_tokens", body.get("max_completion_tokens")),
                "messages": body.get("max_tokens"),
                "gemini": body.get("generationConfig", {}).get("maxOutputTokens"),
                "converse": body.get("inferenceConfig", {}).get("maxTokens"),
            }[wire]
        )
    expected_caps = [1024]
    if scenario == "affordable":
        expected_caps = [1024, 736]
    elif scenario in {"sdk_retry", "auth", "fallback"}:
        expected_caps = [1024, 1024]
    elif scenario == "parameter":
        expected_caps = [None, None]
    assert (response.choices[0].message.content, caps) == ("Answer.", expected_caps)
    if scenario == "deferred":
        assert route.dispatch_requirements == [1024]


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "wire,scenario,required,expected",
    [
        ("chat", "normal", True, None),
        ("responses", "normal", True, None),
        ("messages", "normal", True, "unsupported_route"),
        ("gemini", "normal", True, "unsupported_route"),
        ("messages", "normal", False, None),
        ("gemini", "normal", False, None),
        ("chat", "extra", True, "tool_schema_changed"),
        ("chat", "tools_bool", True, "tool_schema_changed"),
        ("chat", "function_null", True, "tool_schema_changed"),
        ("chat", "unknown", True, "unsupported_route"),
        ("responses", "sanitized", True, "tool_schema_changed"),
        ("chat", "fine_tuned_parallel", True, "unsupported_route"),
        ("chat", "fine_tuned_serial", True, None),
        ("chat", "missing", True, "missing_value"),
        ("converse", "normal", True, "unsupported_route"),
    ],
)
def test_strict_tool_schema_is_preserved(
    native_route, wire, scenario, required, expected, async_mode
):
    from agent.auxiliary_request_requirements import (
        AuxiliaryRequestRequirements,
        AuxiliaryRequestRequirementError,
    )

    route = native_route(wire, async_mode, scenario)
    requests = route.requests
    kwargs = {
        "task": "title",
        "messages": [{"role": "user", "content": "Answer."}],
        "tools": None if scenario == "missing" else [TOOL],
        "requirements": AuxiliaryRequestRequirements(strict_tools=True)
        if required
        else None,
    }
    if scenario == "sanitized":
        import copy

        tools = copy.deepcopy([TOOL])
        tools[0]["function"]["parameters"]["properties"]["value"]["pattern"] = (
            "^[a-z]+$"
        )
        kwargs["tools"] = tools
    if scenario == "extra":
        kwargs["extra_body"] = {"tools": []}
    if scenario == "tools_bool":
        kwargs["extra_body"] = {"tools": True}
    if scenario == "function_null":
        kwargs["extra_body"] = {"tools": [{"type": "function", "function": None}]}
    if scenario == "fine_tuned_serial":
        kwargs["extra_body"] = {"parallel_tool_calls": False}

    def invoke():
        return (
            asyncio.run(aux.async_call_llm(**kwargs))
            if async_mode
            else aux.call_llm(**kwargs)
        )

    if expected is not None:
        with pytest.raises(AuxiliaryRequestRequirementError) as caught:
            invoke()
        assert (caught.value.requirement.value, caught.value.cause.value, requests) == (
            "strict_tools",
            expected,
            [],
        )
        return
    response = invoke()
    body = requests[0]
    if wire == "chat":
        tool = body["tools"][0]["function"]
    elif wire == "gemini":
        tool = body["tools"][0]["functionDeclarations"][0]
    else:
        tool = body["tools"][0]
    if required:
        assert (
            response.choices[0].message.content,
            tool.get("strict"),
            tool.get("parameters"),
        ) == (
            "Answer.",
            True,
            TOOL["function"]["parameters"],
        )
    else:
        assert (response.choices[0].message.content, tool.get("strict")) == (
            "Answer.",
            None,
        )
