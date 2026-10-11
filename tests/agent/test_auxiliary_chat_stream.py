"""Native streamed completion facts must remain distinct from transport EOF."""

import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx
import pytest
from openai.types.completion_usage import CompletionUsage

from agent import auxiliary_client as aux
from agent.native_usage_openai import UsageValidatedAsyncOpenAI, UsageValidatedOpenAI


def _chunk(delta, finish=None):
    return {
        "id": "observed-id",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "observed-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _invoke(chunks, async_mode, done=True):
    requests = []
    frames = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    if done:
        frames += "data: [DONE]\n\n"

    def physical(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=frames
        )

    def prepared(client):
        return aux._PreparedAuxRequest(
            client,
            "configured-model",
            {
                "model": "configured-model",
                "messages": [{"role": "user", "content": "test"}],
            },
            "custom",
            "custom",
            "configured-model",
            str(client.base_url),
            "fixture",
            None,
            30,
            {},
            str(client.base_url),
        )

    def route(client):
        return patch.object(aux, "_prepare_aux_request", return_value=prepared(client))

    transport = httpx.MockTransport(physical)
    if async_mode:

        async def run():
            async with UsageValidatedAsyncOpenAI(
                api_key="fixture",
                base_url="https://copilot.tencent.com/v1",
                http_client=httpx.AsyncClient(transport=transport),
            ) as client:
                with route(client):
                    return await aux.async_call_llm(
                        messages=[{"role": "user", "content": "test"}]
                    )

        response = asyncio.run(run())
    else:
        with UsageValidatedOpenAI(
            api_key="fixture",
            base_url="https://copilot.tencent.com/v1",
            http_client=httpx.Client(transport=transport),
        ) as client:
            with route(client):
                response = aux.call_llm(messages=[{"role": "user", "content": "test"}])
    assert requests == [
        {
            "model": "configured-model",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ]
    return response


def _expected(finish, refusal=None):
    return NS(
        id="observed-id",
        model="observed-model",
        object="chat.completion",
        choices=[
            NS(
                index=0,
                finish_reason=finish,
                message=NS(
                    role="assistant",
                    content="partial",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_details=None,
                    **({"refusal": refusal} if refusal is not None else {}),
                ),
            )
        ],
        usage=None,
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("terminal", ["absent", None, "stop", "length"])
@pytest.mark.parametrize("done", [False, True])
def test_stream_completion_requires_reported_terminal_reason(
    async_mode, terminal, done
):
    chunks = [
        _chunk(
            {
                "content": "partial",
                "reasoning_content": "thought",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_observed",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            }
        )
    ]
    chunks[0]["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if terminal != "absent":
        chunks.append(_chunk({}, terminal))
    response = _invoke(chunks, async_mode, done)
    expected = _expected(None if terminal == "absent" else terminal)
    expected.choices[0].message.reasoning = "thought"
    expected.choices[0].message.tool_calls = [
        NS(
            id="call_observed",
            type="function",
            function=NS(name="inspect", arguments="{}"),
        )
    ]
    expected.usage = CompletionUsage(
        prompt_tokens=0, completion_tokens=0, total_tokens=0
    )
    assert response == expected


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("terminal", [None, "stop", "content_filter"])
@pytest.mark.parametrize("content", ["partial", None])
def test_stream_refusal_fragments_remain_in_terminal_message(
    async_mode, terminal, content
):
    chunks = [
        _chunk(
            {
                **({"content": content} if content is not None else {}),
                "refusal": "Cannot ",
            }
        ),
        _chunk({"refusal": "comply."}),
        _chunk({}, terminal),
    ]
    ticks = []
    with aux.aux_progress_hook(lambda: ticks.append(1)):
        response = _invoke(chunks, async_mode)
    expected = _expected(terminal, "Cannot comply.")
    expected.choices[0].message.content = content or ""
    assert (response, ticks) == (expected, [1, 1])
