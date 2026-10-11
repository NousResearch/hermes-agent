"""Completion provenance remains independent of JSON parsing (regression #136471)."""

import asyncio
from collections import UserDict
from types import SimpleNamespace

import pytest

from agent.plugin_llm import _TrustPolicy, make_plugin_llm_for_test


@pytest.mark.parametrize("method", ["complete", "acomplete", "complete_structured", "acomplete_structured"])
@pytest.mark.parametrize("shape", ["object", "dict", "mapping"])
@pytest.mark.parametrize("reason", ["stop", "length", "content_filter", "tool_calls", "vendor_reason", None, "", 7, {}])
def test_public_result_preserves_completion_evidence(method, shape, reason):
    choice = {"message": SimpleNamespace(content='{"ok": true}'), "finish_reason": reason}
    response = {"choices": [SimpleNamespace(**choice)]}
    if shape == "object":
        response = SimpleNamespace(**response)
    else:
        response["choices"] = [choice if shape == "dict" else UserDict(choice)]
        if shape == "mapping":
            response = UserDict(response)

    async def call_async(**kwargs):
        return "provider", "model", response

    llm = make_plugin_llm_for_test(
        plugin_id="test", policy=_TrustPolicy(plugin_id="test"),
        sync_caller=lambda **kwargs: ("provider", "model", response), async_caller=call_async,
    )
    if method.endswith("structured"):
        result = getattr(llm, method)(instructions="Extract", input=[{"type": "text", "text": "data"}], json_mode=True)
    else:
        result = getattr(llm, method)([{"role": "user", "content": "data"}])
    if method.startswith("a"):
        result = asyncio.run(result)
    expected = reason if isinstance(reason, str) and reason else None
    assert result.finish_reason == expected
    assert result.audit["finish_reason"] == expected
    if shape == "object" and method.endswith("structured"):
        assert result.parsed == {"ok": True}
        assert result.content_type == "json"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reason", [None, "stop", "length", "content_filter", "tool_calls"])
def test_stream_completion_evidence_survives_usage_tail(asynchronous, reason):
    from agent.auxiliary_client import _aggregate_chat_stream, _aggregate_chat_stream_async

    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"ok": true}'), finish_reason=None)]),
        SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason=reason)]),
        SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=5)),
    ]

    async def stream():
        for chunk in chunks:
            yield chunk

    response = (asyncio.run(_aggregate_chat_stream_async(stream())) if asynchronous
                else _aggregate_chat_stream(iter(chunks)))
    assert response.choices[0].finish_reason == reason
    assert response.choices[0].message.content == '{"ok": true}'
    assert response.usage.total_tokens == 5
