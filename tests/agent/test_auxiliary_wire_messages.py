"""The final auxiliary destination owns message sanitation, not virtual MoA."""
import asyncio
import copy
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI, OpenAI

from agent.auxiliary_client import _relay_async_completion, _relay_sync_completion


@pytest.mark.parametrize("async_mode", [False, True])
def test_auxiliary_chat_wire_sanitizes_without_mutating_history(async_mode):
    history = [{"role": "assistant", "content": "answer", "_db_persisted": True,
                "timestamp": 12, "tool_calls": [], "reasoning": "private"}]
    original = copy.deepcopy(history)
    kwargs = {"model": "fixture-model", "messages": history}
    captured = []
    if async_mode:
        async def run():
            async with AsyncOpenAI(api_key="fixture", base_url="http://127.0.0.1:1/v1") as client:
                async def send(request):
                    captured.append(request)
                await _relay_async_completion(client, kwargs, create=send)
        asyncio.run(run())
    else:
        with OpenAI(api_key="fixture", base_url="http://127.0.0.1:1/v1") as client:
            _relay_sync_completion(client, kwargs, create=captured.append)
    assert captured[0]["messages"] == [
        {"role": "assistant", "content": "answer", "reasoning": "private"}
    ]
    assert kwargs["messages"] == original


@pytest.mark.parametrize("async_mode", [False, True])
def test_auxiliary_native_adapters_keep_replay_and_tool_fields(async_mode):
    history = [{"role": "assistant", "content": "", "_db_persisted": True,
                "codex_reasoning_items": [{"type": "reasoning", "id": "rs_fixture"}],
                "thinking_blocks": [{"type": "thinking", "thinking": "native", "signature": "sig"}],
                "tool_calls": [{"id": "call_fixture", "type": "function",
                                "function": {"name": "fixture", "arguments": "{}"}}]}]
    kwargs = {"model": "native", "messages": history}
    captured = []
    if async_mode:
        async def send(request):
            captured.append(request)
        asyncio.run(_relay_async_completion(SimpleNamespace(), kwargs, create=send))
    else:
        _relay_sync_completion(SimpleNamespace(), kwargs, create=captured.append)
    assert captured[0] is kwargs
    assert captured[0]["messages"] is history


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider,strict", [("mistral", True), ("custom", False)])
def test_resolved_auxiliary_provider_owns_opaque_model_role_policy(async_mode, provider, strict):
    history = [
        {"role": "user", "content": "Read"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "read00001", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "read00001", "content": "result"},
        {"role": "user", "content": "New direction"},
    ]
    original = copy.deepcopy(history)
    kwargs = {"model": "opaque-model", "messages": history}
    captured = []
    if async_mode:
        async def run():
            async with AsyncOpenAI(api_key="fixture", base_url="http://127.0.0.1:1/v1") as client:
                async def send(request):
                    captured.append(request)
                await _relay_async_completion(client, kwargs, provider=provider, create=send)
        asyncio.run(run())
    else:
        with OpenAI(api_key="fixture", base_url="http://127.0.0.1:1/v1") as client:
            _relay_sync_completion(client, kwargs, provider=provider, create=captured.append)
    wire = captured[0]["messages"]
    assert [m["role"] for m in wire] == (["user", "assistant", "tool", "assistant", "user"] if strict else ["user", "assistant", "tool", "user"])
    assert wire[-1] == original[-1]
    assert kwargs["messages"] == original
