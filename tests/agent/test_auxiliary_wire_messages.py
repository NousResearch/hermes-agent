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
