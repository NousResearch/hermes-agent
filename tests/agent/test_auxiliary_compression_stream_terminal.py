"""Regression: auxiliary compression must never accept clean EOF as completion.

A stream can carry usable content and then end before the terminal frame.
Stock Hermes previously synthesized finish_reason="stop" at EOF, allowing a
partially transmitted checkpoint into the context-compression commit path.
Other auxiliary tasks retain the previous permissive behavior.
"""
import asyncio
from types import SimpleNamespace

import pytest

from agent.auxiliary_client import _aggregate_chat_stream, _aggregate_chat_stream_async


def _chunk(text="", finish=None, usage=None):
    return SimpleNamespace(
        id="chatcmpl-fixture",
        model="test-model",
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content=text, tool_calls=None),
            finish_reason=finish,
        )],
        usage=usage,
    )


def _usage_only():
    return SimpleNamespace(
        id="chatcmpl-fixture", model="test-model",
        choices=[], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
    )


async def _async_chunks(frames):
    for frame in frames:
        yield frame


def _aggregate(frames, *, task="compression", async_mode=False):
    if async_mode:
        return asyncio.run(_aggregate_chat_stream_async(
            _async_chunks(frames), model="test-model", task=task,
        ))
    return _aggregate_chat_stream(iter(frames), model="test-model", task=task)


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_compression_clean_eof_after_content_is_not_success(async_mode):
    with pytest.raises(ConnectionError, match=r"compression.*terminal.*finish_reason"):
        _aggregate([_chunk("## Partial checkpoint, more sections pending")],
                   async_mode=async_mode)


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_compression_empty_eof_is_not_success(async_mode):
    with pytest.raises(ConnectionError, match=r"compression.*terminal.*finish_reason"):
        _aggregate([], async_mode=async_mode)


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_compression_explicit_stop_then_usage_only_is_accepted(async_mode):
    result = _aggregate([
        _chunk("## Complete checkpoint\n"),
        _chunk("All durable details preserved.", finish="stop"),
        _usage_only(),
    ], async_mode=async_mode)
    assert result.choices[0].message.content == (
        "## Complete checkpoint\nAll durable details preserved."
    )
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens == 11


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_compression_length_terminal_is_not_rewritten_to_stop(async_mode):
    result = _aggregate([_chunk("truncated due to model length", finish="length")],
                        async_mode=async_mode)
    assert result.choices[0].finish_reason == "length"


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_other_auxiliary_tasks_keep_legacy_eof_behavior(async_mode):
    result = _aggregate([_chunk("ordinary auxiliary result")], task="title",
                        async_mode=async_mode)
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].message.content == "ordinary auxiliary result"

@pytest.mark.parametrize("terminal", ["missing", "stop", "error"])
def test_actual_openai_sdk_sse_compression_terminal_contract(terminal):
    """Use the real installed SDK parser, not only SimpleNamespace chunks."""
    import httpx
    from openai import APIError, OpenAI

    checkpoint = (
        "## Historical Task Snapshot\nKnown project state.\n"
        "## Constraints & Preferences\nPreserve tools and source.\n"
        "## Completed Actions\nVerified prior work.\n"
        "## Detailed Session Log (oldest first)\nOne completed action."
    )
    import json

    def event(delta=None, finish_reason=None):
        return "data: " + json.dumps({
            "id": "chatcmpl-aux-fixture",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fixture-model",
            "choices": [{
                "index": 0, "delta": delta or {},
                "finish_reason": finish_reason,
            }],
        }) + "\n\n"

    wire = event({"content": checkpoint})
    if terminal == "stop":
        wire += event({}, "stop") + "data: [DONE]\n\n"
    elif terminal == "error":
        wire += 'data: {"error":{"message":"browser failed","type":"server_error"}}\n\n'

    transport = httpx.MockTransport(lambda _: httpx.Response(
        200, text=wire, headers={"content-type": "text/event-stream"},
    ))
    with httpx.Client(transport=transport) as http_client:
        client = OpenAI(api_key="fixture-key", base_url="http://fixture.invalid/v1",
                        http_client=http_client, max_retries=0)
        stream = client.chat.completions.create(
            model="fixture-model", messages=[{"role": "user", "content": "summarize"}],
            stream=True,
        )
        if terminal == "missing":
            with pytest.raises(ConnectionError, match="terminal finish_reason"):
                _aggregate_chat_stream(stream, model="fixture-model", task="compression")
        elif terminal == "error":
            with pytest.raises(APIError, match="browser failed"):
                _aggregate_chat_stream(stream, model="fixture-model", task="compression")
        else:
            result = _aggregate_chat_stream(stream, model="fixture-model",
                                            task="compression")
            assert result.choices[0].finish_reason == "stop"
            assert result.choices[0].message.content == checkpoint
