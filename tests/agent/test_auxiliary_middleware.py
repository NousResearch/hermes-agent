"""Auxiliary request and execution policies cover every physical attempt (#134252)."""

import asyncio
from copy import deepcopy
import json
import inspect
from pathlib import Path
from types import SimpleNamespace
from threading import Event

import httpx
from openai import APIConnectionError, AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk
import pytest

from agent import auxiliary_client as aux
from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    monkeypatch.setattr(aux, "_resolve_task_provider_model", lambda *a, **k: ("custom", "fixture", "https://fixture.invalid/v1", "test", None))
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    return PluginContext(PluginManifest(name="aux-policy", source="user"), manager)


def _chunk():
    return {"id": "reply", "object": "chat.completion.chunk", "created": 0, "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "Policy title"}}]}


def _response(request):
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                             content="data: " + json.dumps(_chunk()) + "\n\ndata: [DONE]\n\n")
    return httpx.Response(200, json={"id": "reply", "object": "chat.completion", "created": 0, "model": "fixture",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Policy title"}}]})


def _bind_clients(monkeypatch, endpoint):
    sync = OpenAI(api_key="test", base_url="https://fixture.invalid/v1", max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(endpoint)))
    async_client = AsyncOpenAI(api_key="test", base_url="https://fixture.invalid/v1", max_retries=0,
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint)))
    monkeypatch.setattr(aux, "_get_cached_client", lambda *a, **k: (async_client if k.get("async_mode") else sync, "fixture"))
    monkeypatch.setattr(aux, "_to_async_client", lambda *a, **k: (async_client, "fixture"))
    return sync, async_client


def _run(mode, task, messages, async_client):
    if mode == "async":
        async def run():
            try:
                return await aux.async_call_llm(task=task, messages=messages, max_tokens=9)
            finally:
                await async_client.close()
        return asyncio.run(run()).choices[0].message.content
    if mode == "stream":
        return "".join(chunk.choices[0].delta.content or "" for chunk in aux.call_llm(
            task=task, messages=messages, max_tokens=9, stream=True) if chunk.choices)
    return aux.call_llm(task=task, messages=messages, max_tokens=9).choices[0].message.content


@pytest.mark.parametrize("mode", ["sync", "async", "stream"])
@pytest.mark.parametrize("task", ["title_generation", "compression"])
@pytest.mark.parametrize("retry", [False, True])
def test_auxiliary_policies_rewrite_each_wire_attempt_without_mutating_input(context, monkeypatch, mode, task, retry):
    seen, wire, responses, observed = [], [], [], []
    for name in ("pre_auxiliary_call", "post_auxiliary_call", "pre_api_request", "post_api_request"):
        context.register_hook(name, lambda _name=name, **kw: observed.append((_name, kw)))

    def rewrite(**kw):
        seen.append(("request", kw))
        request = {**kw["request"], "max_tokens": 17, "extra_headers": {"x-policy-task": kw["task"]}}
        request.pop("max_completion_tokens", None)
        return {"request": request, "source": "policy"}

    def execute(**kw):
        seen.append(("execution", kw))
        response = kw["next_call"]()
        responses.append(response)
        return response

    context.register_middleware("llm_request", rewrite)
    context.register_middleware("llm_execution", execute)

    def endpoint(request):
        wire.append(request)
        if retry and len(wire) == 1:
            raise httpx.ReadError("fixture transport blip", request=request)
        return _response(request)

    sync, async_client = _bind_clients(monkeypatch, endpoint)
    messages = [{"role": "user", "content": "Summarize this session."}]
    original = deepcopy(messages)
    stream_error = retry and mode == "stream"
    try:
        if stream_error:
            with pytest.raises(APIConnectionError):
                _run(mode, task, messages, async_client)
        else:
            assert _run(mode, task, messages, async_client) == "Policy title"
    finally:
        sync.close()
        if mode != "async":
            asyncio.run(async_client.close())
    assert messages == original
    assert len(responses) == (0 if stream_error else 1)
    assert all(not inspect.isawaitable(r) for r in responses)
    assert len(wire) == 1 + int(retry and not stream_error)
    assert all(json.loads(r.content).get("max_tokens", json.loads(r.content).get("max_completion_tokens")) == 17
               and r.headers.get("x-policy-task") == task for r in wire)
    assert [kind for kind, _ in seen] == ["request", "execution"] * len(wire)
    assert [name for name, _ in observed] == ["pre_auxiliary_call", "post_auxiliary_call"] * len(wire)
    assert all(kw["task"] == kw["aux_task"] == task and kw["middleware_trace"] == [{"source": "policy"}]
               for _, kw in observed)
    assert all(kw["request"]["body"]["max_tokens"] == 17 for name, kw in observed if name == "pre_auxiliary_call")
    ids = {kw["api_request_id"] for _, kw in seen}
    assert len(ids) == 1
    assert all(kw["task"] == task and kw["provider"] == "custom" for _, kw in seen)
    assert [kw["api_call_count"] for kind, kw in seen if kind == "execution"] == list(range(1, len(wire) + 1))


@pytest.mark.parametrize("mode,policy", [("sync", "cache"), ("async", "cache"), ("stream", "cache"), ("async", "cancel"), ("async", "late_cancel")])
def test_execution_policy_short_circuit_and_async_cancellation_never_duplicate_dispatch(context, monkeypatch, mode, policy):
    wire, entered = [], []
    worker_started, release_worker, worker_done = Event(), Event(), Event()
    cached = SimpleNamespace(model="fixture", choices=[SimpleNamespace(message=SimpleNamespace(content="Policy title", tool_calls=None), finish_reason="stop")], usage=None)

    def execute(**kw):
        entered.append(kw)
        if policy == "cache":
            return iter([ChatCompletionChunk(**_chunk())]) if mode == "stream" else cached
        if policy == "late_cancel":
            worker_started.set()
            release_worker.wait(timeout=3)
        try:
            return kw["next_call"]()
        finally:
            worker_done.set()

    context.register_middleware("llm_execution", execute)
    if policy == "cache":
        def endpoint(request):
            wire.append(request)
            return _response(request)
        sync, async_client = _bind_clients(monkeypatch, endpoint)
        try:
            assert _run(mode, "compression", [{"role": "user", "content": "hi"}], async_client) == "Policy title"
        finally:
            sync.close()
            if mode != "async":
                asyncio.run(async_client.close())
        assert wire == []
        assert len(entered) == 1
        return

    async def cancel_run():
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def endpoint(request):
            wire.append(request)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        sync, async_client = _bind_clients(monkeypatch, endpoint)
        try:
            task = asyncio.create_task(aux.async_call_llm(task="compression", messages=[{"role": "user", "content": "hi"}]))
            if policy == "late_cancel":
                assert await asyncio.wait_for(asyncio.to_thread(worker_started.wait, 3), 4)
            else:
                await asyncio.wait_for(started.wait(), 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release_worker.set()
            if policy == "late_cancel":
                assert await asyncio.wait_for(asyncio.to_thread(worker_done.wait, 3), 4)
            else:
                await asyncio.wait_for(cancelled.wait(), 3)
        finally:
            release_worker.set()
            sync.close()
            await async_client.close()
    asyncio.run(cancel_run())
    assert len(entered) == 1 and len(wire) == (0 if policy == "late_cancel" else 1)
