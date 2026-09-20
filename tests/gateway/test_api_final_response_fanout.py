import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.session import SessionSource


def _session_chat_app(adapter):
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


def _source_context(source):
    return {
        "gateway_session_key": None,
        "session_id": "session-1",
        "body": {},
        "user_message": "hello",
        "runtime_request": {"requested": {}, "route_source": "global"},
        "lock_active": False,
        "run_kwargs": {},
        "session_source": source,
    }


async def _wait_for_fanout(adapter):
    await asyncio.sleep(0)
    if adapter._fanout_tasks:
        await asyncio.gather(*adapter._fanout_tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "surface"),
    [
        ("/api/sessions/session-1/chat", "session_chat"),
        ("/api/sessions/session-1/chat/stream", "session_chat_stream"),
    ],
)
async def test_completed_source_bearing_session_chat_delivers_once_to_native_adapter(
    monkeypatch, path, surface,
):
    api = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    calls = []

    class NativeAdapter:
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            calls.append((chat_id, content, reply_to, metadata))
            return SimpleNamespace(success=True)

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", thread_id="thread-7",
    )
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: NativeAdapter()}
    runner._profile_adapters = {}
    api.set_final_response_fanout_handler(runner._deliver_api_final_response)
    monkeypatch.setattr(api, "_prepare_session_chat", AsyncMock(return_value=(_source_context(source), None)))
    monkeypatch.setattr(api, "_conversation_history_for_session", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        api, "_run_agent", AsyncMock(return_value=({"completed": True, "final_response": "  raw final  "}, {})),
    )

    async with TestClient(TestServer(_session_chat_app(api))) as client:
        response = await client.post(path, json={"message": "hello"})
        assert response.status == 200
        await response.read()

    await _wait_for_fanout(api)
    assert calls == [("42", "  raw final  ", None, {"thread_id": "thread-7"})]
    api._response_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        {"completed": False, "final_response": "unfinished"},
        {"completed": True, "failed": True, "final_response": "failed"},
        {"completed": True, "partial": True, "final_response": "partial"},
        {"completed": False, "interrupted": True, "final_response": "interrupted"},
        {"completed": False, "cancelled": True, "final_response": "cancelled"},
    ],
)
@pytest.mark.parametrize("path", ["/api/sessions/session-1/chat", "/api/sessions/session-1/chat/stream"])
async def test_unsuccessful_or_interrupted_session_chat_never_delivers(monkeypatch, result, path):
    api = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    delivered = AsyncMock()
    api.set_final_response_fanout_handler(delivered)
    monkeypatch.setattr(api, "_prepare_session_chat", AsyncMock(return_value=(_source_context(source), None)))
    monkeypatch.setattr(api, "_conversation_history_for_session", AsyncMock(return_value=[]))
    monkeypatch.setattr(api, "_run_agent", AsyncMock(return_value=(result, {})))

    async with TestClient(TestServer(_session_chat_app(api))) as client:
        response = await client.post(path, json={"message": "hello"})
        assert response.status == 200
        await response.read()

    await _wait_for_fanout(api)
    delivered.assert_not_awaited()
    api._response_store.close()


@pytest.mark.asyncio
async def test_missing_completed_defaults_to_delivery(monkeypatch):
    api = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    delivered = AsyncMock()
    api.set_final_response_fanout_handler(delivered)
    monkeypatch.setattr(api, "_prepare_session_chat", AsyncMock(return_value=(_source_context(source), None)))
    monkeypatch.setattr(api, "_conversation_history_for_session", AsyncMock(return_value=[]))
    monkeypatch.setattr(api, "_run_agent", AsyncMock(return_value=({"final_response": "legacy"}, {})))

    async with TestClient(TestServer(_session_chat_app(api))) as client:
        response = await client.post("/api/sessions/session-1/chat", json={"message": "hello"})
        assert response.status == 200
        await response.read()

    await _wait_for_fanout(api)
    delivered.assert_awaited_once_with(session_source=source, content="legacy", surface="session_chat")
    api._response_store.close()


@pytest.mark.asyncio
async def test_delivery_exception_does_not_fail_api_completion(monkeypatch):
    api = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")

    async def broken_delivery(**_kwargs):
        raise RuntimeError("native adapter unavailable")

    api.set_final_response_fanout_handler(broken_delivery)
    monkeypatch.setattr(api, "_prepare_session_chat", AsyncMock(return_value=(_source_context(source), None)))
    monkeypatch.setattr(api, "_conversation_history_for_session", AsyncMock(return_value=[]))
    monkeypatch.setattr(api, "_run_agent", AsyncMock(return_value=({"completed": True, "final_response": "done"}, {})))

    async with TestClient(TestServer(_session_chat_app(api))) as client:
        response = await client.post("/api/sessions/session-1/chat", json={"message": "hello"})
        assert response.status == 200
        assert (await response.json())["message"]["content"] == "done"

    await _wait_for_fanout(api)
    api._response_store.close()


@pytest.mark.asyncio
async def test_runner_uses_named_profile_target_without_default_fallback():
    default_calls, named_calls = [], []

    class Target:
        def __init__(self, calls):
            self.calls = calls

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.calls.append((chat_id, content, reply_to, metadata))
            return SimpleNamespace(success=True)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: Target(default_calls)}
    runner._profile_adapters = {"named": {Platform.DISCORD: Target(named_calls)}}
    source = SessionSource(platform=Platform.DISCORD, chat_id="99", chat_type="group", profile="named")
    await runner._deliver_api_final_response(session_source=source, content="done", surface="session_chat")
    assert named_calls == [("99", "done", None, None)]
    assert default_calls == []

    runner._profile_adapters = {}
    await runner._deliver_api_final_response(session_source=source, content="no fallback", surface="session_chat")
    assert named_calls == [("99", "done", None, None)]
    assert default_calls == []


def test_lifecycle_wires_final_response_fanout_handler():
    installed = []

    class Adapter:
        platform = Platform.TELEGRAM

        def set_message_handler(self, _handler): pass
        def set_fatal_error_handler(self, _handler): pass
        def set_session_store(self, _store): pass
        def set_busy_session_handler(self, _handler): pass
        def set_topic_recovery_fn(self, _handler): pass
        def set_authorization_check(self, _handler): pass
        def set_platform_event_handler(self, _handler): pass
        def set_final_response_fanout_handler(self, handler): installed.append(handler)

    runner = object.__new__(GatewayRunner)
    runner.session_store = object()
    runner._busy_text_mode = "interrupt"
    runner._primary_message_handler = lambda: object()
    runner._handle_adapter_fatal_error = object()
    runner._primary_busy_session_handler = lambda: object()
    runner._recover_telegram_topic_thread_id = object()
    runner._make_adapter_auth_check = lambda _platform: object()
    runner._primary_platform_event_handler = lambda: object()
    GatewayAdapterLifecycleMixin._wire_adapter_handlers(runner, Adapter())
    assert installed == [runner._deliver_api_final_response]
