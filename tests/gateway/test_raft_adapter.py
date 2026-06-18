"""Tests for the Raft channel adapter."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from plugins.platforms.raft.adapter import (
    ACTIVITY_DRAIN_SCHEMA,
    ACTIVITY_EVENT_SCHEMA,
    ActivityQueue,
    BRIDGE_TOKEN_HEADER,
    DEFAULT_PATH,
    RaftAdapter,
    _ACTIVE_ADAPTERS,
    _ACTIVE_ADAPTERS_LOCK,
    _RAFT_CONTEXT_LOCK,
    _RAFT_PROMPT_TURN_IDS,
    _RAFT_SESSION_IDS,
    _RAFT_TURN_IDS,
    _has_content_field,
    _env_enablement,
    _is_connected,
    _on_session_start,
    _on_pre_llm_call,
    _on_pre_tool_call,
    _on_post_llm_call,
    _on_post_tool_call,
    _on_session_end,
    _on_session_finalize,
    check_raft_requirements,
    register,
)
from gateway.session import build_session_key

RAFT_CHANNEL_SCHEMA = "raft-channel-wake.v1"
FUTURE_RAFT_CHANNEL_SCHEMA = "raft-channel-wake.v2"


def _make_config(**extra):
    data = {
        "bridge_token": "bridge-secret",
        "runtime_session": "default",
        "port": 0,
    }
    data.update(extra)
    return PlatformConfig(enabled=True, extra=data)


def _make_adapter(**extra):
    return RaftAdapter(_make_config(**extra))


def _create_app(adapter: RaftAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_post(adapter._path, adapter._handle_wake)
    app.router.add_post("/activity", adapter._handle_activity)
    app.router.add_get("/activity/drain", adapter._handle_activity_drain)
    return app


def _activity_event(event_id: str, **overrides):
    event = {
        "schema": ACTIVITY_EVENT_SCHEMA,
        "eventId": event_id,
        "sessionId": "session-1",
        "hookEventName": "PreToolUse",
        "status": "ok",
        "occurredAt": "2026-06-16T06:00:00Z",
        "toolName": "execute_code",
    }
    event.update(overrides)
    return event


class TestRaftWakePayload:
    def test_detects_content_fields(self):
        assert _has_content_field({"text": "hello"}) is True
        assert _has_content_field({"nested": {"messages": []}}) is True
        assert _has_content_field({"eventId": "evt-1", "messageId": "msg-1"}) is False


class TestRaftWakeHttp:
    @pytest.mark.asyncio
    async def test_send_is_noop_success(self):
        adapter = _make_adapter()

        result = await adapter.send("default", "hello")

        assert result.success is True
        assert result.message_id is None

    @pytest.mark.asyncio
    async def test_rejects_missing_bridge_token(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(DEFAULT_PATH, json={"eventId": "wake-1"})
            assert resp.status == 401
            body = await resp.json()

        assert body["ok"] is False
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_content_bearing_payload(self):
        adapter = _make_adapter()
        adapter.set_message_handler(AsyncMock())
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                DEFAULT_PATH,
                json={"eventId": "wake-1", "text": "do work"},
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert resp.status == 400
            body = await resp.json()

        assert body == {"ok": False, "error": "content_not_allowed"}
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_not_ready_without_gateway_handler(self):
        adapter = _make_adapter()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                DEFAULT_PATH,
                json={"eventId": "wake-1"},
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert resp.status == 503
            body = await resp.json()

        assert body["ok"] is False
        assert body["runtimeSession"] == "default"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("schema", [RAFT_CHANNEL_SCHEMA, FUTURE_RAFT_CHANNEL_SCHEMA])
    async def test_accepts_content_free_wake_as_internal_event(self, schema):
        adapter = _make_adapter()
        adapter.set_message_handler(AsyncMock())
        adapter.handle_message = AsyncMock()

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                DEFAULT_PATH,
                json={
                    "schema": schema,
                    "attemptId": "attempt-1",
                    "eventId": "wake-1",
                    "messageId": "msg-1",
                    "agentId": "agent-1",
                    "profile": "dev",
                    "coreSessionId": "default",
                    "adapterInstance": "hermes",
                    "occurredAt": "2026-06-11T08:00:00Z",
                },
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert resp.status == 202
            body = await resp.json()

        assert body == {"ok": True, "runtimeSession": "default"}

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.internal is True
        assert event.message_id == "wake-1"
        assert event.raw_message["schema"] == schema
        assert event.raw_message["eventId"] == "wake-1"
        assert event.raw_message["attemptId"] == "attempt-1"
        assert event.raw_message["messageId"] == "msg-1"
        assert event.source.platform == Platform("raft")
        assert event.source.chat_id == "default"
        assert "raft manual get" in event.text

    @pytest.mark.asyncio
    async def test_busy_session_queues_without_interrupt(self):
        handler = AsyncMock()
        adapter = _make_adapter()
        adapter.set_message_handler(handler)

        source = adapter.build_source(
            chat_id="default",
            chat_name="Raft channel",
            chat_type="dm",
            user_id="raft-bridge",
            user_name="Raft Bridge",
        )
        session_key = build_session_key(source)
        adapter._active_sessions[session_key] = __import__("asyncio").Event()

        accepted = await adapter._accept_wake({"eventId": "wake-busy"})

        assert accepted is True
        handler.assert_not_called()
        assert session_key in adapter._pending_messages
        pending = adapter._pending_messages[session_key]
        assert pending.message_id == "wake-busy"
        assert "raft manual get" in pending.text


class TestRaftActivityHttp:
    @pytest.mark.asyncio
    async def test_activity_endpoint_auth_validation_and_drain(self):
        adapter = _make_adapter()
        adapter._activity_queue = ActivityQueue(cap=2)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            unauthorized = await client.post("/activity", json=_activity_event("evt-1"))
            assert unauthorized.status == 401

            unknown = await client.post(
                "/activity",
                json={**_activity_event("evt-1"), "transcript_path": "/tmp/session.jsonl"},
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert unknown.status == 400

            for event_id in ["evt-1", "evt-2", "evt-3"]:
                resp = await client.post(
                    "/activity",
                    json=_activity_event(event_id),
                    headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
                )
                assert resp.status == 202

            drain = await client.get(
                "/activity/drain?max=10",
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert drain.status == 200
            body = await drain.json()

        assert body["schema"] == ACTIVITY_DRAIN_SCHEMA
        assert body["dropped"] == 1
        assert [event["eventId"] for event in body["events"]] == ["evt-2", "evt-3"]

    def test_hook_mapping_reports_only_raft_context(self):
        adapter = _make_adapter()
        with _RAFT_CONTEXT_LOCK:
            _RAFT_PROMPT_TURN_IDS.clear()
            _RAFT_SESSION_IDS.clear()
            _RAFT_TURN_IDS.clear()
        with _ACTIVE_ADAPTERS_LOCK:
            _ACTIVE_ADAPTERS.add(adapter)
        try:
            _on_pre_tool_call(
                session_id="session-1",
                turn_id="turn-1",
                tool_name="execute_code",
                args={"cmd": "echo nope"},
            )
            assert adapter._activity_queue.drain(10)["events"] == []

            _on_pre_llm_call(
                platform="raft",
                session_id="session-1",
                turn_id="turn-1",
                user_message="run a probe",
            )
            _on_pre_llm_call(
                platform="raft",
                session_id="session-1",
                turn_id="turn-1",
                user_message="run a follow-up LLM call in the same turn",
            )
            _on_pre_tool_call(
                session_id="session-1",
                turn_id="turn-1",
                tool_name="execute_code",
                args={"cmd": "echo ok"},
            )
            _on_post_tool_call(
                session_id="session-1",
                turn_id="turn-1",
                tool_name="execute_code",
                args={"cmd": "echo ok"},
                result="ok",
                status="ok",
                duration_ms=321,
            )
            _on_post_llm_call(
                platform="raft",
                session_id="session-1",
                turn_id="turn-1",
                assistant_response="done",
            )
            _on_session_end(
                platform="raft",
                session_id="session-1",
                turn_id="turn-1",
                completed=True,
                interrupted=False,
            )
            _on_session_finalize(
                platform="raft",
                session_id="session-1",
                reason="shutdown",
            )
            drain = adapter._activity_queue.drain(10)
        finally:
            with _ACTIVE_ADAPTERS_LOCK:
                _ACTIVE_ADAPTERS.discard(adapter)
            with _RAFT_CONTEXT_LOCK:
                _RAFT_PROMPT_TURN_IDS.clear()
                _RAFT_SESSION_IDS.clear()
                _RAFT_TURN_IDS.clear()

        assert [event["hookEventName"] for event in drain["events"]] == [
            "UserPromptSubmit",
            "PreToolUse",
            "PostToolUse",
            "Stop",
            "SessionEnd",
        ]
        tool_start = drain["events"][1]
        assert tool_start["toolName"] == "execute_code"
        assert '"cmd": "echo ok"' in tool_start["toolInput"]
        tool_result = drain["events"][2]
        assert tool_result["durationMs"] == 321

    def test_session_start_registers_raft_profile_env_passthrough(self):
        import tools.env_passthrough as env_passthrough_mod
        from tools.code_execution_tool import _scrub_child_env
        from tools.environments.local import _make_run_env
        from tools.env_passthrough import clear_env_passthrough, is_env_passthrough

        previous_config_passthrough = env_passthrough_mod._config_passthrough
        clear_env_passthrough()
        env_passthrough_mod._config_passthrough = frozenset()
        with _RAFT_CONTEXT_LOCK:
            _RAFT_PROMPT_TURN_IDS.clear()
            _RAFT_SESSION_IDS.clear()
            _RAFT_TURN_IDS.clear()
        try:
            assert "RAFT_PROFILE" not in _scrub_child_env(
                {"RAFT_PROFILE": "dev"},
                is_windows=False,
            )

            _on_session_start(session_id="session-1", turn_id="turn-1")
            assert not is_env_passthrough("RAFT_PROFILE")

            _on_session_start(platform="raft", session_id="session-1", turn_id="turn-1")

            assert is_env_passthrough("RAFT_PROFILE")
            assert _scrub_child_env({"RAFT_PROFILE": "dev"}, is_windows=False)["RAFT_PROFILE"] == "dev"
            with patch.dict(os.environ, {"PATH": "/usr/bin", "RAFT_PROFILE": "dev"}, clear=True):
                assert _make_run_env({})["RAFT_PROFILE"] == "dev"
        finally:
            clear_env_passthrough()
            env_passthrough_mod._config_passthrough = previous_config_passthrough
            with _RAFT_CONTEXT_LOCK:
                _RAFT_PROMPT_TURN_IDS.clear()
                _RAFT_SESSION_IDS.clear()
                _RAFT_TURN_IDS.clear()

    def test_interrupted_turn_reports_error_stop(self):
        adapter = _make_adapter()
        with _RAFT_CONTEXT_LOCK:
            _RAFT_PROMPT_TURN_IDS.clear()
            _RAFT_SESSION_IDS.clear()
            _RAFT_TURN_IDS.clear()
        with _ACTIVE_ADAPTERS_LOCK:
            _ACTIVE_ADAPTERS.add(adapter)
        try:
            _on_pre_llm_call(
                platform="raft",
                session_id="session-1",
                turn_id="turn-1",
            )
            _on_session_end(
                platform="raft",
                session_id="session-1",
                turn_id="turn-1",
                completed=False,
                interrupted=True,
            )
            drain = adapter._activity_queue.drain(10)
        finally:
            with _ACTIVE_ADAPTERS_LOCK:
                _ACTIVE_ADAPTERS.discard(adapter)
            with _RAFT_CONTEXT_LOCK:
                _RAFT_PROMPT_TURN_IDS.clear()
                _RAFT_SESSION_IDS.clear()
                _RAFT_TURN_IDS.clear()

        assert [event["hookEventName"] for event in drain["events"]] == [
            "UserPromptSubmit",
            "Stop",
        ]
        assert drain["events"][1]["status"] == "error"
        assert drain["events"][1]["errorClass"] == "interrupted"


class TestRaftConfig:
    def test_env_enablement_auto_enables_with_raft_profile(self, monkeypatch):
        monkeypatch.setenv("RAFT_PROFILE", "my-agent")

        extra = _env_enablement()

        assert extra is not None
        assert extra["enabled"] is True

    def test_env_enablement_returns_none_without_profile(self, monkeypatch):
        monkeypatch.delenv("RAFT_PROFILE", raising=False)

        assert _env_enablement() is None

    def test_is_connected_checks_bridge_token_or_enabled(self):
        assert _is_connected(PlatformConfig(enabled=True, extra={"bridge_token": "tok"})) is True
        assert _is_connected(PlatformConfig(enabled=True, extra={"enabled": True})) is True
        assert _is_connected(PlatformConfig(enabled=True, extra={})) is False

    def test_register_calls_register_platform(self):
        registered = {}
        hooks = {}

        class FakeCtx:
            def register_platform(self, **kwargs):
                registered.update(kwargs)

            def register_hook(self, name, handler):
                hooks[name] = handler

        register(FakeCtx())

        assert registered["name"] == "raft"
        assert registered["label"] == "Raft"
        assert registered["emoji"] == "🔔"
        assert "profile show" in registered["platform_hint"]
        assert "manual get" in registered["platform_hint"]
        assert "--profile" in registered["platform_hint"]
        assert hooks == {
            "on_session_start": _on_session_start,
            "pre_llm_call": _on_pre_llm_call,
            "pre_tool_call": _on_pre_tool_call,
            "post_tool_call": _on_post_tool_call,
            "post_llm_call": _on_post_llm_call,
            "on_session_end": _on_session_end,
            "on_session_finalize": _on_session_finalize,
        }
