"""Tests for the Raft channel adapter."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock, patch

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
    RAFT_ADAPTER_INSTANCE,
    RAFT_ADAPTER_VERSION,
    RUNTIME_INTEGRATION_SCHEMA,
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
    interactive_setup,
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
    return adapter._create_http_app()


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
    async def test_runtime_manifest_is_authenticated_and_adapter_owned(self):
        adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            unauthorized = await client.get(DEFAULT_PATH)
            assert unauthorized.status == 401

            response = await client.get(
                DEFAULT_PATH,
                headers={BRIDGE_TOKEN_HEADER: "bridge-secret"},
            )
            assert response.status == 200
            body = await response.json()

        assert body["schema"] == RUNTIME_INTEGRATION_SCHEMA
        assert body["runtimeId"] == "hermes"
        assert body["adapterVersion"] == RAFT_ADAPTER_VERSION
        assert body["wakeAdapter"]["kind"] == "raft-channel"
        assert body["bridgeLifecycle"]["requiresBridgeFailureCodes"] == [
            "BRIDGE_NOT_RUNNING",
            "NO_CORE_SESSION",
        ]

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


class TestBodySize:
    """The wake/activity endpoints enforced max_body_bytes only via the
    Content-Length header; a Transfer-Encoding: chunked request
    (content_length=None) bypassed the cap entirely and read the full body,
    bounded only by aiohttp's implicit 1 MiB default. Mirrors
    gateway/platforms/webhook.py's TestBodySize."""

    @pytest.mark.asyncio
    async def test_wake_chunked_oversized_payload_rejected(self):
        adapter = _make_adapter(max_body_bytes=100)
        adapter.set_message_handler(AsyncMock())
        adapter.handle_message = AsyncMock()

        async def _chunked_body():
            payload = json.dumps({"eventId": "x" * 500}).encode("utf-8")
            for i in range(0, len(payload), 64):
                yield payload[i : i + 64]
                await asyncio.sleep(0)

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                DEFAULT_PATH,
                data=_chunked_body(),
                headers={
                    BRIDGE_TOKEN_HEADER: "bridge-secret",
                    "Content-Type": "application/json",
                },
            )
            assert resp.status == 413
            body = await resp.json()

        assert body == {"ok": False, "error": "payload_too_large"}
        adapter.handle_message.assert_not_awaited()


class TestRaftConfig:
    def test_spawned_bridge_binds_stable_hermes_adapter_instance(self, monkeypatch):
        adapter = _make_adapter()
        process = Mock(pid=123)
        monkeypatch.setenv("RAFT_PROFILE", "hermes-profile")
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/raft")

        with patch("plugins.platforms.raft.adapter.subprocess.Popen", return_value=process) as popen:
            adapter._spawn_bridge(49152)

        command = popen.call_args.args[0]
        assert command[-2:] == ["--adapter-instance", RAFT_ADAPTER_INSTANCE]
        assert command[command.index("--wake-channel-endpoint") + 1] == (
            "http://127.0.0.1:49152/wake"
        )

    def test_env_enablement_auto_enables_with_raft_profile(self, monkeypatch):
        monkeypatch.setenv("RAFT_PROFILE", "my-agent")

        extra = _env_enablement()

        assert extra is not None
        assert extra["enabled"] is True


    def test_interactive_setup_keeps_existing_profile_when_not_reconfigured(
        self, monkeypatch, tmp_path, capsys
    ):
        env_path = tmp_path / ".env"
        env_path.write_text("RAFT_PROFILE=existing\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("RAFT_PROFILE", "existing")
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        interactive_setup()

        assert env_path.read_text(encoding="utf-8") == "RAFT_PROFILE=existing\n"
        assert os.environ["RAFT_PROFILE"] == "existing"
        assert "Keeping RAFT_PROFILE=existing" in capsys.readouterr().out

