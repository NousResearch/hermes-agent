"""Tests for the Raft channel adapter."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
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
    # Mirror connect(): client_max_size enforces the cap on chunked bodies.
    app = web.Application(client_max_size=adapter._max_body_bytes)
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
    async def test_busy_wakes_preserve_runner_fifo_tail_order(self):
        """A later wake must not replace the head across an occupied FIFO tail."""
        adapter = _make_adapter()
        runner = object.__new__(GatewayRunner)
        runner._queued_events = {}
        adapter.set_pending_event_queue_handler(
            lambda session_key, event: runner._queue_or_replace_pending_event(
                session_key, event, adapter
            )
        )
        adapter.set_message_handler(AsyncMock())
        source = SessionSource(
            platform=Platform("raft"),
            chat_id="default",
            chat_type="dm",
            user_id="raft-bridge",
        )
        session_key = build_session_key(source)
        events = [
            MessageEvent(
                text=f"wake-{message_id}",
                message_type=MessageType.TEXT,
                source=source,
                message_id=message_id,
                internal=True,
            )
            for message_id in ("a1", "b1", "a2")
        ]
        runner._enqueue_fifo(session_key, events[0], adapter)
        runner._enqueue_fifo(session_key, events[1], adapter)
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter.handle_message(events[2])

        turns = [adapter._pending_messages[session_key]]
        turns.extend(runner._queued_events.get(session_key, []))
        assert [turn.message_id for turn in turns] == ["a1", "b1", "a2"]


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

