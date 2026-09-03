"""Tests for the Raft channel adapter."""

import asyncio
import json
import os
from types import SimpleNamespace
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


class _ExitedBridgeProc:
    def __init__(self, returncode: int = 1, pid: int = 1234):
        self.pid = pid
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        return None

    def kill(self):
        return None


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


class TestRaftBridgeSupervision:
    @pytest.mark.asyncio
    async def test_bridge_exit_marks_retryable_fatal(self):
        adapter = _make_adapter()
        proc = _ExitedBridgeProc(returncode=7)
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        adapter._bridge_process = proc

        await adapter._monitor_bridge_process(proc)

        assert adapter._bridge_process is None
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "raft_bridge_exited"
        assert adapter.fatal_error_retryable is True
        assert "status 7" in (adapter.fatal_error_message or "")
        fatal_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_planned_shutdown_bridge_exit_is_not_fatal(self):
        adapter = _make_adapter()
        proc = _ExitedBridgeProc(returncode=0)
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = False
        adapter._bridge_process = proc

        await adapter._monitor_bridge_process(proc)

        assert adapter._bridge_process is None
        assert adapter.has_fatal_error is False
        fatal_handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_bridge_monitor_does_not_mark_fatal(self):
        adapter = _make_adapter()
        old_proc = _ExitedBridgeProc(returncode=1, pid=111)
        new_proc = _ExitedBridgeProc(returncode=0, pid=222)
        fatal_handler = AsyncMock()
        adapter.set_fatal_error_handler(fatal_handler)
        adapter._running = True
        adapter._bridge_process = new_proc

        await adapter._monitor_bridge_process(old_proc)

        assert adapter._bridge_process is new_proc
        assert adapter.has_fatal_error is False
        fatal_handler.assert_not_awaited()


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


# ---------------------------------------------------------------------------
# Multiplex secondary-profile scope (RAFT_PROFILE resolution)
# ---------------------------------------------------------------------------
#
# _spawn_bridge, _env_enablement, and register()'s platform_hint all
# previously read RAFT_PROFILE via raw os.environ.get unconditionally. Under
# a multiplexed secondary profile, os.environ holds the DEFAULT profile's
# YAML-to-env bridge output — a secondary profile with its own RAFT_PROFILE
# (set only in its own .env, resolved via the installed secret scope) would
# silently connect the bridge subprocess / CLI hint to the default profile's
# external Raft workspace/agent identity instead of its own. Mirrors the
# Buzz/SimpleX fix for #98738.

@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("RAFT_PROFILE", "default-profile-slug")


class _FakeCtx:
    """Minimal ``ctx`` capturing ``register_platform``'s kwargs."""

    def __init__(self):
        self.platform_kwargs = None

    def register_platform(self, **kwargs):
        self.platform_kwargs = kwargs

    def register_hook(self, *args, **kwargs):
        pass


class TestMultiplexProfileScope:

    def test_secondary_profile_uses_its_own_slug_and_never_borrows_default(
        self, multiplex_scope, default_profile_env, monkeypatch
    ):
        """Bridge spawn, env-enablement and the register() hint all resolve
        the secondary profile's own RAFT_PROFILE; with none of its own the
        profile fails closed (bridge not spawned, not auto-enabled)."""
        import plugins.platforms.raft.adapter as raft_mod

        monkeypatch.setattr(raft_mod.shutil, "which", lambda name: "/usr/bin/raft")
        spawned = []
        monkeypatch.setattr(
            raft_mod.subprocess, "Popen",
            lambda cmd, **kwargs: spawned.append(cmd) or SimpleNamespace(pid=1),
        )

        multiplex_scope({"RAFT_PROFILE": "secondary-profile-slug"})
        _make_adapter()._spawn_bridge(9999)
        assert spawned[-1][:3] == ["/usr/bin/raft", "--profile", "secondary-profile-slug"]
        assert _env_enablement() == {"enabled": True}
        ctx = _FakeCtx()
        register(ctx)
        assert "--profile secondary-profile-slug" in ctx.platform_kwargs["platform_hint"]
        assert "default-profile-slug" not in ctx.platform_kwargs["platform_hint"]

        spawned.clear()
        multiplex_scope({})
        _make_adapter()._spawn_bridge(9999)
        assert spawned == []
        assert _env_enablement() is None

    def test_default_profile_unscoped_keeps_env_precedence(
        self, monkeypatch, default_profile_env
    ):
        """Multiplex ON but no scope (the DEFAULT profile constructs
        unscoped): env is its own bridge output and still wins."""
        from agent.secret_scope import set_multiplex_active

        set_multiplex_active(True)
        try:
            assert _env_enablement() == {"enabled": True}
            ctx = _FakeCtx()
            register(ctx)
            assert "--profile default-profile-slug" in ctx.platform_kwargs["platform_hint"]
        finally:
            set_multiplex_active(False)
