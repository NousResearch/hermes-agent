"""Real adapter/runner dispatch and human approval boundaries for isolated plugin work."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tests.gateway.test_slash_access_dispatch import _make_runner


class Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.DISCORD)
        self.sent = []
    async def connect(self, *, is_reconnect=False):
        return True
    async def disconnect(self):
        self._mark_disconnected()
    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="sent")
    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


@pytest.mark.asyncio
async def test_auth_and_both_busy_guards_preserve_parent_and_legacy_handlers(monkeypatch):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="boundary", source="user"), manager)
    invocations = []
    def contextual(raw_args, *, context=None):
        invocations.append((raw_args, context.source))
        return "handled"
    ctx.register_command("side-probe", contextual, busy_policy="noninterrupting")
    ctx.register_command("legacy-probe", lambda args: "legacy:" + args)
    ctx.register_command("builtin-probe", str)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda: manager)
    runner = _make_runner(platform_extra={"group_allow_admin_from": ["admin"], "group_user_allowed_commands": ["side-probe"]})
    runner._draining = False
    runner._hm_quick_commands = lambda: {}
    runner._is_user_authorized = GatewayRunner._is_user_authorized.__get__(runner)
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "owner,denied")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")
    runner._get_unauthorized_dm_behavior = lambda *args, **kwargs: "ignore"
    runner._is_session_running = lambda key: True
    runner._hm_evict_idle_stale_agent = lambda key: None
    runner._hm_evict_reaped_agent = lambda key: None
    parent = Mock()
    source = SessionSource(Platform.DISCORD, "chat", user_id="owner", thread_id="topic", chat_type="group")
    key = runner._session_key_for_source(source)
    runner._running_agents[key] = parent
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    adapter = Adapter()
    runner.adapters = {Platform.DISCORD: adapter}
    adapter.set_message_handler(runner._handle_message)
    adapter._active_sessions[key] = asyncio.Event()
    pending = MessageEvent(text="parent queued", source=source)
    adapter._pending_messages[key] = pending
    for user in ("outsider", "denied", "owner"):
        runner.config.platforms[Platform.DISCORD].extra["group_user_allowed_commands"] = [] if user == "denied" else ["side-probe"]
        event = MessageEvent(text="/side_probe prompt", source=replace(source, user_id=user), message_id="trigger")
        await adapter._handle_message_while_active(event, key)
    assert [args for args, _ in invocations] == ["prompt"]
    assert invocations[0][1].user_id == "owner"
    parent.interrupt.assert_not_called()
    assert adapter._pending_messages[key] is pending
    assert key in adapter._active_sessions
    assert adapter.sent[-1][1] == "handled"
    runner._is_session_running = lambda key: False
    runner._check_slash_access = lambda *args: None
    handled, result, _ = await runner._hm_dispatch_quick_and_plugin_commands(
        MessageEvent(text="/legacy-probe text", source=source), source, "legacy-probe")
    assert handled and result == "legacy:text"
    handled, result, _ = await runner._hm_dispatch_quick_and_plugin_commands(
        MessageEvent(text="/builtin-probe opaque", source=source), source, "builtin-probe")
    assert handled and result == "opaque"
    manager.unload()


@pytest.mark.asyncio
async def test_approval_request_id_owner_races_and_cancellation(tmp_path, monkeypatch):
    from gateway.side_runs import SideRunService
    from hermes_cli.plugin_side_runs import SideRunConfig
    from hermes_state import SessionDB
    from tools import approval
    from tools.approval_context import get_current_session_key, manual_approval_required
    from tools.approval_gateway_wait import _await_gateway_decision

    db = SessionDB(tmp_path / "state.db")
    decisions = []
    class Agent:
        def __init__(self, **kwargs):
            self.model, self.provider = kwargs["model"], kwargs["provider"]
        def run_conversation(self, **kwargs):
            assert manual_approval_required()
            assert not approval.is_approval_bypass_active()
            key = get_current_session_key()
            result = _await_gateway_decision(key, approval._gateway_notify_cb(key), {
                "command": "rm example.txt", "description": "delete example", "pattern_key": "example"})
            decisions.append(result)
            return {"final_response": "approved" if result["choice"] == "once" else "denied"}
        def interrupt(self, *args, **kwargs):
            pass
        def close(self):
            pass
    monkeypatch.setattr("run_agent.AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {"provider": "openai"})
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(_db=db)
    runner.session_store = SimpleNamespace(_entries={})
    runner._draining = False
    runner._session_key_for_source = lambda source: "parent"
    runner._reply_anchor_for_event = lambda event: "trigger"
    runner._thread_metadata_for_source = lambda *args: {"thread_id": "topic"}
    source = SessionSource(Platform.TELEGRAM, "chat", user_id="owner", thread_id="topic", chat_type="group")
    notices = asyncio.Queue()
    async def send(*args, **kwargs):
        await notices.put(args[1])
        return SendResult(success=True)
    adapter = SimpleNamespace(send=send)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._adapter_for_source = lambda source: adapter
    runner._check_slash_access = lambda *args: None
    service = SideRunService(runner, max_concurrent=2)
    runner._plugin_side_runs = service
    config = SideRunConfig.from_mapping({"provider": "openai", "model": "model", "tools": []})
    parent_cb = Mock()
    approval.register_gateway_notify("parent", parent_cb)
    manager = PluginManager()
    registration_context = PluginContext(PluginManifest(name="test", source="user"), manager)
    ids = [service.start(MessageEvent(text="/side", source=source), "test", "prompt", config, registration_context=registration_context) for _ in range(2)]
    try:
        prompts = [await asyncio.wait_for(notices.get(), 10) for _ in ids]
        assert all("/approve side:" in text for text in prompts)
        runs = list(service.runs.values())
        requests = [approval.list_gateway_approvals(run.key)[0]["request_id"] for run in runs]
        assert len(set(requests)) == len(requests)
        intruder = MessageEvent(text=f"/approve side:{requests[0]}", source=replace(source, user_id="other"))
        assert "unavailable" in service.approval_reply(intruder)
        assert approval.list_gateway_approvals(runs[0].key)
        allowed = replace(intruder, source=source)
        assert "recorded" in service.approval_reply(allowed)
        assert "expired" in service.approval_reply(allowed)
        assert approval.list_gateway_approvals(runs[1].key)
        manager.unload()
        await asyncio.sleep(0)
        assert runs[1].cancelled.is_set()
        await asyncio.wait_for(service.wait(), 10)
        assert sorted(d["choice"] or "deny" for d in decisions) == ["deny", "once"]
        assert approval._gateway_notify_cb("parent") is parent_cb
        # After unload/restart, a stale child id must never approve the parent's FIFO head.
        runner._plugin_side_runs = None
        runner._hm_update_prompt_reply = Mock(side_effect=AssertionError("fell through to parent"))
        assert "expired" in await runner._hm_pending_reply_intercepts(allowed, source, "parent")
    finally:
        service.shutdown()
        await asyncio.wait_for(service.wait(), 10)
        runner._shutdown_executor()
        approval.unregister_gateway_notify("parent")
        db.close()


@pytest.mark.asyncio
async def test_unloaded_async_handler_cannot_launch_with_stale_context(monkeypatch):
    manager = PluginManager()
    registration = PluginContext(PluginManifest(name="delayed", source="user"), manager)
    entered, release = asyncio.Event(), asyncio.Event()
    async def delayed(args, *, context=None):
        entered.set()
        await release.wait()
        return context.start_side_run(args, {"provider": "openai", "model": "fixture"})
    registration.register_command("delayed-side", delayed, busy_policy="noninterrupting")
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda: manager)
    from gateway.plugin_commands import dispatch_plugin_command
    service = SimpleNamespace(start=Mock(return_value="should-not-launch"))
    runner = SimpleNamespace(_check_slash_access=lambda *args: None, _plugin_side_runs=service)
    source = SessionSource(Platform.TELEGRAM, "chat", user_id="owner")
    task = asyncio.create_task(dispatch_plugin_command(runner, MessageEvent(text="/delayed-side prompt", source=source), source, "delayed-side"))
    await entered.wait()
    manager.unload()
    release.set()
    handled, result = await task
    assert handled and "failed" in result.lower()
    service.start.assert_not_called()
