"""Side runs exercise the real gateway boundary and durable child sessions, without model I/O."""
import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.mark.asyncio
async def test_plugin_dispatch_authorizes_and_consumes_errors(monkeypatch):
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="test-side", source="user"), manager)
    calls = []

    def broken(args):
        calls.append(args)
        raise RuntimeError("secret-provider-token")

    ctx.register_command("probe-side", broken)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda: manager)
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    runner._hm_quick_commands = lambda: {}
    runner._check_slash_access = Mock(return_value="Denied")
    source = SessionSource(Platform.TELEGRAM, "chat", user_id="owner")
    event = MessageEvent(text="/probe-side payload", source=source)
    result = await runner._hm_dispatch_quick_and_plugin_commands(event, source, "probe-side")
    assert result[:2] == (True, "Denied")
    assert calls == []
    runner._check_slash_access.return_value = None
    handled, message, _ = await runner._hm_dispatch_quick_and_plugin_commands(event, source, "probe-side")
    assert handled and "failed" in message.lower()
    assert "secret-provider-token" not in message
    assert calls == ["payload"]


@pytest.mark.asyncio
async def test_children_isolated_cancel_retains_capacity_and_owner(tmp_path, monkeypatch):
    from gateway.side_runs import SideRunService
    from hermes_cli.plugin_side_runs import SideRunConfig
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", "telegram", model="parent-model")
    db.append_message("parent", "user", "untouched")
    snapshot = db.get_session("parent")
    messages = db.get_messages("parent")
    started, release, interrupted = threading.Event(), threading.Event(), threading.Event()
    agents = []

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.interrupted = False
            agents.append(self)
        def run_conversation(self, user_message, conversation_history, task_id):
            assert conversation_history is None
            self.kwargs["session_db"].append_message(task_id, "user", user_message)
            started.set()
            assert release.wait(10)
            self.kwargs["session_db"].append_message(task_id, "assistant", "child answer")
            return {"final_response": "child answer"}
        def interrupt(self, *args, **kwargs):
            self.interrupted = True
            interrupted.set()
        def close(self):
            self.closed = True

    monkeypatch.setattr("run_agent.AIAgent", Agent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": kw["requested"], "api_key": "fake", "base_url": "https://invalid.example/v1",
        "api_mode": "chat_completions"})
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(_db=db)
    runner._draining = False
    runner.session_store = SimpleNamespace(_entries={"parent-key": SimpleNamespace(session_id="parent")})
    runner._session_key_for_source = lambda source: "parent-key"
    runner._reply_anchor_for_event = lambda event: "reply"
    runner._thread_metadata_for_source = lambda source, *args: {"thread_id": source.thread_id}
    adapter = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(success=True)))
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._adapter_for_source = lambda source: adapter
    source = SessionSource(Platform.TELEGRAM, "chat", user_id="owner", thread_id="topic", chat_type="group")
    event = MessageEvent(text="/probe prompt", source=source)
    service = SideRunService(runner, max_concurrent=1)
    config = SideRunConfig.from_mapping({"provider": "openai", "model": "child-model", "tools": [], "run_budget_seconds": 1})
    child_id = service.start(event, "plugin", "prompt", config)
    try:
        assert await asyncio.to_thread(started.wait, 10)
        row = db.get_session(child_id)
        assert row["parent_session_id"] == "parent"
        assert row["session_key"] == "parent-key"  # discoverable in the owner's chat listing
        assert agents[0].kwargs["gateway_session_key"] != "parent-key"
        assert json.loads(row["model_config"])["side_run"]["owner"]["user_id"] == "owner"
        assert agents[0].kwargs["fallback_model"] == []
        assert agents[0].kwargs["enabled_toolsets"] == []
        assert not service.cancel(replace(source, user_id="intruder"), child_id)
        assert await asyncio.to_thread(interrupted.wait, 5)
        assert service.cancel(source, child_id)
        assert agents[0].interrupted
        service.runs[child_id].task.cancel()
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="capacity"):
            service.start(event, "plugin", "another", config)
        assert db.get_session("parent") == snapshot
        assert db.get_messages("parent") == messages
    finally:
        release.set()
        await asyncio.wait_for(service.wait(), 10)
        runner._shutdown_executor()
    assert agents[0].closed
    assert not service.runs
    assert "cancel" in adapter.send.await_args.args[1].lower()
    assert adapter.send.await_args.kwargs["metadata"] == {"thread_id": "topic", "_interim_send": True}
    assert adapter.send.await_args.kwargs["reply_to"] == "reply"
    assert all(m["content"] != "untouched" for m in db.get_messages_as_conversation(child_id, include_ancestors=True))
    db.end_session("parent", "compression")
    assert db.resolve_resume_session_id("parent") == "parent"
    db.close()


def test_config_and_strict_resolution_fail_closed(tmp_path, monkeypatch):
    from hermes_cli.plugin_side_runs import SideRunConfig
    from hermes_cli.runtime_provider import resolve_runtime_provider
    import yaml
    for patch in ({"max_iterations": True}, {"max_tokens": False}, {"run_budget_seconds": float("nan")},
                  {"run_budget_seconds": float("inf")}, {"max_iterations": 501}, {"provider": "auto"},
                  {"tools": "all"}, {"reasoning": {"enabled": 1}}, {"fallback_model": "other"}):
        with pytest.raises(ValueError):
            SideRunConfig.from_mapping({"provider": "openai", "model": "test-model", **patch})
    # Real profile config + real resolver; transport is local/config-only, no credential lookup mocks.
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openrouter", "default": "primary-model"},
        "providers": {"test-route": {"base_url": "http://127.0.0.1:9999/v1", "api_key": "test-only"}},
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = resolve_runtime_provider(requested="custom:test-route", target_model="explicit-model", strict=True)
    assert runtime["provider"] == "custom"
    assert runtime["base_url"] == "http://127.0.0.1:9999/v1"
    with pytest.raises(ValueError):
        resolve_runtime_provider(requested="auto", target_model="test", strict=True)
    with pytest.raises(Exception):
        resolve_runtime_provider(requested="missing-route", target_model="test", strict=True)


@pytest.mark.asyncio
async def test_side_run_owner_cannot_be_bypassed_by_shared_group_or_admin(tmp_path):
    from gateway.side_runs import owner_identity
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / "state.db")
    source = SessionSource(Platform.TELEGRAM, "chat", user_id="owner", chat_type="group", scope_id="scope")
    db.create_session("child", "telegram", model_config={"side_run": {"owner": owner_identity(source)}},
                      user_id="owner", chat_id="chat", chat_type="group")
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(get_session=AsyncMock(side_effect=db.get_session))
    runner._resume_caller_is_admin = lambda source: True
    runner._gateway_session_origin_for_id = lambda sid: None
    runner._is_shared_session_source = lambda source: True
    assert await runner._resume_target_allowed(source, "child")
    assert not await runner._resume_target_allowed(replace(source, user_id="other"), "child", allow_override=True)
    assert not await runner._resume_target_allowed(replace(source, scope_id="elsewhere"), "child", allow_override=True)
    assert not await runner._resume_row_visible(replace(source, user_id="other"), db.get_session("child"), allow_all=True)
    matrix = replace(source, platform=Platform.MATRIX)
    db.create_session("matrix-child", "matrix", model_config={"side_run": {"owner": owner_identity(matrix)}})
    runner._same_matrix_room = lambda *args: True
    assert await runner._resume_access_denied_reply(replace(matrix, user_id="other"), "matrix-child", "child", True, True)
    db.close()
