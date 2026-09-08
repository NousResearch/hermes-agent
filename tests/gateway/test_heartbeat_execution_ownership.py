"""Admitted heartbeat work belongs to a conversation, not a reusable route."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evals.heartbeat_idle_wire import WireAdapter
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_cli.heartbeat import HeartbeatState, migrate_heartbeat_to_session, save_heartbeat


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["reset", "suspend", "compression", "prepare-reset", "prepare-suspend", "unchanged"])
async def test_admitted_heartbeat_executes_only_in_own_conversation(tmp_path, boundary):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner._running_agents = {}
    runner._run_in_executor_with_context = asyncio.to_thread
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    adapter = WireAdapter(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
    adapter.wire = []
    runner._adapter_for_source = lambda source: adapter
    runner._is_telegram_topic_lane = lambda source: False
    runner._cache_session_source = lambda *args: None
    runner._clear_session_env = lambda tokens: None
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="boundary", user_id="owner")
    entry = runner.session_store.get_or_create_session(source)
    key, old = entry.session_key, entry.session_id
    await runner._warm_goals_session_db("test")
    save_heartbeat(old, HeartbeatState(prompt="old work", interval_seconds=60, created_at=1))
    executions = []

    def change():
        if "reset" in boundary:
            runner.session_store.reset_session(key)
        elif "suspend" in boundary:
            runner.session_store.suspend_session(key)
        elif boundary == "compression":
            child = old + "-child"
            runner.session_store._db_for_key(key).publish_compression_child(
                parent_session_id=old, child_session_id=child, source="telegram",
                require_compression_lease=False, model="offline", model_config={},
                system_prompt="offline", messages=[{"role": "user", "content": "retained"}],
            )
            assert migrate_heartbeat_to_session(old, child)
            assert runner.session_store.advance_compression_session(key, old, child)

    async def prepare(*args):
        if boundary.startswith("prepare-"):
            change()
        return runner._PreparedTurn([], "", "old work", "old work", None, None), {}

    async def model(**kwargs):
        executions.append(kwargs["session_id"])
        raise asyncio.CancelledError  # stop before unrelated post-turn delivery

    runner._hmwa_prepare_turn = prepare
    runner._run_agent = model
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    adapter.set_message_handler(lambda event: runner._handle_message_with_agent(event, source, key, 1))
    try:
        await runner._heartbeat_poll_once({key: (source, old)})
        assert key in adapter._session_tasks  # prove admission before changing ownership
        if not boundary.startswith("prepare-"):
            change()
        await asyncio.gather(*adapter._background_tasks, return_exceptions=True)
        await asyncio.sleep(0)
        expected = [old + "-child"] if boundary == "compression" else [old] if boundary == "unchanged" else []
        assert executions == expected
        if boundary == "suspend":
            assert runner.session_store.peek_session_id(key) != old  # normal reset policy still runs
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_restart_does_not_restore_suspended_heartbeat(tmp_path):
    from gateway.run_heartbeat_restore import restore_heartbeat_watches

    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="stopped", user_id="owner")
    entry = store.get_or_create_session(source)
    save_heartbeat(entry.session_id, HeartbeatState(prompt="stale work", interval_seconds=60, created_at=1))
    store.suspend_session(entry.session_key)
    store.close_all_db_handles()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = SessionStore(tmp_path / "sessions", config)
    runner._heartbeat_watch = {}
    runner._start_heartbeat_poller = lambda: None
    runner._run_in_executor_with_context = asyncio.to_thread
    try:
        await restore_heartbeat_watches(runner)
        assert runner._heartbeat_watch == {}
        restored = runner.session_store.lookup_by_session_key(entry.session_key)
        assert restored.suspended and restored.session_id == entry.session_id
    finally:
        runner.session_store.close_all_db_handles()
