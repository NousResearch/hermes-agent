"""Restart recovery uses current routing and never borrows another profile's heartbeat."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionStore, SessionSource
from hermes_cli import goals
from hermes_cli.heartbeat import HeartbeatManager
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_restore_retries_persisted_routes_in_their_own_profiles(tmp_path, monkeypatch):
    from gateway.run_heartbeat_restore import restore_heartbeat_watches

    home = tmp_path / '.hermes'
    named = home / 'profiles' / 'work'
    named.mkdir(parents=True)
    (named / 'config.yaml').write_text('{}')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    dbs = {str(p): SessionDB(db_path=p / 'state.db') for p in (home, named)}
    monkeypatch.setattr(goals, '_DB_CACHE', dbs)
    config = GatewayConfig(multiplex_profiles=True)
    store = SessionStore(home / 'sessions', config)
    entries = []
    try:
        for profile, status, topic in [(None, 'active', '11'), ('work', 'active', '22'),
                                       ('work', 'paused', '33'), (None, 'cleared', '44')]:
            source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat',
                                   thread_id=topic, profile=profile, scope_id='workspace')
            with _profile_runtime_scope(named if profile else home):
                entry = store.get_or_create_session(source)
                manager = HeartbeatManager(entry.session_id)
                manager.set('check ' + topic, 60)
                if status == 'paused':
                    manager.pause()
                elif status == 'cleared':
                    manager.clear()
            entries.append(entry)
        # Reload the real persisted routing index as a fresh process would.
        store.close_all_db_handles()
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = config
        runner.session_store = SessionStore(home / 'sessions', config)
        runner._heartbeat_watch = {}
        runner._start_heartbeat_poller = lambda: None
        runner._profile_name_for_source = lambda source: source.profile
        runner._adapter_for_source = lambda source: object()
        runner._run_in_executor_with_context = asyncio.to_thread
        original = runner.session_store.list_sessions
        with monkeypatch.context() as patch:
            patch.setattr(runner.session_store, 'list_sessions', lambda: (_ for _ in ()).throw(OSError('cold index')))
            await restore_heartbeat_watches(runner)
        assert runner._heartbeat_watch == {}
        assert original()
        # A poller inherited from work must still restore the default profile too.
        with _profile_runtime_scope(named):
            await restore_heartbeat_watches(runner)
        expected = {e.session_key: (e.origin, e.session_id) for e in entries[:2]}
        assert runner._heartbeat_watch == expected
        with monkeypatch.context() as patch:
            patch.setattr('hermes_cli.heartbeat.load_heartbeat', lambda sid: None)
            await restore_heartbeat_watches(runner)
        assert runner._heartbeat_watch == expected
        runner._heartbeat_watch.clear()
        await restore_heartbeat_watches(runner)
        assert runner._heartbeat_watch == expected
    finally:
        store.close_all_db_handles()
        if 'runner' in locals():
            runner.session_store.close_all_db_handles()
        for db in dbs.values():
            db.close()


@pytest.mark.asyncio
async def test_startup_arms_retry_poller_even_without_any_watches(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._heartbeat_watch = {}
    runner._background_tasks = set()
    runner._ensure_hosted_room_worker = AsyncMock()
    runner._hosted_room_worker_watcher = AsyncMock()
    runner._spawn_supervised = lambda *args: None
    runner._start_loop_heartbeat_task = lambda: None
    runner.hooks = SimpleNamespace(loaded_hooks=[], emit=AsyncMock())
    runner.adapters = {}
    runner._send_update_notification = AsyncMock(return_value=True)
    runner.session_store = SimpleNamespace(list_sessions=lambda: [])
    runner._run_in_executor_with_context = asyncio.to_thread
    monkeypatch.setattr('gateway.channel_directory.build_channel_directory', AsyncMock(return_value={}))
    await runner._start_post_connect_services(0)
    task = getattr(runner, '_heartbeat_poll_task', None)
    try:
        assert task is not None and not task.done()
        assert runner._heartbeat_watch == {}
    finally:
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
