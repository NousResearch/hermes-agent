"""Heartbeat watches follow their route and profile, not the poller's creator."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat, migrate_heartbeat_to_session
from hermes_constants import get_hermes_home


@pytest.mark.asyncio
async def test_watches_read_and_admit_in_each_owner_profile(tmp_path, monkeypatch):
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._run_in_executor_with_context = asyncio.to_thread
    runner._is_session_running = lambda key: False
    runner._queue_depth = lambda key, adapter: 0
    rows = []

    class Adapter:
        _message_handler = True
        _active_sessions = {}

        async def handle_message(self, event):
            rows.append((event.source.profile, str(get_hermes_home()), event.text))
            self._active_sessions[event.metadata['gateway_session_key']] = True

    runner._adapter_for_source = lambda source: Adapter()
    watch = {}
    for name in ('alpha', 'beta'):
        home = tmp_path / '.hermes' / 'profiles' / name
        home.mkdir(parents=True, exist_ok=True)
        (home / 'config.yaml').write_text('{}\n')
        with _profile_runtime_scope(home):
            await runner._warm_goals_session_db('test')
            save_heartbeat(name, HeartbeatState(prompt=name, interval_seconds=60, created_at=1))
        watch[name] = (SessionSource(platform=Platform.TELEGRAM, chat_id=name, profile=name), name)
    with _profile_runtime_scope(tmp_path / '.hermes' / 'profiles' / 'alpha'):
        await runner._heartbeat_poll_once(watch)
    assert {row[0] for row in rows} == {'alpha', 'beta'}
    assert all(row[1] == str(tmp_path / '.hermes' / 'profiles' / row[0]) for row in rows)
    assert len(watch) == 2


@pytest.mark.asyncio
async def test_watch_tracks_rotated_route_owner_without_reviving_parent():
    runner = object.__new__(GatewayRunner)
    runner._run_in_executor_with_context = asyncio.to_thread
    runner._is_session_running = lambda key: False
    runner._queue_depth = lambda key, adapter: 0
    runner.session_store = SimpleNamespace(peek_session_id=lambda key: 'child')
    events = []

    class Adapter:
        _message_handler = True
        _active_sessions = {}

        async def handle_message(self, event):
            events.append(event)
            self._active_sessions['route'] = True

    runner._adapter_for_source = lambda source: Adapter()
    await runner._warm_goals_session_db('test')
    save_heartbeat('parent', HeartbeatState(prompt='follow rotation', interval_seconds=60, created_at=1))
    assert migrate_heartbeat_to_session('parent', 'child')
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='chat')
    watch = {'route': (source, 'parent')}
    await runner._heartbeat_poll_once(watch)
    assert len(events) == 1
    assert watch['route'][1] == 'child'
    assert HeartbeatManager('parent').state is None
    runner.session_store.peek_session_id = lambda key: None
    await runner._heartbeat_poll_once(watch)
    assert not watch  # a departed route never falls back to its stale heartbeat
