"""Pane controls retain the physical owner while a route can be rebound."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway import session_ancillary as ancillary
from gateway.session_ancillary import AuthorityRefusal
from tools import session_tmux


def setup_owner(monkeypatch):
    live = SimpleNamespace(route='route', event_stream=SimpleNamespace(lock=threading.RLock()))
    store = SimpleNamespace(_lock=threading.Lock(), _entries={'route': SimpleNamespace(session_id='owner')})
    authority = SimpleNamespace(sessions={'owner': live}, runner=SimpleNamespace(session_store=store))
    connection = SimpleNamespace(authority=authority)
    ref = SimpleNamespace(session_id='owner')
    monkeypatch.setattr(ancillary, 'physical_target', lambda authority, ref: 'owner')
    monkeypatch.setattr(ancillary, '_owned_targets', lambda *a, **k: (None, [], 'owner'))
    monkeypatch.setattr(ancillary, '_interrupt_delegations', lambda *a: 0)
    return connection, ref, store


def test_stop_fences_rebind_after_pane_discovery(monkeypatch):
    connection, ref, store = setup_owner(monkeypatch)
    calls = []
    def discover(sid):
        store._entries['route'].session_id = 'replacement'
        return [{'session_id': 'tmux:abc:0', 'status': 'running'}]
    monkeypatch.setattr(session_tmux, 'list_session_panes', discover)
    monkeypatch.setattr(session_tmux, 'stop_session_pane', lambda *args: calls.append(args))
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        asyncio.run(ancillary.stop_processes(connection, ref, {'session_id': 'owner'}))
    assert calls == []


def test_kill_fences_rebind_before_action(monkeypatch):
    connection, ref, store = setup_owner(monkeypatch)
    calls = []
    def selection(*args, **kwargs):
        store._entries['route'].session_id = 'replacement'
        return None, [], 'owner'
    monkeypatch.setattr(ancillary, '_owned_targets', selection)
    monkeypatch.setattr(session_tmux, 'stop_session_pane', lambda *args: calls.append(args))
    with pytest.raises(AuthorityRefusal, match='stale_generation'):
        asyncio.run(ancillary.kill_process(connection, ref, {'session_id': 'owner', 'process_id': 'tmux:abc:0'}))
    assert calls == []


def test_stop_continues_after_bad_pane(monkeypatch):
    connection, ref, _ = setup_owner(monkeypatch)
    monkeypatch.setattr(session_tmux, 'list_session_panes', lambda sid: [
        {'session_id': 'tmux:bad:0', 'status': 'running'},
        {'session_id': 'tmux:good:1', 'status': 'running'},
    ])
    calls = []
    def stop(sid, pid):
        calls.append(pid)
        if pid == 'tmux:bad:0':
            raise OSError('pane disappeared')
        return {'status': 'killed'}
    monkeypatch.setattr(session_tmux, 'stop_session_pane', stop)
    result = asyncio.run(ancillary.stop_processes(connection, ref, {'session_id': 'owner'}))
    assert result['killed'] == 1
    assert calls == ['tmux:bad:0', 'tmux:good:1']
