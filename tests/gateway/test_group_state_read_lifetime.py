"""Real canonical state reads never create, reopen, or borrow a foreign DB.

External execution/coordinator boundaries use the existing inert Retry fixture.
"""
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from gateway.session_hosted_service import CanonicalHostedRoomService
from tui_gateway.hosted_room_driver import HostedRoomRuntime
from tests.gateway.test_canonical_peer_backoff_retry import rpc


@pytest.fixture
def case(tmp_path, monkeypatch):
    # Install the actual original owner before constructing the canonical service.
    # No transport/planner work is needed to exercise the status read boundary.
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError('coordinator/listener/network startup forbidden')
    monkeypatch.setattr(HostedRoomRuntime, 'start', forbidden)
    monkeypatch.setattr('tui_gateway.hosted_room_peer_http._open_roomlink_url', forbidden)
    with SessionDB(tmp_path / 'state.db') as db:
        runner = SimpleNamespace(_draining=False)
        authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='owner',
            db=db, epoch=begin_runtime_epoch(db, instance_id='owner'))
        runner.session_authority = authority
        service = CanonicalHostedRoomService(authority, None)
        authority.hosted_room_service = service
        service.local_profiles = lambda: ('default', 'reviewer')
        client = AuthorityConnection(authority, SimpleNamespace(), dict(user_id='alice'))
        service.authorize_room(client.actor.subject, 'room', create=True)
        service.create_room(room_id='room', name='Read lifetime', members=[
            dict(member_id='writer', profile='default', handle='writer'),
            dict(member_id='reviewer', profile='reviewer', handle='reviewer')])
        # The inert running-coordinator observation presupposes driver schema.
        # Bootstrap via its real accessor in SETUP, never during the guarded read.
        from gateway.hosted_room_driver import list_tasks
        assert list_tasks(db.db_path, room_id='room') == []
        service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        try:
            yield SimpleNamespace(authority=authority, service=service, connection=client)
        finally:
            service.runtime._thread = None



@pytest.mark.parametrize('gate', ['healthy', 'drain', 'same_store_service', 'uninstalled_service',
    'foreign_service', 'closed', 'replaced_db', 'replaced_file', 'quarantine', 'wrong_actor',
    'wrong_room', 'wrong_profile', 'no_read', 'epoch', 'instance'])
def test_state_reads_only_pinned_lifetime(case, tmp_path, monkeypatch, gate):
    c = case
    authority, service, db = c.authority, c.service, c.authority.db
    foreign_home = tmp_path / 'foreign'
    foreign_home.mkdir()
    with SessionDB(foreign_home / 'state.db') as foreign_db:
        foreign = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(foreign_home),
            instance_id='foreign', db=foreign_db,
            epoch=begin_runtime_epoch(foreign_db, instance_id='foreign'))
        same_store = SessionAuthority(authority.runner, profile_id=authority.profile_id,
            instance_id='replacement', db=db, epoch=authority.epoch)
        if gate in {'wrong_actor', 'wrong_profile', 'no_read'}:
            c.connection = AuthorityConnection(authority, SimpleNamespace(), dict(
                user_id='bob' if gate == 'wrong_actor' else 'alice',
                profile_id=foreign.profile_id if gate == 'wrong_profile' else authority.profile_id,
                capabilities=[] if gate == 'no_read' else ['session:read', 'session:control']))
        gates = {
            'drain': lambda: monkeypatch.setattr(authority.runner, '_draining', True),
            'same_store_service': lambda: monkeypatch.setattr(service, 'authority', same_store),
            'uninstalled_service': lambda: monkeypatch.setattr(authority, 'hosted_room_service', None),
            'foreign_service': lambda: monkeypatch.setattr(service, 'authority', foreign),
            'closed': db.close,
            'replaced_db': lambda: monkeypatch.setattr(authority, 'db', foreign_db),
            'replaced_file': lambda: db.db_path.rename(tmp_path / 'detached-state.db'),
            'quarantine': lambda: monkeypatch.setattr(db, '_db_corrupt', True),
            'epoch': lambda: monkeypatch.setattr(authority, 'epoch', authority.epoch + 1),
            'instance': lambda: monkeypatch.setattr(authority, 'instance_id', 'changed'),
        }
        if gate in gates:
            gates[gate]()
        trace, foreign_trace, forbidden_calls = [], [], []
        raw = db._conn
        changes = raw.total_changes if raw is not None else None
        if raw is not None:
            raw.set_trace_callback(trace.append)
        foreign_db._conn.set_trace_callback(foreign_trace.append)
        def forbidden(*args, **kwargs):
            forbidden_calls.append(True)
            raise AssertionError('status must not open, write, unlink, or use network')
        service.runtime._wake.clear()
        with monkeypatch.context() as guarded:
            guarded.setattr(sqlite3, 'connect', forbidden)
            guarded.setattr(db, '_open_writer_conn', forbidden)
            guarded.setattr(db, '_execute_write', forbidden)
            guarded.setattr('pathlib.Path.unlink', forbidden)
            guarded.setattr('tui_gateway.hosted_room_peer_http._open_roomlink_url', forbidden)
            reply = rpc(c, 'groups.state', {'room_id': 'not-owned' if gate == 'wrong_room' else 'room'})
        if raw is not None:
            raw.set_trace_callback(None)
            assert raw.total_changes == changes
        foreign_db._conn.set_trace_callback(None)
        assert forbidden_calls == [] and foreign_trace == []
        assert not service.runtime._wake.is_set()
        assert all(s.lstrip().split()[0].upper() in {'SELECT', 'BEGIN', 'ROLLBACK'} for s in trace), trace
        if gate in {'healthy', 'drain', 'same_store_service', 'uninstalled_service'}:
            assert reply['result']['room']['room_id'] == 'room', reply
            if gate != 'healthy':
                assert not any(a['kind'] in {'retry', 'discard', 'approval'}
                    for a in reply['result']['driver_status']['pending_actions'])
        else:
            assert reply.get('error', {}).get('data', {}).get('reason') in {
                'group_state_unavailable', 'storage_unavailable', 'permission_denied', 'profile_mismatch'}, reply
        # Restore object identity before fixture teardown; never reopen the closed DB.
        authority.db = db
        service.authority = authority


@pytest.mark.parametrize('drift', ['epoch', 'room_owner', 'close'])
def test_state_rechecks_after_projection_with_close_excluded(case, monkeypatch, drift):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    c = case
    db = c.authority.db
    real_status = c.service.status
    raw = db._conn
    # Independent real writer for the post-snapshot epoch/ACL counterexamples.
    entered, release, close_marked = Event(), Event(), Event()
    evict = db._evict_one_idle_read_conn
    def observe_close():
        assert db._read_conns_closed
        close_marked.set()
        return evict()
    if drift == 'close':
        monkeypatch.setattr(db, '_evict_one_idle_read_conn', observe_close)
    def status(*args, **kwargs):
        result = real_status(*args, **kwargs)
        if drift == 'close':
            entered.set()
            assert release.wait(5)
        else:
            sql = ('UPDATE runtime_epoch SET epoch=epoch+1' if drift == 'epoch' else
                   "UPDATE state_meta SET value='bob' WHERE key='gateway.hosted.owner.v1:room'")
            other_thread = sqlite3.connect(db.db_path)
            try:
                other_thread.execute(sql)
                other_thread.commit()
            finally:
                other_thread.close()
        return result
    monkeypatch.setattr(c.service, 'status', status)
    before = raw.total_changes
    try:
        if drift == 'close':
            with ThreadPoolExecutor(max_workers=2) as pool:
                future = pool.submit(rpc, c, 'groups.state')
                assert entered.wait(5)
                # Actual close uses this lock: it cannot retire the borrowed handle.
                acquired = db._lock.acquire(blocking=False)
                if acquired:
                    db._lock.release()
                assert not acquired
                closing = pool.submit(db.close)
                assert close_marked.wait(5)
                assert db._conn is raw  # close published withdrawal but is lock-blocked
                release.set()
                reply = future.result(timeout=5)
                closing.result(timeout=5)
            assert reply['error']['data']['reason'] == 'group_state_unavailable', reply
            assert db._conn is None
            assert rpc(c, 'groups.state')['error']['data']['reason'] == 'group_state_unavailable'
        else:
            reply = rpc(c, 'groups.state')
            assert reply['error']['data']['reason'] in {'stale_epoch', 'permission_denied'}, reply
            assert raw.total_changes == before
    finally:
        release.set()
