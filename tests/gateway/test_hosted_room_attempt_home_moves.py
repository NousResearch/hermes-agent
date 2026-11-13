"""Stop and Retry of a turn whose profile moved: the home that took the attempt decides, and only
that home, over real private sockets."""
import asyncio
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms, session_hosted_transport as transport
from gateway.config import GatewayConfig
from gateway.runtime_ownership import ProfileOwnership
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_group_controls import _disband
from gateway.session_hosted_member_homes import _HOMES, record_member_home
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_agent.hermes_state import SessionDB
from hermes_agent.hermes_state_runtime import (
    begin_runtime_epoch, claim_session_input, list_session_admissions, settle_session_input,
)
from tests.gateway.test_session_hosted_transport import _SourceTask, _server


class _Homes:
    def __init__(self, source, loop, targets):
        self.source, self.loop, self.targets = source, loop, targets
        self.servers, self.owners, self.calls = {}, {}, []
        self.service = source.service
        self.service.runtime.prepare_room = self.service.runtime.publish_terminal = None
        self.now = time.time()
        self.service.runtime.clock = lambda: self.now
        self.service.runtime.indeterminate_defer_seconds = 5
        self.service.runtime.lease_ttl_seconds = 1000
        self.service.runtime.process_generation = 'test'
        self.binding = self.service.bindings()[0]
        self.sids = {}
        hosted_rooms.append_event(source.db.db_path, room_id='room', event_id='input', kind='message.user',
            actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'frozen'},
            authority_gateway_id=source.gateway, authority_epoch=1)
        for home, authority in [(source.authority.profile_id, source.authority), *targets.items()]:
            owner = self.owners[home] = ProfileOwnership()
            owner.reserve([home])
            server = self.servers[home] = _server(home)
            transport.install_hosted_transport(server, authority, loop,
                attest=lambda *args: self.service.attest(*args))
            original = server.private_handlers['hosted-producer']
            def observe(params, peer, home=home, original=original):
                # A source policy lock held across this reverse RPC would deadlock real gateways.
                assert self.service._policy_lock.acquire(blocking=False)
                self.service._policy_lock.release()
                self.calls.append((home, params['operation'], params['params']))
                return original(params, peer)
            server.private_handlers['hosted-producer'] = observe
            self.start(home)

    @property
    def task(self):
        return tasks.get_task(self.source.db.db_path, self.source.identity)

    def start(self, home):
        assert asyncio.run_coroutine_threadsafe(self.servers[home].start(), self.loop).result(10)

    def stop(self, home):
        asyncio.run_coroutine_threadsafe(self.servers[home].stop(), self.loop).result(10)

    def submit(self, home):
        self.source.homes['other'] = str(home)
        rpc = self.service._resolve_member_transport(self.binding, self.task)
        sid = rpc.create(profile='other', source='bot_room', title='Group: room')['session_id']
        self.sids[home] = sid
        rpc.submit(profile='other', source='bot_room', session_id=sid, prompt='frozen',
            task=self.source.identity, execution_generation=self.task['execution_generation'],
            attachments=self.source.bound, on_terminal=None)
        return rpc

    def defer(self, home):
        self.stop(home)  # The daemon still owns its lock; its control socket is unavailable.
        task = self.task
        tasks.begin_task_cancel(self.source.db.db_path, self.source.identity,
            cancel_id=f"stop:{task['execution_generation']}",
            expected_cancel_generation=task['cancel_generation'], clock=self.service.runtime.clock)
        self.now += 10
        runtime = self.service.runtime
        runtime._retry_stopping_tasks(self.binding, runtime._ensure_lease(self.binding))
        assert self.task['status'] == 'deferred'
        self.start(home)

    def disband(self):
        return _disband(self.service, self.service, self.source.db.db_path, self.source.gateway, {'room_id': 'room'})

    def pending(self):
        assert self.service.stop_room('room', cancel_id='stop-again') == 0
        with pytest.raises(hosted_rooms.RoomStopPendingError):
            self.disband()
        assert self.task['status'] == 'deferred'
        assert self.service._room('room').get('disbanded_at') is None

    def rows(self, home):
        return list_session_admissions(self.targets[home].db, session_id=self.sids[home], pending_only=False)

    def claim(self, home):
        authority = self.targets[home]
        return claim_session_input(authority.db, epoch=authority.epoch, session_id=self.sids[home])

    def finish(self, home, row):
        authority = self.targets[home]
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
            generation=row['generation'], outcome='interrupted')

    def restart(self):
        self.source.db.close()
        self.source.db = SessionDB(self.source.db.db_path)
        self.source.authority.db = self.source.db
        self.source.authority.epoch = begin_runtime_epoch(self.source.db, instance_id='restarted')
        self.service = CanonicalHostedRoomService(self.source.authority, None)
        self.service.runtime.prepare_room = self.service.runtime.publish_terminal = None
        self.now += 1001
        self.service.runtime.clock = lambda: self.now

    def close(self):
        for home in self.servers:
            self.stop(home)
            self.owners[home].close()


@pytest.fixture
def homes(hosted_owner, tmp_path, monkeypatch):
    authority, loop, _, agent = hosted_owner
    home_a, home_b = tmp_path / 'profiles' / 'other', tmp_path / 'elsewhere' / 'profiles' / 'other'
    for home in (home_a, home_b):
        home.mkdir(parents=True, mode=0o700)
    authority.profile_id = str(home_a)
    store = SessionStore(home_b / 'sessions', GatewayConfig())
    other_db = SessionDB(home_b / 'state.db')
    other_agent = SimpleNamespace(interrupted=False)
    other_agent.interrupt = lambda: setattr(other_agent, 'interrupted', True)
    runner = SimpleNamespace(session_store=store, _session_db=other_db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: other_agent)
    other = asyncio.run_coroutine_threadsafe(initialize_session_authority(
        runner, profile_id=str(home_b), instance_id='other'), loop).result(10)
    monkeypatch.setattr(other, '_schedule', lambda ref: None)
    source = _SourceTask(tmp_path, monkeypatch, b'attachment')
    fixture = _Homes(source, loop, {home_a: authority, home_b: other})
    fixture.a, fixture.b, fixture.agent = home_a, home_b, agent
    try:
        yield fixture
    finally:
        fixture.close()
        source.db.close()
        other_db.close()


def _gone(homes, home):
    homes.stop(home)
    homes.owners[home].close()  # no daemon holds its lock any more


def test_a_resubmit_after_non_admission_is_a_new_attempt_at_the_new_home(homes):
    # The real lifecycle: a proven non-admission requeues the attempt (requeue_not_admitted_task),
    # and its next start_task is a new generation, so the home that takes it gets its own record.
    homes.submit(homes.a)
    homes.finish(homes.a, homes.claim(homes.a))  # home A doesn't hold this attempt
    db, clock = homes.source.db.db_path, homes.service.runtime.clock
    first = homes.task
    lease = tasks.acquire_lease(db, room_id='room', gateway_id=homes.source.gateway, authority_epoch=1,
                                process_generation='test', ttl_seconds=1000, clock=clock)
    tasks.requeue_not_admitted_task(db, tasks.TaskAttempt(
        homes.source.identity, lease, first['execution_generation'], first['cancel_generation']), clock=clock)
    tasks.start_task(db, homes.source.identity, lease, expected_cancel_generation=first['cancel_generation'],
                     clock=clock)
    assert homes.task['execution_generation'] == first['execution_generation'] + 1
    homes.submit(homes.b)  # the profile moved while it was queued
    homes.claim(homes.b)  # home B runs it
    homes.defer(homes.b)
    homes.calls.clear()
    homes.service.stop_room('room', cancel_id='again')
    asked = {call[0] for call in homes.calls}
    assert homes.b in asked and homes.a not in asked, asked
    assert (homes.b, 'interrupt') in {call[:2] for call in homes.calls}  # the home running it stops it


def test_a_removed_mapping_settles_on_the_admitting_home_alone(homes):
    homes.submit(homes.a)
    homes.defer(homes.a)
    record_member_home(homes.source.db, 'room', 'two', homes.b)  # served there too, for other turns
    del homes.source.homes['other']
    _gone(homes, homes.a)
    assert 'tombstone' in homes.disband()  # home B, still up, never held this attempt


def test_retry_moves_on_when_the_admitting_home_is_gone(homes):
    homes.submit(homes.a)
    homes.defer(homes.a)
    homes.source.homes['other'] = str(homes.b)
    identity = homes.source.identity
    _gone(homes, homes.a)
    homes.service.retry_room_task('room', member_id='two', task_id=identity.task_id, execution_generation=1)
    assert homes.task['status'] == 'queued'  # its next start is a new generation at the current home


def test_a_turn_from_before_attempt_homes_is_stopped_at_a_live_older_home(homes):
    homes.submit(homes.a)
    homes.defer(homes.a)
    with sqlite3.connect(homes.source.db.db_path) as conn:
        for (key,) in conn.execute('SELECT key FROM state_meta WHERE substr(key, 1, ?)=?', (len(_HOMES), _HOMES)).fetchall():
            if len(json.loads(key[len(_HOMES):])) == 4:
                conn.execute('DELETE FROM state_meta WHERE key=?', (key,))
    homes.source.homes['other'] = str(homes.b)  # its "no such session" proves nothing about home A
    # Home A still holds the queued input: its Stop goes there, and once A holds nothing it settles.
    assert 'tombstone' in homes.disband()
    assert (homes.a, 'interrupt') in {call[:2] for call in homes.calls}
    row, = homes.rows(homes.a)
    assert row['status'] == 'terminal'
