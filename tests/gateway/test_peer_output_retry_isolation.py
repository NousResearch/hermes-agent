"""Held Output I/O through real controls and independent manual room cycles."""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_retry import tick, pending, clock
from tests.tui_gateway.test_hosted_room_peer_backoff_progress import LocalRPC
from gateway import hosted_room_driver as tasks
from gateway.run import _profile_runtime_scope
from gateway.session_controls import AuthorityConnection
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError


async def local_control_task(c, monkeypatch, control):
    from hermes_state_runtime import claim_session_input, settle_session_input
    from gateway import run
    from hermes_constants import get_hermes_home
    original_config = run._load_gateway_config
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'},
        'terminal': {'cwd': str(c.home)}, 'platform_toolsets': {'gui': [], 'bot_room': []}}
        if get_hermes_home() == c.home else original_config())
    monkeypatch.setattr(c.authority, '_schedule', lambda ref: None)
    c.authority.runner._cached_agent_for = lambda _: None
    with _profile_runtime_scope(c.home, hydrate_secrets=False):
        await asyncio.to_thread(c.service.send, room_id='room-one', event_id='control-request',
            payload=dict(thread_id='control-thread', text='@reader control task'))
        task, = tasks.list_tasks(c.db.db_path, room_id='room-one', status='queued')
        attempt = tasks.start_task(c.db.db_path, task['identity'], c.attempt.lease,
                                  expected_cancel_generation=0, clock=time.time)
        task = tasks.get_task(c.db.db_path, task['identity'])
        rpc = c.service._resolve_member_transport(c.binding, task)
        coords = dict(profile='default', source='bot_room')
        session = await asyncio.to_thread(rpc.create, **coords, title='Group: room-one')
        await asyncio.to_thread(rpc.submit, **coords, session_id=session['session_id'],
            prompt=task['payload']['prompt'], task=task['identity'],
            execution_generation=attempt.execution_generation, on_terminal=lambda receipt: None)
        row = claim_session_input(c.db, epoch=c.authority.epoch, session_id=session['session_id'])
        assert row is not None
        if control == 'discard':
            # Explicit interrupted-store fixture, not restart or NEW-admission proof.
            c.db._execute_write(lambda conn: conn.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],)))
            c.db._execute_write(lambda conn: conn.execute("UPDATE hosted_room_driver_tasks SET status='indeterminate' WHERE task_id=?", (task['identity'].task_id,)))
        else:
            settle_session_input(c.db, epoch=c.authority.epoch, admission_id=row['admission_id'],
                generation=row['generation'], outcome='completed', result={'result': {'final_response': 'Completed before Stop.'}, 'usage': {}})
        c.service.runtime._leases['room-one'] = c.attempt.lease
        return task


@pytest.mark.asyncio
@pytest.mark.parametrize('control', ['discard', 'stop'])
async def test_control_publication_releases_outer_policy_lock(files_target, monkeypatch, control):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        original = PeerOutputCustody.acknowledge
        with monkeypatch.context() as failure:
            def unavailable(self, *args, **kwargs):
                raise PeerRunsHTTPError('before ACK', retryable=True, status_code=503)
            failure.setattr(PeerOutputCustody, 'acknowledge', unavailable)
            await tick(c)
        other = await local_control_task(c, monkeypatch, control)
        now[0] = pending(c)[0]['next_attempt_at']
        entered, release = threading.Event(), threading.Event()
        def held(self, *args, **kwargs):
            entered.set()
            assert release.wait(10), 'test must release held transport'
            return original(self, *args, **kwargs)
        monkeypatch.setattr(PeerOutputCustody, 'acknowledge', held)
        # Inert coordinator liveness only; no runtime thread/listener is started.
        c.service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
        connection = AuthorityConnection(c.authority, SimpleNamespace(), {'user_id': 'alice', 'provider': 'local',
            'capabilities': ['session:read', 'session:control', 'session:submit'],
            'profile_id': str(c.home), 'instance_id': c.authority.instance_id})
        params = dict(room_id='room-one')
        if control == 'discard':
            params.update(member_id='reader', task_id=other['identity'].task_id,
                          execution_generation=other['execution_generation'])
        else:
            params['cancel_id'] = 'root-stop'
        operation = asyncio.create_task(connection.dispatch(dict(id=1, method='groups.' + control, params=params)))
        try:
            assert await asyncio.to_thread(entered.wait, 8)
            def acquire():
                got = c.service._policy_lock.acquire(blocking=False)
                if got:
                    c.service._policy_lock.release()
                return got
            assert await asyncio.to_thread(acquire), 'control callback retained outer policy lock during ACK'
        finally:
            release.set()
            response = await asyncio.wait_for(operation, 10)
            c.service.runtime._thread = None
        assert 'result' in response, response
        assert pending(c) == []
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
async def test_held_room_output_does_not_serialize_another_room(files_target, monkeypatch):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        clock(c)
        local = LocalRPC()
        with _profile_runtime_scope(c.home, hydrate_secrets=False):
            c.service.authorize_room('alice', 'room-two', create=True)
            c.service.create_room(room_id='room-two', name='Independent', members=[
                dict(member_id='reader', profile='default', handle='reader'),
                {**c.service._room('room-one')['members'][0], 'member_id': 'observer', 'handle': 'observer'}])
            c.service.member_rpcs[('room-two', 'reader', 'default', 'alice', str(c.home))] = local
            second = next(b for b in c.service.bindings() if b.room_id == 'room-two')
        entered, release = threading.Event(), threading.Event()
        original = PeerOutputCustody.read
        def held(self, *args):
            entered.set()
            assert release.wait(10)
            return original(self, *args)
        monkeypatch.setattr(PeerOutputCustody, 'read', held)
        first = asyncio.create_task(tick(c))
        async def healthy():
            with _profile_runtime_scope(c.home, hydrate_secrets=False):
                await asyncio.to_thread(c.service.send, room_id='room-two', event_id='healthy',
                    payload=dict(thread_id='healthy-thread', text='@reader reply'))
                await asyncio.to_thread(c.service.runtime._process_room, second)
        try:
            assert await asyncio.to_thread(entered.wait, 8)
            second_work = asyncio.create_task(healthy())
            done, _ = await asyncio.wait({second_work}, timeout=3)
            assert second_work in done, 'room A transport blocked room B canonical admission/dispatch'
            await second_work
            assert len([x for x in local.calls if x[0] == 'submit']) == 1
            assert any(e['kind'] == 'message.member' for e in c.service._events('room-two'))
        finally:
            release.set()
            await asyncio.wait_for(first, 10)
            if 'second_work' in locals():
                await asyncio.wait_for(second_work, 10)
        assert len(c.executions) == len(c.launched) == 1


@pytest.mark.asyncio
async def test_same_room_overlapping_ticks_share_one_attempt(files_target, monkeypatch):
    from gateway.session_peer_output_custody import PeerOutputCustody
    async with peer_case(files_target, monkeypatch) as c:
        now = clock(c)
        entered, release = threading.Event(), threading.Event()
        original, reads = PeerOutputCustody.read, []
        def held(self, *args):
            reads.append(args)
            entered.set()
            assert release.wait(10)
            return original(self, *args)
        monkeypatch.setattr(PeerOutputCustody, 'read', held)
        first = asyncio.create_task(tick(c))
        second = None
        try:
            assert await asyncio.to_thread(entered.wait, 8)
            now[0] += 10  # pre-I/O due time alone cannot serialize an in-flight attempt
            second = asyncio.create_task(tick(c))
            done, _ = await asyncio.wait({second}, timeout=2)
            assert not done and len(reads) == 1
        finally:
            release.set()
            await asyncio.wait_for(first, 10)
            if second is not None:
                await asyncio.wait_for(second, 10)
        assert len(reads) == 1 and pending(c) == []
        assert len([e for e in c.service._events('room-one') if e['kind'] == 'message.member']) == 1
