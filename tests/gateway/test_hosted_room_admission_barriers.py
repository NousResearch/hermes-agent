"""Durable room barriers fence new admission/claim without destroying receipt replay."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms as rooms, hosted_room_input_preparation
from gateway.session_hosted_service import CanonicalHostedRoomService
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, cancel_session_input, get_session_admission, list_session_admissions,
)
from tests.gateway.test_session_hosted_rpc import owner  # noqa: F401


def local_case(owner, monkeypatch, *, previous_stop=False):
    authority, loop, _, _ = owner
    service = CanonicalHostedRoomService(authority, loop)
    authority.hosted_room_service = service
    monkeypatch.setattr(service, 'profile_homes', lambda: {'default': Path(authority.profile_id), 'other': Path(authority.profile_id)})
    service.authorize_room('alice', 'room', create=True)
    gateway = rooms.local_authority_gateway_id()
    rooms.create_room(authority.db.db_path, room_id='room', name='Room', authority_gateway_id=gateway,
        members=[{'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                 {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
    if previous_stop:
        assert service.stop_room('room', cancel_id='before-source') == 0
    service.send(room_id='room', event_id='input', payload={'text': '@one frozen', 'thread_id': 'thread'})
    task, = tasks.list_tasks(authority.db.db_path, room_id='room', status='queued')
    identity = task['identity']
    binding = service.bindings()[0]
    lease = service.runtime._ensure_lease(binding)
    tasks.start_task(authority.db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
    member = service._resolve_member_transport(binding, tasks.get_task(authority.db.db_path, identity))
    coords = dict(profile='default', source='bot_room')
    sid = member.create(**coords, title='Group: room')['session_id']
    args = dict(**coords, session_id=sid, prompt=task['payload']['prompt'], task=identity, execution_generation=1,
                on_terminal=lambda _: None)
    return SimpleNamespace(authority=authority, loop=loop, service=service, member=member,
                           sid=sid, identity=identity, args=args, lease=lease, gateway=gateway)


def changed_claim_selection(c, admitted, kind, monkeypatch):
    """Inject a cooperating writer after the outer check; not a scheduler reachability claim."""
    from gateway import session_authority
    original_claim = session_authority.claim_session_input
    original_check = c.service.check_admission
    checked, claims, effects, successors = [], [], [], []

    def check(ref, row, **kwargs):
        checked.append(row['admission_id'])
        return original_check(ref, row, **kwargs)

    def claim(db, **kwargs):
        claims.append(True)
        if len(claims) == 1:
            if kind == 'same_id_change':
                db._execute_write(lambda conn: conn.execute(
                    "UPDATE session_admissions SET request_id=request_id||':changed' WHERE admission_id=?",
                    (admitted['admission_id'],)))
            else:
                cancel_session_input(db, epoch=c.authority.epoch, admission_id=admitted['admission_id'])
                tasks.begin_task_cancel(c.service.db_path, c.identity, cancel_id='replace',
                    expected_cancel_generation=0, clock=time.time)
                tasks.complete_task_cancel(c.service.db_path, c.identity, cancel_id='replace',
                    expected_cancel_generation=1, clock=time.time)
                event = rooms.append_event(c.service.db_path, room_id='room', event_id='second', kind='message.user',
                    actor={'kind': 'user', 'id': 'alice'}, payload={'text': 'second', 'thread_id': 'thread'},
                    authority_gateway_id=c.gateway, authority_epoch=1)
                identity = tasks.TaskIdentity('room', 'successor', 'thread', 'next-turn')
                tasks.admit_task(c.service.db_path, identity, payload={'target_profile': 'default',
                    'target_member_id': 'one', 'source_event_seq': event['seq'], 'prompt': 'second'}, clock=time.time)
                tasks.start_task(c.service.db_path, identity, c.lease, expected_cancel_generation=0, clock=time.time)
                request_id = 'hosted:' + json.dumps([asdict(identity), 1], sort_keys=True, separators=(',', ':'))
                payload = deepcopy(admitted['payload'])
                payload['text'] = 'second'
                successors.append(admit_session_input(db, epoch=c.authority.epoch, principal_id='alice',
                    session_id=c.sid, request_id=request_id, payload=payload))
        return original_claim(db, **kwargs)

    async def execute(authority, ref, row):
        effects.append(row['admission_id'])
        return ''

    monkeypatch.setattr(c.service, 'check_admission', check)
    monkeypatch.setattr(session_authority, 'claim_session_input', claim)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    asyncio.run_coroutine_threadsafe(c.authority._drain(c.member.ref), c.loop).result(10)
    if kind == 'same_id_change':
        assert checked == [admitted['admission_id']] and len(claims) == 1 and not effects
        assert get_session_admission(c.authority.db, admission_id=admitted['admission_id'])['status'] == 'queued'
    else:
        successor, = successors
        assert checked == [admitted['admission_id'], successor['admission_id']]
        assert effects == [successor['admission_id']]
        assert get_session_admission(c.authority.db, admission_id=successor['admission_id'])['status'] == 'terminal'


@contextmanager
def barrier(c, kind, monkeypatch):
    if kind == 'retiring':
        c.service.begin_disband('room')
        assert c.service.is_retiring('room')
        yield
    elif kind == 'stop':
        committed = threading.Event()
        original = rooms.request_room_stop

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            committed.set()
            return result

        with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as workers:
            patch.setattr(rooms, 'request_room_stop', observed)
            with c.service._policy_lock:
                stopping = workers.submit(c.service.stop_room, 'room', cancel_id='during-preparation')
                assert committed.wait(10), 'native Stop did not commit its event'
                assert tasks.get_task(c.service.db_path, c.identity)['status'] == 'running'
                yield
            stopping.result(timeout=10)
    else:
        if kind == 'stopping':
            tasks.begin_task_cancel(c.service.db_path, c.identity, cancel_id='already-stopping',
                                    expected_cancel_generation=0, clock=time.time)
        yield


@pytest.mark.parametrize('kind', ['open', 'previous_stop', 'retiring', 'stop', 'stopping'])
def test_new_admission_reads_room_barrier_in_its_writer(owner, monkeypatch, kind):
    c = local_case(owner, monkeypatch, previous_stop=kind == 'previous_stop')
    preparing, release = threading.Event(), threading.Event()
    original = hosted_room_input_preparation.prepare_hosted_input

    def prepare(*args, **kwargs):
        preparing.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(hosted_room_input_preparation, 'prepare_hosted_input', prepare)
    receipt = error = None
    with ThreadPoolExecutor(max_workers=1) as workers:
        submitted = workers.submit(c.member.submit, **c.args)
        try:
            assert preparing.wait(10)
            with barrier(c, kind, monkeypatch):
                release.set()
                try:
                    receipt = submitted.result(timeout=10)
                except RuntimeStoreError as exc:
                    error = exc.reason
                rows = list_session_admissions(c.authority.db, session_id=c.sid, pending_only=False)
                if kind in {'open', 'previous_stop'}:
                    assert error is None and len(rows) == 1
                    assert c.service.check_admission(c.member.ref, rows[0])['identity'] == c.identity
                    assert c.member.submit(**c.args)['admission_id'] == receipt['admission_id']
                else:
                    assert error == 'permission_denied' and not rows, (kind, error, rows)
        finally:
            release.set()


@pytest.mark.parametrize('kind,phase', [
    (kind, phase) for kind in ['retiring', 'stop'] for phase in ['before_check', 'after_check']
] + [('replacement', 'claim_selection'), ('same_id_change', 'claim_selection')])
def test_retained_receipt_replays_but_old_work_cannot_be_claimed(owner, monkeypatch, kind, phase):
    c = local_case(owner, monkeypatch)
    receipt = c.member.submit(**c.args)
    admitted = get_session_admission(c.authority.db, admission_id=receipt['admission_id'])
    if phase == 'claim_selection':
        changed_claim_selection(c, admitted, kind, monkeypatch)
        return
    if phase == 'before_check':
        with barrier(c, kind, monkeypatch):
            assert c.member.submit(**c.args)['admission_id'] == receipt['admission_id']
            assert len(list_session_admissions(c.authority.db, session_id=c.sid, pending_only=False)) == 1
            assert c.member.info(profile='default', source='bot_room', session_id=c.sid)['status'] == 'queued'
            c.member.history(profile='default', source='bot_room', session_id=c.sid)
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                c.service.check_admission(c.member.ref, admitted)
            assert c.member.interrupt(profile='default', source='bot_room', session_id=c.sid,
                expected_task_id=c.identity.task_id)['interrupted']
        return

    checked, release = threading.Event(), threading.Event()
    real_check = c.service.check_admission
    effects = []

    def checked_before_barrier(ref, row, **kwargs):
        result = real_check(ref, row, **kwargs)
        checked.set()
        assert release.wait(10), 'test did not return the completed claim check'
        return result

    async def execution_boundary(authority, ref, row):
        effects.append(row['admission_id'])
        return ''

    monkeypatch.setattr(c.service, 'check_admission', checked_before_barrier)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execution_boundary)
    draining = asyncio.run_coroutine_threadsafe(c.authority._drain(c.member.ref), c.loop)
    try:
        assert checked.wait(10), 'drain did not reach the real same-home claim check'
        with barrier(c, kind, monkeypatch):
            release.set()
            draining.result(timeout=10)
            current = get_session_admission(c.authority.db, admission_id=receipt['admission_id'])
            assert not effects and current['status'] == 'queued', (kind, effects, current['status'])
    finally:
        release.set()
        draining.result(timeout=10)
