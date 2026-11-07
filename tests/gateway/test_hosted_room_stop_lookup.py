"""Public Stop and Retry preserve exact outcomes and close unused admission keys."""
from concurrent.futures import ThreadPoolExecutor
import threading
import json

import pytest

from gateway import hosted_room_driver as tasks, session_hosted_attachments
from gateway.session_hosted_member_homes import _HOMES
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from hermes_state_runtime import (
    RuntimeStoreError, begin_runtime_epoch, claim_session_input, list_session_admissions, recover_session_inputs,
)
from tests.gateway.test_hosted_room_attempt_home_moves import homes as homes_fixture
from tests.gateway.test_hosted_room_stop_owner_socket import _started, _finish
from tests.gateway.test_hosted_room_stop_terminal import _turn

homes = homes_fixture


@pytest.mark.parametrize('route,restart', [('same_process', False), ('owner_socket', False), ('owner_socket', True)])
def test_stop_before_admission_closes_the_key_before_delayed_submit(request, monkeypatch, route, restart):
    if route == 'same_process':
        turn = _turn(request.getfixturevalue('hosted_owner'), monkeypatch, admitted=False)
        service, authority, identity = turn.service, turn.authority, turn.identity
        rpc, sid, payload = turn.member, turn.sid, turn.task()['payload']
    else:
        h = request.getfixturevalue('homes')
        service, authority, identity = h.service, h.targets[h.a], h.source.identity
        h.source.homes['other'] = str(h.a)
        rpc = service._resolve_member_transport(h.binding, h.task)
        sid = rpc.create(profile='other', source='bot_room', title='Group: room')['session_id']
        payload = h.task['payload']
    preparing, release = threading.Event(), threading.Event()
    original = session_hosted_attachments.submission_payload

    def paused(*args, **kwargs):
        preparing.set()  # the destination has already accepted source attestation
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(session_hosted_attachments, 'submission_payload', paused)
    with ThreadPoolExecutor(max_workers=1) as executor:
        submitted = executor.submit(rpc.submit, profile=payload['target_profile'], source='bot_room',
            session_id=sid, prompt=payload['prompt'], attachments=payload.get('attachments'),
            task=identity, execution_generation=1, on_terminal=None)
        try:
            assert preparing.wait(10)
            service.stop_room('room', cancel_id='before-admission', require_acknowledged=True)
            assert tasks.get_task(service.db_path, identity)['status'] == 'cancelled'
            if restart:
                authority.epoch = begin_runtime_epoch(authority.db, instance_id='restarted-destination')
                recover_session_inputs(authority.db, epoch=authority.epoch)
        finally:
            release.set()
        with pytest.raises(RuntimeStoreError, match='permission_denied|stale_generation'):
            submitted.result(timeout=10)
    assert list_session_admissions(authority.db, session_id=sid, pending_only=False) == []


@pytest.mark.parametrize('outcome,expected', [('completed', 'settled'), ('interrupted', 'cancelled')])
def test_retry_after_home_restart_consumes_the_deferred_outcome(homes, outcome, expected):
    _, row = _started(homes)
    homes.defer(homes.a)
    homes.restart()
    _finish(homes, row, outcome, 'finished while home was down')
    homes.source.homes['other'] = str(homes.b)
    published = []
    homes.service.runtime.publish_terminal = lambda binding, task: published.append(task)
    result = homes.service.retry_room_task('room', member_id='two', task_id=homes.source.identity.task_id,
                                           execution_generation=1)
    assert (result['status'], result['execution_generation']) == (expected, 1)
    if expected == 'settled':
        assert [t['result']['text'] for t in published] == ['finished while home was down']
    assert tasks.list_tasks(homes.service.db_path, room_id='room', status='queued') == []
    assert len(homes.rows(homes.a)) == 1


def test_disband_keeps_a_completed_receipt_outside_the_results_budget(homes):
    _, row = _started(homes)
    homes.defer(homes.a)
    homes.restart()
    _finish(homes, row, 'completed', 'old answer')
    service, authority = homes.service, homes.targets[homes.a]
    rpc = service._resolve_member_transport(homes.binding, homes.task)
    sid = rpc.resolve_exact(profile='other', source='bot_room', title='Group: room')['session_id']
    for index in range(9):
        identity = tasks.TaskIdentity('room', f'later-{index}', 'thread', f'later-{index}')
        tasks.admit_task(service.db_path, identity, payload=homes.task['payload'], clock=service.runtime.clock)
        attempt = tasks.start_task(service.db_path, identity, service.runtime._ensure_lease(homes.binding),
                                   expected_cancel_generation=0, clock=service.runtime.clock)
        rpc.submit(profile='other', source='bot_room', session_id=sid, prompt='frozen', attachments=homes.source.bound,
            task=identity, execution_generation=1, on_terminal=None)
        newer = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        _finish(homes, newer, 'completed', 'x' * (70 * 1024))
        tasks.settle_task(service.db_path, attempt, settlement_id=f'later-{index}', status='settled',
                         result={'text': 'later'}, clock=service.runtime.clock)
    assert homes.source.identity.task_id not in {
        r['task_id'] for r in rpc.results(profile='other', source='bot_room', session_id=sid)}
    published = []
    service.runtime.publish_terminal = lambda binding, task: published.append(task)
    assert homes.disband()['tombstone']['room_id'] == 'room'
    assert homes.task['status'] == 'settled'
    assert [t['result']['text'] for t in published] == ['old answer']


@pytest.mark.parametrize('legacy', [False, True])
def test_an_older_owner_keeps_history_recovery_without_an_interrupt(homes, monkeypatch, legacy):
    _, row = _started(homes)
    homes.defer(homes.a)
    if legacy:
        key = _HOMES + json.dumps(['room', 'two', homes.source.identity.task_id, 1], separators=(',', ':'))
        homes.source.db._execute_write(lambda conn: conn.execute('DELETE FROM state_meta WHERE key=?', (key,)))
        homes.source.homes['other'] = str(homes.b)
    homes.restart()
    info = HostedRoomAuthorityRPC._info

    async def old_info(rpc, params):
        result = await info(rpc, params)
        result.pop('lookup_attempt', None)
        result.pop('interrupt_execution_generation', None)
        return result

    monkeypatch.setattr(HostedRoomAuthorityRPC, '_info', old_info)
    homes.calls.clear()
    homes.pending()
    assert not any(op == 'interrupt' for _, op, _ in homes.calls)
    assert not homes.agent.interrupted
    _finish(homes, row, 'interrupted')
    assert homes.disband()['tombstone']['room_id'] == 'room'
