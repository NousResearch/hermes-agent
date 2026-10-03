"""A local pre-dispatch refusal is visible without weakening unknown output custody."""
import asyncio
from copy import deepcopy
from dataclasses import replace

import pytest

from gateway import hosted_room_driver as tasks, hosted_rooms
from gateway.session_group_peers import room_link
from tests.gateway.test_session_group_peers import gateway, call, linked_room  # noqa: F401


def unjoined(gateway, monkeypatch):
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'http://127.0.0.1:9')

    async def create():
        room = await linked_room(gateway, room_link(gateway.authority)['catalog'])
        sent = await call(gateway.owner, 'groups.send', room_id='linked', event_id='early',
                         payload={'text': '@reviewer Please review the launch plan.', 'thread_id': 'launch'})
        assert sent['accepted']
        return room

    room = asyncio.run(create())
    gateway.service.runtime._run_cycle()
    task, = tasks.list_tasks(gateway.service.db_path, room_id='linked')
    assert task['status'] == 'failed'
    assert not gateway.service.peer_routes
    return room, task


def test_unjoined_peer_failure_survives_reopen_and_publishes_once_without_consent(gateway, monkeypatch):
    _, task = unjoined(gateway, monkeypatch)
    with gateway.db._read_ctx() as conn:
        assert conn.execute("SELECT count(*) FROM state_meta WHERE key LIKE 'group.peer-output.v1.%'").fetchone()[0] == 0
    # Reopen before the next policy pass publishes the durable terminal failure.
    gateway.service.runtime._release_idle_leases()
    from gateway.session_hosted_service import CanonicalHostedRoomService
    reopened = CanonicalHostedRoomService(gateway.authority, None)
    reopened.runtime._run_cycle()
    events = hosted_rooms.read_events(gateway.service.db_path, room_id='linked')['events']
    failures = [e for e in events if e['kind'] == 'turn.failed']
    assert len(failures) == 1, 'pre-dispatch refusal was hidden behind missing output consent'
    assert 'has not joined' in failures[0]['payload']['error']
    assert not reopened.status('linked')['pending_actions']
    reopened.runtime._run_cycle()
    events = hosted_rooms.read_events(gateway.service.db_path, room_id='linked')['events']
    assert [e for e in events if e['kind'] == 'turn.failed'] == failures
    assert tasks.get_task(gateway.service.db_path, task['identity']) == task


def test_missing_stale_or_peer_supplied_evidence_cannot_release_output(gateway, monkeypatch):
    from tui_gateway.hosted_room_driver import _bounded_terminal_result
    room, task = unjoined(gateway, monkeypatch)
    assert gateway.service._unreported_output(room, task) is None
    assert not tasks.is_proven_nonadmission(task), 'a failed preflight must not gain deferred Retry semantics'
    proof = task['result']['pre_dispatch_failure']
    cases = [dict(task, result={'error': task['result']['error']}), dict(task, status='indeterminate')]
    for key in ('execution_generation', 'cancel_generation', 'run_lease_generation'):
        cases.append(dict(task, **{key: task[key] + 1}))
    for key in ('run_gateway_id', 'run_process_generation'):
        cases.append(dict(task, **{key: 'another-owner'}))
    cases.append(dict(task, identity=replace(task['identity'], task_id='another-task')))
    for key, value in [('authority_epoch', room['authority_epoch'] + 1), ('cancel_generation', False)]:
        changed = deepcopy(task)
        changed['result']['pre_dispatch_failure'][key] = value
        cases.append(changed)
    forged = _bounded_terminal_result({'text': 'peer reply', 'pre_dispatch_failure': proof})
    assert 'pre_dispatch_failure' not in forged
    cases.append(dict(task, result=forged))
    for unknown in cases:
        assert gateway.service._unreported_output(room, unknown) is not None
    for changed_room in (dict(room, authority_epoch=room['authority_epoch'] + 1),
                         dict(room, authority_gateway_id='another-gateway')):
        assert gateway.service._unreported_output(changed_room, task) is not None

    # A malformed post-submit terminal receipt uses the same failure helper but
    # must not mint pre-dispatch evidence; a later cancellation also wins its fence.
    lease = gateway.service.runtime._leases['linked']
    identity = replace(task['identity'], task_id='later-task', turn_id='later-turn')
    tasks.admit_task(gateway.service.db_path, identity, payload=task['payload'], clock=gateway.service.runtime.clock)
    attempt = tasks.start_task(gateway.service.db_path, identity, lease,
                              expected_cancel_generation=0, clock=gateway.service.runtime.clock)
    binding = gateway.service.bindings()[0]
    gateway.service.runtime._on_terminal(binding, attempt, {'status': 'settled', 'settlement_id': ['invalid'],
                                                           'pre_dispatch_failure': proof})
    failed = tasks.get_task(gateway.service.db_path, identity)
    assert failed['status'] == 'failed' and 'pre_dispatch_failure' not in failed['result']
    assert gateway.service._unreported_output(room, failed) is not None
    identity = replace(identity, task_id='stopped-task', turn_id='stopped-turn')
    tasks.admit_task(gateway.service.db_path, identity, payload=task['payload'], clock=gateway.service.runtime.clock)
    attempt = tasks.start_task(gateway.service.db_path, identity, lease,
                              expected_cancel_generation=0, clock=gateway.service.runtime.clock)
    tasks.begin_task_cancel(gateway.service.db_path, identity, cancel_id='stop',
                           expected_cancel_generation=0, clock=gateway.service.runtime.clock)
    with pytest.raises(tasks.StaleTaskError):
        tasks.settle_unsubmitted_task(gateway.service.db_path, attempt, error='stale refusal', clock=gateway.service.runtime.clock)
    assert tasks.get_task(gateway.service.db_path, identity)['result'] is None
