"""Native transfer fencing, uncertain writes and unread FIFO recovery."""
import asyncio
from dataclasses import replace
from pathlib import Path
import threading

import pytest


from agent.content_steering import ContentSteering
from gateway.session_queued_controls import steer_transfers
from hermes_state_runtime import get_session_admission, settle_session_input
from tests.gateway.test_native_image_transfer import owner
from tests.waiting import async_wait_until

pytest_plugins = ["tests.gateway.test_native_image_steering"]


def settle(authority, running, outcome='completed'):
    sid, generation = running['target_session_id'], running['generation']
    with authority.sessions[sid].event_stream.lock:
        transfers = steer_transfers(authority, sid, generation)
        authority.executing.pop(sid, None)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=running['admission_id'],
                         generation=generation, outcome=outcome, _transfers=transfers)


@pytest.mark.asyncio
async def test_duplicate_and_identical_captions_settle_by_ticket_and_hold_media(setup, monkeypatch):
    from gateway.session_ingress_media import release_admission_media
    runner, agent, images, _adapter, key, event, authority, running = await owner(setup, monkeypatch)
    agent._content_steering = ContentSteering()
    agent._content_steering.open()
    runner._session_state(key).turn.event = replace(event, reply_expected=False)
    event.reply_expected = True
    first = await authority.admit_native(event)
    assert runner._session_state(key).turn.event.reply_expected is True
    again = await authority.admit_native(event)
    second = await authority.admit_native(replace(event, message_id='second-image'))
    assert first.admission_id == again.admission_id != second.admission_id
    row = get_session_admission(authority.db, admission_id=first.admission_id)
    retained = [Path(m['path']) for m in row['payload']['native_text_v1']['media']]
    for path in images:
        path.unlink()
    release_admission_media(authority.db, first.admission_id)
    assert all(path.exists() for path in retained), 'GC must not collect a terminal steering row'
    # Consume just the first ticket; the second identical caption remains unread.
    pending = agent._content_steering._pending[:1]
    messages = []
    agent._content_steering._commit(agent, messages, [], pending, {'role': 'assistant', 'content': 'Candidate'})
    agent._content_steering.seal()
    settle(authority, running)
    a = get_session_admission(authority.db, admission_id=first.admission_id)
    b = get_session_admission(authority.db, admission_id=second.admission_id)
    assert (a['status'], a['outcome']) == ('terminal', 'steered')
    assert b['status'] == 'queued' and b['seq'] > a['seq']
    assert not agent._content_steering._pending
    authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('race', ['generation', 'seal', 'cancel', 'write-error', 'commit-error', 'post-commit-generation'])
async def test_transfer_races_leave_exactly_one_owner(setup, monkeypatch, race):
    import gateway.run_image_steering as native
    _runner, agent, _images, _adapter, _key, event, authority, running = await owner(setup, monkeypatch)
    agent._content_steering = ContentSteering()
    agent._content_steering.open()
    entered, release = threading.Event(), threading.Event()
    prepare = native._prepare
    def held(*args):
        entered.set()
        assert release.wait(10)
        return prepare(*args)
    monkeypatch.setattr(native, '_prepare', held)
    if race.endswith('error') or race == 'post-commit-generation':
        write = native.steer_session_input
        def failing(*args, **kwargs):
            if race == 'post-commit-generation':
                result = write(*args, **kwargs)
                with authority.sessions[agent.session_id].event_stream.lock:
                    authority.executing[agent.session_id] = running['generation'] + 1
                return result
            if race == 'commit-error':
                write(*args, **kwargs)
            raise OSError('synthetic transfer failure')
        monkeypatch.setattr(native, 'steer_session_input', failing)
    task = asyncio.create_task(authority.admit_native(event))
    try:
        await async_wait_until(entered.is_set, timeout=10, what='native prepare')
        if race == 'generation':
            authority.executing[agent.session_id] = running['generation'] + 1
        elif race == 'seal':
            agent._content_steering.seal()
        elif race == 'cancel':
            task.cancel()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await async_wait_until(lambda: not authority._mutation_tasks, timeout=10, what='admission completion')
        from hermes_state_runtime import list_session_admissions
        rows = list_session_admissions(authority.db, session_id=agent.session_id, pending_only=False)
        assert len(rows) == 2
        if race == 'cancel':
            assert rows[1]['outcome'] == 'steering'
        authority.executing[agent.session_id] = running['generation']
        settle(authority, running)
        row = get_session_admission(authority.db, admission_id=rows[1]['admission_id'])
        assert row['status'] == 'queued'
        assert row['seq'] == rows[1]['seq']
        assert len(row['payload']['native_text_v1']['media']) == 2
        assert not agent._content_steering._pending
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        authority.db.close()


@pytest.mark.asyncio
async def test_owner_restart_does_not_replay_a_possibly_consumed_upload(setup, monkeypatch):
    from hermes_state_runtime import begin_runtime_epoch, recover_session_inputs
    _runner, agent, _images, _adapter, _key, event, authority, running = await owner(setup, monkeypatch)
    agent._content_steering = ContentSteering()
    agent._content_steering.open()
    receipt = await authority.admit_native(event)
    epoch = begin_runtime_epoch(authority.db, instance_id='replacement-owner')
    recover_session_inputs(authority.db, epoch=epoch)
    row = get_session_admission(authority.db, admission_id=receipt.admission_id)
    assert (row['status'], row['outcome']) == ('terminal', 'interrupted')
    active = get_session_admission(authority.db, admission_id=running['admission_id'])
    assert active['status'] == 'unknown'
    authority.db.close()
