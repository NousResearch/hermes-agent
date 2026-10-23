"""GW-10 (#139532): an ``unknown`` admission no person is waiting on settles as interrupted.

A turn lost across an owner restart is fenced ``unknown`` and pauses the session's FIFO until a
person resolves it, because only they know whether to resend it. An automation delivery (a cron
run, a ``local_automation_v1`` completion, a native automation envelope) has nobody to ask: it
settles ``interrupted`` (never replayed), the transcript gets a notice, and the FIFO moves on.
Real ``SessionStore`` SQLite, two real owner epochs; no store monkeypatching.
"""
import queue
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from gateway.session_authority import LiveSession, initialize_session_authority
from gateway.session_contract import SessionRef, Submission
from gateway.session_controls import AuthorityConnection
from hermes_state_keys import CRON_PRINCIPAL_ID
from hermes_state_runtime import admit_session_input, claim_session_input, list_session_admissions
from tests.waiting import async_wait_until


class Peer:
    def __init__(self):
        self.frames = queue.Queue()

    def write(self, frame):
        self.frames.put(frame)
        return True


def _runner(tmp_path, monkeypatch, executed):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    return store, runner


async def _lost_head_then_restart(tmp_path, monkeypatch, executed, admit_head):
    """Owner 1 admits a head (via ``admit_head``) and a person's follower, claims the head, dies.
    Owner 2 starts on the same store. Returns (store, owner2, head_id, follower_id)."""
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='gw10')
    db.create_session('s', source='telegram')
    first = await initialize_session_authority(runner, profile_id='p', instance_id='owner-1')
    first.sessions['s'] = LiveSession(source, 's')
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(first, Peer(), {'user_id': 'human'}).actor
    first._schedule = lambda ref: None
    head_id = await admit_head(first, actor, ref)
    follower = await first.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))
    claimed = claim_session_input(db, epoch=first.epoch, session_id='s')
    assert claimed['admission_id'] == head_id
    db.append_message('s', 'user', 'HEAD')  # the lost turn got as far as its user row
    second = await initialize_session_authority(runner, profile_id='p', instance_id='owner-2')
    second.sessions['s'] = LiveSession(source, 's')
    return store, second, head_id, follower.admission_id


async def _automation_head(authority, _actor, _ref):
    row = admit_session_input(authority.db, epoch=authority.epoch, principal_id='automation:agent:main:telegram:dm:gw10',
                              session_id='s', request_id='completion-1',
                              payload={'text': 'HEAD', 'local_automation_v1': {'identity': 'proc-1', 'owner': 's'}})
    return row['admission_id']


async def _cron_head(authority, _actor, _ref):
    row = admit_session_input(authority.db, epoch=authority.epoch, principal_id=CRON_PRINCIPAL_ID,
                              session_id='s', request_id='cron:job:exec-1', payload={'text': 'HEAD'})
    return row['admission_id']


async def _webhook_head(authority, _actor, _ref):
    # A webhook delivery: a messaging principal and a native envelope carrying its delivery
    # destination, no automation block. The provider already got its 202; nobody can resend it.
    row = admit_session_input(authority.db, epoch=authority.epoch, principal_id='messaging:[null,"webhook","r",null,null]',
                              session_id='s', request_id='delivery-1', payload={'text': 'HEAD', 'native_text_v1': {
                                  'event': {'message_id': 'delivery-1'}, 'webhook_delivery': {'deliver': 'log'},
                                  'webhook_route': 'digest'}})
    return row['admission_id']


async def _person_head(authority, actor, ref):
    receipt = await authority.submit(actor, Submission('head', ref, {'text': 'HEAD'}, 'queue'))
    return receipt.admission_id


def _statuses(db):
    return {r['admission_id']: (r['status'], r['outcome'])
            for r in list_session_admissions(db, session_id='s', pending_only=False)}


@pytest.mark.asyncio
@pytest.mark.parametrize('admit_head', [_automation_head, _cron_head, _webhook_head],
                         ids=['local_automation_v1', 'cron', 'webhook_delivery'])
async def test_unknown_automation_settles_interrupted_and_fifo_moves_on(tmp_path, monkeypatch, admit_head):
    executed = []
    store, authority, head_id, follower_id = await _lost_head_then_restart(tmp_path, monkeypatch, executed, admit_head)
    try:
        # The restart settled the lost automation turn: terminal/interrupted, never `unknown`.
        assert _statuses(authority.db)[head_id] == ('terminal', 'interrupted')
        authority._schedule(SessionRef('p', 's'))
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')
        # The follower ran exactly once; the automation input was never replayed.
        assert executed == ['FOLLOWER']
        assert _statuses(authority.db)[follower_id] == ('terminal', 'completed')
        # A Hermes notice in the transcript says the automated delivery was interrupted.
        messages = authority.db.get_messages('s')
        notices = [m for m in messages if m.get('display_kind') == 'failed_turn']
        assert notices and 'automat' in notices[0]['content'].lower(), messages
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_persons_unknown_turn_still_pauses_the_fifo(tmp_path, monkeypatch):
    executed = []
    store, authority, head_id, follower_id = await _lost_head_then_restart(tmp_path, monkeypatch, executed, _person_head)
    try:
        assert _statuses(authority.db)[head_id] == ('unknown', None)
        authority._schedule(SessionRef('p', 's'))
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain paused')
        assert executed == []
        assert _statuses(authority.db)[follower_id] == ('queued', None)
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_automation_fenced_unknown_in_this_owner_settles_on_the_drain(tmp_path, monkeypatch):
    """The drain path: an automation head this owner fences ``unknown`` (a lost managed worker
    commits that verdict without a restart) settles on the next claim instead of pausing."""
    executed = []
    store, runner = _runner(tmp_path, monkeypatch, executed)
    db = store._db
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='gw10')
    db.create_session('s', source='telegram')
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner-1')
    authority.sessions['s'] = LiveSession(source, 's')
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(authority, Peer(), {'user_id': 'human'}).actor
    authority._schedule = lambda ref: None
    head_id = await _automation_head(authority, actor, ref)
    follower = await authority.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))
    claimed = claim_session_input(db, epoch=authority.epoch, session_id='s')
    assert claimed['admission_id'] == head_id
    # The managed-worker verdict: the claimed row is committed ``unknown`` under this same epoch.
    db._execute_write(lambda c: c.execute("UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (head_id,)))
    try:
        del authority._schedule
        authority._schedule(ref)
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')
        assert executed == ['FOLLOWER']
        statuses = _statuses(db)
        assert statuses[head_id] == ('terminal', 'interrupted')
        assert statuses[follower.admission_id] == ('terminal', 'completed')
        # Nothing restarted on this path: the notice must not claim a restart (review #141357).
        notices = [m['content'] for m in db.get_messages('s') if m.get('display_kind') == 'failed_turn']
        assert notices and all('restart' not in n.lower() for n in notices), notices
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_messaging_pause_notice_names_discard_and_discard_resumes_the_fifo(tmp_path, monkeypatch):
    """Messaging: the paused notice says why and offers `/discard` in the chat; `/discard` there
    resolves the person's lost turn through the same ``resolve_unknown`` and the follower runs."""
    from gateway.session_ingress import pause_notice
    from gateway.slash_commands_session import discard_lost_turns
    executed = []
    store, authority, head_id, _follower_id = await _lost_head_then_restart(tmp_path, monkeypatch, executed, _person_head)
    try:
        notice = pause_notice(authority, SessionRef('p', 's'), 'unknown_execution')
        assert '/discard' in notice and 'restarted' in notice, notice
        reply = await discard_lost_turns(authority, 's')
        assert '1' in reply and 'discard' in reply.lower(), reply
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')
        assert executed == ['FOLLOWER']
        assert _statuses(authority.db)[head_id] == ('terminal', 'interrupted')
        assert 'nothing' in (await discard_lost_turns(authority, 's')).lower()
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_captured_completed_automation_result_gets_no_restart_notice(tmp_path, monkeypatch):
    """Review #141357 F6: an automation turn fenced ``unknown`` only because its settlement write
    failed, while this owner holds its completed result, commits that result as completed. Its
    transcript must not also claim the turn was interrupted by a restart."""
    from gateway.session_results import admission_result
    from gateway.session_settlement_recovery import recover_failed_settlement
    from gateway.session_unattended_unknown import settle_unattended_unknown
    store, runner = _runner(tmp_path, monkeypatch, [])
    db = store._db
    db.create_session('s', source='telegram')
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner-1')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='gw10'), 's')
    authority._schedule = lambda ref: None
    try:
        admit_session_input(db, epoch=authority.epoch, principal_id='automation:probe', session_id='s',
                            request_id='done', payload={'text': 'done'})
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        db.append_message('s', 'user', 'done')
        db.append_message('s', 'assistant', 'REAL_FINISHED_ANSWER')
        authority.pending_results[row['admission_id']] = {'result': {
            'final_response': 'REAL_FINISHED_ANSWER', 'completed': True, 'messages': []}, 'usage': {}}
        await recover_failed_settlement(authority, row)
        assert await settle_unattended_unknown(authority, SessionRef('p', 's'))
        assert _statuses(db)[row['admission_id']] == ('terminal', 'completed')
        assert admission_result(db, row['admission_id'])['result']['final_response'] == 'REAL_FINISHED_ANSWER'
        assert not [m for m in db.get_messages('s') if m.get('display_kind') == 'failed_turn']
    finally:
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_lost_automation_worker_settles_on_its_own_exit_and_runs_the_queued_follower(tmp_path, monkeypatch):
    """Review #141357 F2: a managed worker lost mid-turn fences its automation head ``unknown`` and
    raises ManagedExecutionUnknown. That same drain settles it and runs the follower already queued,
    without waiting for an unrelated wake."""
    from gateway.session_managed_worker import ManagedExecutionUnknown
    from gateway.session_settlement_recovery import recover_failed_settlement
    store, runner = _runner(tmp_path, monkeypatch, [])
    db = store._db
    db.create_session('s', source='telegram')
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner-1')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='gw10'), 's')
    ref = SessionRef('p', 's')
    actor = AuthorityConnection(authority, Peer(), {'user_id': 'human'}).actor
    ran = []
    try:
        authority._schedule = lambda ref: None
        head_id = await _automation_head(authority, actor, ref)
        follower = await authority.submit(actor, Submission('follower', ref, {'text': 'FOLLOWER'}, 'queue'))

        async def execute(owner, ref, row):
            if row['admission_id'] == head_id:
                await recover_failed_settlement(owner, row)
                raise ManagedExecutionUnknown('managed_worker_unknown')
            ran.append(row['payload']['text'])
            return 'ACK'
        monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
        del authority._schedule
        authority._schedule(ref)
        live = authority.sessions['s']
        await async_wait_until(lambda: live.task is not None and live.task.done(), what='drain finished')
        assert ran == ['FOLLOWER']
        statuses = _statuses(db)
        assert statuses[head_id] == ('terminal', 'interrupted')
        assert statuses[follower.admission_id][0] == 'terminal'
    finally:
        store.close_all_db_handles()
