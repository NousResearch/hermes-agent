"""prompt.steer_queued / prompt.replace_queued failure windows (RT-35 review): a durable queued prompt
is never reported steered unless its text reached the turn that settled, the steer's storage wait
never holds a lock the owner loop takes, a replacement is bound to the admission it replaced, and
only interactive local producers are offered (or served) either verb."""
import asyncio
import sqlite3
import threading
import time

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway import session_finite, session_queued_controls
from gateway.session_contract import SessionRef
from hermes_state_runtime import (
    begin_runtime_epoch, claim_session_input, find_session_admission, get_session_admission,
    recover_session_inputs,
)
from tests.gateway.test_prompt_queued_controls import Turn, _call, _completions, _connection, _ledger, interactive
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until, wait_until


async def _drained(authority):
    task = authority.sessions['s'].task
    if task is not None:
        await asyncio.wait_for(task, 30)


@pytest.mark.asyncio
async def test_a_steer_before_the_turn_adopts_its_agent_is_refused_and_the_prompt_still_runs(tmp_path, monkeypatch):
    """Reviewer repro: execution held before adoption, steer, then agent construction fails. The text
    must not be reported steered; the durable queued prompt stays queued and runs in FIFO order."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    held, release, ran = asyncio.Event(), asyncio.Event(), []

    async def execute(authority, ref, row):
        ran.append(row['payload']['text'])
        if row['payload']['text'] == 'running':
            held.set()
            await asyncio.wait_for(release.wait(), 30)
            raise RuntimeError('agent construction failed')
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(held.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        reply = await _call(_connection(authority), 'prompt.steer_queued', admission_id=queued.admission_id)
        assert reply.get('error', {}).get('data', {}).get('reason') == 'steer_unavailable', reply
        assert get_session_admission(db, admission_id=queued.admission_id)['status'] == 'queued'
        assert not authority.pending_steers, 'nothing may be latched for an agent that does not exist yet'
        release.set()
        await _drained(authority)
    assert ran == ['running', 'follow-up'], 'the queued prompt must still run as its own turn'
    assert _ledger(db) == [('running', 'terminal', 'failed'), ('follow-up', 'terminal', 'completed')]


@pytest.mark.asyncio
async def test_a_steer_into_a_turn_that_then_fails_settles_interrupted_not_steered(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn, frames = Turn(monkeypatch, fail='running'), []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.claimed.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        conn = _connection(authority)
        reply = await _call(conn, 'prompt.steer_queued', admission_id=queued.admission_id)
        assert 'error' not in reply, reply
        # Transferred into the running turn, not yet steered: it settles with that turn.
        assert (reply['result']['status'], reply['result']['outcome']) == ('started', None), reply
        assert turn.agent.steers == ['follow-up']
        turn.release.set()
        await _drained(authority)
        receipt = await _call(conn, 'prompt.receipt', admission_id=queued.admission_id)
    assert (receipt['result']['status'], receipt['result']['outcome']) == ('terminal', 'interrupted'), receipt
    assert (queued.admission_id, 'steered') not in _completions(frames)
    assert turn.ran == ['running'], 'a steered prompt is never replayed on its own'


class BufferedAgent(InterruptControlMixin):
    """The real steer seam (``agent/interrupt_control.py``): ``steer`` only buffers the text, and the turn
    takes it at its drains. A text no drain took is still buffered when the turn ends."""

    def __init__(self):
        self._pending_steer, self._pending_steer_lock = None, threading.Lock()

    def clear_interrupt(self, *args, **kwargs):
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize('consumed', [True, False])
async def test_a_steer_settles_steered_only_when_a_drain_of_the_turn_took_it(tmp_path, monkeypatch, consumed):
    """Reviewer repro (late steer): the text entered the agent's buffer after the turn's last drain. That
    claim never ran it, so it is not steered: the admission goes back to its FIFO place and runs once as
    its own turn, and the cached agent no longer holds it for an unrelated later turn."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    agent, claimed, release, ran, taken, frames = BufferedAgent(), asyncio.Event(), asyncio.Event(), [], [], []

    async def execute(authority, ref, row):
        ran.append(row['payload']['text'])
        authority.adopt_agent(ref.session_id, row['generation'], agent)
        claimed.set()
        await asyncio.wait_for(release.wait(), 30)
        if consumed:
            taken.append(agent._drain_pending_steer())  # the turn's next drain takes the text
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    authority.sessions['s'].event_stream.observers.add(frames.append)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(claimed.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        reply = await _call(_connection(authority), 'prompt.steer_queued', admission_id=queued.admission_id)
        assert 'error' not in reply, reply
        release.set()
        await _drained(authority)
    assert agent._pending_steer is None, 'an unconsumed steer must not wait for an unrelated later turn'
    if consumed:
        assert (taken, ran) == (['follow-up'], ['running'])
        assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'steered')]
    else:
        assert (queued.admission_id, 'steered') not in _completions(frames)
        assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'completed')]
        assert ran == ['running', 'follow-up'], 'the returned prompt runs once, as its own turn'


@pytest.mark.asyncio
@pytest.mark.parametrize('later', ['alpha\nbeta', 'alpha'])
async def test_each_transfer_settles_by_its_own_identity_whatever_text_another_carries(tmp_path, monkeypatch, later):
    """Reviewer repro (round 3): A ('alpha') is consumed by a drain of the turn; B arrives after that last
    drain carrying A's text inside a multiline prompt, or exactly A's text. A is steered and never runs
    again; B, which the turn never read, goes back to its queue place and runs once as its own turn."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    agent, ran, taken = BufferedAgent(), [], []
    claimed, drain, drained, release = (asyncio.Event() for _ in range(4))

    async def execute(authority, ref, row):
        ran.append(row['payload']['text'])
        authority.adopt_agent(ref.session_id, row['generation'], agent)
        if len(ran) > 1:
            return 'done'
        claimed.set()
        await asyncio.wait_for(drain.wait(), 30)
        taken.append(agent._drain_pending_steer())  # a drain of the turn consumes A
        drained.set()
        await asyncio.wait_for(release.wait(), 30)
        return 'done'  # B was accepted after that last drain
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(claimed.wait(), 30)
        conn = _connection(authority)
        a = await _submit(authority, 'a', 'alpha')
        assert 'error' not in await _call(conn, 'prompt.steer_queued', admission_id=a.admission_id)
        drain.set()
        await asyncio.wait_for(drained.wait(), 30)
        b = await _submit(authority, 'b', later)
        assert 'error' not in await _call(conn, 'prompt.steer_queued', admission_id=b.admission_id)
        release.set()
        await _drained(authority)
        await async_wait_until(lambda: len(ran) == 2, timeout=30, what='the returned admission ran')
        await _drained(authority)
    assert taken == ['alpha']
    assert ran == ['running', later], 'the consumed prompt never runs again; the unread one runs once'
    assert _ledger(db) == [('running', 'terminal', 'completed'), ('a', 'terminal', 'steered'),
                           ('b', 'terminal', 'completed')]
    assert agent._pending_steer is None


@pytest.mark.asyncio
async def test_a_transfer_stays_listed_as_steering_until_its_claim_settles(tmp_path, monkeypatch):
    """Reviewer repro (returned prompt stays hidden): a pending snapshot taken while a steer is in
    transfer still lists it, marked ``steering``, so no client retires an admission that may go back to
    the queue. Once it settles ``steered`` it leaves the pending set."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn, frames = Turn(monkeypatch), []
    authority.sessions['s'].event_stream.observers.add(frames.append)

    def listed(admission_id):
        pending = [f['params']['payload']['pending'] for f in frames if f['params']['type'] == 'session.info'][-1]
        return [(row['status'], row.get('steering'), row['controls']) for row in pending
                if row['admission_id'] == admission_id]
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.claimed.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        reply = await _call(_connection(authority), 'prompt.steer_queued', admission_id=queued.admission_id)
        assert 'error' not in reply, reply
        assert listed(queued.admission_id) == [('started', True, [])]
        turn.release.set()
        await _drained(authority)
    assert listed(queued.admission_id) == []
    assert _ledger(db)[-1] == ('follow-up', 'terminal', 'steered')


@pytest.mark.asyncio
@pytest.mark.parametrize('retry', [False, True])
async def test_a_steer_write_that_committed_but_reported_failure_is_still_settled_by_its_claim(
        tmp_path, monkeypatch, retry):
    """Reviewer repro: the transfer commits, then the write reports a failure. The claim's fence must
    survive it: an exact retry delivers the committed transfer, and without one the claim's settlement
    puts the undelivered prompt back in its queue place (never left ``steering`` until a restart)."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn = Turn(monkeypatch)
    original = session_queued_controls.steer_session_input

    def committed_then_failed(*args, **kwargs):
        original(*args, **kwargs)
        raise sqlite3.OperationalError('disk I/O error')
    monkeypatch.setattr(session_queued_controls, 'steer_session_input', committed_then_failed)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.claimed.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        conn = _connection(authority)
        failed = await _call(conn, 'prompt.steer_queued', admission_id=queued.admission_id)
        assert 'error' in failed, failed
        if retry:
            again = await _call(conn, 'prompt.steer_queued', admission_id=queued.admission_id)
            assert (again['result']['status'], again['result']['outcome']) == ('started', None), again
        turn.release.set()
        await _drained(authority)
    if retry:
        assert (turn.agent.steers, turn.ran) == (['follow-up'], ['running'])
        assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'steered')]
    else:
        assert (turn.agent.steers, turn.ran) == ([], ['running', 'follow-up'])
        assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'completed')]


@pytest.mark.asyncio
async def test_a_turn_that_ends_between_the_steer_commit_and_delivery_gets_its_prompt_back(tmp_path, monkeypatch):
    """The receiving execution returns after the steer committed but before the text reached its agent:
    the admission is put back in its FIFO place (never reported steered) and runs as its own turn."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn = Turn(monkeypatch)
    loop = asyncio.get_running_loop()
    original = session_queued_controls.steer_session_input

    def committed_then_turn_ends(*args, **kwargs):
        committed = original(*args, **kwargs)
        loop.call_soon_threadsafe(turn.release.set)
        wait_until(lambda: authority.executing.get('s') is None, timeout=30, what='the receiving execution returned')
        return committed
    monkeypatch.setattr(session_queued_controls, 'steer_session_input', committed_then_turn_ends)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.claimed.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        reply = await _call(_connection(authority), 'prompt.steer_queued', admission_id=queued.admission_id)
        assert reply.get('error', {}).get('data', {}).get('reason') == 'steer_unavailable', reply
        await _drained(authority)
        await async_wait_until(lambda: turn.ran == ['running', 'follow-up'], timeout=30,
                               what="the restored row's own drain ran it")
        await _drained(authority)
    assert turn.agent.steers == []
    assert turn.ran == ['running', 'follow-up']
    assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'completed')]


def test_an_owner_restart_mid_transfer_settles_the_steered_admission_interrupted(tmp_path, monkeypatch):
    """Committed but not confirmed consumed when the owner died: interrupted, never auto-replayed."""
    from hermes_state_runtime import admit_session_input
    db, authority = _authority(tmp_path, monkeypatch)
    with db:
        for request_id in ('running', 'follow-up'):
            admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='s',
                                request_id=request_id, payload={'text': request_id})
        running = claim_session_input(db, epoch=authority.epoch, session_id='s')
        follow = find_session_admission(db, principal_id='human', session_id='s', request_id='follow-up')
        session_queued_controls.steer_session_input(db, epoch=authority.epoch, admission_id=follow['admission_id'],
                                                    generation=running['generation'])
        epoch = begin_runtime_epoch(db, instance_id='next')
        recover_session_inputs(db, epoch=epoch)
    assert _ledger(db) == [('running', 'unknown', None), ('follow-up', 'terminal', 'interrupted')]


@pytest.mark.asyncio
@pytest.mark.parametrize('verb', ['steer', 'replace'])
async def test_a_held_sqlite_writer_never_stalls_the_owner_loop_behind_a_queued_control(tmp_path, monkeypatch, verb):
    """Reviewer repro: another connection holds BEGIN IMMEDIATE while a steer/replace waits on the writer.
    A loop-side receipt (which takes the session's stream lock) and an unrelated ticker stay responsive."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn = Turn(monkeypatch)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.claimed.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        other = await _submit(authority, 'other')
        holding, release, done, writing = threading.Event(), threading.Event(), threading.Event(), threading.Event()

        def hold_writer():
            blocker = sqlite3.connect(str(db.db_path), isolation_level=None, timeout=10)
            try:
                blocker.execute('BEGIN IMMEDIATE')
                holding.set()
                release.wait(timeout=30)  # bounded: a stalled loop cannot release it itself
                blocker.execute('ROLLBACK')
            finally:
                blocker.close()
                done.set()
        # The control's own store write starts (and then waits on the held writer).
        from gateway import session_authority
        target, name = ((session_queued_controls, 'steer_session_input') if verb == 'steer'
                        else (session_authority, 'admit_session_input'))
        original = getattr(target, name)

        def announced(*args, **kwargs):
            writing.set()
            return original(*args, **kwargs)
        monkeypatch.setattr(target, name, announced)
        threading.Thread(target=hold_writer, daemon=True).start()
        assert await asyncio.to_thread(holding.wait, 30)
        params = ({'admission_id': queued.admission_id} if verb == 'steer' else
                  {'admission_id': queued.admission_id, 'submission_id': 'edited', 'text': 'edited'})
        control = asyncio.create_task(_call(_connection(authority), f'prompt.{verb}_queued', **params))
        assert await asyncio.to_thread(writing.wait, 30)
        # The writer stays held until this loop-side receipt returns: a receipt (or a loop) stuck behind
        # the held writer could only finish when the 30 s hold timed out.
        started = time.monotonic()
        await authority.receipt(ACTOR, REF, other.admission_id)
        receipt_latency = time.monotonic() - started
        release.set()
        reply = await asyncio.wait_for(control, 30)
        assert await asyncio.to_thread(done.wait, 30)
        turn.release.set()
        await _drained(authority)
    assert 'error' not in reply, reply
    assert receipt_latency < 10, f'a loop-side receipt waited {receipt_latency:.2f}s on the held writer'


@pytest.mark.asyncio
async def test_concurrent_replacements_of_different_rows_under_one_submission_id_replace_exactly_one(tmp_path, monkeypatch):
    """Reviewer repro: both initial lookups see no replacement; one replacement is admitted and binds the
    row it retired, the other is refused admission_conflict and its row stays queued."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    barrier = threading.Barrier(2, timeout=30)
    original = session_queued_controls.find_session_admission

    def synchronized(*args, **kwargs):
        found = original(*args, **kwargs)
        barrier.wait()
        return found
    monkeypatch.setattr(session_queued_controls, 'find_session_admission', synchronized)
    with db:
        a, b = await _submit(authority, 'a'), await _submit(authority, 'b')
        replies = await asyncio.gather(*(
            _call(_connection(authority), 'prompt.replace_queued', admission_id=old.admission_id,
                  submission_id='edit', text='same words') for old in (a, b)))
    succeeded = [reply for reply in replies if 'error' not in reply]
    refused = [reply['error']['data']['reason'] for reply in replies if 'error' in reply]
    assert len(succeeded) == 1 and refused == ['admission_conflict'], replies
    ledger = dict((request_id, (status, outcome)) for request_id, status, outcome in _ledger(db))
    assert sorted([ledger['a'], ledger['b']]) == [('queued', None), ('terminal', 'cancelled')], ledger
    assert ledger['edit'] == ('queued', None)


@pytest.mark.asyncio
async def test_a_replacement_retry_naming_another_admission_is_a_conflict(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        a, b = await _submit(authority, 'a'), await _submit(authority, 'b')
        conn = _connection(authority)
        edit = {'submission_id': 'edit', 'text': 'same words'}
        first = await _call(conn, 'prompt.replace_queued', admission_id=a.admission_id, **edit)
        assert 'error' not in first, first
        # An exact retry (same old admission) answers the same replacement.
        again = await _call(conn, 'prompt.replace_queued', admission_id=a.admission_id, **edit)
        assert again.get('result', {}).get('admission_id') == first['result']['admission_id'], again
        # Another row, independently cancelled, never borrows that replacement's receipt.
        await _call(conn, 'prompt.cancel', admission_id=b.admission_id)
        other = await _call(conn, 'prompt.replace_queued', admission_id=b.admission_id, **edit)
        assert other.get('error', {}).get('data', {}).get('reason') == 'admission_conflict', other


def test_a_hosted_room_queued_row_is_offered_no_controls_and_both_verbs_refuse_it(hosted_owner):
    """A hosted-room settlement has no ``steered`` (or replaced) result: its rows are never offered either
    verb and the verbs refuse them by name, so the room's terminal/results reads stay healthy."""
    from dataclasses import asdict
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_contract import Principal
    from gateway.session_controls import AuthorityConnection
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default',
                                 principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    callbacks = []
    rows = [rpc.submit(**coords, session_id=sid, prompt=text, task=TaskIdentity('room', text, 'thread', text),
                       execution_generation=1, on_terminal=callbacks.append) for text in ('first', 'second')]
    claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
    ref = SessionRef(authority.profile_id, sid)

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result(10)

    async def pending():
        return [asdict(authority._pending_receipt(row)) for row in authority._pending_rows(ref)]
    assert [row['controls'] for row in run(pending())] == [[], []]

    conn = AuthorityConnection(authority, object(), {'user_id': principal.subject})
    conn.actor = Principal(principal.subject, authority.profile_id, principal.capabilities, conn.actor.transport_id)
    conn.subscriptions[sid] = 'attached'
    for method, params in (('prompt.steer_queued', {}),
                           ('prompt.replace_queued', {'submission_id': 'edited', 'text': 'edited'})):
        reply = run(conn.dispatch({'id': 1, 'method': method, 'params': {
            'session_id': sid, 'admission_id': rows[1]['admission_id'], **params}}))
        assert reply.get('error', {}).get('data', {}).get('reason') == 'unsupported_control', (method, reply)
    assert get_session_admission(authority.db, admission_id=rows[1]['admission_id'])['status'] == 'queued'
    # The room's reads stay healthy over its ledger.
    assert isinstance(rpc.history(**coords, session_id=sid), list)
    assert isinstance(rpc.info(**coords, session_id=sid), dict)


def test_only_interactive_local_sessions_are_offered_queued_controls(tmp_path, monkeypatch):
    """cron, kanban, a2a and bot_room rows settle without a ``steered``/replaced result: never offered."""
    db, authority = _authority(tmp_path, monkeypatch)
    row = {'status': 'queued', 'target_session_id': 's', 'payload': {'text': 'hello'}}
    with db:
        assert session_queued_controls.queued_controls(authority, row) == [], 'no launch policy: not interactive'
        # ``desktop``: a Desktop chat migrated from main keeps main's source label.
        for source, offered in (('gui', True), ('desktop', True), ('tui', True), ('cli', True), ('acp', True),
                                ('cron', False), ('kanban', False), ('a2a', False), ('bot_room', False)):
            authority.launch_sources = {}
            interactive(db, source=source)
            controls = session_queued_controls.queued_controls(authority, row)
            assert controls == (['steer', 'replace'] if offered else []), (source, controls)
