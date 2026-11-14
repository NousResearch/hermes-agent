"""prompt.steer_queued / prompt.replace_queued: Desktop's Steer and Edit on a canonical queued row
(RT-35). Both verbs consume the still-queued admission in the owner's own write transaction, so it
can never both run as a queued turn and be steered or replaced, and a retry replays the outcome."""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from hermes_state_runtime import (
    begin_runtime_epoch, claim_session_input, find_session_admission, list_session_admissions,
    recover_session_inputs,
)
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit


class Agent:
    def __init__(self):
        self.steers = []

    def steer(self, text):
        self.steers.append(text)
        return True

    def clear_interrupt(self, *args, **kwargs):
        pass


class Turn:
    """Every turn the drain claims adopts one agent and runs until ``release`` (``fail``: that turn's
    text raises after adoption, so its execution fails)."""

    def __init__(self, monkeypatch, fail=None):
        from gateway import session_finite
        self.agent, self.claimed, self.release, self.ran = Agent(), asyncio.Event(), asyncio.Event(), []

        async def execute(authority, ref, row):
            self.ran.append(row['payload']['text'])
            authority.adopt_agent(ref.session_id, row['generation'], self.agent)
            self.claimed.set()
            await asyncio.wait_for(self.release.wait(), 30)
            if row['payload']['text'] == fail:
                raise RuntimeError('the receiving turn failed')
            return 'done'
        monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)


def interactive(db, sid='s', source='gui'):
    """The session was launched from an interactive local surface (Desktop by default): a local
    session's row records its frozen launch policy's source."""
    db._execute_write(lambda conn: conn.execute('UPDATE sessions SET source=? WHERE id=?', (source, sid)))


def _connection(authority):
    from gateway.session_controls import AuthorityConnection
    conn = AuthorityConnection(authority, object(), {'user_id': 'human'})
    conn.subscriptions['s'] = 'attached'  # prompt.replace_queued admits like prompt.submit
    return conn


async def _call(conn, method, **params):
    return await conn.dispatch({'id': 1, 'method': method, 'params': {'session_id': 's', **params}})


def _ledger(db):
    return [(row['request_id'], row['status'], row['outcome'])
            for row in list_session_admissions(db, session_id='s', pending_only=False)]


def _completions(frames):
    return [(f['params']['payload'].get('admission_id'), f['params']['payload'].get('outcome'))
            for f in frames if f['params']['type'] == 'message.complete']


@pytest.mark.asyncio
async def test_steer_queued_delivers_the_queued_text_into_the_running_turn_exactly_once(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    turn, frames = Turn(monkeypatch), []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(turn.claimed.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        conn = _connection(authority)

        reply = await _call(conn, 'prompt.steer_queued', admission_id=queued.admission_id)
        assert 'error' not in reply, reply
        # Transferred into the running turn: it reads as that execution until the turn settles.
        assert (reply['result']['status'], reply['result']['outcome']) == ('started', None), reply
        assert turn.agent.steers == ['follow-up']
        assert (queued.admission_id, 'steered') not in _completions(frames)

        # A retry of the same request replays the recorded transfer and steers nothing twice.
        again = await _call(conn, 'prompt.steer_queued', admission_id=queued.admission_id)
        assert again.get('result') == reply['result'], again
        assert turn.agent.steers == ['follow-up']

        turn.release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 30)
        settled = await _call(conn, 'prompt.steer_queued', admission_id=queued.admission_id)
        assert (settled['result']['status'], settled['result']['outcome']) == ('terminal', 'steered'), settled
        assert (queued.admission_id, 'steered') in _completions(frames)
    assert turn.ran == ['running'], 'the steered admission also ran as its own queued turn'
    assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'steered')]


@pytest.mark.asyncio
async def test_steer_queued_refuses_by_name_and_leaves_the_admission_queued(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        idle = await _submit(authority, 'waiting')
        conn = _connection(authority)
        reply = await _call(conn, 'prompt.steer_queued', admission_id=idle.admission_id)
        assert reply['error']['data']['reason'] == 'session_idle', reply

        missing = await _call(conn, 'prompt.steer_queued', admission_id='no-such-admission')
        assert missing['error']['data']['reason'] == 'not_found', missing

        # The drain claimed it first: it runs as its own turn and is never also steered.
        claim_session_input(db, epoch=authority.epoch, session_id='s')
        started = await _call(conn, 'prompt.steer_queued', admission_id=idle.admission_id)
        assert started['error']['data']['reason'] == 'admission_started', started
    assert _ledger(db) == [('waiting', 'started', None)]


def test_the_steer_transaction_fences_on_the_queued_row_and_the_running_generation(tmp_path, monkeypatch):
    """Atomic in the store: queued -> steered only while the named generation is the running one."""
    import hermes_state_runtime as store
    from hermes_state_runtime import RuntimeStoreError, admit_session_input
    assert hasattr(store, 'steer_session_input'), 'no store transaction steers a queued admission'
    steer_session_input = store.steer_session_input
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    with db:
        for request_id in ('running', 'follow-up'):
            admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='s',
                                request_id=request_id, payload={'text': request_id})
        running = claim_session_input(db, epoch=authority.epoch, session_id='s')
        follow = find_session_admission(db, principal_id='human', session_id='s', request_id='follow-up')
        with pytest.raises(RuntimeStoreError) as stale:
            steer_session_input(db, epoch=authority.epoch, admission_id=follow['admission_id'],
                                generation=running['generation'] + 1)
        assert stale.value.reason == 'session_idle'
        with pytest.raises(RuntimeStoreError) as own:
            steer_session_input(db, epoch=authority.epoch, admission_id=running['admission_id'],
                                generation=running['generation'])
        assert own.value.reason == 'admission_started'
        row, fresh = steer_session_input(db, epoch=authority.epoch, admission_id=follow['admission_id'],
                                         generation=running['generation'])
        # Bound to the receiving claim until that execution settles (never claimable, never steered yet).
        assert (row['status'], row['outcome'], row['generation'], fresh) == (
            'terminal', 'steering', running['generation'], True)
        replay, fresh = steer_session_input(db, epoch=authority.epoch, admission_id=follow['admission_id'],
                                            generation=running['generation'])
        assert (replay['outcome'], fresh) == ('steering', False)


@pytest.mark.asyncio
async def test_replace_queued_retires_the_old_admission_and_admits_the_edit_at_the_back(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    with db:
        first = await _submit(authority, 'first')
        await _submit(authority, 'second')
        conn = _connection(authority)
        edit = {'admission_id': first.admission_id, 'submission_id': 'first-edited', 'text': 'first, edited'}

        reply = await _call(conn, 'prompt.replace_queued', **edit)
        assert 'error' not in reply, reply
        new_id = reply['result']['admission_id']
        assert new_id != first.admission_id and reply['result']['status'] == 'queued'
        assert (first.admission_id, 'cancelled') in _completions(frames)
        queued = [r for r in list_session_admissions(db, session_id='s') if r['status'] == 'queued']
        assert [(r['request_id'], r['payload']['text']) for r in queued] == [
            ('second', 'second'), ('first-edited', 'first, edited')], 'the edit must join the back of the FIFO'

        # Exact retry: the same new admission, nothing admitted twice; the old id stays cancelled.
        again = await _call(conn, 'prompt.replace_queued', **edit)
        assert again.get('result', {}).get('admission_id') == new_id, again
        old = await _call(conn, 'prompt.cancel', admission_id=first.admission_id)
        assert (old['result']['status'], old['result']['outcome']) == ('terminal', 'cancelled')
    assert _ledger(db) == [('first', 'terminal', 'cancelled'), ('second', 'queued', None),
                           ('first-edited', 'queued', None)]


@pytest.mark.asyncio
async def test_replace_queued_refuses_a_started_admission_and_admits_nothing(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        first = await _submit(authority, 'first')
        claim_session_input(db, epoch=authority.epoch, session_id='s')
        conn = _connection(authority)
        reply = await _call(conn, 'prompt.replace_queued', admission_id=first.admission_id,
                            submission_id='too-late', text='edited')
        assert reply['error']['data']['reason'] == 'admission_started', reply
        missing = await _call(conn, 'prompt.replace_queued', admission_id='no-such-admission',
                              submission_id='orphan', text='edited')
        assert missing['error']['data']['reason'] == 'not_found', missing
    assert find_session_admission(db, principal_id='human', session_id='s', request_id='too-late') is None
    assert _ledger(db) == [('first', 'started', None)]


@pytest.mark.asyncio
async def test_a_queued_admission_survives_an_owner_restart_and_is_steered_by_the_next_owner(tmp_path, monkeypatch):
    db, before = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(before, '_schedule', lambda ref: None)
    with db:
        await _submit(before, 'head')
        follower = await _submit(before, 'follower')
        # Owner restart: a new epoch over the same store; both rows were still queued.
        epoch = begin_runtime_epoch(db, instance_id='next')
        recover_session_inputs(db, epoch=epoch)
        after = SessionAuthority(before.runner, profile_id='owned', instance_id='next', db=db, epoch=epoch)
        after.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
        turn = Turn(monkeypatch)
        after._schedule(REF)
        await asyncio.wait_for(turn.claimed.wait(), 30)

        reply = await _call(_connection(after), 'prompt.steer_queued', admission_id=follower.admission_id)
        assert 'error' not in reply, reply
        assert turn.agent.steers == ['follower']
        turn.release.set()
        await asyncio.wait_for(after.sessions['s'].task, 30)
    assert turn.ran == ['head']
    assert _ledger(db) == [('head', 'terminal', 'completed'), ('follower', 'terminal', 'steered')]


@pytest.mark.asyncio
async def test_pending_rows_name_the_controls_the_owner_serves(tmp_path, monkeypatch):
    """Desktop shows Steer / Edit only for rows that list them: an older owner lists none."""
    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    with db:
        await _submit(authority, 'head')
        await _submit(authority, 'follower')
        claim_session_input(db, epoch=authority.epoch, session_id='s')
        pending = {row['input_id']: row for row in (asdict(authority._pending_receipt(r))
                                                     for r in authority._pending_rows(REF))}
    assert pending['follower'].get('controls') == ['steer', 'replace']
    assert not pending['head'].get('controls'), 'a started row has no queued-row controls'
