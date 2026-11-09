"""A claimed turn's fence covers all the work the turn started, for that work's whole lifetime.

The turn's identity is one value: the bound ``TurnScope`` (session, turn, fence), plus the store
that issued the claim. Work the turn starts (a copied asyncio task, the title-upgrade thread, a
queued token update) carries that identity with it, so once another owner takes the profile over
or a newer claim moves the session's generation, that work's writes are refused, also after the
turn's body returned. Work whose turn still holds its lease writes as before. The owner's own
settlement runs as an owner operation under its own epoch/generation checks.
"""
import asyncio
import contextvars
import threading
from contextlib import closing
from pathlib import Path

import pytest

import hermes_state_runtime as rt
from agent.runtime_scope import bind_profile_scope, bind_turn, claim_fence, owner_operation
from hermes_scope import Fence, current_turn, push_state
from hermes_state import SessionDB
from hermes_state_errors import TurnFenceSupersededError


@pytest.fixture
def claimed(tmp_path):
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        db.create_session('s', 'cli')
        epoch = rt.begin_runtime_epoch(db, instance_id='one')
        rt.admit_session_input(db, epoch=epoch, principal_id='u', session_id='s', request_id='r', payload={})
        row = rt.claim_session_input(db, epoch=epoch, session_id='s')
        yield db, row


def _elsewhere(fn, *args):
    return contextvars.Context().run(fn, *args)


def _take_over(db):
    """Another owner takes the profile over."""
    _elsewhere(lambda: rt.begin_runtime_epoch(db, instance_id='two'))


def _next_claim(db, row):
    """The same owner settles *row*'s claim and claims the session's next input: a new generation."""
    def run():
        epoch, sid = row['owner_epoch'], row['target_session_id']
        rt.settle_session_input(db, epoch=epoch, admission_id=row['admission_id'],
                                generation=row['generation'], outcome='completed', result={})
        rt.admit_session_input(db, epoch=epoch, principal_id='u', session_id=sid, request_id='r2', payload={})
        claimed = rt.claim_session_input(db, epoch=epoch, session_id=sid)
        assert claimed['generation'] > row['generation']
        return claimed
    return _elsewhere(run)


def _fence(row):
    return claim_fence(row)


# --- queued token accounting carries the claim it was queued under ------------------------------

def test_a_superseded_turns_queued_tokens_are_refused_like_its_synchronous_ones(claimed):
    db, row = claimed
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        _take_over(db)
        with pytest.raises(TurnFenceSupersededError):
            db.update_token_counts('s', input_tokens=17)
        db.queue_token_counts('s', input_tokens=17)
        assert db.flush_token_counts()
    assert db.flush_token_counts()
    assert db.get_session('s')['input_tokens'] == 0


def test_queued_tokens_of_a_live_turn_land_after_its_body_returned(claimed):
    db, row = claimed
    gate = threading.Event()
    original = db._apply_token_batch

    def held(batch):
        assert gate.wait(10)
        original(batch)
    db._apply_token_batch = held
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        db.queue_token_counts('s', input_tokens=5, api_call_count=1)
    gate.set()  # the writer applies after the body exited, in its own thread
    assert db.flush_token_counts()
    assert db.get_session('s')['input_tokens'] == 5


def test_queued_tokens_never_merge_across_claims(claimed):
    """A delta queued by the superseded claim and one queued by its successor, in one writer batch:
    the writer refuses the first and lands the second, never a merged sum under either fence."""
    from hermes_state_fence import writing_turn
    db, row = claimed
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        old = writing_turn()
    nxt = _next_claim(db, row)
    with bind_turn('s', nxt['admission_id'], _fence(nxt), store=db.db_path):
        new = writing_turn()
    batch = [('s', dict(input_tokens=3, api_call_count=1), old), ('s', dict(input_tokens=4, api_call_count=1), new)]
    assert db._coalesce_token_deltas(list(batch)) == batch
    db._apply_token_batch(batch)
    assert db.get_session('s')['input_tokens'] == 4
    # Same claim, same route: still merged into one write.
    assert len(db._coalesce_token_deltas([batch[1], batch[1]])) == 1


def test_tokens_queued_outside_any_turn_are_unfenced(claimed):
    db, _row = claimed
    _take_over(db)
    db.queue_token_counts('s', input_tokens=2)
    assert db.flush_token_counts()
    assert db.get_session('s')['input_tokens'] == 2


# --- work the turn started keeps the turn's fence after the body exits ---------------------------

@pytest.mark.asyncio
async def test_a_copied_task_cannot_land_a_late_reply_after_takeover(claimed):
    db, row = claimed
    go = asyncio.Event()
    outcome = []

    async def late():
        await go.wait()
        assert current_turn() is not None  # still the turn's work
        try:
            db.append_message('s', 'assistant', 'stale late reply')
            outcome.append('written')
        except TurnFenceSupersededError:
            outcome.append('refused')
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        task = asyncio.create_task(late())
    _take_over(db)
    go.set()
    await task
    assert outcome == ['refused']
    assert db.get_messages('s') == []


@pytest.mark.asyncio
async def test_a_copied_task_of_a_turn_still_holding_its_lease_writes(claimed):
    db, row = claimed
    go = asyncio.Event()

    async def late():
        await go.wait()
        db.append_message('s', 'assistant', 'late but current')
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        task = asyncio.create_task(late())
    go.set()
    await task
    assert [m['content'] for m in db.get_messages('s')] == ['late but current']


def _title_thread(db, monkeypatch, title):
    from agent import title_generator as tg
    entered, go = threading.Event(), threading.Event()

    def generate(*_a, **_k):
        entered.set()
        assert go.wait(10)
        return title
    monkeypatch.setattr(tg, 'generate_title', generate)
    monkeypatch.setattr(tg, '_auto_title_enabled', lambda: True)
    monkeypatch.setattr(tg, '_model_title_upgrade_enabled', lambda: True)
    monkeypatch.setattr(tg, '_kanban_task_title', lambda: None)
    return tg, entered, go


def test_the_title_thread_of_a_turn_whose_owner_lost_the_profile_publishes_nothing(claimed, monkeypatch):
    db, row = claimed
    tg, entered, go = _title_thread(db, monkeypatch, 'Old owner title')
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        thread = tg.maybe_auto_title(db, 's', 'Explain transactional leases')
        assert entered.wait(10)
        instant = db.get_session_title('s')
    _take_over(db)
    go.set()
    thread.join(10)
    assert not thread.is_alive()
    assert db.get_session_title('s') == instant != 'Old owner title'


def test_a_late_title_survives_the_sessions_next_claim(claimed, monkeypatch):
    """The title names the session from this turn's ask; the same owner claiming the next input
    must not drop it (and its paid call) as if it were a stale reply."""
    db, row = claimed
    tg, entered, go = _title_thread(db, monkeypatch, 'Next claim title')
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        thread = tg.maybe_auto_title(db, 's', 'Explain transactional leases')
        assert entered.wait(10)
    _next_claim(db, row)
    go.set()
    thread.join(10)
    assert db.get_session_title('s') == 'Next claim title'


def test_a_late_title_of_a_turn_not_superseded_still_lands(claimed, monkeypatch):
    db, row = claimed
    tg, entered, go = _title_thread(db, monkeypatch, 'Current owner title')
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        thread = tg.maybe_auto_title(db, 's', 'Explain transactional leases')
        assert entered.wait(10)
    go.set()  # the body already returned; nothing superseded the claim
    thread.join(10)
    assert db.get_session_title('s') == 'Current owner title'


# --- one turn identity: the bound TurnScope ------------------------------------------------------

def _turn_scope(db, row):
    from agent.runtime_scope import scope_for_profile
    return scope_for_profile(Path(db.db_path).parent).for_turn('s', row['admission_id'], _fence(row))


def test_a_bound_turn_scope_alone_fences_its_profiles_store(claimed):
    db, row = claimed
    _take_over(db)
    with push_state(scope=_turn_scope(db, row)):
        with pytest.raises(TurnFenceSupersededError):
            db.append_message('s', 'assistant', 'unmarked turn')
    assert db.get_messages('s') == []


def test_bind_turn_naming_no_store_fences_the_bound_profiles_store(claimed):
    db, row = claimed
    with bind_profile_scope(Path(db.db_path).parent):
        with bind_turn('s', row['admission_id'], _fence(row), store=None):
            _take_over(db)
            with pytest.raises(TurnFenceSupersededError):
                db.append_message('s', 'assistant', 'x')


# --- settlement is the owner's operation, under its own checks -----------------------------------

def test_settlement_from_a_copied_turn_context_runs_as_the_owners_operation(claimed):
    db, row = claimed
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        copied = contextvars.copy_context()

    def settle():
        with owner_operation():
            assert current_turn() is None
            return rt.settle_session_input(db, epoch=row['owner_epoch'], admission_id=row['admission_id'],
                                           generation=row['generation'], outcome='completed', result={})
    assert copied.run(settle)['status'] == 'terminal'


def test_an_owner_operation_keeps_the_ledgers_own_fence(claimed):
    db, row = claimed
    _take_over(db)

    def settle():
        with owner_operation():
            return rt.settle_session_input(db, epoch=row['owner_epoch'], admission_id=row['admission_id'],
                                           generation=row['generation'], outcome='completed', result={})
    with pytest.raises(rt.RuntimeStoreError):
        settle()


def test_the_turn_identity_is_immutable(claimed):
    db, row = claimed
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        mark = current_turn()
    with pytest.raises(AttributeError):
        mark.store = None
    assert (mark.session, mark.turn, mark.fence) == ('s', row['admission_id'], Fence(row['owner_epoch'],
                                                                                  row['generation']))


# --- work designed to outlive the turn keeps the owner half of the fence -------------------------

def test_work_that_outlives_the_turn_survives_the_next_claim_but_not_a_takeover(claimed):
    from agent.runtime_scope import outlive_turn
    db, row = claimed
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        copied = contextvars.copy_context()

    def report(text):
        with outlive_turn():
            db.append_message('s', 'assistant', text)
    _next_claim(db, row)
    with pytest.raises(TurnFenceSupersededError):  # without outlive_turn the next claim fences it
        copied.run(db.append_message, 's', 'assistant', 'plain late write')
    copied.run(report, 'delegation result')
    _take_over(db)
    with pytest.raises(TurnFenceSupersededError):
        copied.run(report, 'after takeover')
    assert [m['content'] for m in db.get_messages('s')] == ['delegation result']


def test_an_async_delegation_runs_as_work_that_outlives_its_turn(claimed, monkeypatch):
    import tools.async_delegation as ad
    db, row = claimed
    seen = []

    def runner():
        seen.append(current_turn())
        return {'status': 'completed'}
    monkeypatch.setattr(ad, '_persist_dispatch', lambda record: None)
    monkeypatch.setattr(ad, '_finalize', lambda *a, **k: None)
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        handle = ad.dispatch_async_delegation(goal='g', context=None, toolsets=None, role='leaf', model=None,
                                              session_key='', parent_session_id='s', runner=runner)
    assert handle['status'] == 'dispatched'
    for _ in range(200):
        if seen:
            break
        threading.Event().wait(0.05)
    assert seen and seen[0].outlives and seen[0].turn == row['admission_id']


def test_a_background_review_runs_as_work_that_outlives_its_turn(claimed, monkeypatch):
    import types

    import agent.background_review as br
    import run_agent
    db, row = claimed
    seen, done = [], threading.Event()

    def target():
        seen.append(current_turn())
        done.set()
    monkeypatch.setattr(br, 'prepare_background_review_run', lambda agent: object())
    monkeypatch.setattr(br, 'spawn_background_review_thread', lambda *a, **k: (target, ''))
    agent = types.SimpleNamespace(_maybe_requeue_preempted_review=lambda *a: None)
    spawn = types.MethodType(run_agent.AIAgent._spawn_background_review_now, agent)
    with bind_turn('s', row['admission_id'], _fence(row), store=db.db_path):
        spawn([])
    assert done.wait(10)
    assert seen[0].outlives and seen[0].turn == row['admission_id']
