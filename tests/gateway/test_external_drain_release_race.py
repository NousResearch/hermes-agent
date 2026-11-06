"""Cancel between the real transaction refusal and owner-loop publication.

Retains the event schedule from independent_review_probe.py::
release_before_refusal_publication. Storage, claim, pump and cancellation are
real; only finite turn execution is inert. No evidence sinks in canonical tests.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.session import SessionSource
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import SessionRef
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError, admit_session_input, begin_runtime_epoch, cancel_session_input,
    claim_session_input, get_session_admission,
)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', [
    'released', 'shutdown', 'not-running', 'retiring', 'retired', 'removed', 'replaced',
    'cancelled-head', 'unknown', 'stale-epoch', 'cancelled-pump',
])
async def test_release_before_real_refusal_publication(tmp_path, monkeypatch, change):
    from gateway import session_authority as module

    db = SessionDB(tmp_path / 'release-race.db')
    runner = SimpleNamespace(_running=True, _draining=False, _external_drain_active=False,
                             _update_runtime_status=lambda *a, **k: None,
                             _persist_active_agents=lambda: None, _serving_state=lambda: 'running')
    owner = SessionAuthority(runner, profile_id='fixture', instance_id='independent', db=db,
                             epoch=begin_runtime_epoch(db, instance_id='independent'))
    runner.session_authority = owner
    db.create_session('s', source='telegram')
    live = owner.sessions['s'] = LiveSession(
        SessionSource(Platform.TELEGRAM, 'chat', user_id='human'), 'route')
    ref = SessionRef(owner.profile_id, 's')
    row = admit_session_input(db, epoch=owner.epoch, principal_id='human', session_id='s',
                              request_id='follower', payload={'text': 'private synthetic follower'})
    following = None
    if change == 'cancelled-head':
        following = admit_session_input(db, epoch=owner.epoch, principal_id='human', session_id='s',
                                         request_id='next', payload={'text': 'next synthetic input'})
    before = get_session_admission(db, admission_id=row['admission_id'])
    loop = asyncio.get_running_loop()
    reached, refused = asyncio.Event(), asyncio.Event()
    permit_guard, permit_return = threading.Event(), threading.Event()
    real_claim = module.claim_session_input
    refusals, executed = [], []

    def held_claim(*args, **kwargs):
        loop.call_soon_threadsafe(reached.set)
        if not permit_guard.wait(3):
            raise TimeoutError('private guard release missing')
        try:
            return real_claim(*args, **kwargs)
        except RuntimeStoreError as exc:
            refusals.append(exc.reason)
            loop.call_soon_threadsafe(refused.set)
            if not permit_return.wait(3):
                raise TimeoutError('private refusal release missing')
            raise

    async def inert_execution(authority, session, claimed):
        executed.append((claimed['admission_id'], claimed['owner_epoch'], claimed['generation']))
        return 'private synthetic completion'

    monkeypatch.setattr(module, 'claim_session_input', held_claim)
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', inert_execution)
    task = asyncio.create_task(owner._drain(ref))
    live.task = task
    try:
        await asyncio.wait_for(reached.wait(), 3)
        runner._external_drain_active = True
        permit_guard.set()
        await asyncio.wait_for(refused.wait(), 3)
        assert refusals == ['runtime_draining']  # real transaction guard, not a fabricated refusal
        assert live.external_drain_paused is False
        assert get_session_admission(db, admission_id=row['admission_id']) == before
        def mark_unknown():
            from gateway.session_settlement_recovery import _mark_unknown
            claimed = claim_session_input(db, epoch=owner.epoch, session_id='s')
            assert claimed['generation'] == 1
            assert _mark_unknown(owner, claimed) == 'unknown'

        mutations = {
            'shutdown': lambda: vars(runner).update(_draining=True, _running=False),
            'not-running': lambda: setattr(runner, '_running', False),
            'retiring': lambda: setattr(owner, 'retiring', True),
            'retired': lambda: setattr(owner, 'retired', True),
            'removed': lambda: owner.sessions.pop('s'),
            'replaced': lambda: owner.sessions.update(s=LiveSession(live.source, live.route)),
            'cancelled-head': lambda: cancel_session_input(db, epoch=owner.epoch, admission_id=row['admission_id']),
            'unknown': mark_unknown,
            'stale-epoch': lambda: begin_runtime_epoch(db, instance_id='replacement'),
            'cancelled-pump': task.cancel,
        }
        if action := mutations.get(change):
            action()
        GatewayShutdownMixin._exit_external_drain(runner)
        assert runner._external_drain_active is False
        permit_return.set()
        if change == 'cancelled-pump':
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        else:
            await asyncio.wait_for(task, 3)
        assert task.done() and live.task is task  # no unrelated scheduling event
        current = get_session_admission(db, admission_id=row['admission_id'])
        if change in {'released', 'cancelled-head'}:
            target = following or row
            assert get_session_admission(db, admission_id=target['admission_id'])['outcome'] == 'completed', (
                'cancel before refusal publication stranded the durable FIFO')
            assert executed == [(target['admission_id'], owner.epoch, 1)]
            assert current['outcome'] == ('cancelled' if following else 'completed')
        else:
            assert executed == []
            assert current['status'] == ('unknown' if change == 'unknown' else 'queued')
        if change == 'replaced':
            assert owner.sessions['s'].external_drain_paused is False
        if change == 'removed':
            assert 's' not in owner.sessions
    finally:
        permit_guard.set()
        permit_return.set()
        await asyncio.gather(task, return_exceptions=True)
        pending = list(getattr(owner, '_mutation_tasks', ()))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['runtime_draining', 'permission_denied', 'unknown_execution'])
async def test_unrelated_refusal_is_not_retried(tmp_path, monkeypatch, reason):
    db = SessionDB(tmp_path / 'other-refusal.db')
    runner = SimpleNamespace(_running=True, _draining=False, _external_drain_active=False)
    owner = SessionAuthority(runner, profile_id='fixture', instance_id='independent', db=db,
                             epoch=begin_runtime_epoch(db, instance_id='independent'))
    db.create_session('s', source='telegram')
    live = owner.sessions['s'] = LiveSession(SessionSource(Platform.TELEGRAM, 'chat'), 'route')
    calls = []

    async def refusal(*args):
        calls.append(reason)
        if len(calls) != 1:
            raise AssertionError('a non-external refusal must not be retried')
        raise RuntimeStoreError(reason)

    monkeypatch.setattr(owner, '_claim_next', refusal)
    try:
        assert await owner._claim_with_retry(SessionRef(owner.profile_id, 's'), live) is None
        assert calls == [reason] and live.external_drain_paused is False
    finally:
        db.close()
