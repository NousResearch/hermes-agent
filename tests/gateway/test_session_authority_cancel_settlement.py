"""Cancelling a queued admission settles every observer kind in the authority, not per caller."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch, list_session_admissions

ACTOR = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
REF = SessionRef('owned', 's')


def _authority(tmp_path, monkeypatch, platform=None):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=platform, user_id='human'), 'route')
    return db, authority


def _submit(authority, request_id, text=None):
    return authority.submit(ACTOR, Submission(request_id=request_id, ref=REF,
                                              payload={'text': text or request_id}, intent='queue'))


@pytest.mark.asyncio
async def test_cancel_queued_settles_native_waiter_and_publishes_terminal_completion(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    frames = []
    with db:
        receipt = await _submit(authority, 'queued-then-cancelled')
        # The native ingress delivery waiter (session_ingress.admit_message) and an
        # event-stream observer (ACP prompt(), API run projections) both wait on this row.
        authority.native_waiters.add(receipt.admission_id)
        waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
        authority.sessions['s'].event_stream.observers.add(frames.append)

        cancelled = await authority.cancel_queued(ACTOR, REF, receipt.admission_id)
        assert (cancelled.status, cancelled.outcome) == ('terminal', 'cancelled')

        assert waiter.done() and receipt.admission_id not in authority.native_waiters
        assert receipt.admission_id not in authority.waiters
        terminal = [f['params'] for f in frames if f['params']['type'] == 'message.complete']
        assert len(terminal) == 1, [f['params']['type'] for f in frames]
        assert terminal[0]['admission_id'] == receipt.admission_id
        assert terminal[0]['payload']['outcome'] == 'cancelled'
        assert terminal[0]['payload']['admission_id'] == receipt.admission_id

        # Idempotent: a repeated cancel of the terminal row is not a second completion.
        await authority.cancel_queued(ACTOR, REF, receipt.admission_id)
        assert len([f for f in frames if f['params']['type'] == 'message.complete']) == 1


@pytest.mark.asyncio
async def test_cancelling_the_head_of_a_paused_fifo_resumes_its_successor(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway import session_api_turn, session_finite

    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.API_SERVER)
    executed = []

    def check_api_turn(authority, ref, payload):
        if payload['text'] == 'blocked':
            raise RuntimeStoreError('permission_denied')
    monkeypatch.setattr(session_api_turn, 'check_api_turn', check_api_turn)

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        head = await _submit(authority, 'blocked')
        await _submit(authority, 'follower')
        # The drain pauses on the head's preclaim refusal and its task ends.
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert executed == []

        await authority.cancel_queued(ACTOR, REF, head.admission_id)
        task = authority.sessions['s'].task
        assert task is not None and not task.done(), 'cancelling the blocking head must reschedule the drain'
        await asyncio.wait_for(task, 5)
        assert executed == ['follower']
        statuses = {r['request_id']: (r['status'], r['outcome'])
                    for r in list_session_admissions(db, session_id='s', pending_only=False)}
        assert statuses == {'blocked': ('terminal', 'cancelled'), 'follower': ('terminal', 'completed')}


@pytest.mark.asyncio
async def test_hosted_head_cancelled_during_preclaim_does_not_let_successor_skip_its_check(tmp_path, monkeypatch):
    import threading
    from gateway import session_finite, session_hosted_transport, session_local_recovery
    from gateway.config import Platform

    db, authority = _authority(tmp_path, monkeypatch, platform=Platform.LOCAL)
    monkeypatch.setattr(session_local_recovery, 'restore_local_session', lambda authority, sid: None)
    validated, executed = [], []
    head_checking, release_head = threading.Event(), threading.Event()

    def check_remote_hosted_admission(authority, ref, row):
        validated.append(row['request_id'])
        if row['request_id'] == 'hosted:A':
            head_checking.set()
            release_head.wait(5)
            return True
        # B's own reauthorization is revoked; it must be asked, never inherit A's verdict.
        raise RuntimeStoreError('permission_denied')
    monkeypatch.setattr(session_hosted_transport, 'check_remote_hosted_admission', check_remote_hosted_admission)

    async def execute(authority, ref, row):
        executed.append(row['request_id'])
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        head = await _submit(authority, 'hosted:A')
        await _submit(authority, 'hosted:B')
        await asyncio.get_running_loop().run_in_executor(None, head_checking.wait, 5)
        await authority.cancel_queued(ACTOR, REF, head.admission_id)
        release_head.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)

    assert executed == [], 'the successor reached execution without its own hosted validation'
    assert validated == ['hosted:A', 'hosted:B']
    statuses = {r['request_id']: r['status'] for r in list_session_admissions(db, session_id='s', pending_only=False)}
    assert statuses == {'hosted:A': 'terminal', 'hosted:B': 'queued'}


def _wire_turn_agent(authority, generation, agent):
    """Run the real per-turn agent wiring with this authority as the turn's approval owner."""
    from gateway.run_turn_runner import TurnRunner
    ctx = SimpleNamespace(
        progress_callback=None, native_tool_start_callback=None, voice_ack_callback=None,
        _voice_ack_guild=[None], _native_slack_task_cards=False, native_tool_complete_callback=None,
        _step_callback_sync=None, _hooks_ref=SimpleNamespace(loaded_hooks=[]), _status_callback_sync=None,
        _event_callback_sync=None, _status_adapter=None, session_key='route', user_config={},
        _thinking_enabled=False, agent_holder=[None], tools_holder=[None], process_task_id=None,
        process_baseline=None, run_generation=0)
    holder = SimpleNamespace(
        _ctx=ctx, _approval_owner=(authority, 's', generation),
        _runner=SimpleNamespace(_service_tier=None, _consume_pending_turn_sidecar_notes=lambda key: []),
        _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
        _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
        _clarify_callback_sync=lambda *a, **k: None, _notice_callback_sync=lambda *a, **k: None,
        _attach_session_title_callback=lambda agent, ctx: None,
        combined_tool_start_callback=None, combined_tool_complete_callback=None)
    TurnRunner._wire_turn_agent_callbacks(holder, agent, {}, None, None, None, False)


@pytest.mark.asyncio
async def test_stop_before_the_turn_agent_exists_reaches_the_agent_once_it_is_wired(tmp_path, monkeypatch):
    from gateway import session_finite

    db, authority = _authority(tmp_path, monkeypatch)
    cache = {}
    authority.runner._cached_agent_for = cache.get
    agent = SimpleNamespace(interrupted=0)
    agent.interrupt = lambda *a, **k: setattr(agent, 'interrupted', agent.interrupted + 1)
    stopped = asyncio.Event()

    async def execute(authority, ref, row):
        # First turn: the claim is committed and message.start published, but agent
        # construction has not finished. Stop arrives in exactly this window.
        await asyncio.wait_for(stopped.wait(), 5)
        cache['route'] = agent
        _wire_turn_agent(authority, row['generation'], agent)
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'first')
        task = authority.sessions['s'].task
        while authority._handle(REF).execution_state != 'running':
            await asyncio.sleep(0)
        handle = await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        assert handle.execution_state == 'running'
        stopped.set()
        await asyncio.wait_for(task, 5)
    assert agent.interrupted == 1, 'a Stop accepted for the running generation must reach its agent'

    # The latch belongs to that exact generation: the next turn's agent is not interrupted.
    agent.interrupted = 0
    with db:
        await _submit(authority, 'second')
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
    assert agent.interrupted == 0
