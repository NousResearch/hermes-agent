"""The owner runs a session's end-of-conversation lifecycle when its last viewer closes it.

Regression for #134889 LC-1/LC-2/LC-6. Main's classic CLI and TUI ran the memory providers'
``on_session_end`` commit and plugin ``on_session_finalize`` (which closes the Relay scope) in their own
process at exit and ``/new``. Under the gateway the client is a viewer with no memory manager, so
nothing ran until idle eviction; and a close sent while a turn was live left ``ended_at`` unset for good.
"""
import asyncio
import threading
from types import MethodType, SimpleNamespace

import pytest

from tests.waiting import async_wait_until


async def _owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.run_shutdown_session_end import GatewaySessionEndMixin
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    calls = []

    async def finalize(**kwargs):
        calls.append(('finalize', kwargs))
        return []

    async def housekeeping(func, *args):
        return func(*args)

    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', lambda name, **kwargs: calls.append((name, kwargs)))
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None,
                             _evict_cached_agent_at_boundary=lambda key: calls.append(('commit', key)),
                             _finalize_session_off_loop=finalize, _FINALIZE_TIMEOUT_S=10.0,
                             _run_housekeeping_in_executor=housekeeping,
                             _run_release_in_profile_scope=lambda target, args, key: target(*args))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._reset_session_off_loop = MethodType(GatewaySessionEndMixin._reset_session_off_loop, runner)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    return store._db, authority, calls


def _viewer(authority, tmp_path):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {
        'provider': 'local', 'user_id': 'native', 'profile_id': str(tmp_path), 'instance_id': 'owner'})


async def _call(conn, method, **params):
    reply = await conn.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return reply['result']


async def _open(db, authority, tmp_path):
    db.create_session('chat', source='cli', model='fake-model', cwd=str(tmp_path))
    db.append_message('chat', 'user', 'q')
    db.append_message('chat', 'assistant', 'a')
    viewer = _viewer(authority, tmp_path)
    resumed = await _call(viewer, 'session.resume', session_id='chat')
    db.reopen_session('chat')
    return viewer, resumed['subscription_id']


@pytest.mark.asyncio
async def test_last_viewer_close_commits_memory_and_finalizes_on_the_owner(tmp_path, monkeypatch):
    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    other = _viewer(authority, tmp_path)
    second = await _call(other, 'session.resume', session_id='chat')
    route = authority.sessions['chat'].route

    # Another viewer still has the conversation open: nothing ends.
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='new_session')
    assert calls == []
    # A plain detach (a viewer leaving, not closing) never ends it either.
    await _call(other, 'session.detach', session_id='chat', subscription_id=second['subscription_id'])
    assert calls == [] and db.get_session('chat')['end_reason'] is None

    again = await _call(other, 'session.resume', session_id='chat')
    await _call(other, 'session.detach', session_id='chat', subscription_id=again['subscription_id'],
                end_reason='cli_close')
    assert db.get_session('chat')['end_reason'] == 'cli_close'
    assert calls == [('commit', route), ('finalize', {
        'session_id': 'chat', 'platform': 'cli', 'reason': 'cli_close', 'session_key': route})]


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['new_session', 'tui_close'])  # classic CLI /new; Ink /new or resume switch
async def test_a_viewer_new_resets_the_session_on_the_owner_after_finalize(tmp_path, monkeypatch, reason):
    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    route = authority.sessions['chat'].route

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason=reason)

    # Once, after finalize, with the gateway's own /new arguments (the next session is not open yet).
    assert calls == [('commit', route),
                     ('finalize', {'session_id': 'chat', 'platform': 'cli', 'reason': reason, 'session_key': route}),
                     ('on_session_reset', {'session_id': None, 'reason': 'new_session', 'platform': 'cli',
                                           'old_session_id': 'chat', 'new_session_id': None})]


@pytest.mark.asyncio
async def test_close_while_a_turn_is_queued_completes_when_the_fifo_goes_idle(tmp_path, monkeypatch):
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input, cancel_session_input

    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    row = admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                              request_id='r1', payload={'text': 'still running'})

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')
    assert db.get_session('chat')['ended_at'] is None and calls == []

    # The turn ends (here: cancelled) and the drain reaches its idle point.
    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
    authority._schedule(SessionRef(authority.profile_id, 'chat'))
    await async_wait_until(lambda: len(calls) == 2)
    assert db.get_session('chat')['end_reason'] == 'cli_close'
    assert [name for name, _ in calls] == ['commit', 'finalize']



@pytest.mark.asyncio
async def test_a_turn_queued_during_a_slow_finalize_starts_after_it(tmp_path, monkeypatch):
    from gateway import session_finite
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input

    db, authority, _calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    executed, finalizing, release = [], asyncio.Event(), asyncio.Event()

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'done'

    async def slow_finalize(**kwargs):
        finalizing.set()
        await release.wait()  # a plugin's on_session_finalize, up to the finalize timeout
        return []

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    authority.runner._finalize_session_off_loop = slow_finalize
    ref = SessionRef(authority.profile_id, 'chat')
    user = authority.sessions['chat'].source.user_id  # the session's own operator may run turns
    admit_session_input(db, epoch=authority.epoch, principal_id=user, session_id='chat',
                        request_id='r1', payload={'text': 'mid-turn'})
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')
    authority._schedule(ref)
    # The turn ends, the drain's idle point completes the close and blocks in finalize.
    await asyncio.wait_for(finalizing.wait(), 10)

    admit_session_input(db, epoch=authority.epoch, principal_id=user, session_id='chat',
                        request_id='r2', payload={'text': 'automation'})
    authority._schedule(ref)
    assert executed == ['mid-turn']  # it waits for the plugins, as the review notes
    release.set()
    await async_wait_until(lambda: executed == ['mid-turn', 'automation'])


def _gate_busy_check(monkeypatch):
    """Hold every settler inside its off-loop idle check until the test releases it."""
    import gateway.session_viewer_close as viewer_close
    entered, release = [], threading.Event()

    def busy(*args):
        entered.append(args)
        release.wait(timeout=30)
        return False

    monkeypatch.setattr(viewer_close, '_busy', busy)
    return entered, release


@pytest.mark.asyncio
async def test_a_viewer_reattaching_during_the_idle_check_cancels_the_close(tmp_path, monkeypatch):
    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    entered, release = _gate_busy_check(monkeypatch)

    closing = asyncio.create_task(_call(viewer, 'session.detach', session_id='chat',
                                        subscription_id=subscription, end_reason='cli_close'))
    await async_wait_until(lambda: len(entered) == 1)
    await _call(_viewer(authority, tmp_path), 'session.resume', session_id='chat')
    release.set()
    await closing

    # The conversation is being watched again: its agent stays cached, nothing is finalized and
    # the row is not left ended under the attached viewer.
    assert calls == [] and authority.pending_closes == {}
    row = db.get_session('chat')
    assert row['ended_at'] is None and row['end_reason'] is None


@pytest.mark.asyncio
async def test_a_close_stamps_the_row_only_once_it_is_claimed(tmp_path, monkeypatch):
    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    entered, release = _gate_busy_check(monkeypatch)

    closing = asyncio.create_task(_call(viewer, 'session.detach', session_id='chat',
                                        subscription_id=subscription, end_reason='cli_close'))
    await async_wait_until(lambda: len(entered) == 1)
    # Still cancellable: a viewer may yet reattach, so nothing is stamped.
    assert db.get_session('chat')['ended_at'] is None
    release.set()
    await closing

    row = db.get_session('chat')
    assert row['ended_at'] is not None and row['end_reason'] == 'cli_close'
    assert [name for name, _ in calls] == ['commit', 'finalize']


def _gate_stamp(monkeypatch):
    """Hold the stamp's write (past its viewer check) until the test releases it."""
    import gateway.session_acp_lifecycle as lifecycle
    stamp, entered, release = lifecycle.end_idle_local_session, [], threading.Event()

    def gated(*args, **kwargs):
        entered.append(kwargs)
        release.wait(timeout=30)
        return stamp(*args, **kwargs)

    monkeypatch.setattr(lifecycle, 'end_idle_local_session', gated)
    return entered, release


@pytest.mark.asyncio
async def test_a_viewer_reattaching_during_the_stamp_keeps_the_session_live(tmp_path, monkeypatch):
    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    entered, release = _gate_stamp(monkeypatch)

    closing = asyncio.create_task(_call(viewer, 'session.detach', session_id='chat',
                                        subscription_id=subscription, end_reason='cli_close'))
    await async_wait_until(lambda: len(entered) == 1)
    await _call(_viewer(authority, tmp_path), 'session.resume', session_id='chat')
    release.set()
    await closing

    # The row is reopened under the new viewer, and its agent and Relay scope are left alone.
    assert len(authority.sessions['chat'].subscribers) == 1
    assert calls == [] and authority.pending_closes == {}
    assert db.get_session('chat')['ended_at'] is None


@pytest.mark.asyncio
async def test_input_admitted_during_the_stamp_defers_the_close_to_the_idle_point(tmp_path, monkeypatch):
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input, cancel_session_input

    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    entered, release = _gate_stamp(monkeypatch)

    closing = asyncio.create_task(_call(viewer, 'session.detach', session_id='chat',
                                        subscription_id=subscription, end_reason='cli_close'))
    await async_wait_until(lambda: len(entered) == 1)
    row = admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                              request_id='r1', payload={'text': 'automation'})
    release.set()
    await closing

    # The busy FIFO refused the stamp: nothing ran yet and the close waits for the idle point.
    assert calls == [] and 'chat' in authority.pending_closes
    assert db.get_session('chat')['ended_at'] is None

    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
    authority._schedule(SessionRef(authority.profile_id, 'chat'))
    await async_wait_until(lambda: len(calls) == 2)
    assert db.get_session('chat')['end_reason'] == 'cli_close'
    assert [name for name, _ in calls] == ['commit', 'finalize']


@pytest.mark.asyncio
async def test_concurrent_settlers_run_the_close_lifecycle_once(tmp_path, monkeypatch):
    from gateway.session_viewer_close import settle_viewer_close

    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    entered, release = _gate_busy_check(monkeypatch)

    # The detach's own settle and a drain idle point both reach the idle check.
    closing = asyncio.create_task(_call(viewer, 'session.detach', session_id='chat',
                                        subscription_id=subscription, end_reason='cli_close'))
    await async_wait_until(lambda: len(entered) == 1)
    drain_idle = asyncio.create_task(settle_viewer_close(authority, 'chat'))
    await async_wait_until(lambda: len(entered) == 2)
    release.set()
    await asyncio.gather(closing, drain_idle)

    assert [name for name, _ in calls] == ['commit', 'finalize']
    assert db.get_session('chat')['end_reason'] == 'cli_close'


def _stored(db):
    import json
    from gateway.session_viewer_close import CLOSE_PREFIX
    record = db.get_meta(CLOSE_PREFIX + 'chat')
    if record is None:
        return None
    try:
        return json.loads(record)
    except ValueError:
        return {'reason': record}  # a bare-reason record (before operation ids)


async def _restarted_owner(tmp_path, monkeypatch, db):
    """The next gateway process on the same state.db: a new authority whose memory starts empty."""
    db2, authority, calls = await _owner(tmp_path, monkeypatch)
    assert authority.pending_closes == {}
    return db2, authority, calls


@pytest.mark.asyncio
async def test_a_close_pending_at_gateway_stop_completes_after_restart(tmp_path, monkeypatch):
    """#134889 LC-6b, independent review P2-4: Ctrl+D mid-turn, then the gateway stops before the
    turn's idle point. The deferred close lived only in memory, so the row stayed open forever."""
    from gateway.session_local_recovery import recover_local_sessions
    from hermes_state_runtime import admit_session_input, cancel_session_input

    db, authority, _calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    row = admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                              request_id='r1', payload={'text': 'mid-turn'})
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')
    assert db.get_session('chat')['ended_at'] is None and 'chat' in authority.pending_closes
    # The shutdown settles the turn (interrupted) without reaching the drain's idle point.
    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])

    db2, restarted, calls2 = await _restarted_owner(tmp_path, monkeypatch, db)
    recover_local_sessions(restarted, schedule=True)
    await async_wait_until(lambda: len(calls2) == 2, timeout=10)
    assert db2.get_session('chat')['end_reason'] == 'cli_close'
    assert [name for name, _ in calls2] == ['commit', 'finalize']
    assert db2.get_meta('gateway.viewer_close.v1:chat') is None  # settled: nothing left to re-arm


@pytest.mark.asyncio
async def test_a_viewer_reattaching_cancels_the_persisted_close(tmp_path, monkeypatch):
    from gateway.session_local_recovery import recover_local_sessions
    from gateway.session_viewer_close import CLOSE_PREFIX
    from hermes_state_runtime import admit_session_input, cancel_session_input

    db, authority, _calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    row = admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                              request_id='r1', payload={'text': 'mid-turn'})
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')
    assert _stored(db)['reason'] == 'cli_close'

    # Re-attach before settlement (same process): the stored close goes with the in-memory one.
    await _call(_viewer(authority, tmp_path), 'session.resume', session_id='chat')
    assert authority.pending_closes == {} and db.get_meta(CLOSE_PREFIX + 'chat') is None

    # And after a restart (the turn settled meanwhile) nothing re-arms it.
    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
    db2, restarted, calls2 = await _restarted_owner(tmp_path, monkeypatch, db)
    recover_local_sessions(restarted, schedule=True)
    assert restarted.sessions['chat'].task is None  # no queued turn and no close: no drain at all
    assert restarted.pending_closes == {} and calls2 == []
    assert db2.get_session('chat')['ended_at'] is None


def _gate_persist(monkeypatch, reason):
    """Hold *reason*'s close after its real SQLite commit, before its off-loop write returns."""
    import gateway.session_viewer_close as close
    original = close._persist_close
    entered, release = threading.Event(), threading.Event()

    def gated(authority, session_id, record):
        original(authority, session_id, record)
        if reason in record:
            entered.set()
            assert release.wait(10)
    monkeypatch.setattr(close, '_persist_close', gated)
    return entered, release


@pytest.mark.asyncio
async def test_a_cancelled_older_close_keeps_the_newer_close_durable(tmp_path, monkeypatch):
    """#134889 LC-6b, independent review round 2: close A commits, a viewer reattaches (cancelling A)
    and closes again as B; A's late continuation deleted B's stored close by key, so a restart
    before the idle point left the row open for good."""
    from gateway.session_local_recovery import recover_local_sessions
    from hermes_state_runtime import admit_session_input, cancel_session_input

    db, authority, _calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    row = admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                              request_id='r1', payload={'text': 'mid-turn'})
    entered, release = _gate_persist(monkeypatch, 'close_a')
    old = asyncio.create_task(_call(viewer, 'session.detach', session_id='chat',
                                    subscription_id=subscription, end_reason='close_a'))
    await async_wait_until(entered.is_set)
    newcomer = _viewer(authority, tmp_path)
    resumed = await _call(newcomer, 'session.resume', session_id='chat')
    await _call(newcomer, 'session.detach', session_id='chat', subscription_id=resumed['subscription_id'],
                end_reason='close_b')
    stored_b = _stored(db)
    assert stored_b['reason'] == 'close_b'
    release.set()
    await old

    # A's continuation sees it lost and removes only its own record: B's stays, op id and all.
    assert _stored(db) == stored_b
    assert authority.pending_closes['chat'].reason == 'close_b'

    # The old owner's memory is gone; the next authority recovers B and ends the row.
    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
    db2, restarted, calls2 = await _restarted_owner(tmp_path, monkeypatch, db)
    recover_local_sessions(restarted, schedule=True)
    await async_wait_until(lambda: len(calls2) == 2, timeout=10)
    assert db2.get_session('chat')['end_reason'] == 'close_b'
    assert db2.get_session('chat')['ended_at'] is not None
    assert [name for name, _ in calls2] == ['commit', 'finalize'] and _stored(db2) is None


@pytest.mark.asyncio
async def test_the_newer_close_settles_and_the_older_one_leaves_no_trace(tmp_path, monkeypatch):
    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    entered, release = _gate_persist(monkeypatch, 'close_a')
    old = asyncio.create_task(_call(viewer, 'session.detach', session_id='chat',
                                    subscription_id=subscription, end_reason='close_a'))
    await async_wait_until(entered.is_set)
    newcomer = _viewer(authority, tmp_path)
    resumed = await _call(newcomer, 'session.resume', session_id='chat')
    # Idle FIFO: B settles immediately, while A is still parked after its commit.
    await _call(newcomer, 'session.detach', session_id='chat', subscription_id=resumed['subscription_id'],
                end_reason='close_b')
    assert db.get_session('chat')['end_reason'] == 'close_b' and _stored(db) is None
    release.set()
    await old
    assert [name for name, _ in calls] == ['commit', 'finalize']  # once, for B only
    assert db.get_session('chat')['end_reason'] == 'close_b'
    assert authority.pending_closes == {} and _stored(db) is None


@pytest.mark.asyncio
async def test_a_reattach_during_the_close_write_cancels_it_durably(tmp_path, monkeypatch):
    from gateway.session_local_recovery import recover_local_sessions
    from hermes_state_runtime import admit_session_input, cancel_session_input

    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    row = admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                              request_id='r1', payload={'text': 'mid-turn'})
    entered, release = _gate_persist(monkeypatch, 'close_a')
    old = asyncio.create_task(_call(viewer, 'session.detach', session_id='chat',
                                    subscription_id=subscription, end_reason='close_a'))
    await async_wait_until(entered.is_set)
    await _call(_viewer(authority, tmp_path), 'session.resume', session_id='chat')
    release.set()
    await old
    assert authority.pending_closes == {} and _stored(db) is None and calls == []

    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
    db2, restarted, calls2 = await _restarted_owner(tmp_path, monkeypatch, db)
    recover_local_sessions(restarted, schedule=True)
    assert restarted.pending_closes == {} and calls2 == []
    assert db2.get_session('chat')['ended_at'] is None


@pytest.mark.asyncio
async def test_a_stale_owner_cannot_delete_a_stored_close(tmp_path, monkeypatch):
    """Compare-and-delete is epoch-fenced: an owner that lost the store leaves its successor's close."""
    import json
    from gateway.session_viewer_close import CLOSE_PREFIX, forget_persisted_close

    db, authority, _calls = await _owner(tmp_path, monkeypatch)
    record = json.dumps({'op': 'x', 'reason': 'cli_close'})
    db.set_meta(CLOSE_PREFIX + 'chat', record)
    stale = SimpleNamespace(db=db, epoch=authority.epoch - 1)
    assert forget_persisted_close(stale, 'chat', record) is False
    assert forget_persisted_close(authority, 'chat', 'another close') is False
    assert db.get_meta(CLOSE_PREFIX + 'chat') == record
    assert forget_persisted_close(authority, 'chat', record) is True
    assert db.get_meta(CLOSE_PREFIX + 'chat') is None


async def _hard_killed_mid_turn(tmp_path, monkeypatch):
    """Ctrl+D mid-turn, then the owner dies with no shutdown code (SIGKILL): the started admission
    and the stored close are all that is left; the next owner recovers the admission ``unknown``."""
    from hermes_state_runtime import admit_session_input, claim_session_input

    db, authority, _calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                        request_id='r1', payload={'text': 'mid-turn'})
    row = claim_session_input(db, epoch=authority.epoch, session_id='chat')
    assert row['status'] == 'started'
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')
    assert _stored(db)['reason'] == 'cli_close' and db.get_session('chat')['ended_at'] is None
    db2, restarted, calls2 = await _restarted_owner(tmp_path, monkeypatch, db)
    from hermes_state_runtime import get_session_admission
    assert get_session_admission(db2, admission_id=row['admission_id'])['status'] == 'unknown'
    return db2, restarted, calls2, row


@pytest.mark.asyncio
async def test_a_close_stored_before_a_hard_kill_ends_the_row_past_the_unknown_turn(tmp_path, monkeypatch):
    """#134889 LC-6b, @alt-glitch warning 1 on #140489: after SIGKILL the in-flight turn recovers
    ``unknown``, the drain pauses on ``unknown_execution`` before the idle point, and the stored close
    never applied: the row stayed open forever. Nobody is left to Discard the turn (the viewer closed
    the session), so the owner resolves it as interrupted (never replayed) and the close completes."""
    from gateway.session_local_recovery import recover_local_sessions
    from hermes_state_runtime import get_session_admission

    db2, restarted, calls2, row = await _hard_killed_mid_turn(tmp_path, monkeypatch)
    recover_local_sessions(restarted, schedule=True)
    await async_wait_until(lambda: len(calls2) == 2, timeout=10)
    assert db2.get_session('chat')['end_reason'] == 'cli_close'
    assert db2.get_session('chat')['ended_at'] is not None
    settled = get_session_admission(db2, admission_id=row['admission_id'])
    assert (settled['status'], settled['outcome']) == ('terminal', 'interrupted')
    assert [name for name, _ in calls2] == ['commit', 'finalize'] and _stored(db2) is None


@pytest.mark.asyncio
async def test_a_viewer_back_before_the_unknown_is_resolved_keeps_the_turn_for_its_own_discard(tmp_path, monkeypatch):
    """The owner only resolves an unknown turn for a close nobody cancelled: a viewer that attached
    after the restart cancels the close and decides about the turn itself (Discard)."""
    from hermes_state_runtime import get_session_admission
    from gateway.session_viewer_close import rearm_persisted_closes

    db2, restarted, calls2, row = await _hard_killed_mid_turn(tmp_path, monkeypatch)
    from gateway.session_local_recovery import restore_local_session
    restore_local_session(restarted, 'chat')
    await _call(_viewer(restarted, tmp_path), 'session.resume', session_id='chat')
    assert rearm_persisted_closes(restarted) == [] and restarted.pending_closes == {}
    from gateway.session_contract import SessionRef
    restarted._schedule(SessionRef(restarted.profile_id, 'chat'))
    await asyncio.wait_for(restarted.sessions['chat'].task, 10)  # the drain pauses on the unknown turn
    assert get_session_admission(db2, admission_id=row['admission_id'])['status'] == 'unknown'
    assert db2.get_session('chat')['ended_at'] is None and calls2 == [] and _stored(db2) is None


@pytest.mark.asyncio
async def test_attaching_a_session_restored_on_demand_cancels_its_stored_close(tmp_path, monkeypatch):
    """Why attach reads the store instead of trusting ``pending_closes.pop``: a stored close is only
    re-armed by the scheduled boot recovery, for sessions it could restore. One mounted any other
    way (here the authority's own unscheduled mount; in production a session paused at boot and
    restored by ``authorize`` on first use) has a stored close that never reached memory. Left in
    place, the next boot would close a session its viewer came back to."""
    import json
    from gateway.session_viewer_close import CLOSE_PREFIX

    db, authority, _calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription)
    db.set_meta(CLOSE_PREFIX + 'chat', json.dumps({'op': 'x', 'reason': 'cli_close'}))
    db2, restarted, calls2 = await _restarted_owner(tmp_path, monkeypatch, db)
    assert restarted.pending_closes == {} and db2.get_meta(CLOSE_PREFIX + 'chat') is not None
    await _call(_viewer(restarted, tmp_path), 'session.resume', session_id='chat')
    assert db2.get_meta(CLOSE_PREFIX + 'chat') is None and calls2 == []
