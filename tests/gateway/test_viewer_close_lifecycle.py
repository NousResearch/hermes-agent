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
