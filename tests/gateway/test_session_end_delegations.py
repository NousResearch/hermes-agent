"""``exit`` and ``/new`` end that session's own async delegations in the gateway; a completion for a
local session no viewer is watching is written into it as a note, not an unattended turn.

Regression for #134889 LC-7. Main's classic CLI interrupted its async delegations at exit and main's
TUI ended a session's own delegations when it closed it; under the gateway the client is a viewer, so
the delegations kept running after it left and their completions (and background-process notices)
started turns nobody watched.
"""
import json
from types import MethodType, SimpleNamespace

import pytest

from tools import async_delegation as delegation


@pytest.fixture(autouse=True)
def _no_delegations():
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


async def _owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.run_shutdown_session_end import GatewaySessionEndMixin
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    monkeypatch.setattr('hermes_cli.lifecycle.invoke_hook', lambda name, **kwargs: None)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())

    async def finalize(**kwargs):
        return []

    async def housekeeping(func, *args):
        return func(*args)

    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None,
                             _evict_cached_agent_at_boundary=lambda key: None,
                             _finalize_session_off_loop=finalize, _FINALIZE_TIMEOUT_S=10.0,
                             _run_housekeeping_in_executor=housekeeping,
                             _run_release_in_profile_scope=lambda target, args, key: target(*args))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._reset_session_off_loop = MethodType(GatewaySessionEndMixin._reset_session_off_loop, runner)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    return store._db, authority


async def _call(conn, method, **params):
    reply = await conn.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return reply['result']


async def _open(db, authority, tmp_path, session_id):
    from gateway.session_controls import AuthorityConnection
    db.create_session(session_id, source='cli', model='fake-model', cwd=str(tmp_path))
    db.append_message(session_id, 'user', 'q')
    viewer = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {
        'provider': 'local', 'user_id': 'native', 'profile_id': str(tmp_path), 'instance_id': 'owner'})
    resumed = await _call(viewer, 'session.resume', session_id=session_id)
    db.reopen_session(session_id)
    return viewer, resumed['subscription_id']


def _running(stopped, delegation_id, *, parent='', route=''):
    """A live async delegation as ``dispatch_async_delegation`` registers it."""
    delegation._records[delegation_id] = {
        'delegation_id': delegation_id, 'status': 'running', 'parent_session_id': parent,
        'session_key': route, 'interrupt_fn': lambda: stopped.append(delegation_id)}


# Classic CLI exit and /new; Ink /new (or a resume switch) and exit.
@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['cli_close', 'new_session', 'tui_close', 'tui_shutdown'])
async def test_exit_and_new_cancel_only_that_sessions_delegations(tmp_path, monkeypatch, reason):
    db, authority = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    await _open(db, authority, tmp_path, 'other')
    stopped = []
    _running(stopped, 'deleg_parent', parent='chat')
    _running(stopped, 'deleg_route', route=authority.sessions['chat'].route)
    _running(stopped, 'deleg_other', parent='other', route=authority.sessions['other'].route)

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason=reason)

    assert sorted(stopped) == ['deleg_parent', 'deleg_route']


@pytest.mark.asyncio
async def test_leaving_without_closing_cancels_nothing(tmp_path, monkeypatch):
    """Desktop closing or a dropped connection detaches without an end reason; another viewer still
    watching keeps the session's work alive even when one viewer exits."""
    db, authority = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    stopped = []
    _running(stopped, 'deleg_parent', parent='chat')

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription)
    assert stopped == []

    first = await _call(viewer, 'session.resume', session_id='chat')
    second, _ = await _open(db, authority, tmp_path, 'unrelated')
    await _call(second, 'session.resume', session_id='chat')
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=first['subscription_id'],
                end_reason='cli_close')
    assert stopped == []


@pytest.mark.asyncio
async def test_viewer_reattaching_during_detach_cleanup_keeps_delegations(tmp_path, monkeypatch):
    """The last viewer's detach awaits the editor-MCP release off the loop; a viewer that attaches
    meanwhile keeps the session's work: the viewer check sits right before the interruption."""
    import asyncio
    import threading
    from gateway import session_local_mcp
    db, authority = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    stopped = []
    _running(stopped, 'deleg_parent', parent='chat')
    entered, release = threading.Event(), threading.Event()

    def suspended(_authority, _session_ids):
        entered.set()
        assert release.wait(10)
    monkeypatch.setattr(session_local_mcp, 'release_editor_mcp', suspended)
    detach = asyncio.ensure_future(_call(viewer, 'session.detach', session_id='chat', subscription_id=subscription,
                                         end_reason='cli_close'))
    assert await asyncio.to_thread(entered.wait, 10)
    other, _ = await _open(db, authority, tmp_path, 'unrelated')
    await _call(other, 'session.resume', session_id='chat')
    release.set()
    await detach

    assert stopped == []
    assert 'chat' not in authority.pending_closes
    assert db.get_session('chat')['ended_at'] is None


def _dispatch(route, parent):
    return delegation.dispatch_async_delegation(goal='late', context=None, toolsets=None, role='leaf', model=None,
                                                session_key=route, parent_session_id=parent,
                                                runner=lambda: {'status': 'completed'})


@pytest.mark.asyncio
async def test_delegation_started_after_the_close_is_refused_until_a_viewer_returns(tmp_path, monkeypatch):
    """The close interrupts what is registered; the parent turn can still dispatch more (the close
    waits for an idle FIFO, and a fan-out may be mid-build). Those are refused for that session only,
    until a viewer attaches again."""
    db, authority = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    await _open(db, authority, tmp_path, 'other')
    route = authority.sessions['chat'].route

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')

    assert _dispatch(route, 'chat')['status'] == 'rejected'
    assert _dispatch('', 'chat')['status'] == 'rejected', 'matched by the parent session too'
    assert _dispatch(authority.sessions['other'].route, 'other')['status'] == 'dispatched'
    await _call(viewer, 'session.resume', session_id='chat')
    assert _dispatch(route, 'chat')['status'] == 'dispatched', 'a viewer that came back keeps its work'


@pytest.mark.asyncio
async def test_a_closed_sessions_refusal_goes_with_its_live_state(tmp_path, monkeypatch):
    """A session closed and never resumed keeps refusing its still-running turn's dispatches until
    the owner retires its live state (delete, prune); then the refusal goes too, so closed sessions
    don't accumulate refusals. Another closed session keeps its own."""
    from gateway.session_mutations import retire_live_sessions
    db, authority = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path, 'chat')
    other, other_subscription = await _open(db, authority, tmp_path, 'other')
    route, other_route = authority.sessions['chat'].route, authority.sessions['other'].route
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')
    await _call(other, 'session.detach', session_id='other', subscription_id=other_subscription,
                end_reason='new_session')
    assert _dispatch(route, 'chat')['status'] == 'rejected', 'its turn may still be running'

    retire_live_sessions(authority, ['chat'])

    assert (authority.profile_id, 'chat') not in delegation._refused, 'retired with its live state'
    assert _dispatch(route, 'chat')['status'] == 'dispatched'
    assert _dispatch(other_route, 'other')['status'] == 'rejected', 'other closed sessions keep theirs'
    assert list(delegation._refused) == [(authority.profile_id, 'other')]


# --- completions for a local session with no viewer --------------------------------------------

async def _local(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_state import AsyncSessionDB, SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)  # admitted rows stay queued
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    db = SessionDB(tmp_path / 'state.db')
    runner.session_store._db = db
    runner._session_db = AsyncSessionDB(db)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='lc7', db=db)
    actor = Principal('test-owner', 'default', frozenset({'session:create', 'session:submit', 'session:read'}),
                      'fixture')
    ref = create_local_session(authority, actor, dict(request_id='lc7', source='cli', model='fixture', toolsets=[]))
    return runner, authority, ref, actor


_PROCESS = ('completion', 'proc_lc7', 1700000000.0)
_DELEGATION = ('async_delegation', 'deleg_lc7', '')
_CARDS = {'completion': 'process_complete', 'async_delegation': 'async_delegation_complete'}


async def _deliver(runner, authority, ref, producer):
    """What ``_inject_watch_notification`` admits for a finished background process or delegation."""
    from gateway.platforms.event import MessageEvent
    identity = json.dumps(producer, separators=(',', ':'))
    live = authority.sessions[ref.session_id]
    entry = runner.session_store.lookup_by_session_key(live.route)
    event = MessageEvent(text=f'[finished] RESULT_{producer[1]}', source=live.source, internal=True, metadata={
        'gateway_session_key': live.route, 'gateway_session_id': entry.session_id,
        'display_kind': _CARDS[producer[0]], 'display_text': f'Finished: {producer[1]}'})
    await authority.admit_automation(runner._adapter_for_source(live.source), event, identity)
    return entry.session_id


def _admissions(authority, ref):
    from hermes_state_runtime import list_session_admissions
    return [r for r in list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
            if r['principal_id'].startswith('automation:')]


def _notes(authority, target):
    return [m for m in authority.db.get_messages(target) if 'RESULT_' in str(m.get('content'))]


def _record_turns(runner):
    """The turns the drain actually executes, by their input text (no model call)."""
    seen = []

    async def handle(event):
        seen.append(event.text)
        return 'ok'
    runner._handle_message = handle
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize('producer', [_PROCESS, _DELEGATION], ids=['process', 'delegation'])
async def test_unviewed_local_completion_is_a_note_not_a_turn(tmp_path, monkeypatch, producer):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        async def no_turn(event):
            raise AssertionError('no viewer: the completion must not start a turn')
        runner._handle_message = no_turn
        target = await _deliver(runner, authority, ref, producer)
        await authority._drain(ref)
        notes = _notes(authority, target)
        assert len(notes) == 1, notes
        assert notes[0]['role'] == 'user' and notes[0]['display_kind'] == _CARDS[producer[0]]
        assert notes[0]['display_metadata']['display_text'] == f'Finished: {producer[1]}'
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']

        # A producer retry (lost ACK) is the same settled admission, even once a viewer is back:
        # a completion is a note or a turn, never both.
        await _deliver(runner, authority, ref, producer)
        authority.sessions[ref.session_id].subscribers['back'] = _actor
        await _deliver(runner, authority, ref, producer)
        await authority._drain(ref)
        assert len(_notes(authority, target)) == 1 and len(_admissions(authority, ref)) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_viewed_local_completion_still_starts_its_turn(tmp_path, monkeypatch):
    runner, authority, ref, actor = await _local(tmp_path, monkeypatch)
    try:
        authority.sessions[ref.session_id].subscribers['watching'] = actor
        target = await _deliver(runner, authority, ref, _PROCESS)
        assert [r['status'] for r in _admissions(authority, ref)] == ['queued']
        seen = _record_turns(runner)
        await authority._drain(ref)
        assert seen == [f'[finished] RESULT_{_PROCESS[1]}'], 'a watched completion runs as its turn'
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
        assert _notes(authority, target) == []
        # Its retry after the viewer left is the same settled admission: no second turn, no note.
        authority.sessions[ref.session_id].subscribers.clear()
        await _deliver(runner, authority, ref, _PROCESS)
        await authority._drain(ref)
        assert len(seen) == 1 and len(_admissions(authority, ref)) == 1 and _notes(authority, target) == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_messaging_completion_still_starts_its_turn(tmp_path, monkeypatch):
    """A messaging chat has no viewers at all: its completion keeps admitting a turn, as on main."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter
    from tests.gateway.test_completion_admission import pending

    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='lc7-messaging')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    human = MessageEvent(text='human', source=source, message_id='human')
    with native_callback(runner, human, get_hermes_home()):
        await authority.admit_native(human)
    key = runner.session_store._generate_session_key(source)
    sid = runner.session_store.peek_session_id(key)
    assert await runner._deliver_async_delegation_group([dict(pending(key, 'deleg_chat'), parent_session_id=sid)])
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert [r['principal_id'].startswith('automation:') for r in rows] == [False, True], rows
    seen = _record_turns(runner)
    from gateway.session_contract import SessionRef
    await authority._drain(SessionRef(authority.profile_id, sid))
    assert seen[0] == 'human' and len(seen) == 2 and 'deleg_chat' in seen[1], 'the completion ran as a turn'
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert [r['outcome'] for r in rows] == ['completed', 'completed']
    assert not [m for m in authority.db.get_messages(sid) if 'deleg_chat' in str(m.get('content'))]


@pytest.mark.asyncio
async def test_other_local_automation_is_unchanged(tmp_path, monkeypatch):
    """Only completions become notes: a /loop tick or Kanban wake keeps its turn with no viewer."""
    from gateway.session_automation import admit_route_wake
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        live = authority.sessions[ref.session_id]
        assert await admit_route_wake(runner, runner._adapter_for_source(live.source), live.source,
                                      'TICK', 'loop-tick')
        assert [r['status'] for r in _admissions(authority, ref)] == ['queued']
        assert not live.subscribers
        seen = _record_turns(runner)
        await authority._drain(ref)
        assert seen == ['TICK'], 'no viewer, and the wake still runs as its turn'
        assert [r['outcome'] for r in _admissions(authority, ref)] == ['completed']
        target = runner.session_store.lookup_by_session_key(live.route).session_id
        assert not [m for m in authority.db.get_messages(target) if 'TICK' in str(m.get('content'))]
    finally:
        authority.db.close()
