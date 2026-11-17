"""Local sessions nobody closed end on the gateway, and only open ones come back at boot (#143547).

LC-35: a killed terminal or dropped socket detached with no ``end_reason``, and nothing ever ended
the row. LC-37: a gateway stop with a viewer attached left the row open for good. LC-36: every
receipt, ended or not, was restored into the live registry on each boot. The turn itself keeps
running (#106742); the row ends once it settles, through the epoch-fenced viewer-close path.
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


async def _owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    finalized = []

    async def finalize(**kwargs):
        finalized.append(kwargs['reason'])
        return []

    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _finalize_session_off_loop=finalize)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    return authority, finalized


def _viewer(authority):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})


async def _attach(authority, sid):
    viewer = _viewer(authority)
    reply = await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
    assert 'result' in reply, reply
    return viewer, reply['result']


def _create(authority, text='earlier turn'):
    from gateway.session_local import create_local_session
    ref = create_local_session(authority, _viewer(authority).actor, {
        'request_id': text, 'source': 'gui', 'cwd': str(Path(authority.db.db_path).parent), 'model': 'm',
        'toolsets': []})
    authority.db.append_message(ref.session_id, 'user', text)
    return ref


def _capture_timers(monkeypatch):
    """The grace timer's (delay, callback), fired by the test instead of after a real wait."""
    import gateway.session_viewer_close as viewer_close
    armed = []

    def call_later(loop, delay, callback):
        armed.append((delay, callback))
        return loop.call_later(3600, callback)
    monkeypatch.setattr(viewer_close, '_call_later', call_later, raising=False)
    return armed


def _stale_rows(monkeypatch):
    """Every restored row already idle past the legacy TTL, so the startup arm uses the plain grace."""
    import gateway.session_viewer_close as viewer_close
    monkeypatch.setattr(viewer_close, 'STARTUP_IDLE_S', 0.0, raising=False)


def _stored_close(db, sid):
    from gateway.session_viewer_close import CLOSE_PREFIX
    return db.get_meta(CLOSE_PREFIX + sid)


@pytest.mark.asyncio
async def test_a_dropped_viewer_ends_the_session_after_its_turn_settles(tmp_path, monkeypatch):
    import gateway.session_finite as finite
    import tools.async_delegation as delegation
    from hermes_state_runtime import admit_session_input, list_session_admissions

    armed = _capture_timers(monkeypatch)
    authority, finalized = await _owner(tmp_path, monkeypatch)
    db, ref = authority.db, _create(authority)
    sid, user = ref.session_id, authority.sessions[ref.session_id].source.user_id
    viewer, _ = await _attach(authority, sid)
    started, release, executed, refused = asyncio.Event(), asyncio.Event(), [], []

    async def execute(authority, ref, row):
        started.set()
        await release.wait()
        executed.append(row['payload']['text'])
        return 'done'
    monkeypatch.setattr(finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(delegation, 'refuse_for_session', lambda *a, **k: refused.append(a))
    monkeypatch.setattr(delegation, 'interrupt_for_session', lambda *a, **k: refused.append(k))
    admit_session_input(db, epoch=authority.epoch, principal_id=user, session_id=sid,
                        request_id='r1', payload={'text': 'long turn'})
    authority._schedule(ref)
    await asyncio.wait_for(started.wait(), 10)

    await viewer.close()  # the terminal is killed: no end_reason
    assert [delay for delay, _ in armed] == [300.0]
    armed[0][1]()
    await async_wait_until(lambda: _stored_close(db, sid) is not None)
    # The turn keeps running and nothing it started is refused; the row stays open meanwhile.
    assert db.get_session(sid)['ended_at'] is None and executed == [] and refused == []

    release.set()
    await async_wait_until(lambda: finalized == ['ws_orphan_reap'])  # the end hooks run after the stamp
    assert db.get_session(sid)['end_reason'] == 'ws_orphan_reap'
    assert executed == ['long turn'] and _stored_close(db, sid) is None

    admit_session_input(db, epoch=authority.epoch, principal_id=user, session_id=sid,
                        request_id='r2', payload={'text': 'next turn'})
    authority._schedule(ref)
    await async_wait_until(lambda: executed == ['long turn', 'next turn'])
    await async_wait_until(lambda: not list_session_admissions(db, session_id=sid))
    assert db.get_session(sid)['ended_at'] is None  # reopened on the same id


@pytest.mark.asyncio
async def test_a_viewer_back_within_the_grace_keeps_the_session_open(tmp_path, monkeypatch):
    armed = _capture_timers(monkeypatch)
    authority, finalized = await _owner(tmp_path, monkeypatch)
    sid = _create(authority).session_id
    viewer, _ = await _attach(authority, sid)
    await viewer.close()
    await _attach(authority, sid)
    assert authority.orphan_closes == {}

    armed[0][1]()  # a timer that fired anyway finds the viewer back
    await asyncio.sleep(0)
    assert authority.pending_closes == {} and _stored_close(authority.db, sid) is None
    assert authority.db.get_session(sid)['ended_at'] is None and finalized == []


@pytest.mark.asyncio
async def test_an_orphan_end_leaves_an_unknown_turn_for_its_user(tmp_path, monkeypatch):
    """Nobody closed the session, so nobody decided to drop the turn a hard-killed owner lost."""
    from hermes_state_runtime import admit_session_input, claim_session_input, get_session_admission

    armed = _capture_timers(monkeypatch)
    first, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(first).session_id
    admit_session_input(first.db, epoch=first.epoch, principal_id=first.sessions[sid].source.user_id,
                        session_id=sid, request_id='r1', payload={'text': 'mid-turn'})
    row = claim_session_input(first.db, epoch=first.epoch, session_id=sid)
    authority, finalized = await _owner(tmp_path, monkeypatch)  # SIGKILL, then the next owner
    assert get_session_admission(authority.db, admission_id=row['admission_id'])['status'] == 'unknown'

    viewer, _ = await _attach(authority, sid)
    await viewer.close()
    armed[-1][1]()
    from gateway.session_runtime_workers import mutation_tasks
    await asyncio.gather(*mutation_tasks(authority))  # the close is stored and its settle has run
    assert _stored_close(authority.db, sid) is not None
    assert get_session_admission(authority.db, admission_id=row['admission_id'])['status'] == 'unknown'
    assert authority.db.get_session(sid)['ended_at'] is None and finalized == []


@pytest.mark.asyncio
async def test_boot_restores_only_open_sessions_and_an_ended_one_on_first_touch(tmp_path, monkeypatch):
    from hermes_state_local import end_idle_local_session

    first, _ = await _owner(tmp_path, monkeypatch)
    ended, live = _create(first, 'finished').session_id, _create(first, 'ongoing').session_id
    assert end_idle_local_session(first.db, epoch=first.epoch, session_id=ended, target_id=ended, reason='cli_close')

    authority, _ = await _owner(tmp_path, monkeypatch)
    assert live in authority.sessions and ended not in authority.sessions
    _, resumed = await _attach(authority, ended)
    assert [message['content'] for message in resumed['messages']] == ['finished']
    assert ended in authority.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize('producer', ['delegation', 'process'])
async def test_work_finishing_after_close_and_restart_is_noted_on_the_ended_session(tmp_path, monkeypatch, producer):
    """Boot leaves ended sessions unrestored, but not one whose background work has not reported yet."""
    from hermes_state_local import end_idle_local_session
    from tests.gateway.test_completion_admission import pending
    from tests.gateway.test_session_end_delegations import _local
    from tests.gateway.test_unviewed_completion_notes import _delegation_note, _process_event, _restart, _target
    import tools.async_delegation as delegation

    delegation._reset_for_tests()
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
    if producer == 'delegation':
        evt = dict(pending(route, 'deleg_late'), parent_session_id=target)
    else:
        evt = _process_event(authority, ref, target, 'proc_late')
    assert end_idle_local_session(authority.db, epoch=authority.epoch, session_id=ref.session_id,
                                  target_id=target, reason='cli_close')
    runner, authority = await _restart(tmp_path, monkeypatch, authority)
    try:
        if producer == 'delegation':
            assert await runner._deliver_async_delegation_group([evt]) is True
        else:
            assert await runner._deliver_completion_notification('[finished] RESULT_proc_late', evt) is True
        await authority._drain(ref)
        assert len(_delegation_note(authority, target, evt.get('delegation_id', 'proc_late'))) == 1
        assert authority.db.get_session(target)['end_reason'] == 'cli_close'
    finally:
        authority.db.close()
        delegation._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize('viewer_returns', [False, True])
async def test_a_session_open_when_the_gateway_stopped_ends_after_boot(tmp_path, monkeypatch, viewer_returns):
    from gateway.session_local_recovery import recover_local_sessions

    armed = _capture_timers(monkeypatch)
    _stale_rows(monkeypatch)
    first, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(first).session_id
    await _attach(first, sid)  # the gateway stops with this viewer attached: no detach ever runs

    authority, finalized = await _owner(tmp_path, monkeypatch)
    recover_local_sessions(authority, schedule=True)
    assert [delay for delay, _ in armed] == [300.0]
    if viewer_returns:
        await _attach(authority, sid)
    armed[0][1]()
    if viewer_returns:
        await asyncio.sleep(0)
        assert authority.db.get_session(sid)['ended_at'] is None and finalized == []
    else:
        await async_wait_until(lambda: finalized == ['startup_orphan_reap'])
        assert authority.db.get_session(sid)['end_reason'] == 'startup_orphan_reap'


@pytest.mark.asyncio
async def test_a_session_active_before_the_restart_is_kept_for_its_idle_classic_cli(tmp_path, monkeypatch):
    """An idle classic CLI redials only on its next call: boot must not end its session in minutes."""
    from gateway.session_local_recovery import recover_local_sessions
    from gateway.session_viewer_close import STARTUP_IDLE_S

    armed = _capture_timers(monkeypatch)
    first, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(first).session_id
    await _attach(first, sid)

    authority, _ = await _owner(tmp_path, monkeypatch)
    recover_local_sessions(authority, schedule=True)
    [(delay, _)] = armed
    assert STARTUP_IDLE_S - 60 < delay <= STARTUP_IDLE_S


def test_the_startup_idle_probe_does_not_nest_its_read_context(tmp_path, monkeypatch):
    """``_idle_s`` runs on the boot thread, and a SessionDB read must not open inside another.

    A read that lands on the degraded path (no WAL: DELETE journal / NFS fallback) waits on the
    writer's non-reentrant lock. Resolving the session id inside ``_idle_s``'s own read context
    re-entered ``_read_ctx`` and waited on the lock this thread already held, so boot never
    reached the control socket and the liveness watchdog killed the gateway on every restart.
    Re-entry fails here instead of wedging the runner on that deadlock.
    """
    from contextlib import contextmanager
    from gateway.session_viewer_close import _idle_s
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session(session_id='s1', source='cli', model='m')
    original = db._read_ctx
    depth = 0

    @contextmanager
    def guarded_read_ctx():
        nonlocal depth
        assert depth == 0, '_idle_s opened a read context inside another read context'
        depth += 1
        try:
            with original() as conn:
                yield conn
        finally:
            depth -= 1

    monkeypatch.setattr(db, '_read_ctx', guarded_read_ctx)
    try:
        assert _idle_s(SimpleNamespace(db=db), 's1') >= 0
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('config, startup, detach', [
    ('dashboard:\n  ws_orphan_reap_grace_s: 0\n', [], []),
    ('dashboard:\n  startup_orphan_sweep: false\n', [], [300.0]),
    ('dashboard:\n  ws_orphan_reap_grace_s: 0\n  startup_orphan_sweep: false\n'
     'sessions:\n  orphan_end_grace_s: 60\n', [60.0], [60.0]),
])
async def test_the_legacy_reap_opt_outs_hold_until_the_new_key_is_set(tmp_path, monkeypatch, config, startup, detach):
    from gateway.session_local_recovery import recover_local_sessions

    armed = _capture_timers(monkeypatch)
    _stale_rows(monkeypatch)
    first, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(first).session_id
    await _attach(first, sid)
    (Path(first.db.db_path).parent / 'config.yaml').write_text(config)

    authority, _ = await _owner(tmp_path, monkeypatch)
    recover_local_sessions(authority, schedule=True)
    assert [delay for delay, _ in armed] == startup
    viewer, _ = await _attach(authority, sid)
    await viewer.close()
    assert [delay for delay, _ in armed[len(startup):]] == detach


@pytest.mark.asyncio
async def test_a_session_left_with_queued_input_ends_after_boot_once_it_drains(tmp_path, monkeypatch):
    import gateway.session_finite as finite
    from gateway.session_local_recovery import recover_local_sessions
    from hermes_state_runtime import admit_session_input, list_session_admissions

    armed = _capture_timers(monkeypatch)
    _stale_rows(monkeypatch)
    first, _ = await _owner(tmp_path, monkeypatch)
    sid = _create(first).session_id
    admit_session_input(first.db, epoch=first.epoch, principal_id=first.sessions[sid].source.user_id,
                        session_id=sid, request_id='r1', payload={'text': 'queued'})
    executed = []

    async def execute(authority, ref, row):
        executed.append(row['payload']['text'])
        return 'done'
    monkeypatch.setattr(finite, 'execute_finite_admission', execute)

    authority, finalized = await _owner(tmp_path, monkeypatch)  # the gateway stopped before running it
    recover_local_sessions(authority, schedule=True)
    await async_wait_until(lambda: executed == ['queued']
                           and not list_session_admissions(authority.db, session_id=sid))
    assert [delay for delay, _ in armed] == [300.0]
    armed[0][1]()
    await async_wait_until(lambda: finalized == ['startup_orphan_reap'])
    assert authority.db.get_session(sid)['end_reason'] == 'startup_orphan_reap'


@pytest.mark.asyncio
async def test_a_dropped_viewer_expands_the_grace_against_its_own_profile(tmp_path, monkeypatch):
    """The socket's close detaches outside the RPC's owner scope; the launch env must not answer."""
    armed = _capture_timers(monkeypatch)
    authority, _ = await _owner(tmp_path, monkeypatch)
    home = Path(authority.db.db_path).parent
    (home / 'config.yaml').write_text('sessions:\n  orphan_end_grace_s: ${ORPHAN_GRACE}\n')
    (home / '.env').write_text('ORPHAN_GRACE=7\n')
    monkeypatch.setenv('ORPHAN_GRACE', '900')
    viewer, _ = await _attach(authority, _create(authority).session_id)
    await viewer.close()
    assert [delay for delay, _ in armed] == [7.0]


@pytest.mark.asyncio
async def test_a_sibling_torn_down_with_the_shared_socket_reads_its_own_grace(tmp_path, monkeypatch):
    """A shared socket's close tears its sibling connections down under the primary profile's scope;
    each still detaches as its own owner, so the sibling's grace expands against its own ``.env``."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import get_hermes_home

    armed = _capture_timers(monkeypatch)
    home, primary = get_hermes_home(), tmp_path / 'primary'
    primary.mkdir()
    (home / 'config.yaml').write_text('sessions:\n  orphan_end_grace_s: ${ORPHAN_GRACE}\n')
    (home / '.env').write_text('ORPHAN_GRACE=900\n')
    (primary / '.env').write_text('ORPHAN_GRACE=1\n')
    sibling, _ = await _owner(home, monkeypatch)  # profile id, store and config share one home
    sibling.runner.config = SimpleNamespace(multiplex_profiles=True)
    viewer, _ = await _attach(sibling, _create(sibling).session_id)

    token = set_secret_scope(build_profile_secret_scope(primary), profile_home=str(primary))
    try:
        await viewer.close()  # reached from the primary connection's close: its scope is bound
    finally:
        reset_secret_scope(token)
    assert [delay for delay, _ in armed] == [900.0]


@pytest.mark.asyncio
async def test_a_zero_grace_never_ends_a_detached_session(tmp_path, monkeypatch):
    armed = _capture_timers(monkeypatch)
    authority, _ = await _owner(tmp_path, monkeypatch)
    (Path(authority.db.db_path).parent / 'config.yaml').write_text('sessions:\n  orphan_end_grace_s: 0\n')
    sid = _create(authority).session_id
    viewer, _ = await _attach(authority, sid)
    await viewer.close()
    assert armed == [] and authority.orphan_closes == {}
    assert authority.db.get_session(sid)['ended_at'] is None


@pytest.mark.asyncio
async def test_a_session_the_tui_keeps_live_is_only_orphaned_when_its_connection_drops(tmp_path, monkeypatch):
    armed = _capture_timers(monkeypatch)
    authority, _ = await _owner(tmp_path, monkeypatch)
    kept, front = _create(authority, 'kept live').session_id, _create(authority, 'in front').session_id
    tui = _viewer(authority)
    held = await tui.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': kept}})
    await tui.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': front}})
    # `/sessions new` or the live switcher: the previous session is handed back with a plain detach.
    out = await tui.dispatch({'id': 3, 'method': 'session.detach', 'params': {
        'session_id': kept, 'subscription_id': held['result']['subscription_id']}})
    assert out['result']['detached'] is True
    assert armed == [] and authority.orphan_closes == {}
    listed = await tui.dispatch({'id': 4, 'method': 'session.list', 'params': {}})
    assert {row['id'] for row in listed['result']['sessions']} == {kept, front}

    await tui.close()  # the terminal is killed while it keeps `kept` in the background
    assert sorted(authority.orphan_closes) == sorted([kept, front])
    assert [delay for delay, _ in armed] == [300.0, 300.0]


@pytest.mark.asyncio
async def test_a_watch_match_after_close_and_restart_is_noted_on_the_ended_session(tmp_path, monkeypatch):
    import queue
    from hermes_state_local import end_idle_local_session
    from tests.gateway.test_session_end_delegations import _local
    from tests.gateway.test_unviewed_completion_notes import _delegation_note, _restart, _target

    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
    evt = {'type': 'watch_match', 'session_id': 'proc_watch', 'event_id': 'evt-1', 'command': 'serve',
           'pattern': 'READY', 'output': 'READY_proc_watch', 'session_key': route, 'parent_session_id': target}
    assert end_idle_local_session(authority.db, epoch=authority.epoch, session_id=ref.session_id,
                                  target_id=target, reason='cli_close')
    runner, authority = await _restart(tmp_path, monkeypatch, authority)
    try:
        completion_queue = queue.Queue()
        completion_queue.put(evt)
        await runner._drain_watch_notifications(completion_queue)
        assert completion_queue.empty()  # admitted, not requeued
        await authority._drain(ref)
        assert len(_delegation_note(authority, target, 'READY_proc_watch')) == 1
        assert authority.db.get_session(target)['end_reason'] == 'cli_close'
    finally:
        authority.db.close()
