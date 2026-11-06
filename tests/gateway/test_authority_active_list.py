"""``session.active_list`` is answered by the session authority from its own live turns.

Desktop's silent-turn check (``session-states.ts`` ``onEventSilence``) asks this verb whether a
turn that went 45 s without an event is still running. The authority never registered it, so the
WS fell back to the legacy sidecar's handler, whose ``_sessions`` never holds an authority-owned
session: a foreground tool quiet for 45 s read as a reaped turn, and Desktop painted "Hermes ended
this turn without a reply" over a turn that was still running.
"""
import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    # This checkout sits under the operator's real home. A synthetic turn must not
    # probe its parent manifest while resolving a managed-worker environment.
    monkeypatch.setattr('pm.environments._payload_manifest', lambda root: None)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    started, release = asyncio.Event(), asyncio.Event()

    async def answer(event):
        # A long foreground tool: the turn emits nothing until it returns.
        started.set()
        await release.wait()
        return event.text

    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='live', instance_id='owner')
    store._db.create_session('chat', source='telegram')
    authority.sessions['chat'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='chat'), 'chat')
    connection = AuthorityConnection(authority, object(), {'user_id': 'viewer'})

    async def active(**params):
        reply = await connection.dispatch({'id': 1, 'method': 'session.active_list', 'params': params})
        assert 'result' in reply, reply
        return {row['id']: row for row in reply['result']['sessions']}

    async def submit(text):
        from gateway.session_contract import SessionRef, Submission
        await authority.submit(connection.actor, Submission(text, SessionRef('live', 'chat'), {'text': text}, 'queue'))
        await asyncio.wait_for(started.wait(), 5)

    try:
        yield SimpleNamespace(authority=authority, active=active, submit=submit, release=release)
    finally:
        release.set()
        tasks = [live.task for live in authority.sessions.values() if live.task is not None]
        if tasks:
            await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_a_running_turn_reads_working_until_it_settles(owner):
    assert (await owner.active())['chat']['status'] == 'idle'
    await owner.submit('hi')
    row = (await owner.active(current_session_id='chat'))['chat']
    assert (row['status'], row['session_key'], row['current']) == ('working', 'chat', True), row
    owner.release.set()
    await owner.authority.sessions['chat'].task
    assert (await owner.active())['chat']['status'] == 'idle'


@pytest.mark.asyncio
async def test_a_turn_blocked_on_a_prompt_reads_waiting(owner, monkeypatch):
    await owner.submit('hi')
    controls = owner.authority.sessions['chat'].controls
    monkeypatch.setattr(controls, 'snapshot', lambda session_id, generation: ({'kind': 'approval'},))
    assert (await owner.active())['chat']['status'] == 'waiting'


@pytest.mark.asyncio
@pytest.mark.parametrize('capture', ['session', 'admissions', 'already-claimed'])
@pytest.mark.parametrize('kind,remote', [('clarify', False), ('clarify', True), ('approval', True)],
                         ids=['native-clarify', 'remote-clarify', 'remote-approval'])
async def test_active_list_preserves_current_claim_prompt(owner, monkeypatch, capture, kind, remote):
    """Return actual old storage rows after the owner claims and installs its prompt.

    The session-row case is the independent review's unchanged race schedule. Holding
    admissions too covers a claim after both storage projections have been captured.
    """
    from gateway import session_active_list
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input, claim_session_input

    authority = owner.authority
    db, live = authority.db, authority.sessions['chat']
    ref = SessionRef('live', 'chat')
    queued = admit_session_input(db, epoch=authority.epoch, principal_id='viewer', session_id='chat',
                                 request_id='private-input', payload={'text': 'synthetic'}, intent='queue')
    loop = asyncio.get_running_loop()
    captured, resume, finished = asyncio.Event(), threading.Event(), threading.Event()
    reads, frames, responses = [], [], []
    live.event_stream.observers.add(frames.append)
    task = None
    armed = True

    def held_read(read, *args, **kwargs):
        nonlocal armed
        value = read(*args, **kwargs)
        if armed:
            armed = False
            reads.append(value)
            loop.call_soon_threadsafe(captured.set)
            try:
                assert resume.wait(5), 'counterexample rendezvous timed out'
            finally:
                finished.set()
        return value

    async def start_and_prompt():
        claimed = await asyncio.to_thread(claim_session_input, db, epoch=authority.epoch,
                                          session_id='chat', head=queued['admission_id'])
        authority._stamp_claim(ref, live, claimed)
        # Valid PendingControls shape; avoid importing gateway.run solely for redaction.
        with live.event_stream.lock:
            prompt = {'kind': kind, 'prompt_id': 'private-prompt',
                      'execution_generation': claimed['generation']}
            if kind == 'clarify':
                prompt.update(question='synthetic', choices=[], multi_select=False)
            else:
                prompt.update(command='synthetic', description='', choices=['once', 'deny'])
            live.controls.pending['private-prompt'] = (SimpleNamespace(event=threading.Event()), prompt)
            if remote:
                live.controls.remote_responders['private-prompt'] = lambda *args: responses.append(args)
        return claimed, prompt

    try:
        if capture == 'already-claimed':
            claimed, prompt = await start_and_prompt()
            rows = await owner.active()
        else:
            if capture == 'session':
                read = db.get_session
                monkeypatch.setattr(db, 'get_session', lambda *a, **kw: held_read(read, *a, **kw))
            else:
                read = session_active_list.list_session_admissions
                monkeypatch.setattr(session_active_list, 'list_session_admissions',
                                    lambda *a, **kw: held_read(read, *a, **kw))
            task = asyncio.create_task(owner.active())
            await asyncio.wait_for(captured.wait(), 5)
            claimed, prompt = await start_and_prompt()
            assert live.controls.pending['private-prompt'][1] == prompt
            resume.set()
            rows = await asyncio.wait_for(task, 5)
            if capture == 'session':
                assert reads[0]['runtime_generation'] == 0
            else:
                assert reads[0][0]['status'] == 'queued'
        assert claimed['generation'] == live.event_stream.execution['execution_generation'] == 1
        assert rows['chat']['status'] == 'waiting', 'poll used stale storage instead of the current claim'
        assert live.controls.pending['private-prompt'][1] == prompt
        assert ('private-prompt' in live.controls.remote_responders) is remote
        assert responses == []
        assert not any(frame['params']['type'].endswith('.settled') for frame in frames)
    finally:
        resume.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
            assert await asyncio.to_thread(finished.wait, 5), 'held storage worker did not settle'
        live.event_stream.observers.discard(frames.append)


@pytest.mark.asyncio
async def test_active_list_keeps_loop_responsive_during_storage_read(owner, monkeypatch):
    """A real SQLite read is held until another loop task can release it.

    A thread deadline releases a regressed synchronous call without deadlocking pytest;
    the oracle is the heartbeat observing the still-held read, not a latency threshold.
    """
    loop = asyncio.get_running_loop()
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    notified = asyncio.Event()
    db = owner.authority.db
    original = db._read_ctx
    held = False
    heartbeat = []

    @contextmanager
    def read_ctx():
        nonlocal held
        with original() as conn:
            if not held:
                held = True
                entered.set()
                loop.call_soon_threadsafe(notified.set)
                release.wait(3)  # failure-only deadlock escape, not the responsiveness oracle
                returned.set()
            yield conn

    async def tick():
        await notified.wait()
        heartbeat.append(entered.is_set() and not returned.is_set())
        release.set()

    monkeypatch.setattr(db, '_read_ctx', read_ctx)
    ticker = asyncio.create_task(tick())
    try:
        rows = await owner.active(current_session_id='chat')
        await asyncio.wait_for(ticker, 5)
    finally:
        release.set()
        if not ticker.done():
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
    assert heartbeat == [True], 'active_list blocked the loop until the SQLite read hold expired'
    assert rows['chat']['current'] is True
    assert rows['chat']['status'] == 'idle'


@pytest.mark.asyncio
async def test_active_list_keeps_authorization_and_controls_on_loop(owner, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    await owner.submit('hi')
    thread = threading.get_ident()
    authority = owner.authority
    original_read = authority.db._read_ctx
    original_authorize = authority.authorize
    original_snapshot = authority.sessions['chat'].controls.snapshot
    reads, authorizations, snapshots = [], [], []

    @contextmanager
    def read_ctx():
        reads.append(threading.get_ident())
        with original_read() as conn:
            yield conn

    def authorize(*args):
        authorizations.append(threading.get_ident())
        return original_authorize(*args)

    def snapshot(*args):
        snapshots.append(threading.get_ident())
        return original_snapshot(*args)

    monkeypatch.setattr(authority.db, '_read_ctx', read_ctx)
    monkeypatch.setattr(authority, 'authorize', authorize)
    monkeypatch.setattr(authority.sessions['chat'].controls, 'snapshot', snapshot)
    rows = await owner.active(current_session_id='chat')
    assert reads and all(t != thread for t in reads), 'SQLite read ran on the owner loop'
    assert authorizations and all(t == thread for t in authorizations)
    assert snapshots and all(t == thread for t in snapshots)
    assert rows['chat']['current'] is True

    reads.clear()
    denied = AuthorityConnection(authority, object(), {'user_id': 'viewer', 'capabilities': []})
    reply = await denied.dispatch({'id': 2, 'method': 'session.active_list', 'params': {}})
    assert reply['error']['data']['reason'] == 'permission_denied'
    assert reads == []


@pytest.mark.asyncio
async def test_active_list_preserves_session_actor_and_profile_boundaries(owner):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_controls import AuthorityConnection
    authority = owner.authority
    authority.db.create_session('private', source='cli')
    authority.sessions['private'] = LiveSession(
        SessionSource(platform=Platform.LOCAL, chat_id='private', user_id='other-user'), 'private')
    assert set(await owner.active(current_session_id='private')) == {'chat'}

    foreign = AuthorityConnection(authority, object(), {'user_id': 'viewer', 'profile_id': 'foreign'})
    reply = await foreign.dispatch({'id': 2, 'method': 'session.active_list', 'params': {}})
    assert reply['result']['sessions'] == []
    reply = await foreign.dispatch({'id': 3, 'method': 'session.active_list', 'params': {'profile': 'foreign'}})
    assert reply['error']['data']['reason'] == 'profile_mismatch'

    operator = AuthorityConnection(authority, object(), {'user_id': 'viewer'}, operator=True)
    reply = await operator.dispatch({'id': 4, 'method': 'session.active_list',
                                     'params': {'current_session_id': 'private'}})
    rows = {row['id']: row for row in reply['result']['sessions']}
    assert set(rows) == {'chat', 'private'}
    assert rows['private']['current'] is True
    assert rows['chat']['current'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['remove', 'replace', 'revoke'], ids=['removed', 'replaced', 'permission-revoked'])
@pytest.mark.parametrize('method', ['session.active_list', 'session.list'])
async def test_active_list_drops_stale_session_after_read(owner, monkeypatch, change, method):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    loop = asyncio.get_running_loop()
    authority = owner.authority
    original = authority.db._read_ctx
    released = threading.Event()
    changed = []

    def mutate():
        changed.append(change)
        if change == 'remove':
            authority.sessions.pop('chat')
        elif change == 'replace':
            authority.sessions['chat'] = LiveSession(None, 'replacement')
        else:
            authority.sessions['chat'].source = SessionSource(
                platform=Platform.LOCAL, chat_id='chat', user_id='other-user')
        released.set()

    @contextmanager
    def read_ctx():
        with original() as conn:
            if not changed:
                loop.call_soon_threadsafe(mutate)
                assert released.wait(5), 'owner loop could not process a session change during its read'
            yield conn

    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(authority, object(), {'user_id': 'viewer'})
    monkeypatch.setattr(authority.db, '_read_ctx', read_ctx)
    reply = await connection.dispatch({'id': 1, 'method': method, 'params': {}})
    assert reply['result']['sessions'] == []
    assert changed == [change]


@pytest.mark.asyncio
async def test_active_list_preserves_queued_settling_and_closed_visibility(owner):
    import json
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_runtime import admit_session_input
    authority = owner.authority
    queued = admit_session_input(authority.db, epoch=authority.epoch, principal_id='viewer',
                                 session_id='chat', request_id='queued', payload={'text': 'hi'}, intent='queue')
    assert (await owner.active())['chat']['status'] == 'starting'
    authority.sessions['chat'].settling[queued['admission_id']] = dict(queued, status='started')
    assert (await owner.active())['chat']['status'] == 'working'
    authority.sessions['chat'].settling.clear()
    with authority.sessions['chat'].event_stream.lock:
        authority.sessions['chat'].event_stream.execution = {'execution_generation': 1}
    assert (await owner.active())['chat']['status'] == 'working'
    with authority.sessions['chat'].event_stream.lock:
        authority.sessions['chat'].event_stream.execution = {}

    for sid, source in [('closed', 'cli'), ('hidden', 'telegram'), ('internal', 'tool')]:
        authority.db.create_session(sid, source=source)
        authority.sessions[sid] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id=sid), sid)
    authority.db.set_session_hidden('hidden', True)
    authority.db.end_session('closed', end_reason='tui_close')
    authority.db._execute_write(lambda conn: conn.execute(
        'INSERT INTO state_meta(key,value) VALUES (?,?)',
        (POLICY_PREFIX + 'closed', json.dumps({'entry': {'session_id': 'closed'}}))))
    assert set(await owner.active()) == {'chat', 'hidden'}
    authority.sessions['closed'].subscribers['pane'] = object()
    assert set(await owner.active()) == {'chat', 'hidden', 'closed'}


@pytest.mark.asyncio
async def test_session_list_keeps_loop_responsive_during_storage_read(owner, monkeypatch):
    loop = asyncio.get_running_loop()
    db = owner.authority.db
    original = db._read_ctx
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    notified = asyncio.Event()
    heartbeat = []
    held = False

    @contextmanager
    def read_ctx():
        nonlocal held
        with original() as conn:
            if not held:
                held = True
                entered.set()
                loop.call_soon_threadsafe(notified.set)
                release.wait(3)  # Escape a regression that blocks the loop.
                returned.set()
            yield conn

    async def tick():
        await notified.wait()
        heartbeat.append(entered.is_set() and not returned.is_set())
        release.set()

    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(owner.authority, object(), {'user_id': 'viewer'})
    monkeypatch.setattr(db, '_read_ctx', read_ctx)
    ticker = asyncio.create_task(tick())
    try:
        reply = await connection.dispatch({'id': 1, 'method': 'session.list', 'params': {}})
        await asyncio.wait_for(ticker, 5)
    finally:
        release.set()
        if not ticker.done():
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
    assert heartbeat == [True]
    assert [row['id'] for row in reply['result']['sessions']] == ['chat']


@pytest.mark.asyncio
async def test_session_list_preserves_picker_filters_and_current_state(owner):
    import json
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_controls import AuthorityConnection
    from hermes_state_local import POLICY_PREFIX
    authority = owner.authority
    connection = AuthorityConnection(authority, object(), {'user_id': 'viewer'})

    async def listing(**params):
        reply = await connection.dispatch({'id': 1, 'method': 'session.list', 'params': params})
        assert reply['result']['scope'] == 'live'
        return {row['id']: row for row in reply['result']['sessions']}

    for sid, source in [('closed', 'cli'), ('hidden', 'telegram'), ('archived', 'telegram'), ('internal', 'tool')]:
        authority.db.create_session(sid, source=source)
        authority.sessions[sid] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id=sid), sid)
    authority.db.set_session_hidden('hidden', True)
    authority.db.set_session_archived('archived', True)
    authority.db.end_session('closed', end_reason='tui_close')
    authority.db._execute_write(lambda conn: conn.execute(
        'INSERT INTO state_meta(key,value) VALUES (?,?)',
        (POLICY_PREFIX + 'closed', json.dumps({'entry': {'session_id': 'closed'}}))))
    assert set(await listing()) == {'chat'}
    assert set(await listing(include_hidden=True)) == {'chat', 'hidden', 'archived'}
    live = authority.sessions['closed']
    live.subscribers['pane'] = object()
    assert set(await listing()) == {'chat', 'closed'}
    live.subscribers.clear()
    live.event_stream.execution = {'execution_generation': 2}
    assert (await listing())['closed']['running'] is True
    live.event_stream.execution = {}
    live.settling['receipt'] = {'admission_id': 'receipt', 'status': 'started'}
    assert (await listing())['closed']['running'] is True
    live.settling['receipt']['status'] = 'unknown'
    assert (await listing())['closed']['running'] is False
    live.settling.clear()
    rows = await listing(include_hidden=True, limit=1)
    expected = max(('chat', 'hidden', 'archived'), key=lambda sid: authority.db.get_session(sid)['started_at'])
    assert list(rows) == [expected]
    assert set(rows[expected]) == {'session_id', 'id', 'title', 'source', 'started_at', 'message_count', 'model', 'running'}


@pytest.mark.asyncio
async def test_cancelled_session_list_worker_finishes_without_live_mutation(owner, monkeypatch):
    from gateway import session_active_list
    from gateway.session_controls import AuthorityConnection
    loop = asyncio.get_running_loop()
    entered, finished = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    original = session_active_list._read_state
    authority = owner.authority
    live = authority.sessions['chat']
    connection = AuthorityConnection(authority, object(), {'user_id': 'viewer'})

    def held_read(*args):
        try:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            return original(*args)
        finally:
            loop.call_soon_threadsafe(finished.set)

    monkeypatch.setattr(session_active_list, '_read_state', held_read)
    task = asyncio.create_task(connection.dispatch({'id': 1, 'method': 'session.list', 'params': {}}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        live.event_stream.execution = {'execution_generation': 4}
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 5)
        await asyncio.gather(task, return_exceptions=True)
    assert authority.sessions['chat'] is live
    assert live.event_stream.execution == {'execution_generation': 4}
    assert live.settling == {} and live.subscribers == {}
