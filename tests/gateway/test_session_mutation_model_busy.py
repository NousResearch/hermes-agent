"""A model pick on a session with a running turn lands for the next turn (RT-12, #138771)."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


def _owner(tmp_path, monkeypatch, handle):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import model_switch
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k['raw_input'], target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _handle_message=handle,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='gui', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    return store, db, authority, owner, ref


def _running_model(authority, ref):
    from gateway.config import Platform
    from gateway.session_local_recovery import local_adapter_map
    live = authority.sessions[ref.session_id]
    return local_adapter_map(authority)[Platform.LOCAL].policies[live.source.chat_id].model


@pytest.mark.asyncio
async def test_model_pick_during_a_running_turn_applies_to_the_next_turn(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    import hermes_state_runtime as rt
    started, release = threading.Event(), asyncio.Event()
    seen = []
    state = {}

    async def handle(event):
        seen.append((event.text, _running_model(state['authority'], state['ref'])))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    try:
        first = await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        snap = db.get_session(ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'pick', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': 'picked'}}})
        assert 'result' in reply, reply
        assert reply['result']['status'] == 'deferred'
        # The running turn keeps its route.
        assert _running_model(authority, ref) == 'frozen'
        release.set()
        await async_wait_until(
            lambda: rt.get_session_admission(db, admission_id=first.admission_id)['status'] == 'terminal',
            timeout=10, what='the running turn settled')
        await async_wait_until(lambda: _running_model(authority, ref) == 'picked',
                               timeout=10, what='the pick published at the turn boundary')
        second = await authority.submit(owner.actor, Submission('next', ref, {'text': 'NEXT'}, 'queue'))
        await async_wait_until(
            lambda: rt.get_session_admission(db, admission_id=second.admission_id)['status'] == 'terminal',
            timeout=10, what='the next turn ran')
        assert seen == [('RUNNING', 'frozen'), ('NEXT', 'picked')]
        assert db.get_session(ref.session_id)['model'] == 'picked'
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_revision_conflict_refusal_carries_the_fresh_fences(tmp_path, monkeypatch):
    async def handle(event):
        return 'ok'
    store, db, _authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        snap = db.get_session(ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'stale', 'expected_revision': snap['runtime_revision'] + 7,
            'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': 'picked'}}})
        data = reply['error']['data']
        assert data['reason'] == 'revision_conflict'
        assert (data['revision'], data['execution_generation']) == (snap['runtime_revision'], snap['runtime_generation'])
        retry = await owner.dispatch({'id': 2, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'retry', 'expected_revision': data['revision'],
            'expected_generation': data['execution_generation'], 'operation': 'model', 'payload': {'model': 'picked'}}})
        assert retry['result']['model'] == 'picked'
    finally:
        await owner.close()
        store.close_all_db_handles()


async def _mutate(owner, db, ref, rid, model, snap=None, *, call=1):
    snap = snap or db.get_session(ref.session_id)
    return await owner.dispatch({'id': call, 'method': 'session.mutate', 'params': {
        'session_id': ref.session_id, 'request_id': rid, 'expected_revision': snap['runtime_revision'],
        'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': model}}})


async def _run_turn(authority, owner, db, ref, name):
    from gateway.session_contract import Submission
    import hermes_state_runtime as rt
    turn = await authority.submit(owner.actor, Submission(name, ref, {'text': name}, 'queue'))
    await async_wait_until(
        lambda: rt.get_session_admission(db, admission_id=turn.admission_id)['status'] == 'terminal',
        timeout=10, what=f'turn {name} settled')


@pytest.mark.asyncio
async def test_a_retried_deferred_pick_does_not_undo_a_newer_pick(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    started, release = threading.Event(), asyncio.Event()
    seen = []
    state = {}

    async def handle(event):
        seen.append((event.text, _running_model(state['authority'], state['ref'])))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    try:
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        snap = db.get_session(ref.session_id)
        first = await _mutate(owner, db, ref, 'pick-a', 'model-a', snap)
        assert first['result']['status'] == 'deferred', first
        newer = await _mutate(owner, db, ref, 'pick-b', 'model-b', call=2)
        assert newer['result']['status'] == 'deferred', newer
        # A's reply was lost: the client retries it with the same identity and fences.
        retry = await _mutate(owner, db, ref, 'pick-a', 'model-a', snap, call=3)
        assert retry['result'] == first['result']
        release.set()
        await _run_turn(authority, owner, db, ref, 'NEXT')
        assert seen[-1] == ('NEXT', 'model-b')
        assert db.get_session(ref.session_id)['model'] == 'model-b'
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_newer_idle_pick_replaces_an_older_pending_pick(tmp_path, monkeypatch):
    from gateway.session_deferred_pick import defer_pick
    seen = []
    state = {}

    async def handle(event):
        seen.append((event.text, _running_model(state['authority'], state['ref'])))
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    try:
        # An older pick still pending when the drain is between turns (awaiting delivery).
        snap = db.get_session(ref.session_id)
        defer_pick(authority, owner.actor, ref, {
            'session_id': ref.session_id, 'request_id': 'old', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': 'older'}})
        newer = await _mutate(owner, db, ref, 'new', 'newer')
        assert newer['result']['model'] == 'newer', newer
        await _run_turn(authority, owner, db, ref, 'NEXT')
        assert seen == [('NEXT', 'newer')]
        assert db.get_session(ref.session_id)['model'] == 'newer'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_pending_pick_writes_do_not_stall_the_event_loop(tmp_path, monkeypatch):
    import time
    from gateway.session_contract import Submission
    started, release = threading.Event(), asyncio.Event()
    state = {}

    async def handle(event):
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    gaps = []

    async def heartbeat():
        while True:
            before = time.monotonic()
            await asyncio.sleep(0.01)  # health: allow HX013 -- the probe measures loop latency itself
            gaps.append(time.monotonic() - before)
    try:
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        write = db._execute_write

        def contended(fn, *a, **k):  # a writer lock held elsewhere (transcript write, maintenance)
            time.sleep(0.4)  # health: allow HX013 -- a writer lock held elsewhere, the stall under test
            return write(fn, *a, **k)
        monkeypatch.setattr(db, '_execute_write', contended)
        beat = asyncio.create_task(heartbeat())
        reply = await _mutate(owner, db, ref, 'pick', 'picked')
        assert reply['result']['status'] == 'deferred', reply
        release.set()
        await async_wait_until(lambda: _running_model(authority, ref) == 'picked',
                               timeout=20, what='the pick published at the turn boundary')
        beat.cancel()
        assert max(gaps) < 0.3, max(gaps)
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_guarded_deferred_pick_asks_for_confirmation_instead_of_being_dropped(tmp_path, monkeypatch):
    from agent.usage_anchor import USAGE_ANCHOR_MODEL_CONFIG_KEY, capture_usage_anchor
    from gateway.session_contract import Submission
    from hermes_cli import model_selection_guards
    monkeypatch.setattr(model_selection_guards, '_context_cache_threshold', lambda: 1000)
    started, release = threading.Event(), asyncio.Event()
    seen, events = [], []
    state = {}

    async def handle(event):
        seen.append((event.text, _running_model(state['authority'], state['ref'])))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    authority.runner._cached_agent_for = lambda route: None
    stream = authority.sessions[ref.session_id].event_stream
    publish = stream.publish
    monkeypatch.setattr(stream, 'publish', lambda sid, payload, **k: (
        events.append((k.get('event_type'), payload)), publish(sid, payload, **k))[1])
    try:
        db.append_message(ref.session_id, role='user', content='hello')
        db.append_message(ref.session_id, role='assistant', content='hi')
        db.patch_session_model_config(ref.session_id, {USAGE_ANCHOR_MODEL_CONFIG_KEY: capture_usage_anchor(
            250_000, 10, db.get_messages_as_conversation(ref.session_id))})
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        assert (await _mutate(owner, db, ref, 'pick', 'guarded'))['result']['status'] == 'deferred'
        release.set()

        def asks():
            return [p for t, p in events if t == 'session.updated' and p.get('status') == 'confirmation_required']
        # The running turn settled: the drain resolves the pick and asks instead of dropping it.
        await async_wait_until(lambda: asks(), timeout=10, what='the guarded pick asked for confirmation')
        ask = asks()[0]
        assert ask['target_model'] == 'guarded' and ask['payload'] == {'model': 'guarded'}
        assert ask['committed_model'] == 'frozen', ask
        assert not [p for t, p in events if p.get('status') == 'refused']
        assert _running_model(authority, ref) == 'frozen'
        # The user confirms: the canonical resend carries the owner's token and commits.
        snap = db.get_session(ref.session_id)
        confirmed = await owner.dispatch({'id': 9, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'confirmed', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model',
            'payload': {'model': 'guarded', 'confirm': ask['confirm']}}})
        assert confirmed['result']['model'] == 'guarded', confirmed
        assert len(asks()) == 1
        await _run_turn(authority, owner, db, ref, 'AFTER')
        assert seen[-1] == ('AFTER', 'guarded')
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_refused_deferred_pick_names_the_reason_and_the_committed_model(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    from hermes_cli import model_switch
    started, release = threading.Event(), asyncio.Event()
    events = []

    async def handle(event):
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    switch = model_switch.switch_model
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=False, new_model='', target_provider='') if k['raw_input'] == 'gone' else switch(**k))
    stream = authority.sessions[ref.session_id].event_stream
    publish = stream.publish
    monkeypatch.setattr(stream, 'publish', lambda sid, payload, **k: (
        events.append((k.get('event_type'), payload)), publish(sid, payload, **k))[1])
    try:
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        assert (await _mutate(owner, db, ref, 'pick', 'gone'))['result']['status'] == 'deferred'
        release.set()

        def refused():
            return [p for t, p in events if t == 'session.updated' and p.get('status') == 'refused']
        await async_wait_until(lambda: refused(), timeout=10, what='the refusal was announced')
        assert refused()[0].items() >= {'operation': 'model', 'reason': 'model_resolution_failed',
                                     'target_model': 'gone', 'model': 'frozen'}.items()
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [RuntimeError, OSError])
async def test_a_deferred_pick_that_raises_is_refused_and_the_queue_keeps_draining(tmp_path, monkeypatch, error):
    from gateway.session_contract import Submission
    from gateway.session_deferred_pick import _PREFIX
    from hermes_cli import model_switch
    import hermes_state_runtime as rt
    started, release = threading.Event(), asyncio.Event()
    events, seen, state, raising = [], [], {}, threading.Event()

    async def handle(event):
        seen.append((event.text, _running_model(state['authority'], state['ref'])))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    switch = model_switch.switch_model

    def resolve(**k):
        if raising.is_set() and k['raw_input'] == 'boom':
            raise error('resolver exploded')
        return switch(**k)
    monkeypatch.setattr(model_switch, 'switch_model', resolve)
    stream = authority.sessions[ref.session_id].event_stream
    publish = stream.publish
    monkeypatch.setattr(stream, 'publish', lambda sid, payload, **k: (
        events.append((k.get('event_type'), payload)), publish(sid, payload, **k))[1])
    try:
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        assert (await _mutate(owner, db, ref, 'pick', 'boom'))['result']['status'] == 'deferred'
        raising.set()
        nxt = await authority.submit(owner.actor, Submission('next', ref, {'text': 'NEXT'}, 'queue'))
        release.set()
        await async_wait_until(
            lambda: rt.get_session_admission(db, admission_id=nxt.admission_id)['status'] == 'terminal',
            timeout=10, what='the queued turn ran despite the failing pick')
        assert seen == [('RUNNING', 'frozen'), ('NEXT', 'frozen')]
        with db._read_ctx() as conn:
            assert conn.execute('SELECT 1 FROM state_meta WHERE key=?',
                                (_PREFIX + ref.session_id,)).fetchone() is None
        refused = [p for t, p in events if t == 'session.updated' and p.get('status') == 'refused']
        assert refused and refused[0].items() >= {'operation': 'model', 'reason': 'model_resolution_failed',
                                                 'target_model': 'boom', 'model': 'frozen'}.items()
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_deferred_model_and_reasoning_pick_commits_both_together(tmp_path, monkeypatch):
    from gateway.session_contract import Submission
    from gateway.session_policy import policy_for_source
    started, release = threading.Event(), asyncio.Event()
    seen = []
    state = {}

    async def handle(event):
        live = state['authority'].sessions[state['ref'].session_id]
        effort = policy_for_source(state['authority'].runner, live.source).reasoning_config
        seen.append((event.text, _running_model(state['authority'], state['ref']), (effort or {}).get('effort')))
        if event.text == 'RUNNING':
            started.set()
            await release.wait()
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    state.update(authority=authority, ref=ref)
    authority.runner._set_session_reasoning_override = lambda *a: None
    try:
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await async_wait_until(started.is_set, timeout=10, what='the first turn is running')
        snap = db.get_session(ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'pick', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model',
            'payload': {'model': 'picked', 'reasoning': 'high'}}})
        assert reply.get('result', {}).get('status') == 'deferred', reply
        release.set()
        await _run_turn(authority, owner, db, ref, 'NEXT')
        assert seen[-1] == ('NEXT', 'picked', 'high'), seen
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()
