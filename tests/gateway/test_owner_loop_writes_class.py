"""Bug class: a synchronous SQLite write reached from an ``async def`` on the owner loop.

Every row drives the real authority verb end to end and asserts the named write ran off the
owner-loop thread (the population is listed in SWEEP_owner_loop_writes.md; the contended-writer
stall itself is shown by test_mutation_prepare_off_loop / test_claim_reopen_off_loop).
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest


def _owner(tmp_path, monkeypatch, handle):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _handle_message=handle,
                             _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    return store, authority, AuthorityConnection(authority, object(), {'user_id': 'human'})


def _local(authority, owner, tmp_path, source='cli', request_id='m'):
    from gateway.session_local import create_local_session
    return create_local_session(authority, owner.actor, dict(
        request_id=request_id, source=source, cwd=str(tmp_path), model='frozen', toolsets=[]))


async def _terminal(db, receipt):
    import hermes_state_runtime as rt
    async with asyncio.timeout(10):
        while rt.get_session_admission(db, admission_id=receipt.admission_id)['status'] != 'terminal':
            await asyncio.sleep(0.02)


async def _model_prepare(authority, owner, tmp_path, db):
    from hermes_cli import model_switch
    model_switch_real = model_switch.switch_model
    model_switch.switch_model = lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='switched', target_provider='custom', base_url='http://127.0.0.1:9/v1')
    try:
        ref = _local(authority, owner, tmp_path)
        snap = db.get_session(ref.session_id)
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'switch', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model',
            'payload': {'model': 'switched'}}})
    finally:
        model_switch.switch_model = model_switch_real
    assert reply['result']['model'] == 'switched', reply


async def _claim_reopen(authority, owner, tmp_path, db):
    from gateway.session_contract import Submission
    ref = _local(authority, owner, tmp_path)
    db.end_session(ref.session_id, 'tui_shutdown')
    await _terminal(db, await authority.submit(owner.actor, Submission('first', ref, {'text': 'FIRST'}, 'queue')))
    assert db.get_session(ref.session_id)['end_reason'] is None


async def _acp_idle_after_turn(authority, owner, tmp_path, db):
    from gateway.session_contract import Submission
    ref = _local(authority, owner, tmp_path, source='acp')
    await _terminal(db, await authority.submit(owner.actor, Submission('acp', ref, {'text': 'ACP'}, 'queue')))
    async with asyncio.timeout(10):
        await authority.sessions[ref.session_id].task
    assert db.get_session(ref.session_id)['end_reason'] is not None


async def _acp_detach(authority, owner, tmp_path, db):
    ref = _local(authority, owner, tmp_path, source='acp')
    snapshot = await authority.attach(owner.actor, ref)
    await authority.detach(owner.actor, snapshot.subscription_id)
    assert db.get_session(ref.session_id)['end_reason'] is not None


async def _create_titled_hidden(authority, owner, tmp_path, db):
    reply = await owner.dispatch({'id': 1, 'method': 'session.create', 'params': {
        'request_id': 'titled', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': [],
        'title': 'My Title', 'hidden': True}})
    assert 'result' in reply, reply


async def _native_register(authority, owner, tmp_path, db):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway import session_envelope
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c', user_id='u')
    payload = {'native_text_v1': {'event': {'message_id': 'm1', 'text': 'hi'}}}

    async def prepare_native(runner, event):
        return payload
    session_envelope_prepare = session_envelope.prepare_native
    session_envelope.prepare_native = prepare_native
    original_restore = session_envelope.restore_native
    session_envelope.restore_native = lambda payload, runner=None: SimpleNamespace(source=source)
    authority._admit_native_write = lambda **k: {'admission_id': 'a', 'status': 'queued'}
    authority._admitted = lambda ref, event, row: row
    authority._schedule = lambda ref: None
    try:
        await authority.admit_native(SimpleNamespace())
    finally:
        session_envelope.prepare_native = session_envelope_prepare
        session_envelope.restore_native = original_restore


async def _initialize(authority, owner, tmp_path, db):
    from gateway.session_authority import initialize_session_authority
    authority.runner.session_authority = None
    authority.runner._session_db = db
    await initialize_session_authority(authority.runner, profile_id='owned', instance_id='second')


# (case, module holding the write's name, name, driver): the writer lock is taken just before
# the named call runs.
CASES = [
    ('model-prepare', 'gateway.session_mutations', 'mutate_runtime_session', _model_prepare),
    ('claim-reopen', 'gateway.session_local_recovery', 'reopen_local_session', _claim_reopen),
    ('acp-idle-after-turn', 'gateway.session_acp_lifecycle', 'end_idle_local_session', _acp_idle_after_turn),
    ('acp-detach', 'gateway.session_acp_lifecycle', 'end_idle_local_session', _acp_detach),
    ('create-title-hidden', 'gateway.session_local_title', 'title_new_session', _create_titled_hidden),
    ('native-register', 'gateway.session.SessionStore', 'get_or_create_session', _native_register),
    ('initialize-epoch', 'gateway.session_authority', 'begin_runtime_epoch', _initialize),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('case,module,name,drive', CASES, ids=[c[0] for c in CASES])
async def test_in_class_write_never_runs_on_the_owner_loop_thread(tmp_path, monkeypatch, case, module, name, drive):
    """Each in-class write is wrapped where it is called; the wrapper records the thread it ran on.
    On the owner-loop thread, a contended writer lock (up to _WRITE_PATIENCE_S) freezes every session."""
    import importlib
    async def handle(event):
        return 'ok'
    store, authority, owner = _owner(tmp_path, monkeypatch, handle)
    loop_thread = threading.get_ident()
    if module.endswith('.SessionStore'):
        target = importlib.import_module(module.rsplit('.', 1)[0]).SessionStore
    else:
        target = importlib.import_module(module)
    real, threads = getattr(target, name), []

    def recorded(*args, **kwargs):
        if name != 'mutate_runtime_session' or kwargs.get('_prepare_only'):
            threads.append(threading.get_ident())
        return real(*args, **kwargs)
    monkeypatch.setattr(target, name, recorded)
    try:
        await drive(authority, owner, tmp_path, store._db)
        assert threads, f'{case}: the in-class write never ran'
        assert loop_thread not in threads, f'{case}: the write ran on the owner loop thread'
    finally:
        await owner.close()
        store.close_all_db_handles()
