"""Classic CLI ``--resume <root>`` of a twice-compressed local conversation addresses the gateway by
the id it owns a receipt for (live-pass finding, package 10).

The CLI reads the transcript from the compression tip (a physical segment), but the creation receipt
lives on the logical owner. Ink and ``session.resume`` already answer with that owner; the CLI used to
keep the segment id it read from, so every turn answered ``permission_denied`` and ``/undo`` /
``/rollback`` answered ``storage_unavailable`` (``local_receipt`` found no receipt for the segment).

Real CLI resume resolution (``_preload_resumed_session`` / ``_load_resumed_history_late`` on a
``GatewaySessionStore``) against a real session authority and its real SQLite receipt path.
"""
import asyncio
import io
import types
from types import SimpleNamespace

import pytest


def _runner(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return runner


def _compress(db, parent, child, source):
    history = [{'role': 'user', 'content': f'retained {child}'}, {'role': 'assistant', 'content': 'history'}]
    assert db.try_acquire_compression_lock(parent, 'fixture')
    try:
        db.publish_compression_child(parent_session_id=parent, child_session_id=child,
                                     source=source, messages=history, compression_lock_holder='fixture')
    finally:
        db.release_compression_lock(parent, 'fixture')


async def _compressed(tmp_path):
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    authority = await initialize_session_authority(_runner(tmp_path), profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(authority, actor, dict(request_id='row10', source='cli', cwd=str(tmp_path),
                                                      model='frozen', toolsets=[]))
    live = authority.sessions[ref.session_id]
    store = authority.runner.session_store
    for parent, child in ((ref.session_id, 'seg-2'), ('seg-2', 'seg-3')):
        store._record_gateway_session_peer(parent, live.route, live.source)
        _compress(authority.db, parent, child, authority.db.get_session(parent)['source'])
        store._record_gateway_session_peer(child, live.route, live.source)
    return authority, ref.session_id


def _gateway_store(loop, authority, tmp_path):
    """A ``GatewaySessionStore`` whose verbs run on the real authority (no socket) and whose reads use
    the real read-only state.db reader, as the classic CLI's does."""
    from gateway.session_controls import AuthorityConnection
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_connection import GatewayConnection, SessionState
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from hermes_state import SessionDB
    states = {}
    connection = types.SimpleNamespace(final=False, launch_api_key=None, launch_api_key_typed=False,
                                       state=lambda sid: states.setdefault(sid, SessionState(sid)))
    connection.resume = types.MethodType(GatewayConnection.resume, connection)  # the store's attach helper
    gateway = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': 'owner', 'profile_id': 'fixture', 'instance_id': authority.instance_id,
        'capabilities': ['session:create', 'session:read', 'session:control']})
    calls = []

    def call(method, **params):
        calls.append((method, params))
        reply = loop.run_until_complete(gateway.dispatch({'id': len(calls), 'method': method, 'params': params}))
        if 'error' in reply:
            raise GatewayClientError(reply['error']['message'])
        return reply['result']

    store = GatewaySessionStore(connection)
    store.call = call
    store._reader = SessionDB(db_path=authority.db.db_path, read_only=True)
    return store, calls


def _cli(store, session_id):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli._resumed, cli._session_db, cli.session_id = True, store, session_id
    cli.session_start = None
    cli.conversation_history = []
    cli.tool_progress_mode = 'off'
    cli._console_print = lambda *a, **k: None
    cli._restore_session_state = lambda *a, **k: None
    return cli


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from gateway import run
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    loop = asyncio.new_event_loop()
    try:
        authority, owner = loop.run_until_complete(_compressed(tmp_path))
        store, calls = _gateway_store(loop, authority, tmp_path)
        # _ensure_gateway_session: the CLI opens the gateway session for the --resume target first.
        assert store.attach(owner)['stored_session_id'] == owner
        yield SimpleNamespace(loop=loop, authority=authority, owner=owner, store=store, calls=calls)
    finally:
        loop.close()


def _gateway_accepts(rig, session_id):
    """The admission/mutation gate every turn, /undo and /rollback reach: ``authorize`` restores the
    local session from its stored receipt (``restore_local_session`` -> ``local_receipt``)."""
    from gateway.session_contract import Principal, SessionRef
    actor = Principal('owner', 'fixture', frozenset({'session:read', 'session:control'}), 'socket')
    rig.authority.authorize(actor, SessionRef('fixture', session_id), 'session:control')


@pytest.mark.parametrize('path', ['_preload_resumed_session', '_load_resumed_history_late'])
def test_resume_of_a_compressed_root_addresses_the_receipt_owner(rig, path, monkeypatch):
    monkeypatch.setattr('sys.stderr', io.StringIO())
    cli = _cli(rig.store, rig.owner)
    assert getattr(cli, path)() is True
    # History comes from the live tip segment (what main shows) ...
    assert [m['content'] for m in cli.conversation_history] == ['retained seg-3', 'history']
    # ... but the CLI keeps addressing the gateway by the id it owns a receipt for, as Ink and
    # session.resume do. The segment id fails the receipt check with storage_unavailable.
    assert cli.session_id == rig.owner
    _gateway_accepts(rig, cli.session_id)


def test_the_tip_segment_alone_still_has_no_receipt(rig):
    """The receipt check is not weakened: addressing the physical segment is still refused by name."""
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError) as refused:
        _gateway_accepts(rig, 'seg-3')
    assert refused.value.reason == 'storage_unavailable'
