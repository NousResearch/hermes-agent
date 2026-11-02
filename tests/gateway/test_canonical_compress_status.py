"""A canonical compress narrates its phase to the requesting surface (#147491).

The desktop's /compress travels as canonical ``session.mutate`` and ``prepare_compress``
runs the summarizer inline — minutes on a large session. The legacy ``session.compress``
RPC narrates the phase (tui_gateway ``_status_update`` emits compressing/ready status
frames); the canonical path emitted nothing, so the desktop never showed its thread status
row. The connection now writes the same frames to its own transport, and retires the phase
even when the mutation is refused — a stuck row is worse than no row.
"""
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionActor, SessionRef
from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import RuntimeStoreError

NATIVE_IDENTITY = {'user_id': 'owner', 'provider': 'local', 'capabilities': [
    'session:create', 'session:read', 'session:submit', 'session:control'], 'native_bootstrap': True}


class _RecordingTransport:
    def __init__(self):
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True


def _wired_connection(monkeypatch, delegate):
    """A connection whose mutation delegate is replaced: the contract under test is the
    frame narration the connection wraps AROUND the delegate, not the delegate itself."""
    import gateway.session_mutations
    monkeypatch.setattr(gateway.session_mutations, 'mutate_session', delegate)
    conn = object.__new__(AuthorityConnection)
    conn.authority = SimpleNamespace(db=SimpleNamespace(
        get_session=lambda sid: {'runtime_revision': 1, 'runtime_generation': 1}))
    conn.actor = SessionActor('basic:owner', 'p', frozenset({'session:control'}), 'ws')
    conn.subscriptions = {}
    transport = _RecordingTransport()
    conn.transport = transport
    return conn, SessionRef('p', 'local-1'), transport


def _status_kinds(transport):
    return [f['params']['payload']['kind'] for f in transport.frames
            if f.get('method') == 'event' and f['params'].get('type') == 'status.update']


@pytest.mark.asyncio
async def test_canonical_compress_narrates_compressing_then_ready(monkeypatch):
    seen = {}

    async def delegate(authority, actor, ref, params):
        seen['params'] = params
        return {'operation': 'compress', 'summary': {'lines': ['Compressed: 14 → 3 messages']}}

    conn, ref, transport = _wired_connection(monkeypatch, delegate)
    result = await conn.mutate(ref, {'session_id': 'local-1', 'request_id': 'c1', 'operation': 'compress', 'payload': {}})
    assert _status_kinds(transport) == ['compressing', 'ready']
    assert result['operation'] == 'compress'
    assert seen['params']['operation'] == 'compress'


@pytest.mark.asyncio
async def test_a_failed_compress_still_retires_the_phase(monkeypatch):
    async def delegate(authority, actor, ref, params):
        raise RuntimeStoreError('stale_generation')

    conn, ref, transport = _wired_connection(monkeypatch, delegate)
    with pytest.raises(RuntimeStoreError):
        await conn.mutate(ref, {'session_id': 'local-1', 'request_id': 'c2', 'operation': 'compress', 'payload': {}})
    assert _status_kinds(transport) == ['compressing', 'ready']


@pytest.mark.asyncio
async def test_non_compress_mutations_emit_no_status_frames(monkeypatch):
    async def delegate(authority, actor, ref, params):
        return {'operation': 'archive'}

    conn, ref, transport = _wired_connection(monkeypatch, delegate)
    await conn.mutate(ref, {'session_id': 'local-1', 'request_id': 'a1', 'operation': 'archive', 'payload': {'archived': True}})
    assert _status_kinds(transport) == []


@pytest.mark.asyncio
async def test_a_refused_compress_retires_the_phase_through_the_real_authority(tmp_path, monkeypatch):
    """The refusal path with the REAL authorize/prepare stack: a compress the authority
    refuses still gets its compressing frame retired."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'm1', 'provider': 'openrouter'}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    db.create_session('local-1', source='local', model='m1')
    db._write_sql("UPDATE sessions SET session_key='agent:main:local:dm:local-1', chat_id='local-1' WHERE id='local-1'")
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _adapter_for_source=lambda source: None, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda source, session_key: ('m1', {}))
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='i')
    transport = _RecordingTransport()
    conn = AuthorityConnection(authority, transport, {**NATIVE_IDENTITY, 'profile_id': str(tmp_path), 'instance_id': 'i'})
    ref = SessionRef(str(tmp_path), 'local-1')
    with pytest.raises(RuntimeStoreError):
        await conn.mutate(ref, {'session_id': 'local-1', 'request_id': 'c3', 'operation': 'compress', 'payload': {},
                                'expected_revision': 999_999, 'expected_generation': 999_999})
    assert _status_kinds(transport) == ['compressing', 'ready']
