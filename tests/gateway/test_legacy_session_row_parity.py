"""Resuming a session main created leaves its row as main would (TUI parity, PR #106742 row 35).

Adoption binds the old transcript to the owner; it must not rewrite the row's ``source`` label
(``hermes sessions list --source tui``, ``-c`` / ``_resolve_last_session`` key on it) nor clear the
``end_reason`` / ``ended_at`` stamp on a mere mount. The first admitted turn reopens the row
(``reopen_local_session``), exactly as main's sidecar did.
"""
from types import SimpleNamespace

import pytest
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session


async def _owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'owner'})
    return store, authority, native


@pytest.mark.asyncio
@pytest.mark.parametrize('label', ['tui', 'cli', 'gui'])
async def test_resumed_main_session_keeps_source_and_end_stamp(tmp_path, monkeypatch, label):
    store, authority, native = await _owner(tmp_path, monkeypatch)
    db = store.persistence
    sid = 'old-' + label
    seed_session(db, sid, label, model='fake-model', cwd=str(tmp_path))
    seed_messages(db, sid, {'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'})
    db.sessions.end(sid, 'tui_shutdown')
    before = session_row(db, sid)

    reply = await native.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
    assert 'result' in reply, reply
    row = session_row(db, sid)
    assert (row['source'], row['end_reason'], row['ended_at']) == (label, 'tui_shutdown', before['ended_at'])

    # Every turn refreshes the routing peer through the LOCAL transport; the label survives it.
    live = authority.sessions[sid]
    store._record_gateway_session_peer(sid, live.route, live.source)
    store._record_gateway_session_peer(sid, live.route, live.source, include_compression_ancestors=True)
    assert session_row(db, sid)['source'] == label

    # Main's `-c` / `--resume latest` and `sessions list --source <label>` still find it.
    assert sid in {r.session.session_id for r in db.sessions.search_sessions(sources=[label])}

    # The first admitted turn reopens the row, as main's sidecar did.
    from gateway.session_local_recovery import reopen_local_session
    from gateway.session_contract import SessionRef
    reopen_local_session(authority, SessionRef(authority.profile_id, sid))
    assert session_row(db, sid)['end_reason'] is None
