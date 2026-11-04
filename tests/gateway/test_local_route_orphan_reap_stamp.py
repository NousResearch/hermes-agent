"""A local route the authority owns keeps its conversation when its row is stamped ended (#140642).

A viewer process stamped a Desktop session ``ws_orphan_reap`` while the gateway still ran its turn.
The SessionStore's stale-route self-heal (#54878) then dropped the local route, minted a fresh
session id (whose peer record the authority refuses: ``admission_conflict``, after the row was
already inserted), and the next restart's prune repointed ``sessions.json`` at that empty row, so
every later resume was refused ``admission_conflict``. A local route belongs to its receipt: only
the authority advances it (compression, reset), and the first admitted turn reopens a stamped row.
"""
from types import SimpleNamespace

import pytest


async def _local_session(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_local import create_local_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    ref = create_local_session(authority, connection.actor, {'request_id': 'r', 'source': 'gui',
                                                             'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    authority.db.append_message(ref.session_id, 'user', 'earlier turn')
    return store, authority, connection, ref


def _rows(db):
    with db._read_ctx() as conn:
        return [row[0] for row in conn.execute('SELECT id FROM sessions ORDER BY started_at')]


@pytest.mark.asyncio
async def test_stamped_local_route_is_not_replaced_by_a_fresh_session(tmp_path, monkeypatch):
    store, authority, connection, ref = await _local_session(tmp_path, monkeypatch)
    try:
        live = authority.sessions[ref.session_id]
        authority.db.end_session(ref.session_id, 'ws_orphan_reap')

        entry = store.get_or_create_session(live.source)

        assert entry.session_id == ref.session_id
        assert _rows(authority.db) == [ref.session_id], 'no replacement row may be minted for a local route'
        # The conversation still resumes through the authority.
        reply = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                           'params': {'session_id': ref.session_id}})
        assert reply['result']['messages'][0]['content'] == 'earlier turn', reply
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_restart_prune_keeps_a_stamped_local_route_on_its_receipt_target(tmp_path, monkeypatch):
    store, authority, connection, ref = await _local_session(tmp_path, monkeypatch)
    try:
        live = authority.sessions[ref.session_id]
        authority.db.end_session(ref.session_id, 'ws_orphan_reap')
        # The empty replacement row a pre-fix self-heal left behind on the same route.
        authority.db.create_session('stray-replacement', 'local', session_key=live.route,
                                    chat_id=live.source.chat_id, chat_type='dm', user_id=live.source.user_id)
        with store._lock:
            entry = store._entries[live.route]
            verdict = store._stale_entry_verdict(live.route, entry, authority.db.get_session(ref.session_id))
        assert verdict is None, verdict
        assert store._entries[live.route].session_id == ref.session_id
    finally:
        await connection.close()
        authority.db.close()


@pytest.mark.asyncio
async def test_resume_heals_a_route_repointed_at_an_empty_stray_row(tmp_path, monkeypatch):
    """Installs already wedged by the old repoint: the route names an empty row outside the
    receipt's lineage. The authority's receipt is the truth; the stray entry is dropped."""
    store, authority, connection, ref = await _local_session(tmp_path, monkeypatch)
    try:
        live = authority.sessions.pop(ref.session_id)
        authority.db.create_session('stray-replacement', 'local', session_key=live.route,
                                    chat_id=live.source.chat_id, chat_type='dm', user_id=live.source.user_id)
        with store._lock:
            store._entries[live.route].session_id = 'stray-replacement'

        reply = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                           'params': {'session_id': ref.session_id}})

        assert reply['result']['messages'][0]['content'] == 'earlier turn', reply
        assert store._entries[live.route].session_id == ref.session_id
    finally:
        await connection.close()
        authority.db.close()
