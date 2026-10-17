"""A local session created under 0.22.1 and then used under 0.22.2 opens again on the gateway (#140060).

0.22.1 was cut from the gateway line, so its sessions carry a ``gateway.local_policy.v1`` receipt.
0.22.2 is the classic line again: it resumes such a session by id, and when it compacts it ends the
tip with ``compression`` and publishes the child with the parent's route columns
(``hermes_state_compression._publish_child_session_row`` at v0.22.2), but it never advances the
receipt. Back on the gateway, the receipt's tip then has a compression child, and every restore
refused the session with ``storage_unavailable``.
"""
import json
import time
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection


async def _owner(tmp_path, monkeypatch, instance_id):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id=instance_id)
    return authority, AuthorityConnection(authority, object(), {'user_id': 'owner'})


def _compact_like_0_22_2(db, parent_id, child_id, *, route_columns=True):
    """0.22.2's compaction of a session it resumed by id: the parent ends with ``compression`` and the
    child copies the parent's route columns. No receipt, routing entry or epoch is touched."""
    def write(conn):
        parent = conn.execute('SELECT * FROM sessions WHERE id=?', (parent_id,)).fetchone()
        route = (parent['user_id'], parent['session_key'], parent['chat_id'], parent['chat_type'],
                 parent['origin_json']) if route_columns else (None,) * 5
        conn.execute("UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?", (time.time(), parent_id))
        conn.execute('''INSERT INTO sessions(id,source,model,parent_session_id,cwd,profile_name,
            user_id,session_key,chat_id,chat_type,origin_json,started_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
            (child_id, 'tui', parent['model'], parent_id, parent['cwd'], parent['profile_name'], *route, time.time()))
    db._execute_write(write)
    db.append_message(child_id, 'user', f'after compression into {child_id}')


def _receipt(db, sid):
    from hermes_state_local import local_receipt
    return local_receipt(db, sid)


@pytest.mark.asyncio
async def test_a_session_compacted_by_0_22_2_resumes_on_its_compressed_tip(tmp_path, monkeypatch):
    from gateway.session_local import create_local_session

    first, connection = await _owner(tmp_path, monkeypatch, 'first')
    ref = create_local_session(first, connection.actor, {'request_id': 'r', 'source': 'tui',
                                                         'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    first.db.append_message(ref.session_id, 'user', 'before the downgrade')
    db_path = first.db.db_path
    await connection.close()
    first.db.close()

    from hermes_state import SessionDB
    classic = SessionDB(db_path=db_path)
    _compact_like_0_22_2(classic, ref.session_id, 'classic-child-1')
    _compact_like_0_22_2(classic, 'classic-child-1', 'classic-child-2')
    classic.close()

    second, connection = await _owner(tmp_path, monkeypatch, 'second')
    try:
        assert ref.session_id in second.sessions, 'the receipt was paused on cold recovery'
        receipt = _receipt(second.db, ref.session_id)
        assert receipt['lineage'] == [ref.session_id, 'classic-child-1', 'classic-child-2']
        assert receipt['entry']['session_id'] == 'classic-child-2'
        reply = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                           'params': {'session_id': ref.session_id}})
        assert 'error' not in reply, reply
        assert reply['result']['messages'][-1]['content'] == 'after compression into classic-child-2', reply
    finally:
        await connection.close()
        second.db.close()


@pytest.mark.asyncio
async def test_a_child_without_the_receipts_route_is_not_adopted(tmp_path, monkeypatch):
    """Only a child that carries the receipt's own route, owner and principal continues it. Any other
    compression child (a row that never had the route) keeps the receipt refused, unchanged."""
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_runtime import RuntimeStoreError

    first, connection = await _owner(tmp_path, monkeypatch, 'first')
    ref = create_local_session(first, connection.actor, {'request_id': 'r', 'source': 'tui',
                                                         'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    before = json.loads(json.dumps(_receipt(first.db, ref.session_id)))
    db_path = first.db.db_path
    await connection.close()
    first.db.close()

    from hermes_state import SessionDB
    classic = SessionDB(db_path=db_path)
    _compact_like_0_22_2(classic, ref.session_id, 'unrouted-child', route_columns=False)
    classic.close()

    second, connection = await _owner(tmp_path, monkeypatch, 'second')
    try:
        assert ref.session_id not in second.sessions
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            restore_local_session(second, ref.session_id)
        assert _receipt(second.db, ref.session_id) == before
    finally:
        await connection.close()
        second.db.close()


@pytest.mark.asyncio
async def test_a_compacted_receipt_with_a_foreign_origin_stays_refused(tmp_path, monkeypatch):
    """The catch-up runs only for a receipt that already matches this owner: one naming another
    profile in its origin is refused, unchanged, even with a well-formed compression child."""
    from gateway.session_local import create_local_session
    from hermes_state_local import POLICY_PREFIX

    first, connection = await _owner(tmp_path, monkeypatch, 'first')
    ref = create_local_session(first, connection.actor, {'request_id': 'r', 'source': 'tui',
                                                         'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    db_path = first.db.db_path
    await connection.close()
    first.db.close()

    from hermes_state import SessionDB
    classic = SessionDB(db_path=db_path)
    _compact_like_0_22_2(classic, ref.session_id, 'classic-child')
    receipt = _receipt(classic, ref.session_id)
    receipt['entry']['origin']['profile'] = 'other-profile'
    classic._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                                                     (json.dumps(receipt), POLICY_PREFIX + ref.session_id)))
    classic.close()

    second, connection = await _owner(tmp_path, monkeypatch, 'second')
    try:
        assert ref.session_id not in second.sessions, 'a foreign origin was replaced by the catch-up'
        assert _receipt(second.db, ref.session_id) == receipt
    finally:
        await connection.close()
        second.db.close()


@pytest.mark.asyncio
async def test_a_compacted_secondary_receipt_missing_its_origin_profile_restores(tmp_path, monkeypatch):
    """LC-18's one tolerance (an origin without ``profile``) still composes with the catch-up: the
    receipt advances to the compressed tip and keeps its own entry, changed only in its target."""
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_state_local import POLICY_PREFIX
    from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    process_ownership.reserve([home for _, home in homes])
    runners = []
    try:
        first = _runner(root, homes)
        runners.append(first)
        await initialize_gateway_runtime(first)
        home = homes[1][1]
        authority = first.session_authorities.for_home(home)
        actor = Principal('owner', authority.profile_id,
                          frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(authority, actor, {'request_id': 'old-desktop', 'source': 'gui',
                'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        receipt = _receipt(authority.db, ref.session_id)
        receipt['entry']['origin'].pop('profile')
        authority.db._execute_write(lambda conn: conn.execute(
            'UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), POLICY_PREFIX + ref.session_id)))
        _compact_like_0_22_2(authority.db, ref.session_id, 'classic-child')
        for owner in first.session_authorities:
            owner.db.close()

        second = _runner(root, homes)
        runners.append(second)
        await initialize_gateway_runtime(second)
        restored = second.session_authorities.for_home(home)
        assert ref.session_id in restored.sessions, 'the compacted receipt was paused on cold recovery'
        assert restored.sessions[ref.session_id].source.profile == 'alpha'
        after = _receipt(restored.db, ref.session_id)
        assert after['lineage'] == [ref.session_id, 'classic-child']
        assert after['entry'] == dict(receipt['entry'], session_id='classic-child')
    finally:
        for runner in runners:
            for owner in getattr(runner, 'session_authorities', ()):
                owner.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_the_resume_hint_the_tui_prints_opens_the_session(tmp_path, monkeypatch):
    """The TUI's exit epilogue prints ``_short_resume_id`` (``local-`` + 16 hex); in-TUI ``/resume``
    sends that hint as ``session_id``."""
    from gateway.session_local import create_local_session
    from hermes_cli.main_tui_launch import _short_resume_id

    authority, connection = await _owner(tmp_path, monkeypatch, 'first')
    try:
        ref = create_local_session(authority, connection.actor, {'request_id': 'r', 'source': 'tui',
                                                                 'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
        hint = _short_resume_id(authority.db, ref.session_id)
        assert hint != ref.session_id and len(hint) == 22
        reply = await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': hint}})
        assert 'error' not in reply, reply
        assert reply['result']['session_id'] == ref.session_id
    finally:
        await connection.close()
        authority.db.close()
