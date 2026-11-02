"""Preview rejects unknown session routing keys before cold receipt adoption."""

from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
from hermes_state import SessionDB
from tests.gateway.test_local_receipt_untracked_compression import (
    _compact_like_0_22_2,
    _owner,
    _receipt,
    _unrecovered_owner,
)
from tui_gateway.contracts.registry import CANONICAL_METHODS, METHODS


def _state(db, session_id):
    receipt = _receipt(db, session_id)
    with db._read_ctx() as conn:
        route = conn.execute(
            "SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?",
            (receipt['route'],),
        ).fetchone()
        revision = conn.execute(
            'SELECT runtime_revision FROM sessions WHERE id=?', (session_id,),
        ).fetchone()[0]
        dump = tuple(conn.iterdump())
    assert route is not None
    return {'receipt': receipt, 'route': route[0], 'revision': revision, 'dump': dump}


@pytest.mark.asyncio
@pytest.mark.parametrize('no_read', [False, True], ids=['ordinary', 'no-read'])
async def test_preview_extra_session_id_cannot_adopt_cold_receipt(tmp_path, monkeypatch, no_read):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    first, creator = await _owner(tmp_path, monkeypatch, 'first')
    ref = create_local_session(first, creator.actor, {
        'request_id': 'r', 'source': 'tui', 'cwd': str(tmp_path),
        'model': 'no-provider', 'toolsets': [],
    })
    path = first.db.db_path
    await creator.close()
    first.db.close()
    with SessionDB(db_path=path) as classic:
        _compact_like_0_22_2(classic, ref.session_id, 'classic-child')
    cold, connection = _unrecovered_owner(tmp_path, 'cold')
    if no_read:
        connection = AuthorityConnection(cold, object(), {'user_id': 'owner', 'capabilities': []})
    try:
        before = _state(cold.db, ref.session_id)
        assert before['receipt']['entry']['session_id'] == ref.session_id
        reply = await connection.dispatch({'id': 1, 'method': 'groups.preview', 'params': {
            'room_id': 'unowned', 'text': '@two', 'session_id': ref.session_id,
        }})
        after = _state(cold.db, ref.session_id)
        # Check individual components as well as the complete SQLite image.
        assert after['receipt'] == before['receipt']
        assert after['route'] == before['route']
        assert after['revision'] == before['revision']
        assert after['dump'] == before['dump']
        assert reply['error']['message'] == 'invalid_params'
        assert 'session_id' in reply['error']['data']['fields']
    finally:
        await connection.close()
        cold.db.close()


@pytest.mark.asyncio
async def test_preview_is_advertised_by_canonical_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(profile_id=str(tmp_path), instance_id='owner', db=db, events={})
        connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        try:
            reply = await connection.dispatch({'id': 1, 'method': 'groups.capabilities', 'params': {}})
            assert 'result' in reply, reply
            assert 'groups.preview' in reply['result']['methods']
        finally:
            await connection.close()


def test_preview_shared_and_canonical_wire_shapes_match():
    shared = METHODS['groups.preview']
    canonical = CANONICAL_METHODS['groups.preview']
    assert shared.params.model_json_schema() == canonical.params.model_json_schema()
    assert shared.result.model_json_schema() == canonical.result.model_json_schema()
