"""A local receipt whose session row is gone is released at boot, not refused forever (#145499).

Deleting a local session removes its rows, but a receipt can outlive them (a delete path that
predates the receipt fence, a profile move that removed the row in the source store). Every
recovery pass then refused that receipt with the same ``stored identity mismatch`` a real identity
conflict raises, one WARNING per orphan per pass, forever, and the operator cannot tell the benign
orphan from an integrity signal.

Boot now retires such a receipt when its id holds nothing (no transcript, ledger row or child),
and a missing row that still holds something reports ``session_missing`` instead of the misleading
mismatch.
"""
import logging
from types import SimpleNamespace

import pytest


async def _boot(tmp_path, monkeypatch, instance_id='owner'):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(
        session_store=store, _session_db=store._db, adapters={}, _draining=False,
        _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
        _finalize_session_off_loop=lambda **kwargs: [])
    authority = await initialize_session_authority(
        runner, profile_id=str(tmp_path), instance_id=instance_id)
    return authority, store


def _create(authority, tmp_path, request_id='leaked-receipt'):
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    viewer = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    return create_local_session(authority, viewer.actor, {
        'request_id': request_id, 'source': 'gui', 'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})


def _drop_rows_without_fence(db, sid):
    """The end state a delete path that never fenced receipts leaves: no row, receipt kept."""
    def write(conn):
        conn.execute('DELETE FROM messages WHERE session_id = ?', (sid,))
        conn.execute('DELETE FROM sessions WHERE id = ?', (sid,))
    db._execute_write(write)


@pytest.mark.asyncio
async def test_orphaned_receipt_is_released_at_boot(tmp_path, monkeypatch, caplog):
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError

    authority, _ = await _boot(tmp_path, monkeypatch)
    ref = _create(authority, tmp_path)
    assert local_receipt(authority.db, ref.session_id)['entry']['session_id'] == ref.session_id
    _drop_rows_without_fence(authority.db, ref.session_id)

    with caplog.at_level(logging.INFO, logger='gateway.session_local_recovery'):
        second, _ = await _boot(tmp_path, monkeypatch, instance_id='second')
    messages = [record.getMessage() for record in caplog.records]
    assert not [line for line in messages if 'receipt refused' in line], messages
    assert [line for line in messages if 'receipt dropped' in line and ref.session_id in line], messages
    with pytest.raises(RuntimeStoreError):
        local_receipt(second.db, ref.session_id)


@pytest.mark.asyncio
async def test_a_missing_row_is_named_apart_from_an_identity_mismatch(tmp_path, monkeypatch):
    """``_checked_entry``: the benign orphan and a real identity conflict are different reasons."""
    from gateway.session_local_recovery import _checked_entry
    from hermes_state_local import local_receipt

    authority, _ = await _boot(tmp_path, monkeypatch)
    ref = _create(authority, tmp_path)
    live = authority.sessions[ref.session_id]
    receipt = local_receipt(authority.db, ref.session_id)
    assert _checked_entry(authority, receipt, live.source, live.route).session_id == ref.session_id

    orphan = SimpleNamespace(db=SimpleNamespace(get_session=lambda sid: None))
    with pytest.raises(ValueError, match='session_missing'):
        _checked_entry(orphan, receipt, live.source, live.route)

    foreign = SimpleNamespace(db=SimpleNamespace(get_session=lambda sid: {
        'session_key': 'agent:other:local:elsewhere', 'chat_id': 'elsewhere', 'user_id': 'somebody'}))
    with pytest.raises(ValueError, match='stored identity mismatch'):
        _checked_entry(foreign, receipt, live.source, live.route)
