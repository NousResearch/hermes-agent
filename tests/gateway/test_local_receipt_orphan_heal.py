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


@pytest.mark.asyncio
async def test_the_heal_leaves_a_live_receipt_alone(tmp_path, monkeypatch):
    """The heal retires only the orphan: a sibling whose row is still there keeps its receipt, and
    recovery still restores it. Retiring a live receipt would drop the session's launch policy."""
    from gateway.session_local_recovery import local_receipt_ids
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError

    authority, _ = await _boot(tmp_path, monkeypatch)
    leaked = _create(authority, tmp_path, 'leaked-receipt')
    live = _create(authority, tmp_path, 'live-receipt')
    _drop_rows_without_fence(authority.db, leaked.session_id)

    second, _ = await _boot(tmp_path, monkeypatch, instance_id='second')
    with pytest.raises(RuntimeStoreError):
        local_receipt(second.db, leaked.session_id)
    assert local_receipt(second.db, live.session_id)['entry']['session_id'] == live.session_id
    assert live.session_id in local_receipt_ids(second)


@pytest.mark.asyncio
async def test_a_missing_row_that_still_holds_history_keeps_its_receipt(tmp_path, monkeypatch, caplog):
    """The safety valve: a receipt is retired only when its id holds nothing. A row that is gone
    while its transcript survives (a store written before the FK was enforced, a repair that removed
    the row alone) keeps the receipt, so the conversation's policy is not lost, and reports the new
    reason instead of the mismatch a real conflict raises."""
    import sqlite3

    from hermes_state_local import local_receipt

    authority, _ = await _boot(tmp_path, monkeypatch)
    ref = _create(authority, tmp_path)
    sid = ref.session_id
    # The FK on messages, admissions and workers is what normally makes an empty row the only
    # reachable "gone" state; a store written with foreign_keys off still carries the history.
    conn = sqlite3.connect(authority.db.db_path)
    conn.execute('PRAGMA foreign_keys=OFF')
    conn.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'kept', 0)",
                 (sid,))
    conn.execute('DELETE FROM sessions WHERE id = ?', (sid,))
    conn.commit()
    conn.close()

    with caplog.at_level(logging.WARNING, logger='gateway.session_local_recovery'):
        second, _ = await _boot(tmp_path, monkeypatch, instance_id='second')
    messages = [record.getMessage() for record in caplog.records]
    assert local_receipt(second.db, sid)['entry']['session_id'] == sid
    assert [line for line in messages if 'session_missing' in line], messages
    assert not [line for line in messages if 'stored identity mismatch' in line], messages
