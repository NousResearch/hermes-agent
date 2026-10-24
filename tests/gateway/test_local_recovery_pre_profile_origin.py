"""A served profile's refused local receipt names its failing check in gateway.log (#140099).

The restore cases themselves (an absent origin ``profile`` restores; a null or foreign one, or a
mismatched route, home or user, refuses) live in ``test_local_receipt_missing_profile.py``.
"""
import json
from types import SimpleNamespace

import pytest


def _build_runtime(tmp_path, monkeypatch):
    """A fresh runner/registry/store over the reserved homes (launch ``default`` + ``secondary``):
    a cold restart builds a second one over the same on-disk state."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authorities import SessionAuthorities
    from gateway import run
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700, exist_ok=True)
    monkeypatch.setenv('HERMES_HOME', str(root))
    # The autouse isolation fixture pins hermes_state.DEFAULT_DB_PATH to its own fake home once
    # hermes_state is imported; per-home stores must follow the scoped HERMES_HOME here.
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    secondary = root / 'profiles' / 'secondary'
    secondary.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not (secondary / 'config.yaml').exists():
        (secondary / 'config.yaml').write_text('{}')
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    registry = SessionAuthorities(root, multiplexed=True)
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={},
                             _draining=False, session_authorities=registry)
    return runner, registry, root, secondary


async def _served_authority(tmp_path, monkeypatch, instance_id, db=None):
    """Launch home plus one served secondary sharing the runner's store, as a multiplex gateway
    builds them; returns (served authority, its state db, the secondary home path)."""
    from gateway.session_authority import initialize_session_authority
    from hermes_state import SessionDB
    runner, registry, root, secondary = _build_runtime(tmp_path, monkeypatch)
    launch = await initialize_session_authority(runner, profile_id=str(root), instance_id=instance_id)
    registry.add(root, launch, name='default')
    if db is None:
        db = SessionDB(db_path=secondary / 'state.db')
    # Reserve-then-replace, as the multiplex bring-up does: recovery inside initialize must
    # already see the served slot (and thus the profile name) to rebuild a profiled source.
    registry.add(secondary, None, name='secondary')
    served = await initialize_session_authority(runner, profile_id=str(secondary),
                                                instance_id=instance_id, register=False, db=db)
    registry.replace(secondary, served)
    return served, db, secondary


def _age_receipt_origins(db, sid, mutate):
    """Rewrite every persisted origin copy of *sid*'s receipt through *mutate* (a dict -> dict
    function), the way an older build's on-disk spelling differs from the recomputed source."""
    from hermes_state_local import POLICY_PREFIX
    with db._read_ctx() as conn:
        row = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()
    receipt = json.loads(row[0])
    receipt['entry']['origin'] = mutate(receipt['entry']['origin'])
    db._write_sql('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), POLICY_PREFIX + sid))

    def age_routing(conn):
        rows = conn.execute('SELECT session_key, entry_json FROM gateway_routing WHERE session_key=?',
                            (receipt['route'],)).fetchall()
        for key, entry_json in rows:
            entry = json.loads(entry_json)
            if entry.get('origin') is not None:
                entry['origin'] = mutate(entry['origin'])
                conn.execute('UPDATE gateway_routing SET entry_json=? WHERE session_key=?',
                             (json.dumps(entry), key))
    db._execute_write(age_routing)
    with db._read_ctx() as conn:
        row = conn.execute('SELECT origin_json FROM sessions WHERE id=?', (sid,)).fetchone()
    db._write_sql('UPDATE sessions SET origin_json=? WHERE id=?', (json.dumps(mutate(json.loads(row[0]))), sid))
    return receipt


def _store_receipt(db, sid, mutate):
    """Rewrite *sid*'s stored receipt text through *mutate* (a str -> str function)."""
    from hermes_state_local import POLICY_PREFIX
    with db._read_ctx() as conn:
        stored = conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()[0]
    db._write_sql('UPDATE state_meta SET value=? WHERE key=?', (mutate(stored), POLICY_PREFIX + sid))


def _with_receipt_field(path, value):
    """A receipt-text mutation that sets the nested field at *path* to *value*."""
    def mutate(stored):
        receipt = json.loads(stored)
        node = receipt
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return json.dumps(receipt)
    return mutate


_RECEIPT_DAMAGE = {
    'foreign_origin': lambda db, sid: _age_receipt_origins(db, sid, lambda origin: {**origin, 'profile': 'other-profile'}),
    'lineage': lambda db, sid: _store_receipt(db, sid, _with_receipt_field(('lineage',), [sid, 'gone'])),
    # restore_policy wraps this ValueError into a bare storage_unavailable.
    'terminal_policy': lambda db, sid: _store_receipt(db, sid, _with_receipt_field(
        ('policy', 'terminal_json'), json.dumps({'TERMINAL_ENV': 'local', 'TERMINAL_CWD': '/elsewhere'}))),
    # local_receipt itself refuses undecodable text.
    'malformed': lambda db, sid: _store_receipt(db, sid, lambda stored: stored[:-1]),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('damage, clause', [
    ('foreign_origin', 'entry mismatch'),
    ('lineage', 'validate_local_lineage'),
    ('terminal_policy', 'ValueError: terminal policy mismatch'),
    ('malformed', 'JSONDecodeError'),
])
async def test_a_refused_receipt_names_its_failing_check_in_the_log(tmp_path, monkeypatch, caplog, damage, clause):
    """#140099: the client only ever sees ``storage_unavailable``; gateway.log must name the check that
    refused the receipt (the wrapped cause, or where an inner check raised), on cold recovery and on
    every later restore, so a field report pins the layer without offline replication."""
    import logging
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_runtime import RuntimeStoreError

    served, db, secondary = await _served_authority(tmp_path, monkeypatch, 'first')
    actor = Principal('owner', str(secondary), frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(served, actor, {'request_id': damage, 'source': 'gui',
                                               'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
    _RECEIPT_DAMAGE[damage](db, ref.session_id)

    with caplog.at_level(logging.WARNING, logger='gateway.session_local_recovery'):
        cold, _, _ = await _served_authority(tmp_path, monkeypatch, 'second', db=db)
        assert ref.session_id not in cold.sessions
        recovery = [r.getMessage() for r in caplog.records if ref.session_id in r.getMessage()]
        assert any(clause in message for message in recovery), recovery
        assert len(recovery) == 1, f'one boot warning per refused receipt: {recovery}'
        caplog.clear()
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            restore_local_session(cold, ref.session_id)
    resumed = [r.getMessage() for r in caplog.records if ref.session_id in r.getMessage()]
    assert any(clause in message for message in resumed), resumed
