"""A served profile's receipts from before origin composition still resume (#140099)."""
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


@pytest.mark.asyncio
async def test_served_receipt_without_origin_profile_resumes(tmp_path, monkeypatch):
    """Receipts a served profile wrote before ``local_source`` composed ``profile`` into the
    origin (every persisted copy missing the key) must still cold-resume once the recomputed
    source carries the profile; the route, row identity and lineage never changed."""
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session

    served, db, secondary = await _served_authority(tmp_path, monkeypatch, 'first')
    actor = Principal('owner', str(secondary), frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(served, actor, {'request_id': 'legacy', 'source': 'gui',
                                               'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
    _age_receipt_origins(db, ref.session_id, lambda origin: {k: v for k, v in origin.items() if k != 'profile'})

    cold, _, _ = await _served_authority(tmp_path, monkeypatch, 'second', db=db)
    assert ref.session_id in cold.sessions, 'pre-profile receipt failed to cold-resume'
    assert cold.sessions[ref.session_id].source.profile == 'secondary'


@pytest.mark.asyncio
async def test_pre_profile_tolerance_rejects_a_foreign_profile(tmp_path, monkeypatch):
    """The tolerance is for the missing key only: an origin that names a DIFFERENT profile is a
    different session's receipt and must keep failing closed, not borrow this authority."""
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_runtime import RuntimeStoreError

    served, db, secondary = await _served_authority(tmp_path, monkeypatch, 'first')
    actor = Principal('owner', str(secondary), frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(served, actor, {'request_id': 'foreign', 'source': 'gui',
                                               'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
    _age_receipt_origins(db, ref.session_id, lambda origin: {**origin, 'profile': 'other-profile'})

    cold, _, _ = await _served_authority(tmp_path, monkeypatch, 'second', db=db)
    assert ref.session_id not in cold.sessions
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        restore_local_session(cold, ref.session_id)
