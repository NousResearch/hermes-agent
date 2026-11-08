"""An IDLE phone (WebUI) chat hands its slot to a native resume; a mid-turn one keeps refusing.

The dashboard/phone claims a ``webui`` active-session slot on its first turn and holds it for the
life of the open socket, so a chat the user had finished on the phone refused every desktop resume
with ``runtime_coordination_required``. The handoff is gated on the durable session turn lease, the
only cross-process proof that no turn is running.
"""
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _registry_entries(home):
    """The registry file as another process would read it (no lock: the guard may hold it)."""
    path = Path(home) / 'runtime' / 'active_sessions.json'
    return json.loads(path.read_text(encoding='utf-8'))['entries'] if path.exists() else []


def _authority_for(tmp_path, monkeypatch, sid, *, source='webui'):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    db.create_session(sid, source=source, model='old-model', cwd=str(tmp_path))
    db.append_message(sid, role='user', content='FROM THE PHONE')
    db.append_message(sid, role='assistant', content='PHONE ANSWER')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = asyncio.run(
        initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='first'))
    native = Principal('native', str(tmp_path), frozenset({'session:read', 'session:create'}), 'socket')
    authority._native_legacy_transports = {native.transport_id: native}
    return authority, db, native


def _hold_webui_slot(db, sid):
    from hermes_cli.active_sessions import try_acquire_active_session
    lease, refusal = try_acquire_active_session(
        session_id=sid, surface='webui', config={}, registry_home=Path(db.db_path).parent,
        metadata={'live_session_id': 'phone-socket'})
    assert lease is not None and refusal is None, refusal
    return lease


def _stack_turn(db, sid):
    """A turn is running on the phone: its durable turn lease is live (the same call the agent's
    turn admission makes, so the row is committed for other processes to see)."""
    assert db.try_acquire_session_turn_lease(sid, f'pid={time.time_ns() % 60000}:turn=phone:platform=webui')


def test_idle_webui_holder_hands_the_session_to_a_native_resume(tmp_path, monkeypatch):
    from gateway.session_local_title import resolve_titled_session
    from hermes_cli.active_sessions import (
        active_session_liveness_guard, active_session_registry_snapshot, lease_is_registered)

    sid = '20261107_231452_3effe7'
    authority, db, native = _authority_for(tmp_path, monkeypatch, sid)
    lease = _hold_webui_slot(db, sid)
    home = Path(db.db_path).parent

    # Today's refusal: while the holder's entry is in the registry, the session is "live owned".
    with active_session_liveness_guard(sid, registry_home=home) as active:
        assert active

    # The handoff happens INSIDE the liveness lock and BEFORE the adoption commit: capture what the
    # registry says at the moment the row is bound.
    import hermes_state_local
    committed = {}
    real_commit = hermes_state_local.commit_local_session

    def spy(database, *, epoch, receipt):
        committed['registry'] = _registry_entries(home)
        return real_commit(database, epoch=epoch, receipt=receipt)

    monkeypatch.setattr(hermes_state_local, 'commit_local_session', spy)

    ref = resolve_titled_session(authority, native, sid)

    assert ref.session_id == sid
    assert [e for e in committed['registry'] if e.get('session_id') == sid] == []
    assert active_session_registry_snapshot(registry_home=home) == []
    assert db.get_session(sid)['session_key']  # bound: the desktop's route owns the row now
    # The phone's own token no longer names a slot, so its runtime has nothing to run a turn under.
    assert lease_is_registered(lease) is False


def test_a_mid_turn_webui_holder_keeps_the_refusal_and_its_slot(tmp_path, monkeypatch):
    from gateway.session_local_title import resolve_titled_session
    from hermes_state_runtime import RuntimeStoreError

    sid = '20261108_012937_4af988'
    authority, db, native = _authority_for(tmp_path, monkeypatch, sid)
    lease = _hold_webui_slot(db, sid)
    _stack_turn(db, sid)
    home = Path(db.db_path).parent

    with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
        resolve_titled_session(authority, native, sid)

    # Nothing was handed over for a running turn: the holder keeps its slot (this is what makes the
    # test fail if the idle condition is dropped from the yield, not just the refusal below it).
    assert [e for e in _registry_entries(home) if e.get('session_id') == sid]
    assert not db.get_session(sid)['session_key']
    lease.release()
