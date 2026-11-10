"""An idle Desktop pane on a dashboard backend no longer fences opening its chat elsewhere (#148450).

A ``hermes dashboard`` backend keeps a chat's active-session lease for as long as its pane stays
mounted, and its pid stays alive for days. The chat itself is idle: no turn holds the durable turn
lease. Opening that chat through the gateway (Desktop on the Webapp URL, ``/handoff``) adopts the
row; that must not be refused just because the pane is still mounted. A pane that is mid-turn, and
any holder that writes outside the durable turn lease, still refuses.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

CHAT = '20261108_111343_ab5461'


@pytest.fixture
def owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from hermes_cli.active_sessions import release_active_session

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    # A chat the dashboard's Desktop pane ran: a plain desktop row, no canonical receipt.
    db.create_session(CHAT, source='desktop', model='m', cwd=str(tmp_path))
    db.set_session_title(CHAT, 'Rodar dag-ulw via interface desktop')
    db.append_message(CHAT, role='user', content='Q')
    db.append_message(CHAT, role='assistant', content='A')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False)
    authority = asyncio.run(initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='gateway'))
    caps = frozenset({'session:create', 'session:read', 'session:submit', 'session:control'})
    native = SessionActor('native', str(tmp_path), caps, 'native-socket')
    authority._native_legacy_transports = {native.transport_id: native}
    leases = []
    try:
        yield SimpleNamespace(authority=authority, db=db, home=Path(db.db_path).parent, native=native, leases=leases)
    finally:
        for lease in leases:
            release_active_session(lease)


def _mount_dashboard_pane(owner):
    """The dashboard's claim on the pane's first turn (tui_gateway ``_ensure_active_session_slot``)."""
    from tui_gateway import server
    lease, refusal = server._claim_active_session_slot(
        CHAT, live_session_id='1ce47391', surface='desktop', profile_home=owner.home)
    assert lease is not None, refusal
    owner.leases.append(lease)
    return lease


@pytest.mark.asyncio
async def test_an_idle_dashboard_pane_does_not_fence_opening_its_chat_through_the_gateway(owner):
    from gateway.session_local_migration import resolve_local_target
    lease = _mount_dashboard_pane(owner)

    ref = resolve_local_target(owner.authority, owner.native, CHAT)

    snapshot = await owner.authority.attach(owner.native, ref)
    assert [m['content'] for m in snapshot.history] == ['Q', 'A']
    assert not lease.released, 'the pane keeps its lease beside the gateway'


@pytest.mark.asyncio
async def test_a_dashboard_pane_mid_turn_still_refuses_and_binds_nothing(owner):
    from gateway.session_local_migration import resolve_local_target
    from hermes_state_runtime import RuntimeStoreError
    _mount_dashboard_pane(owner)
    holder = f'pid={os.getpid()}:turn=1ce47391:platform=desktop'
    assert owner.db.try_acquire_session_turn_lease(CHAT, holder)

    with pytest.raises(RuntimeStoreError, match='runtime_coordination_required'):
        resolve_local_target(owner.authority, owner.native, CHAT)
    with owner.db._read_ctx() as conn:
        assert not conn.execute("SELECT 1 FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'").fetchall()

    owner.db.release_session_turn_lease(CHAT, holder)
    assert resolve_local_target(owner.authority, owner.native, CHAT).session_id == CHAT
