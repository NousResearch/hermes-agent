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
    from gateway.session_controls import AuthorityConnection
    from hermes_cli.active_sessions import release_active_session
    from tests.fixtures.storage_seed import seed_messages, seed_session

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    # A chat the dashboard's Desktop pane ran: a plain desktop row, no canonical receipt.
    seed_session(db, CHAT, 'desktop', model='m', cwd=str(tmp_path))
    seed_messages(db, CHAT, {'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'})
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _adapter_for_source=lambda source: None, _cached_agent_for=lambda route: None,
                             _evict_cached_agent=lambda route: None)
    identity = {'user_id': 'native', 'provider': 'local', 'profile_id': str(tmp_path),
                'instance_id': 'gateway'}
    authority = asyncio.run(initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='gateway'))
    connection = AuthorityConnection(authority, object(), identity)
    native = connection.actor
    leases = []
    try:
        yield SimpleNamespace(authority=authority, db=db, home=Path(db.scope.profile_dir), native=native,
                               leases=leases, connection=connection)
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
    lease = _mount_dashboard_pane(owner)

    reply = await owner.connection.dispatch({'id': 1, 'method': 'session.resume',
                                             'params': {'session_id': CHAT, 'omit_messages': True}})

    assert 'result' in reply, reply
    assert reply['result']['session_id'] == CHAT
    assert reply['result']['messages_omitted'] is True
    assert reply['result']['message_count'] == 2
    assert not lease.released, 'the pane keeps its lease beside the gateway'


@pytest.mark.asyncio
async def test_a_dashboard_pane_mid_turn_still_refuses_and_binds_nothing(owner):
    _mount_dashboard_pane(owner)
    holder = f'pid={os.getpid()}:turn=1ce47391:platform=desktop'
    assert owner.db.coordination.try_acquire_turn_lease(session_id=CHAT, holder=holder, ttl_seconds=60)

    reply = await owner.connection.dispatch({'id': 2, 'method': 'session.resume',
                                             'params': {'session_id': CHAT, 'omit_messages': True}})
    assert reply.get('error', {}).get('data', {}).get('reason') == 'runtime_coordination_required', reply
    assert not owner.db.bindings.binding_marks(CHAT).policy

    owner.db.coordination.release_turn_lease(session_id=CHAT, holder=holder)
