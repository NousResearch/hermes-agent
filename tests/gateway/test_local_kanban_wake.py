"""Local Kanban wakes reach the owning authority FIFO without a legacy handler.

Real adapters, profile authorities and SQLite admission; only model execution is stubbed.
"""
import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_authorities import owner_scope
from gateway.session_local_recovery import local_adapter_map
from gateway.wake import deliver_wake
from tests.gateway.test_local_producer_admission import _Gateway


@pytest.mark.asyncio
@pytest.mark.parametrize('profile', ['default', 'beta'])
async def test_local_kanban_wake_reaches_owning_profile_without_a_legacy_handler(
    tmp_path, monkeypatch, profile,
):
    gateway = _Gateway(tmp_path, monkeypatch)
    await gateway.start()
    try:
        authority = gateway.primary if profile == 'default' else gateway.secondary
        created = gateway.create(authority, gateway.connect(authority), 'kanban-origin', 'gui')
        session_id = created.session_id
        await gateway.watch(authority, session_id)
        live = authority.sessions[session_id]
        adapter = local_adapter_map(authority)[Platform.LOCAL]
        # The notifier reconstructs coordinates; direct ingress still requires source identity.
        source = SessionSource(
            platform=Platform.LOCAL, chat_id=live.source.chat_id,
            chat_type=live.source.chat_type, user_id=live.source.user_id,
            profile=live.source.profile,
        )
        assert source is not live.source
        assert not adapter.authorize_source(source)
        assert not gateway.rows(authority, session_id)

        # The notifier binds its subscriber's owner before waking the canonical route.
        with owner_scope(authority):
            await deliver_wake(adapter, source=source, text='SYNTHETIC_KANBAN_HANDOFF')
        await gateway.settled(authority, session_id, 1)

        [row] = gateway.rows(authority, session_id)
        assert (row['status'], row['outcome']) == ('terminal', 'completed')
        assert 'local_automation_v1' in row['payload']
        [turn] = gateway.turns
        assert (turn.profile, turn.chat, turn.text, turn.policy) == (
            live.source.profile, session_id, 'SYNTHETIC_KANBAN_HANDOFF',
            (f'{profile}-model', str(tmp_path)),
        )
        other = gateway.primary if profile == 'beta' else gateway.secondary
        assert not gateway.rows(other, session_id)
        # Canonical automation must not broaden direct local-source authorization.
        assert not adapter.authorize_source(source)
    finally:
        await gateway.stop()
