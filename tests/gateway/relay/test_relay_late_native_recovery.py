"""Queued relay work is retried on late connection; unknown work stays paused."""
import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from tests.gateway.relay.test_relay_native_recovery_and_passthrough import (
    _Connector, _rows, _stub_turns, _wired_relay, websockets,
)
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('unknown', [False, True], ids=['queued', 'unknown-paused'])
async def test_late_relay_connection_recovers_durable_queue(tmp_path, monkeypatch, unknown):
    from gateway.run_runtime import recover_gateway_native_sessions

    connector, turns = _Connector('discord'), []
    config = GatewayConfig(sessions_dir=tmp_path / 'sessions')
    raw = {'text': 'late connection', 'message_id': 'late-1', 'source': {
        'platform': 'discord', 'chat_id': 'late-chat', 'user_id': 'late-owner', 'scope_id': 'late-guild'}}
    async with websockets.serve(connector, '127.0.0.1', 0) as server:
        url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay'
        runner = GatewayRunner(config)
        with monkeypatch.context() as held:
            held.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
            authority = await initialize_session_authority(runner, profile_id='default', instance_id='late-1')
            try:
                async with _wired_relay(runner, connector, url) as (_, ws):
                    await ws.send(json.dumps({'type': 'inbound', 'event': raw, 'bufferId': 'late'}) + '\n')
                    await async_wait_until(lambda: any(f.get('type') == 'inbound_ack' for f in connector.outgoing),
                                           timeout=10, what='queued admission ACK')
                if unknown:
                    with authority.db._conn:
                        authority.db._conn.execute("UPDATE session_admissions SET status='unknown'")
            finally:
                authority.db.close()

        runner = GatewayRunner(config)
        _stub_turns(runner, monkeypatch, turns)
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='late-2')
        async def no_hosted(_runner):
            return None
        monkeypatch.setattr('gateway.session_hosted_service.ensure_hosted_service', no_hosted)
        monkeypatch.setattr(runner, '_schedule_planned_restart_replay', lambda: None)
        try:
            assert set((await recover_gateway_native_sessions(runner)).values()) == {'not_found'}
            runner._failed_platforms[Platform.RELAY] = {}
            async with _wired_relay(runner, connector, url) as (adapter, _):
                await runner._install_reconnected_adapter(Platform.RELAY, adapter)
                if unknown:
                    sid = authority.db._read_one('SELECT target_session_id FROM session_admissions')['target_session_id']
                    assert sid in authority.sessions, 'late connection did not bind the paused session'
                    assert [row['status'] for row in _rows(authority)] == ['unknown']
                    assert turns == []
                else:
                    await async_wait_until(lambda: turns and connector.replies('late-chat'), timeout=20,
                                           what='late reconnect drains the acknowledged queue')
                    assert turns == ['late connection']
                    assert connector.replies('late-chat')[-1]['metadata']['user_id'] == 'late-owner'
        finally:
            authority.db.close()
