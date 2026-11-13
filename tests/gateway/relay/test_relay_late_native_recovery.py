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
@pytest.mark.parametrize(('unknown', 'recovery_error'), [(False, False), (True, False), (False, True)],
                         ids=['queued', 'unknown-paused', 'recovery-error-retried'])
async def test_late_relay_connection_recovers_durable_queue(tmp_path, monkeypatch, unknown, recovery_error):
    """``recovery-error-retried``: a store error in the first recovery pass is not a connection failure:
    the published adapter stays installed and recovery retries in the background."""
    from gateway import run_runtime
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
            if recovery_error:
                _fail_first_recovery(monkeypatch, run_runtime)
                runner._running = True
            async with _wired_relay(runner, connector, url) as (adapter, _):
                await runner._install_reconnected_adapter(Platform.RELAY, adapter)
                assert runner.adapters.get(Platform.RELAY) is adapter
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


def _fail_first_recovery(monkeypatch, run_runtime):
    import gateway.session_native_recovery_routes as routes

    real, calls = run_runtime.recover_profile_native_sessions, []

    async def flaky(runner, authority):
        calls.append(authority)
        if len(calls) == 1:
            raise RuntimeError('database is locked')
        return await real(runner, authority)

    monkeypatch.setattr(run_runtime, 'recover_profile_native_sessions', flaky)
    monkeypatch.setattr(routes, '_RECOVERY_RETRY_DELAYS', (0,))


def test_latest_session_admission_reads_newest_first_and_stops_at_the_first_match():
    """A restored lane reads only its newest native envelope, not the session's whole history."""
    import contextlib
    import sqlite3

    from hermes_state_runtime import latest_session_admission

    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE session_admissions (target_session_id, seq, payload_json, payload_digest, lineage_json)')
    payloads = [{'native_text_v1': {'n': n}} for n in range(50)] + [{'automation': True}]
    conn.executemany('INSERT INTO session_admissions VALUES (?, ?, ?, ?, ?)',
                     [('s1', seq, json.dumps(p), '', '[]') for seq, p in enumerate(payloads)])
    db = type('Db', (), {'_read_ctx': lambda self: contextlib.nullcontext(conn)})()
    seen = []

    def accept(payload):
        seen.append(payload)
        return 'native_text_v1' in payload

    row = latest_session_admission(db, session_id='s1', accept=accept)

    assert row['payload'] == {'native_text_v1': {'n': 49}}
    assert len(seen) == 2  # the newer non-native row, then the newest native one
    assert latest_session_admission(db, session_id='other', accept=accept) is None
