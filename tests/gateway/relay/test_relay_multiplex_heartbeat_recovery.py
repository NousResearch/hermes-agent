"""Settled connector-stamped routes still deliver their first restored heartbeat."""
import json

import pytest

from gateway.runtime_ownership import process_ownership
from tests.gateway.relay.test_relay_native_recovery_and_passthrough import (
    _Connector, _multiplexed_runner, _reserve_homes, _rows, _stub_turns, _wired_relay, websockets,
)
from tests.waiting import async_wait_until


def _replace_connector(authority) -> None:
    payload = _rows(authority)[0]['payload']
    payload['native_text_v1']['provenance']['connector'] = 'replaced-connector'
    with authority.db._conn:
        authority.db._conn.execute('UPDATE session_admissions SET payload_json=?', (json.dumps(payload),))


@pytest.mark.asyncio
@pytest.mark.parametrize('valid_connector', [True, False], ids=['verified', 'changed-connector'])
async def test_settled_multiplex_relay_heartbeat_restores_verified_owner(tmp_path, monkeypatch, valid_connector):
    from gateway.run_runtime import initialize_gateway_runtime
    import gateway.run_heartbeat_acceptance as acceptance

    root, work = _reserve_homes(tmp_path, monkeypatch)
    homes = (('default', root), ('work', work))
    connector, turns = _Connector('discord'), []
    raw = {'text': 'settled work', 'message_id': 'settled-work', 'source': {
        'platform': 'discord', 'chat_id': 'work-heartbeat', 'user_id': 'work-owner',
        'scope_id': 'work-guild', 'profile': 'work'}}
    async with websockets.serve(connector, '127.0.0.1', 0) as server:
        url = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/relay'
        for boot in range(2):
            process_ownership.reserve([home for _, home in homes])
            runner = _multiplexed_runner(tmp_path, homes)
            _stub_turns(runner, monkeypatch, turns)
            try:
                await initialize_gateway_runtime(runner)
                async with _wired_relay(runner, connector, url) as (_, ws):
                    if boot == 0:
                        await ws.send(json.dumps({'type': 'inbound', 'event': raw, 'bufferId': 'settled'}) + '\n')
                        await async_wait_until(lambda: _rows(runner.session_authorities.for_home(work)) and all(
                            row['status'] == 'terminal' for row in _rows(runner.session_authorities.for_home(work))),
                            timeout=20, what='work input settled')
                        if not valid_connector:
                            _replace_connector(runner.session_authorities.for_home(work))
                    else:
                        seen = []
                        async def admit(_runner, adapter, source, session_id, route):
                            seen.append(adapter._with_scope(source.chat_id, None))
                        monkeypatch.setattr(acceptance, 'admit_heartbeat', admit)
                        entry = next(e for e in runner.session_store.list_sessions()
                                     if e.origin and e.origin.chat_id == 'work-heartbeat')
                        source = runner._restored_source(entry)
                        await runner._heartbeat_poll_watch({entry.session_key: (source, entry.session_id)},
                                                           entry.session_key, source, entry.session_id)
                        if valid_connector:
                            assert seen, 'the restored connector-stamped heartbeat was dropped'
                            assert seen[0] == {'scope_id': 'work-guild', 'user_id': 'work-owner', 'profile': 'work'}
                        else:
                            assert seen == [], 'unverified admission provenance authorized a Relay fallback'
            finally:
                for authority in list(runner.session_authorities):
                    authority.db.close()
                for _, home in homes:
                    process_ownership.release(home)
