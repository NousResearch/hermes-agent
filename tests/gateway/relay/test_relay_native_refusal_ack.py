"""A refused relay native admission is acknowledged on the same socket, never a replay loop."""
import asyncio
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_refused_relay_frame_is_acked_and_the_reader_keeps_serving(tmp_path, monkeypatch, caplog):
    websockets = pytest.importorskip('websockets')
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport

    sockets, outgoing, ready = [], [], asyncio.Queue()
    descriptor = CapabilityDescriptor(1, 'telegram', 'Fixture', 4096, False, True, False, 'plain', 'chars')

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(descriptor.to_json())}) + '\n')
        sockets.append(ws)
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame['type'] == 'outbound':
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                                          'result': {'success': True, 'message_id': 'sent'}}) + '\n')

    turns = []

    async def turn(event, *_args, **_kwargs):
        turns.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    async with websockets.serve(connector, '127.0.0.1', 0) as relay:
        transport = WebSocketRelayTransport(
            f'ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay', 'telegram', 'fixture-bot',
            gateway_id='fixture-gateway', upgrade_secret='fixture-secret', reconnect=True, reconnect_backoff_s=.05)
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='relay-refusal')
        adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        try:
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)
            source = {'platform': 'telegram', 'chat_id': 'relay-chat', 'chat_type': 'dm', 'user_id': 'relay-user'}
            # A platform this socket never advertised: refused (permission_denied) on every replay.
            refused = {'text': 'for another platform', 'message_id': 'r1', 'source': {**source, 'platform': 'discord'}}
            ok = {'text': 'hello', 'message_id': 'r2', 'source': source}
            for event, buffer_id in ((refused, 'buf-refused'), (ok, 'buf-ok')):
                await ws.send(json.dumps({'type': 'inbound', 'event': event, 'bufferId': buffer_id}) + '\n')
            acked = lambda: {f.get('bufferId') for f in outgoing if f.get('type') == 'inbound_ack'}
            await async_wait_until(lambda: {'buf-refused', 'buf-ok'} <= acked() and turns,
                                   timeout=20, what='both frames acknowledged and the valid one run')
            await async_wait_until(lambda: 'hello' in turns, timeout=10, what='the valid frame run')
            assert turns == ['hello']
            assert 'relay inbound refused (permission_denied)' in caplog.text
            assert len(sockets) == 1, 'the refusal tore the relay socket down and re-dialed'
        finally:
            await adapter.disconnect()
            authority.db.close()
