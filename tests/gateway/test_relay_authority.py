"""Real authenticated relay ingress shares the ordinary execution authority."""
import asyncio
import base64
from copy import deepcopy
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def test_authenticated_relay_uses_canonical_admission(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(tmp_path), PYTHONUNBUFFERED='1')
    result = subprocess.run([sys.executable, __file__, 'peer'], env=env,
                            capture_output=True, text=True, timeout=100)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / 'relay-receipt.json').read_text())
    assert receipt['same_agent'] and receipt['negative_controls']
    print(json.dumps(receipt))


async def probe(peer):
    import websockets
    from gateway.config import Platform, PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport, _event_from_wire
    from gateway.run import GatewayRunner
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_envelope import prepare_native, restore_native
    from hermes_cli import web_server
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions

    sockets, outgoing, frames = [], [], []
    secret = 'owned-relay-fixture-secret'
    ready = asyncio.Queue()
    revoke = False
    descriptor = CapabilityDescriptor(1, 'telegram', 'Fixture', 4096, False, True,
                                      False, 'plain', 'chars')

    async def connector(ws):
        token = ws.request.headers['Authorization'].removeprefix('Bearer ')
        decoded = base64.urlsafe_b64decode(token + '=' * (-len(token) % 4)).decode()
        identity, expiry, signature = decoded.rsplit(':', 2)
        assert identity == 'fixture-gateway' and int(expiry) > time.time()
        assert hmac.compare_digest(signature, hmac.new(secret.encode(),
            f'{identity}:{expiry}'.encode(), hashlib.sha256).hexdigest())
        hello = json.loads(await ws.recv())
        assert hello['platform'] == 'telegram'
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(descriptor.to_json())}) + '\n')
        if revoke:
            await ws.close(code=4401, reason='revoked')
            return
        sockets.append(ws)
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame['type'] == 'outbound':
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                    'result': {'success': True, 'message_id': 'one-final'}}) + '\n')
            elif frame['type'] == 'going_idle':
                await ws.send(json.dumps({'type': 'going_idle_ack'}) + '\n')

    async with websockets.serve(connector, '127.0.0.1', 0) as relay:
        transport = WebSocketRelayTransport(
            f'ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay', 'telegram', 'fixture-bot',
            gateway_id='fixture-gateway', upgrade_secret=secret,
            reconnect=True, reconnect_backoff_s=.05)
        runner = GatewayRunner()
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='relay-test')
        adapter = RelayAdapter(PlatformConfig(enabled=True),
            descriptor, transport)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        api = await start_gateway_api(runner)
        assert await adapter.connect()
        connector_ws = await asyncio.wait_for(ready.get(), 5)
        raw = {'text': 'MESSAGING_WARM', 'message_id': 'relay-1', 'source': {
            'platform': 'telegram', 'chat_id': 'relay-chat', 'chat_type': 'dm', 'user_id': 'relay-user'}}

        async def send(raw_event):
            await connector_ws.send(json.dumps({'type': 'inbound', 'event': raw_event,
                                               'bufferId': raw_event['message_id']}) + '\n')
            async with asyncio.timeout(25):
                while True:
                    entries = list(authority.sessions.values())
                    rows = list_session_admissions(authority.db, session_id=next(iter(authority.sessions)),
                        pending_only=False) if entries else []
                    if any(r['request_id'] == raw_event['message_id'] and r['status'] == 'terminal' for r in rows):
                        while adapter._active_sessions:
                            await asyncio.sleep(.01)
                        assert any(f.get('type') == 'inbound_ack' and f.get('bufferId') == raw_event['message_id']
                                   for f in outgoing), outgoing
                        return rows
                    # Fail on the real adapter's completed rejection, rather than waiting for a timeout.
                    errors = [f for f in outgoing if 'invalid_params' in json.dumps(f) or 'not_found' in json.dumps(f)]
                    assert not errors, errors
                    await asyncio.sleep(.02)

        try:
            failed_insert = asyncio.Event()
            authority.db._execute_write(lambda conn: conn.create_function(
                'ingress_failure_witness', 0, lambda: failed_insert.set() or 1))
            authority.db._execute_write(lambda conn: conn.execute("CREATE TRIGGER fail_relay BEFORE INSERT ON session_admissions BEGIN SELECT ingress_failure_witness(); SELECT RAISE(ABORT, 'owned storage failure'); END"))
            await connector_ws.send(json.dumps({'type': 'inbound', 'event': raw,
                                               'bufferId': 'durable-buffer-1'}) + '\n')
            await asyncio.wait_for(failed_insert.wait(), 10)
            await connector_ws.close()
            connector_ws = await asyncio.wait_for(ready.get(), 5)
            assert not any(f.get('type') == 'inbound_ack' for f in outgoing), outgoing
            assert not peer.requests
            authority.db._execute_write(lambda conn: conn.execute('DROP TRIGGER fail_relay'))
            rows = await send(raw)
            sid = next(iter(authority.sessions))
            warm_agent = authority.agent(SessionRef('default', sid))
            assert warm_agent is not None
            port = api.socket.getsockname()[1]
            async with websockets.connect(f'ws://127.0.0.1:{port}/api/ws?token={web_server._SESSION_TOKEN}') as ws:
                async def rpc(method, **params):
                    await ws.send(json.dumps({'jsonrpc': '2.0', 'id': method, 'method': method, 'params': params}))
                    async with asyncio.timeout(20):
                        while True:
                            frame = json.loads(await ws.recv())
                            frames.append(frame)
                            if frame.get('id') == method:
                                return frame
                resumed = await rpc('session.resume', session_id=sid)
                assert 'result' in resumed, resumed
                await connector_ws.close()
                connector_ws = await asyncio.wait_for(ready.get(), 5)
                # Replayed delivery must not infer or produce another final.
                await connector_ws.send(json.dumps({'type': 'inbound', 'event': raw}) + '\n')
                second = deepcopy(raw)
                second.update(message_id='relay-2', text='WS_SHARED')
                rows = await send(second)
                # Settlement and an RPC reply do not drain the fanout writer.
                # Observe this admission's terminal event before taking a snapshot.
                admission_id = next(r['admission_id'] for r in rows if r['request_id'] == 'relay-2')
                async with asyncio.timeout(10):
                    while not any(f.get('params', {}).get('type') == 'message.complete'
                                  and f['params'].get('admission_id') == admission_id for f in frames):
                        frames.append(json.loads(await ws.recv()))
                snapshot = await rpc('session.resume', session_id=sid)
                assert 'LOCAL_ACK_WS_SHARED' in json.dumps(snapshot), snapshot
                assert [r['request_id'] for r in rows] == ['relay-1', 'relay-2'], rows
                assert len(peer.requests) == 2, peer.requests
                assert authority.agent(SessionRef('default', sid)) is warm_agent
                # Public authority transport cannot submit private/trusted source fields.
                for extra in ({'internal': True}, {'source': raw['source']}, {'profile': 'foreign'},
                              {'native_text_v1': rows[0]['payload']['native_text_v1']}):
                    denied = await rpc('prompt.submit', session_id=sid, input_id='forged', text='forged', **extra)
                    assert 'error' in denied, denied
                # A busy relay callback must not block the same reader that receives
                # outbound ACKs. The FIFO is the authority's, not a second adapter queue.
                blocked = deepcopy(raw)
                blocked.update(message_id='relay-3', text='BLOCK_FIFO')
                await connector_ws.send(json.dumps({'type': 'inbound', 'event': blocked}) + '\n')
                assert await asyncio.to_thread(peer.blocked.wait, 5)
                queued = deepcopy(raw)
                queued.update(message_id='relay-4', text='QUEUED_AFTER_BLOCK')
                await connector_ws.send(json.dumps({'type': 'inbound', 'event': queued}) + '\n')
                try:
                    async with asyncio.timeout(5):
                        while not any(r['request_id'] == 'relay-4' for r in list_session_admissions(
                                authority.db, session_id=sid, pending_only=False)):
                            await asyncio.sleep(.01)
                    assert not adapter._pending_messages
                finally:
                    peer.release.set()
                async with asyncio.timeout(10):
                    while any(not task.done() for task in adapter._background_tasks) or any(r['status'] != 'terminal' for r in
                            list_session_admissions(authority.db, session_id=sid, pending_only=False)):
                        await asyncio.sleep(.01)
                rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
                assert len(peer.requests) == 4, peer.requests
                payload = rows[0]['payload']
                for mutate in ('user_id', 'profile', 'platform'):
                    forged = deepcopy(payload)
                    forged['native_text_v1']['source'][mutate] = 'foreign'
                    try:
                        restore_native(forged, runner)
                    except (RuntimeStoreError, ValueError):
                        pass
                    else:
                        raise AssertionError('forged source accepted: ' + mutate)
                from gateway.session import SessionSource
                from gateway.session_ingress_context import restore_provenance
                untrusted = SessionSource.from_dict(payload['native_text_v1']['source'])
                forged_proof = dict(payload['native_text_v1']['provenance'], relay='forged')
                try:
                    restore_provenance(runner, untrusted, forged_proof)
                except RuntimeStoreError:
                    pass
                else:
                    raise AssertionError('forged seal accepted')
                assert not untrusted.delivered_via_upstream_relay, 'rejected seal installed trust'
                for internal in (False, True):
                    event = _event_from_wire(raw)
                    event.internal = internal
                    try:
                        await prepare_native(runner, event)
                    except RuntimeStoreError:
                        pass
                    else:
                        raise AssertionError('unproven relay accepted')
                secret = 'replacement-credential'
                replacement_transport = WebSocketRelayTransport(transport._url, 'telegram', 'fixture-bot',
                    gateway_id='fixture-gateway', upgrade_secret=secret)
                replacement = RelayAdapter(PlatformConfig(enabled=True), descriptor, replacement_transport)
                runner.adapters[Platform.RELAY] = replacement
                runner._wire_adapter_handlers(replacement)
                try:
                    assert await replacement.connect()
                    await asyncio.wait_for(ready.get(), 5)
                    try:
                        restore_native(payload, runner)
                    except RuntimeStoreError:
                        pass
                    else:
                        raise AssertionError('stale connector replay accepted')
                finally:
                    await replacement.disconnect()
                    runner.adapters[Platform.RELAY] = adapter
                    secret = 'owned-relay-fixture-secret'
                # A real post-handshake 4401 plus the freshly authenticated retry's
                # 4401 must revoke queued provenance, not grant on a cached descriptor.
                revoke = True
                await connector_ws.close(code=4401, reason='revoked')
                async with asyncio.timeout(5):
                    while not transport.auth_revoked:
                        await asyncio.sleep(.01)
                try:
                    restore_native(payload, runner)
                except RuntimeStoreError:
                    pass
                else:
                    raise AssertionError('revoked connector replay accepted')
                finals = [f for f in outgoing if f.get('type') == 'outbound'
                          and f.get('action', {}).get('op') in ('send', 'edit')
                          and 'LOCAL_ACK_' in f.get('action', {}).get('content', '')]
                assert len(finals) == 4, outgoing
                receipt = {'same_agent': True, 'negative_controls': True,
                    'admissions': [r['admission_id'] for r in rows], 'model_requests': len(peer.requests),
                    'finals': finals, 'observer_complete': True, 'reconnect_dedupe': True,
                    'busy_fifo': True, 'replacement_rejected': True, 'ws_revocation_rejected': True}
                Path(os.environ['HERMES_HOME'], 'relay-receipt.json').write_text(json.dumps(receipt))
        finally:
            await adapter.disconnect()
            await stop_gateway_api(api)


def main():
    from http.server import ThreadingHTTPServer
    import threading
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(1, str(Path(__file__).parent / 'fixtures'))
    from shared_authority_peer import ModelPeer
    peer = ThreadingHTTPServer(('127.0.0.1', 0), ModelPeer)
    peer.requests, peer.metadata_requests = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    Path(os.environ['HERMES_HOME'], 'config.yaml').write_text(
        f'model:\n  default: local-wire-stub\n  provider: custom\n  base_url: {url}\n'
        'streaming:\n  enabled: false\nauxiliary:\n  title_generation:\n    enabled: false\n')
    try:
        asyncio.run(probe(peer))
    finally:
        peer.shutdown()
        peer.server_close()


if __name__ == '__main__':
    import traceback
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
