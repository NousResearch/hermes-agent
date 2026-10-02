"""Real signed output transfer interrupted before publication and after source ACK."""
import asyncio
import base64
import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_peer_document_daemons import _gateway, configure
from tests.gateway.test_peer_output_daemons import CONTENT, model
from tests.gateway.test_session_group_peer_daemons import _join_pair, _events


class Proxy(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if self.command == 'POST' and self.path == '/v1/runs':
            self.server.run_posts = getattr(self.server, 'run_posts', 0) + 1
        if self.path.endswith('/artifacts/ack') and self.server.mode == 'block_ack':
            self.send_response(503); self.end_headers(); return
        request = urllib.request.Request(self.server.target + self.path, method=self.command,
            data=body if self.command == 'POST' else None,
            headers={k: v for k, v in self.headers.items() if k.lower() not in {'host', 'connection'}})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read()
            if self.command == 'GET' and '/v1/runs/' in self.path and response.status == 200:
                self.server.unknown_observed.set()
            if self.path.endswith('/artifacts/read') and response.status == 200 and self.server.mode == 'hold_read':
                self.server.entered.set(); self.server.release.wait(30)
            if self.path.endswith('/artifacts/ack') and response.status == 200:
                self.server.acks += 1
                if self.server.mode == 'lose_ack':
                    self.server.mode = 'block_ack'; self.server.entered.set()
                    self.connection.shutdown(socket.SHUT_RDWR); return
            if (self.server.mode == 'offline_after_feature' and response.status == 200
                    and self.headers.get('Hermes-Room-Features') == 'document-output-v1'):
                self.server.mode = 'offline'
                self.server.shutdown(); self.server.server_close()
                self.server.entered.set()
            self.send_response(response.status)
            for key in ('Content-Type', 'Hermes-Room-Proof', 'Hermes-Room-Nonce'):
                if response.headers.get(key): self.send_header(key, response.headers[key])
            self.send_header('Content-Length', str(len(data))); self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass


def proxy_server(target, *, mode='', address=('127.0.0.1', 0)):
    server = ThreadingHTTPServer(address, Proxy)
    server.mode, server.target, server.acks, server.run_posts = mode, target, 0, 0
    server.entered, server.release, server.unknown_observed = threading.Event(), threading.Event(), threading.Event()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@contextmanager
def pair(tmp_path, mode):
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[2]
    hm, pm = model('home'), model('peer')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    home, he = _gateway(tmp_path, 'home', hm, root)
    peer, pe = _gateway(tmp_path, 'peer', pm, root, api_port=port)
    pm.path = peer / 'checklist.md'
    configure(home, lambda c: c['platform_toolsets'].update(bot_room=['file']))
    configure(peer, lambda c: c['platform_toolsets'].update(api_server=['file', 'bot_room']))
    proxy = proxy_server(f'http://127.0.0.1:{port}', mode=mode)
    try:
        yield SimpleNamespace(root=root, home=home, peer=peer, he=he, pe=pe, hm=hm, pm=pm, proxy=proxy,
                              url=f'http://127.0.0.1:{proxy.server_port}')
    finally:
        pm.release.set(); pm.active_release.set(); proxy.release.set()
        for server in (hm, pm, proxy): server.shutdown(); server.server_close()


async def begin(p, hd, pd):
    async with websocket(p.home, hd) as hw, websocket(p.peer, pd) as pw:
        _, invite = await _join_pair(hw, pw, peer_first=True)
        registered = await rpc(hw, 'groups.peer.register', room_id='linked', member_id='reviewer',
            target_url=p.url, target_profile='default', grant=invite['grant'], catalog=invite['catalog'])
        assert registered['result']['registered']
        sent = await rpc(hw, 'groups.send', room_id='linked', event_id='generate',
            payload={'text': '@all Create a checklist and have the next Bot read it.', 'thread_id': 'thread'})
        assert sent['result']['accepted']
        assert await asyncio.to_thread(p.pm.shared_event.wait, 20)
        assert p.pm.shared['ok'], p.pm.shared
        p.pm.release.set()
        assert await asyncio.to_thread(p.proxy.entered.wait, 20)
        if p.proxy.mode == 'block_ack':
            await _events(hw, 'message.member', count=2, timeout=30)
        else:
            events = (await rpc(hw, 'groups.log', room_id='linked'))['result']['events']
            assert not any(e['kind'] == 'message.member' for e in events)
        return invite


async def end(p, hw):
    async with asyncio.timeout(25):
        while True:
            result = await rpc(hw, 'groups.disband', room_id='linked')
            if result.get('result', {}).get('tombstone'):
                return
            rooms = (await rpc(hw, 'groups.list'))['result']['rooms']
            if not any(room['room_id'] == 'linked' for room in rooms):
                return
            await asyncio.sleep(.2)


@pytest.mark.parametrize('mode', ['hold_read', 'lose_ack'])
def test_home_restart_preserves_exact_output_and_lost_ack_is_idempotent(tmp_path, mode):
    with pair(tmp_path, mode) as p:
        with daemon(p.root, p.peer, p.pe, barrier=False) as (_, pd):
            with daemon(p.root, p.home, p.he, barrier=False) as (hp, hd):
                invite = asyncio.run(begin(p, hd, pd))
                hp.kill(); hp.wait(timeout=10)
            if mode == 'hold_read':
                from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
                with sqlite3.connect(p.home / 'state.db') as db:
                    db.execute('DELETE FROM hosted_room_remote_runs')
                    dispatch = json.loads(db.execute("SELECT value FROM state_meta WHERE key LIKE 'group.peer-output.v1.%'").fetchone()[0])['dispatch']
                with sqlite3.connect(p.peer / 'runs_idempotency.db') as db:
                    db.execute('DELETE FROM run_idempotency')
                client = PeerRunsHTTPClient(base_url=p.proxy.target, api_key='', receipt_db_path=p.home / 'state.db',
                    proof_install_id=invite['catalog']['installation_id'])
                replay = client.recover_dispatch(dispatch=dispatch, grant=invite['grant'])
                assert replay['replayed']
                with sqlite3.connect(p.peer / 'state.db') as db:
                    assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 1
                # Publication/ACK itself must recover without an execution POST too.
                with sqlite3.connect(p.home / 'state.db') as db:
                    db.execute('DELETE FROM hosted_room_remote_runs')
            if mode == 'lose_ack':
                assert p.proxy.acks == 1
                with sqlite3.connect(p.peer / 'state.db') as db:
                    assert db.execute('SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL OR blob_reclaimed_at IS NULL').fetchone()[0] == 0
            p.proxy.mode = ''; p.proxy.release.set()
            async def reopened(hd):
                async with websocket(p.home, hd) as hw:
                    replies = await _events(hw, 'message.member', count=2, timeout=45)
                    output = next(e for e in replies if e['actor']['id'] == 'reviewer')
                    attachment, = output['payload']['attachments']
                    result = await rpc(hw, 'groups.attachment.download', room_id='linked', event_id=output['event_id'], attachment_id=attachment['attachment_id'])
                    assert base64.b64decode(result['result']['data_base64']) == CONTENT.encode()
                    assert len(replies) == 2
                    assert 'Review the schedule' in p.hm.read_result
                    await end(p, hw)
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 1
                        assert db.execute("SELECT COUNT(*) FROM state_meta WHERE key LIKE 'gateway.peer-output-disposition.v1.%'").fetchone()[0] == 1
            with daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
                asyncio.run(reopened(hd))


def test_corrupt_published_copy_blocks_ack_replay_and_end_until_repaired(tmp_path):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    with pair(tmp_path, 'lose_ack') as p:
        with daemon(p.root, p.peer, p.pe, barrier=False) as (_, pd), daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
            invite = asyncio.run(begin(p, hd, pd))
            with sqlite3.connect(p.home / 'state.db') as db:
                blob, = db.execute('SELECT blob_id FROM hosted_room_attachments WHERE upload_id LIKE ?', ('bot-output:%',)).fetchone()
            path = HostedRoomAttachmentStore(p.home / 'state.db')._blob_path(blob)
            path.write_bytes(b'corrupted bytes')
            async def blocked_then_repaired():
                async with websocket(p.home, hd) as hw:
                    refused = await rpc(hw, 'groups.disband', room_id='linked')
                    assert 'error' in refused, refused
                    with sqlite3.connect(p.home / 'state.db') as db:
                        assert db.execute("SELECT disbanded_at FROM hosted_rooms WHERE room_id='linked'").fetchone()[0] is None
                        assert db.execute('SELECT COUNT(*) FROM hosted_room_links').fetchone()[0] == 1
                    client = PeerRunsHTTPClient(base_url=p.proxy.target, api_key='', proof_install_id=invite['catalog']['installation_id'])
                    assert (await asyncio.to_thread(client.probe, grant=invite['grant']))['room_id'] == 'linked'
                    assert p.proxy.acks == 1
                    path.write_bytes(CONTENT.encode())
                    p.proxy.mode = ''
                    await end(p, hw)
            asyncio.run(blocked_then_repaired())


@pytest.mark.parametrize('action', ['discard', 'retry'])
def test_proven_unreceived_output_intent_does_not_block_healthy_member_or_controls(tmp_path, action):
    with pair(tmp_path, 'offline_after_feature') as p:
        replacement = None
        try:
            with daemon(p.root, p.peer, p.pe, barrier=False) as (_, pd), daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
                async def exercise():
                    nonlocal replacement
                    async with websocket(p.home, hd) as hw, websocket(p.peer, pd) as pw:
                        _, invite = await _join_pair(hw, pw, peer_first=True)
                        assert (await rpc(hw, 'groups.peer.register', room_id='linked', member_id='reviewer',
                            target_url=p.url, target_profile='default', grant=invite['grant'], catalog=invite['catalog']))['result']['registered']
                        assert (await rpc(hw, 'groups.send', room_id='linked', event_id='offline',
                            payload={'text': '@all Create a checklist and read it.', 'thread_id': 'thread'}))['result']['accepted']
                        await _events(hw, 'turn.deferred', timeout=20)
                        healthy, = await _events(hw, 'message.member', timeout=20)
                        assert healthy['actor']['id'] == 'host' and not p.pm.requests
                        state = (await rpc(hw, 'groups.state', room_id='linked'))['result']
                        pending = next(a for a in state['driver_status']['pending_actions'] if a['kind'] == 'discard')
                        if action == 'retry':
                            replacement = proxy_server(p.proxy.target, address=p.proxy.server_address)
                            p.pm.release.set()
                        control = await rpc(hw, 'groups.' + action, room_id='linked',
                            **{k: pending[k] for k in ('member_id', 'task_id', 'execution_generation')})
                        assert 'result' in control, control
                        if action == 'retry':
                            async with asyncio.timeout(30):
                                while True:
                                    events = (await rpc(hw, 'groups.log', room_id='linked'))['result']['events']
                                    if any(e['kind'] == 'message.member' and e['actor']['id'] == 'reviewer' for e in events): break
                                    await asyncio.sleep(.1)
                            assert p.pm.shared['ok']
                        else:
                            await _events(hw, 'turn.cancelled', timeout=10)
                            state = (await rpc(hw, 'groups.state', room_id='linked'))['result']
                            assert not any(a['kind'] == 'output_retry' for a in state['driver_status']['pending_actions'])
                        await end(p, hw)
                asyncio.run(exercise())
        finally:
            if replacement is not None: replacement.shutdown(); replacement.server_close()


def test_delayed_export_does_not_hold_policy_lock_or_block_stop_of_an_admitted_peer(tmp_path):
    with pair(tmp_path, 'hold_read') as p:
        with daemon(p.root, p.peer, p.pe, barrier=False) as (_, pd), daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
            asyncio.run(begin(p, hd, pd))
            async def stop_while_read_is_held():
                async with websocket(p.home, hd) as hw:
                    sent = await rpc(hw, 'groups.send', room_id='linked', event_id='other-thread',
                        payload={'text': '@reviewer HOLD_ACTIVE', 'thread_id': 'other-thread'})
                    assert sent['result']['accepted']
                    assert await asyncio.to_thread(p.pm.active_event.wait, 15)
                    # The first real export response remains blocked throughout this interval.
                    start = time.monotonic()
                    await asyncio.sleep(2.1)
                    stopped = await asyncio.wait_for(rpc(hw, 'groups.stop', room_id='linked', cancel_id='stop-held-export'), 3)
                    assert stopped['result']['cancelled'] >= 1, stopped
                    assert time.monotonic() - start >= 2 and not p.proxy.release.is_set()
                    p.pm.active_release.set()
                    p.proxy.mode = ''; p.proxy.release.set()
                    await _events(hw, 'turn.cancelled', timeout=20)
                    await end(p, hw)
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        assert db.execute('SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL OR blob_reclaimed_at IS NULL').fetchone()[0] == 0
            asyncio.run(stop_while_read_is_held())


def test_producer_crash_after_share_keeps_unknown_output_until_exact_operator_resolution(tmp_path):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    with pair(tmp_path, '') as p:
        with daemon(p.root, p.home, p.he, barrier=False) as (_, hd):
            async def share_then_crash(pd, proc):
                async with websocket(p.home, hd) as hw, websocket(p.peer, pd) as pw:
                    _, invite = await _join_pair(hw, pw, peer_first=True)
                    registered = await rpc(hw, 'groups.peer.register', room_id='linked', member_id='reviewer',
                        target_url=p.url, target_profile='default', grant=invite['grant'], catalog=invite['catalog'])
                    assert registered['result']['registered']
                    sent = await rpc(hw, 'groups.send', room_id='linked', event_id='producer-crash',
                        payload={'text': '@reviewer Create and share the checklist.', 'thread_id': 'thread'})
                    assert sent['result']['accepted']
                    assert await asyncio.to_thread(p.pm.shared_event.wait, 20)
                    assert p.pm.shared['ok']
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        assert db.execute('SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL').fetchone()[0] == 1
                    proc.kill(); proc.wait(timeout=10)
                    return invite
            with daemon(p.root, p.peer, p.pe, barrier=False) as (pp, pd):
                invite = asyncio.run(share_then_crash(pd, pp))
            p.pm.release.set()
            p.proxy.unknown_observed.clear()
            calls = len(p.pm.requests)
            async def unresolved_then_resolved():
                async with websocket(p.home, hd) as hw:
                    assert await asyncio.to_thread(p.proxy.unknown_observed.wait, 110), await rpc(hw, 'groups.state', room_id='linked')
                    await asyncio.sleep(2.2)
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        row = db.execute("SELECT request_id,admission_id,generation,status FROM session_admissions WHERE principal_id='api'").fetchone()
                        assert row[3] == 'unknown', row
                        assert db.execute('SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL').fetchone()[0] == 1
                    events = (await rpc(hw, 'groups.log', room_id='linked'))['result']['events']
                    assert not any(e['kind'] == 'message.member' for e in events)
                    state = (await rpc(hw, 'groups.state', room_id='linked'))['result']['driver_status']
                    assert any(a['kind'] == 'unknown' for a in state['pending_actions']), state
                    assert not any(a['kind'] in {'retry', 'discard'} for a in state['pending_actions']), state
                    refused = await rpc(hw, 'groups.disband', room_id='linked')
                    assert 'error' in refused, refused
                    await asyncio.sleep(12)  # Cross the normal peer observation/backoff interval.
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        assert db.execute('SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL').fetchone()[0] == 1
                    with sqlite3.connect(p.home / 'state.db') as db:
                        assert db.execute("SELECT disbanded_at FROM hosted_rooms WHERE room_id='linked'").fetchone()[0] is None
                    assert len(p.pm.requests) == calls
                    # A human/operator explicitly resolves this exact unknown accepted execution.
                    client = PeerRunsHTTPClient(base_url=p.proxy.target, api_key='', proof_install_id=invite['catalog']['installation_id'])
                    resolved = await asyncio.to_thread(client._request, '/v1/runs/' + row[0] + '/resolve-unknown',
                        method='POST', room_grant=invite['grant'], body={'admission_id': row[1], 'execution_generation': row[2]})
                    assert resolved['status'] == 'terminal' and resolved['outcome'] == 'interrupted', resolved
                    projection = await asyncio.to_thread(client._request, '/v1/runs/' + row[0], room_grant=invite['grant'])
                    assert projection['status'] == 'cancelled' and 'peer_output_unresolved' not in projection
                    try:
                        await end(p, hw)
                    except TimeoutError:
                        target = await asyncio.to_thread(client._request, '/v1/runs/' + row[0], room_grant=invite['grant'])
                        pytest.fail(str({'target': target, 'home': await rpc(hw, 'groups.state', room_id='linked'),
                            'log': (p.home / 'restart.log').read_text()[-7000:]}))
                    with sqlite3.connect(p.peer / 'state.db') as db:
                        assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 1
                        assert db.execute('SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL OR blob_reclaimed_at IS NULL').fetchone()[0] == 0
                    assert len(p.pm.requests) == calls
                    assert p.proxy.run_posts == 1
            with daemon(p.root, p.peer, p.pe, barrier=False):
                asyncio.run(unresolved_then_resolved())
