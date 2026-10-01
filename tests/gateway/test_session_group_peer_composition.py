"""Two-gateway schedules requiring the separately owned runtime/participant controls."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import sqlite3
import threading
import urllib.error
import urllib.request

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_session_group_peer_daemons import _model, _gateway, _join_pair, _send, _events


def _http(url, path, *, body=None, authorization='Bearer target-gateway-owned-secret'):
    request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization': authorization, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


@pytest.mark.skipif(importlib.util.find_spec('gateway.platforms.api_server_group_owner_stop') is None,
                    reason='composition requires participant Stop owner #105079')
def test_participant_lists_exact_home_scope_freezes_and_stops_peer(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home_model, target_model = _model('HOME_REPLY'), _model('PEER_REPLY', 'PARTICIPANT_STOP')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    target, target_env = _gateway(tmp_path, 'target', target_model, root, api_port=port)
    url = f'http://127.0.0.1:{port}'
    async def exercise(hd, td):
        async with websocket(home, hd) as hw, websocket(target, td) as tw:
            room, invitation = await _join_pair(hw, tw)
            await _send(hw, 'participant-stop', '@reviewer PARTICIPANT_STOP')
            assert await asyncio.to_thread(target_model.gates['PARTICIPANT_STOP'][0].wait, 30)
            listing = await asyncio.to_thread(_http, url, '/v1/group-participants')
            participant, = listing['data']
            identity = participant['participant']
            assert identity == {
                'room_id': 'linked', 'home_install_id': room['authority_gateway_id'],
                'authority_gateway_id': room['authority_gateway_id'], 'authority_epoch': room['authority_epoch'],
                'member_id': 'reviewer', 'target_install_id': invitation['catalog']['installation_id'],
                'target_profile': 'default'}
            stopped = await asyncio.to_thread(_http, url, '/v1/group-participants/stop', body={
                'participant': identity, 'command_id': 'owner-stop', 'confirm': True})
            assert stopped['admissions_frozen'] and stopped['participant'] == identity
            async with asyncio.timeout(10):
                while True:
                    state = (await rpc(hw, 'groups.state', room_id='linked'))['result']['driver_status']
                    if state['counts'].get('cancelled') or any(a['kind'] == 'stopping' for a in state['pending_actions']):
                        break
                    await asyncio.sleep(.1)
            target_model.gates['PARTICIPANT_STOP'][1].set()
            await _events(hw, 'turn.cancelled', timeout=10)
            await _send(hw, 'after-freeze', '@reviewer MUST_NOT_EXECUTE')
            async with asyncio.timeout(30):
                while not (state := (await rpc(hw, 'groups.state', room_id='linked'))['result']['driver_status'])['counts'].get('deferred'):
                    await asyncio.sleep(.1)
            assert len(target_model.requests) == 1
            assert (await asyncio.to_thread(_http, url, '/v1/group-participants'))['data'][0]['admissions_frozen']
    try:
        with daemon(root, target, target_env, barrier=False) as (_, td):
            with daemon(root, home, home_env, barrier=False) as (_, hd):
                asyncio.run(exercise(hd, td))
    finally:
        target_model.gates['PARTICIPANT_STOP'][1].set()
        for model in (home_model, target_model):
            model.shutdown()
            model.server_close()


class _LostReplyProxy(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        request = urllib.request.Request(self.server.target + self.path, method=self.command,
            data=body if self.command == 'POST' else None,
            headers={k: v for k, v in self.headers.items() if k.lower() not in {'host', 'connection'}})
        try:
            response = urllib.request.urlopen(request, timeout=15)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read()
            if self.server.drop and self.path == '/v1/runs' and response.status == 202:
                # Target has admitted it; remove this listening endpoint before the exact replay.
                self.server.drop = False
                self.server.shutdown()
                self.server.server_close()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.server.accepted.set()
                return
            self.send_response(response.status)
            for key in ('Content-Type', 'Hermes-Room-Proof'):
                if response.headers.get(key):
                    self.send_header(key, response.headers[key])
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def _proxy(target, port=0):
    server = ThreadingHTTPServer(('127.0.0.1', port), _LostReplyProxy)
    server.target, server.drop, server.accepted = target, False, threading.Event()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / 'tui_gateway/test_hosted_room_peer_recovery.py').is_file(),
                    reason='lost-response composition requires runtime owner #99960')
def test_accepted_lost_reply_then_preconnect_failure_keeps_original_attempt(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home_model, target_model = _model('HOME_REPLY'), _model('PEER_REPLY', 'LOST_REPLY')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    target, target_env = _gateway(tmp_path, 'target', target_model, root, api_port=port)
    proxy = _proxy(f'http://127.0.0.1:{port}')
    proxy_port = proxy.server_port
    resumed = []
    def attempts():
        with sqlite3.connect(home / 'state.db') as db:
            return db.execute('SELECT task_id, execution_generation, status FROM hosted_room_driver_tasks').fetchall()
    async def exercise(hd, td):
        async with websocket(home, hd) as hw, websocket(target, td) as tw:
            _, invitation = await _join_pair(hw, tw)
            changed = await rpc(hw, 'groups.peer.register', room_id='linked', member_id='reviewer',
                target_url=f'http://127.0.0.1:{proxy_port}', target_profile='default',
                grant=invitation['grant'], catalog=invitation['catalog'])
            assert changed['result']['registered'], changed
            proxy.drop = True
            await _send(hw, 'lost', '@reviewer LOST_REPLY')
            assert await asyncio.to_thread(proxy.accepted.wait, 30)
            assert await asyncio.to_thread(target_model.gates['LOST_REPLY'][0].wait, 30)
            async with asyncio.timeout(30):
                while not any(row[2] == 'indeterminate' for row in await asyncio.to_thread(attempts)):
                    await asyncio.sleep(.1)
            before = await asyncio.to_thread(attempts)
            assert len(before) == 1 and before[0][1:] == (1, 'indeterminate'), before
            assert len(target_model.requests) == 1
            task_id = before[0][0]
            refused = await rpc(hw, 'groups.discard', room_id='linked', member_id='reviewer', task_id=task_id,
                                execution_generation=1)
            assert refused.get('error', {}).get('message') == 'unknown_execution', refused
            restored = _proxy(f'http://127.0.0.1:{port}', proxy_port)
            resumed.append(restored)
            target_model.gates['LOST_REPLY'][1].set()
            reply, = await _events(hw, 'message.member', timeout=45)
            assert reply['payload']['text'] == 'PEER_REPLY'
            assert len(target_model.requests) == 1
            after = await asyncio.to_thread(attempts)
            assert after == [(task_id, 1, 'settled')], after
    try:
        with daemon(root, target, target_env, barrier=False) as (_, td):
            with daemon(root, home, home_env, barrier=False) as (_, hd):
                asyncio.run(exercise(hd, td))
    finally:
        target_model.gates['LOST_REPLY'][1].set()
        if not proxy.accepted.is_set():
            proxy.shutdown()
            proxy.server_close()
        for server in resumed:
            server.shutdown()
            server.server_close()
        for model in (home_model, target_model):
            model.shutdown()
            model.server_close()
