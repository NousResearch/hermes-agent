"""Stop winning after an actual signed GET cannot resume unaccepted preparation."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sqlite3
import threading
import urllib.error
import urllib.request
import base64

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_peer_document_daemons import _gateway
from tests.gateway.test_session_group_peer_daemons import _model, _join_pair


class ResumeBarrier(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if self.command == 'POST' and self.path == '/v1/runs':
            self.server.posts += 1
        request = urllib.request.Request(self.server.target + self.path, method=self.command,
            data=body if self.command == 'POST' else None,
            headers={k: v for k, v in self.headers.items() if k.lower() not in {'host', 'connection'}})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        except (urllib.error.URLError, ConnectionError):
            self.send_response(503); self.end_headers(); return
        with response:
            data = response.read()
            boundary = (self.path.startswith('/v1/runs/') if self.server.boundary == 'after_get'
                        else self.headers.get('Hermes-Room-Features') == 'document-input-v1')
            if self.server.recovering and self.command == 'GET' and boundary and not self.server.entered.is_set():
                self.server.entered.set()
                self.server.release.wait(25)
            self.send_response(response.status)
            for key in ('Content-Type', 'Hermes-Room-Proof', 'Hermes-Room-Nonce'):
                if response.headers.get(key): self.send_header(key, response.headers[key])
            self.send_header('Content-Length', str(len(data))); self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass


@pytest.mark.parametrize('boundary', ['after_get', 'before_fulfillment'])
def test_stop_after_unproven_get_never_posts_preparation_resume(tmp_path, boundary):
    root = Path(__file__).resolve().parents[2]
    hm, pm = _model('HOME'), _model('PEER')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    home, he = _gateway(tmp_path, 'home', hm, root)
    peer, pe = _gateway(tmp_path, 'peer', pm, root, api_port=port)
    (peer / 'crash-document-preparation').write_text('once')
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), ResumeBarrier)
    proxy.target, proxy.posts = f'http://127.0.0.1:{port}', 0
    proxy.boundary, proxy.recovering = boundary, False
    proxy.entered, proxy.release = threading.Event(), threading.Event()
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    async def send(hd, pd, pp):
        async with websocket(home, hd) as hw, websocket(peer, pd) as pw:
            _, invite = await _join_pair(hw, pw)
            assert (await rpc(hw, 'groups.peer.register', room_id='linked', member_id='reviewer',
                target_url=f'http://127.0.0.1:{proxy.server_port}', target_profile='default',
                grant=invite['grant'], catalog=invite['catalog']))['result']['registered']
            item = (await rpc(hw, 'groups.attachment.upload', room_id='linked', upload_id='stop-prep',
                kind='file', name='notes.txt', mime='text/plain', data_base64=base64.b64encode(b'retain prepared bytes').decode()))['result']
            assert (await rpc(hw, 'groups.send', room_id='linked', event_id='prepare', payload={
                'text': '@reviewer read the document', 'thread_id': 'thread',
                'attachments': [{k: item[k] for k in ('attachment_id', 'kind', 'name', 'mime', 'size')}]}))['result']['accepted']
            assert await asyncio.to_thread(pp.wait, 25) == 77
    async def stop_wins(hd):
        async with websocket(home, hd) as hw:
            assert await asyncio.to_thread(proxy.entered.wait, 110)
            before = proxy.posts
            stop = asyncio.create_task(rpc(hw, 'groups.stop', room_id='linked', cancel_id='stop-before-resume'))
            async with asyncio.timeout(10):
                while True:
                    with sqlite3.connect(home / 'state.db') as db:
                        status, generation = db.execute('SELECT status,cancel_generation FROM hosted_room_driver_tasks').fetchone()
                    if status == 'stopping': break
                    await asyncio.sleep(.05)
            assert generation > 0
            proxy.release.set()
            assert 'result' in await stop
            await asyncio.sleep(6)  # Include another actual reconciliation cycle.
            assert proxy.posts == before and not pm.requests
            with sqlite3.connect(peer / 'state.db') as db:
                assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 0
                assert db.execute('SELECT COUNT(*) FROM input_custody_preparations').fetchone()[0] == 1
                assert db.execute('SELECT COUNT(*) FROM input_custody_copies').fetchone()[0] == 1
            assert 'error' in await rpc(hw, 'groups.disband', room_id='linked')
    try:
        with daemon(root, home, he, barrier=False) as (_, hd):
            with daemon(root, peer, pe, barrier=True, fixture='peer_document_preparation_crash.py') as (pp, pd):
                asyncio.run(send(hd, pd, pp))
            proxy.recovering = True
            with daemon(root, peer, pe, barrier=False):
                asyncio.run(stop_wins(hd))
    finally:
        proxy.release.set()
        for server in (proxy, hm, pm): server.shutdown(); server.server_close()
