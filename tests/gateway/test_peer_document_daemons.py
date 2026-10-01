"""Document input crosses two actual gateway owners, never a Desktop-owned worker."""
import asyncio
import base64
import json
from pathlib import Path
import socket
import sqlite3
import copy
import hashlib
import io
import re
import shlex
import sys
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import hermes_yaml as yaml

from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_session_group_peer_daemons import _model, _gateway as base_gateway, _join_pair, _events, _last_user_text, Model


def _gateway(*args, **kwargs):
    home, env = base_gateway(*args, **kwargs)
    env['HERMES_DISABLE_LAZY_INSTALLS'] = '1'
    return home, env


def configure(home, update):
    path = home / 'config.yaml'
    config = yaml.safe_load(path.read_text())
    update(config)
    temporary = home / 'fixture-config.yaml'
    temporary.write_text(yaml.safe_dump(config))
    temporary.replace(path)


class DocumentModel(Model):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length']))
        body = json.loads(raw)
        paths = [json.loads(match[1]) for line in _last_user_text(body).splitlines()
                 if (match := re.fullmatch(r'- ".*": (".*")', line))]
        if paths:
            self.server.approval_command = shlex.join([sys.executable, '-c',
                'import hashlib,pathlib,sys; print("\\n".join(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sys.argv[1:]))', *paths])
        self.rfile = io.BytesIO(raw)
        super().do_POST()


def document_model():
    server = ThreadingHTTPServer(('127.0.0.1', 0), DocumentModel)
    server.requests, server.reply, server.gates = [], 'Reviewed both documents', {}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class LostReply(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        if getattr(self.server, 'blocked', False):
            self.send_response(503); self.end_headers()
            return
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        request = urllib.request.Request(self.server.target + self.path, method=self.command,
            data=body if self.command == 'POST' else None,
            headers={k: v for k, v in self.headers.items() if k.lower() not in {'host', 'connection'}})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read()
            self.server.seen.append((self.path, response.status, len(body)))
            if self.path == '/v1/runs' and response.status == self.server.drop:
                self.server.drop = None
                if getattr(self.server, 'freeze_on_drop', False):
                    self.server.blocked = True
                    self.server.accepted.set()
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            self.send_response(response.status)
            for key in ('Content-Type', 'Hermes-Room-Proof', 'Hermes-Room-Nonce'):
                if response.headers.get(key):
                    self.send_header(key, response.headers[key])
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)


@pytest.mark.parametrize('lost_status', [None, 409, 202])
def test_document_versions_execute_from_receiving_custody_and_remain_downloadable(tmp_path, lost_status):
    root = Path(__file__).resolve().parents[2]
    home_model, peer_model = _model('HOME'), document_model()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    peer, peer_env = _gateway(tmp_path, 'peer', peer_model, root, api_port=port)
    configure(peer, lambda config: config['platform_toolsets'].update(api_server=['terminal']))
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), LostReply)
    proxy.target, proxy.drop, proxy.seen = f'http://127.0.0.1:{port}', lost_status, []
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    proxy_url = f'http://127.0.0.1:{proxy.server_port}'

    async def exercise(hd, pd):
        async with websocket(home, hd) as hw, websocket(peer, pd) as pw:
            room, invite = await _join_pair(hw, pw)
            assert 'document_inputs' not in invite['catalog']
            feature_client = PeerRunsHTTPClient(base_url=proxy_url, api_key='', proof_install_id=invite['catalog']['installation_id'])
            feature = await asyncio.to_thread(feature_client._request, '/v1/room-members/capabilities',
                room_grant=invite['grant'], headers={'Hermes-Room-Features': 'document-input-v1'})
            assert feature['document_inputs']['kinds'] == ['file', 'pdf']
            assert invite['catalog']['attachments'] is False
            registered = await rpc(hw, 'groups.peer.register', room_id='linked', member_id='reviewer',
                target_url=proxy_url, target_profile='default', grant=invite['grant'], catalog=invite['catalog'])
            assert registered['result']['registered']
            versions = []
            data_versions = (b'A' * 3_000_000, b'B' * 3_000_000) if lost_status is None else (b'First document version', b'Changed document version')
            for index, data in enumerate(data_versions):
                result = await rpc(hw, 'groups.attachment.upload', room_id='linked', upload_id=f'upload-{index}',
                    kind='file', name='notes.txt', mime='text/plain', data_base64=base64.b64encode(data).decode())
                uploaded = result['result']
                item = {k: uploaded[k] for k in ('attachment_id', 'kind', 'name', 'mime', 'size')}
                versions.append((item, data))
            result = await rpc(hw, 'groups.send', room_id='linked', event_id='documents', payload={
                'text': '@reviewer APPROVAL_WAIT Compare these documents', 'thread_id': 'thread',
                'attachments': [item for item, _ in versions]})
            assert result['result']['accepted'], result
            source_event_id = result['result']['event']['event_id']
        # Both viewers detach; only the canonical gateways own the work.
        async with websocket(home, hd) as hw:
            async with asyncio.timeout(40):
                while True:
                    events = (await rpc(hw, 'groups.log', room_id='linked'))['result']['events']
                    failed = [e for e in events if e['kind'] == 'turn.failed']
                    assert not failed, failed
                    if any(e['kind'] == 'message.member' for e in events):
                        break
                    await asyncio.sleep(.1)
            assert len(peer_model.requests) == 2 and not home_model.requests
            tool_result = json.dumps([m for m in peer_model.requests[-1]['messages'] if m.get('role') == 'tool'])
            assert all(hashlib.sha256(data).hexdigest() in tool_result for _, data in versions)
            text = _last_user_text(peer_model.requests[0])
            with sqlite3.connect(f'file:{peer / "state.db"}?mode=ro', uri=True) as db:
                (encoded,), = db.execute("SELECT payload_json FROM session_admissions WHERE principal_id='api'").fetchall()
                payload = json.loads(encoded)
                refs = payload['api_turn_v1']['settings']['room_document_inputs']['references']
                assert db.execute('SELECT COUNT(*) FROM input_custody_refs').fetchone()[0] == 2
            assert len({ref['path'] for ref in refs}) == 2
            for ref, (item, data) in zip(refs, versions):
                assert Path(ref['path']).is_relative_to(peer)
                assert Path(ref['path']).read_bytes() == data
                assert ref['path'] in text
                if lost_status is not None:
                    saved = await rpc(hw, 'groups.attachment.download', room_id='linked', event_id=source_event_id, attachment_id=item['attachment_id'])
                    assert 'result' in saved, saved
                    assert base64.b64decode(saved['result']['data_base64']) == data
            assert str(home) not in text
            # Replay with a fresh client that has neither source storage nor a local Run receipt.
            replay_client = PeerRunsHTTPClient(base_url=proxy_url, api_key='',
                proof_install_id=invite['catalog']['installation_id'])
            reply = await asyncio.to_thread(replay_client.recover_dispatch,
                dispatch=payload['api_turn_v1']['settings']['room_dispatch'], grant=invite['grant'])
            assert reply['replayed']
            assert len(peer_model.requests) == 2
            if lost_status == 409:
                dispatch = payload['api_turn_v1']['settings']['room_dispatch']
                # The manifest participates in the exact HTTP fingerprint, even without bytes.
                changed = copy.deepcopy(dispatch)
                changed['document_inputs'][0]['sha256'] = '0' * 64
                with pytest.raises(PeerRunsHTTPError) as conflict:
                    await asyncio.to_thread(replay_client._request, '/v1/runs', method='POST',
                        body={'input': changed['prompt'], 'hosted_room_dispatch': changed},
                        headers={'Idempotency-Key': f"room:{changed['task_id']}:{changed['execution_generation']}"}, room_grant=invite['grant'])
                assert conflict.value.error_code == 'idempotency_key_conflict'
                fresh = copy.deepcopy(dispatch)
                fresh['task_id'] += '-new'
                with pytest.raises(PeerRunsHTTPError) as corrupt:
                    await asyncio.to_thread(replay_client._request, '/v1/runs', method='POST',
                        body={'input': fresh['prompt'], 'hosted_room_dispatch': fresh,
                              'document_bytes': [base64.b64encode(b'X' * i['size']).decode() for i in fresh['document_inputs']]},
                        headers={'Idempotency-Key': f"room:{fresh['task_id']}:{fresh['execution_generation']}"}, room_grant=invite['grant'])
                assert corrupt.value.error_code == 'invalid_room_document_input'
                with sqlite3.connect(f'file:{peer / "state.db"}?mode=ro', uri=True) as db:
                    assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 1
                    assert db.execute('SELECT COUNT(*) FROM input_custody_preparations').fetchone()[0] == 1
            if lost_status == 202:
                # Ingress policy can shrink after acceptance without revoking accepted custody.
                configure(peer, lambda config: config['gateway'].update(max_inbound_media_bytes=1))
                scoped = await asyncio.to_thread(replay_client._request, '/v1/room-members/capabilities',
                    room_grant=invite['grant'], headers={'Hermes-Room-Features': 'document-input-v1'})
                assert scoped['document_inputs']['max_batch_bytes'] == 1
                fresh_client = PeerRunsHTTPClient(base_url=proxy_url, api_key='', proof_install_id=invite['catalog']['installation_id'])
                replay = await asyncio.to_thread(fresh_client.recover_dispatch,
                    dispatch=payload['api_turn_v1']['settings']['room_dispatch'], grant=invite['grant'])
                assert replay['replayed'] and len(peer_model.requests) == 2
                configure(peer, lambda config: config['gateway'].update(max_inbound_media_bytes=0))
                scoped = await asyncio.to_thread(replay_client._request, '/v1/room-members/capabilities',
                    room_grant=invite['grant'], headers={'Hermes-Room-Features': 'document-input-v1'})
                assert scoped['document_inputs']['max_batch_bytes'] == 6_000_000
            if lost_status:
                assert proxy.drop is None
            statuses = [status for path, status, _ in proxy.seen if path == '/v1/runs']
            assert 409 in statuses and 202 in statuses
            if lost_status is None:
                from gateway.hosted_room_documents import DOCUMENT_HTTP_MAX_BYTES
                assert 8_000_000 < max(size for _, _, size in proxy.seen) <= DOCUMENT_HTTP_MAX_BYTES + 16
    try:
        with daemon(root, peer, peer_env, barrier=False) as (_, pd), daemon(root, home, home_env, barrier=False) as (_, hd):
            asyncio.run(exercise(hd, pd))
    finally:
        proxy.shutdown()
        proxy.server_close()
        for model in (home_model, peer_model):
            model.shutdown()
            model.server_close()


def test_home_restart_recovers_accepted_documents_without_source_bytes_or_local_receipt_then_stops(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home_model, peer_model = _model('HOME'), _model('PEER', 'HOLD_DOC')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    peer, peer_env = _gateway(tmp_path, 'peer', peer_model, root, api_port=port)
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), LostReply)
    proxy.target, proxy.drop, proxy.seen = f'http://127.0.0.1:{port}', 202, []
    proxy.freeze_on_drop, proxy.accepted = True, threading.Event()
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    proxy_url = f'http://127.0.0.1:{proxy.server_port}'
    saved = {}

    async def send(hd, pd):
        async with websocket(home, hd) as hw, websocket(peer, pd) as pw:
            _, invite = await _join_pair(hw, pw)
            saved.update(invite)
            assert (await rpc(hw, 'groups.peer.register', room_id='linked', member_id='reviewer',
                target_url=proxy_url, target_profile='default', grant=invite['grant'], catalog=invite['catalog']))['result']['registered']
            file = (await rpc(hw, 'groups.attachment.upload', room_id='linked', upload_id='doc',
                kind='file', name='notes.txt', mime='text/plain', data_base64=base64.b64encode(b'private target copy').decode()))['result']
            result = await rpc(hw, 'groups.send', room_id='linked', event_id='held', payload={
                'text': '@reviewer HOLD_DOC', 'thread_id': 'thread',
                'attachments': [{k: file[k] for k in ('attachment_id', 'kind', 'name', 'mime', 'size')}]})
            assert result['result']['accepted']
            assert await asyncio.to_thread(proxy.accepted.wait, 20)
            assert await asyncio.to_thread(peer_model.gates['HOLD_DOC'][0].wait, 20)

    async def recover_and_stop(hd):
        async with websocket(home, hd) as hw:
            for _ in range(450):
                with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                    receipts = db.execute('SELECT COUNT(*) FROM hosted_room_remote_runs').fetchone()[0]
                if receipts:
                    break
                await asyncio.sleep(.1)
            assert receipts, ((await rpc(hw, 'groups.state', room_id='linked')), proxy.seen)
            assert len(peer_model.requests) == 1
            stopped = await rpc(hw, 'groups.stop', room_id='linked', cancel_id='stop-doc')
            assert stopped['result']['cancelled'] == 1, stopped
            # Stop requests cannot prematurely erase bytes a live worker may still need.
            with sqlite3.connect(f'file:{peer / "state.db"}?mode=ro', uri=True) as db:
                payload = json.loads(db.execute("SELECT payload_json FROM session_admissions WHERE principal_id='api'").fetchone()[0])
            refs = payload['api_turn_v1']['settings']['room_document_inputs']['references']
            assert all(Path(ref['path']).read_bytes() == b'private target copy' for ref in refs)
            peer_model.gates['HOLD_DOC'][1].set()
            try:
                await _events(hw, 'turn.cancelled', timeout=25)
            except TimeoutError:
                pytest.fail(str({'state': await rpc(hw, 'groups.state', room_id='linked'),
                                 'events': await rpc(hw, 'groups.log', room_id='linked'), 'http': proxy.seen[-15:], 'peer_log': (peer / 'restart.log').read_text()[-16000:]}))
            ended = await rpc(hw, 'groups.disband', room_id='linked', cancel_id='end-doc')
            assert ended['result']['tombstone']['room_id'] == 'linked', ended
    try:
        with daemon(root, peer, peer_env, barrier=False) as (_, pd):
            with daemon(root, home, home_env, barrier=False) as (hp, hd):
                asyncio.run(send(hd, pd))
                hp.kill(); hp.wait(timeout=10)
            # The actual lost acceptance left no receipt at the source owner.
            with sqlite3.connect(home / 'state.db') as db:
                assert db.execute('SELECT COUNT(*) FROM hosted_room_remote_runs').fetchone()[0] == 0
                blobs = [row[0] for row in db.execute('SELECT blob_id FROM hosted_room_attachment_blobs')]
            from gateway.hosted_room_attachments import HostedRoomAttachmentStore
            source = HostedRoomAttachmentStore(home / 'state.db')
            for blob in blobs:
                source._blob_path(blob).unlink()
            proxy.blocked = False
            with daemon(root, home, home_env, barrier=False) as (_, hd):
                asyncio.run(recover_and_stop(hd))
    finally:
        peer_model.gates['HOLD_DOC'][1].set()
        proxy.shutdown(); proxy.server_close()
        for model in (home_model, peer_model):
            model.shutdown(); model.server_close()


def test_receiver_crash_during_preparation_reuses_verified_copy_without_duplicate_admission(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home_model, peer_model = _model('HOME'), _model('Recovered document')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    peer, peer_env = _gateway(tmp_path, 'peer', peer_model, root, api_port=port)
    (peer / 'crash-document-preparation').write_text('once')

    async def send(hd, pd, proc):
        async with websocket(home, hd) as hw, websocket(peer, pd) as pw:
            await _join_pair(hw, pw)
            uploaded = (await rpc(hw, 'groups.attachment.upload', room_id='linked', upload_id='crash',
                kind='file', name='review.txt', mime='text/plain', data_base64=base64.b64encode(b'survives preparation crash').decode()))['result']
            result = await rpc(hw, 'groups.send', room_id='linked', event_id='crash', payload={
                'text': '@reviewer review after restart', 'thread_id': 'thread',
                'attachments': [{k: uploaded[k] for k in ('attachment_id', 'kind', 'name', 'mime', 'size')}]})
            assert result['result']['accepted']
            assert await asyncio.to_thread(proc.wait, 25) == 77

    async def recovered(hd):
        async with websocket(home, hd) as hw:
            try:
                reply, = await _events(hw, 'message.member', timeout=90)
            except TimeoutError:
                with sqlite3.connect(peer / 'runs_idempotency.db') as records:
                    runs = records.execute('SELECT run_id,status_json,owner_pid,owner_started FROM run_idempotency').fetchall()
                pytest.fail(str({'state': await rpc(hw, 'groups.state', room_id='linked'), 'events': await rpc(hw, 'groups.log', room_id='linked'),
                                 'peer_log': (peer / 'restart.log').read_text()[-16000:], 'home_log': (home / 'first.log').read_text()[-12000:], 'runs': runs}))
            assert reply['payload']['text'] == 'Recovered document'
            assert len(peer_model.requests) == 1
            with sqlite3.connect(f'file:{peer / "state.db"}?mode=ro', uri=True) as db:
                assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 1
                assert db.execute('SELECT COUNT(*) FROM input_custody_copies').fetchone()[0] == 1
                assert db.execute('SELECT COUNT(*) FROM input_custody_refs').fetchone()[0] == 1
            with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                attempts = db.execute('SELECT execution_generation,status FROM hosted_room_driver_tasks').fetchall()
                assert attempts == [(1, 'settled')], attempts
    try:
        with daemon(root, home, home_env, barrier=True, fixture='peer_document_preparation_crash.py') as (_, hd):
            with daemon(root, peer, peer_env, barrier=True, fixture='peer_document_preparation_crash.py') as (pp, pd):
                asyncio.run(send(hd, pd, pp))
            with sqlite3.connect(peer / 'state.db') as db:
                assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 0
                assert db.execute('SELECT COUNT(*) FROM input_custody_preparations').fetchone()[0] == 1
            with daemon(root, peer, peer_env, barrier=False):
                asyncio.run(recovered(hd))
    finally:
        for model in (home_model, peer_model):
            model.shutdown(); model.server_close()
