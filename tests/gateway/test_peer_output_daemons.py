"""A real peer creates and shares a document; the following Bot reads that exact version."""
import asyncio
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sqlite3
import socket
import threading

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_peer_document_daemons import _gateway, configure
from tests.gateway.test_session_group_peer_daemons import _join_pair

CONTENT = '# Launch checklist\n- Review the schedule\n- Confirm the owners\n'


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if not messages:
            data = b'{"context_length":256000}'
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data))); self.end_headers()
            self.wfile.write(data)
            return
        self.server.requests.append(body)
        last_user = max((i for i, m in enumerate(messages) if m.get('role') == 'user'), default=-1)
        tools = [m for m in messages[last_user + 1:] if m.get('role') == 'tool']
        call = None
        if 'HOLD_ACTIVE' in str(messages[last_user].get('content', '')):
            self.server.active_event.set()
            self.server.active_release.wait(30)
        elif self.server.actor == 'peer':
            if not tools:
                call = ('write_file', {'path': str(self.server.path), 'content': CONTENT})
            elif len(tools) == 1:
                call = ('share_group_file', {'path': str(self.server.path)})
            else:
                shared = json.loads(tools[-1]['content'])
                self.server.shared = shared
                self.server.shared_event.set()
                self.server.release.wait(30)
        else:
            if not tools:
                text = str(messages[last_user].get('content', ''))
                match = re.search(r'\[Shared attachment\] file: ([^\n]+)', text)
                self.server.received_prompt = text
                if match:
                    call = ('read_file', {'path': match[1].strip()})
            else:
                self.server.read_result = tools[-1]['content']
        message = {'role': 'assistant', 'content': 'Shared the checklist. @host please read it.' if self.server.actor == 'peer' else 'Read the exact checklist.'}
        if call:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0,
                'id': f'{self.server.actor}-{len(tools)}', 'type': 'function',
                'function': {'name': call[0], 'arguments': json.dumps(call[1])}}]}
        payload = {'id': 'fixture', 'choices': [{'index': 0, 'message': message,
            'finish_reason': 'tool_calls' if call else 'stop'}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11}}
        data, mime = json.dumps(payload).encode(), 'application/json'
        if body.get('stream'):
            chunk = {'id': 'fixture', 'choices': [{'index': 0, 'delta': message, 'finish_reason': 'tool_calls' if call else 'stop'}]}
            data, mime = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode(), 'text/event-stream'
        self.send_response(200); self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(data))); self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def model(actor):
    server = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    server.actor, server.requests = actor, []
    server.shared_event, server.release = threading.Event(), threading.Event()
    server.active_event, server.active_release = threading.Event(), threading.Event()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_peer_output_publishes_once_after_detach_and_next_bot_reads_exact_version(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home_model, peer_model = model('home'), model('peer')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    home, home_env = _gateway(tmp_path, 'home', home_model, root)
    peer, peer_env = _gateway(tmp_path, 'peer', peer_model, root, api_port=port)
    peer_model.path = peer / 'launch-checklist.md'
    configure(home, lambda config: config['platform_toolsets'].update(bot_room=['file']))
    configure(peer, lambda config: config['platform_toolsets'].update(api_server=['file', 'bot_room']))

    async def exercise(hd, pd):
        async with websocket(home, hd) as hw, websocket(peer, pd) as pw:
            await _join_pair(hw, pw, peer_first=True)
            sent = await rpc(hw, 'groups.send', room_id='linked', event_id='make-checklist',
                payload={'text': '@all Create a launch checklist and have the next Bot read it.', 'thread_id': 'thread'})
            assert sent['result']['accepted'], sent
            async with asyncio.timeout(30):
                while not peer_model.shared_event.is_set():
                    state = (await rpc(hw, 'groups.state', room_id='linked'))['result']
                    for action in state['driver_status']['pending_actions']:
                        if action['kind'] == 'approval':
                            await rpc(hw, 'groups.approve', room_id='linked', choice='once',
                                **{k: action[k] for k in ('member_id', 'task_id', 'execution_generation', 'request_id')})
                    await asyncio.sleep(.1)
            assert peer_model.shared['ok'] is True, peer_model.shared
            assert peer_model.shared['sha256'] == hashlib.sha256(CONTENT.encode()).hexdigest()
        # Source workspace changes after share; the copied version and work survive viewer detach.
        peer_model.path.write_text('A later private edit')
        peer_model.release.set()
        async with websocket(home, hd) as hw:
            async with asyncio.timeout(45):
                while True:
                    events = (await rpc(hw, 'groups.log', room_id='linked'))['result']['events']
                    replies = [event for event in events if event['kind'] == 'message.member']
                    if len(replies) >= 2:
                        break
                    state = (await rpc(hw, 'groups.state', room_id='linked'))['result']
                    assert not state['driver_status']['counts'].get('failed'), state
                    await asyncio.sleep(.1)
            peer_reply = next(event for event in replies if event['actor']['id'] == 'reviewer')
            attachment, = peer_reply['payload']['attachments']
            saved = await rpc(hw, 'groups.attachment.download', room_id='linked', event_id=peer_reply['event_id'],
                              attachment_id=attachment['attachment_id'])
            assert base64.b64decode(saved['result']['data_base64']) == CONTENT.encode()
            assert 'Review the schedule' in home_model.read_result and 'Confirm the owners' in home_model.read_result
            assert 'A later private edit' not in home_model.read_result
            assert str(peer) not in home_model.received_prompt
            with sqlite3.connect(peer / 'state.db') as db:
                assert db.execute("SELECT COUNT(*) FROM session_admissions WHERE principal_id='api'").fetchone()[0] == 1
            async with asyncio.timeout(20):
                while True:
                    with sqlite3.connect(peer / 'state.db') as db:
                        open_items = db.execute('SELECT COUNT(*) FROM hosted_room_output_artifacts WHERE acknowledged_at IS NULL OR blob_reclaimed_at IS NULL').fetchone()[0]
                    if not open_items:
                        break
                    await asyncio.sleep(.1)
            ended = await rpc(hw, 'groups.disband', room_id='linked')
            assert ended['result']['tombstone']['room_id'] == 'linked', ended
    try:
        with daemon(root, peer, peer_env, barrier=False) as (_, pd), daemon(root, home, home_env, barrier=False) as (_, hd):
            asyncio.run(exercise(hd, pd))
    finally:
        peer_model.release.set()
        for server in (home_model, peer_model):
            server.shutdown(); server.server_close()


def test_negotiated_no_file_reply_needs_no_remote_outbox_or_discard(tmp_path):
    root = Path(__file__).resolve().parents[2]
    hm, pm = model('home'), model('plain')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    home, he = _gateway(tmp_path, 'home', hm, root)
    peer, pe = _gateway(tmp_path, 'peer', pm, root, api_port=port)
    async def exercise(hd, pd):
        async with websocket(home, hd) as hw, websocket(peer, pd) as pw:
            await _join_pair(hw, pw, peer_first=True)
            assert (await rpc(hw, 'groups.send', room_id='linked', event_id='text',
                payload={'text': '@reviewer Say hello without sharing a file.', 'thread_id': 'thread'}))['result']['accepted']
            from tests.gateway.test_session_group_peer_daemons import _events
            reply, = await _events(hw, 'message.member', timeout=20)
            assert reply['actor']['id'] == 'reviewer' and not reply['payload'].get('attachments')
            with sqlite3.connect(peer / 'state.db') as db:
                assert db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_output_artifacts'").fetchone() is None
                raw = db.execute("SELECT payload_json FROM session_admissions WHERE principal_id='api'").fetchone()[0]
                assert json.loads(raw)['api_turn_v1']['settings']['room_dispatch']['document_output']
            state = (await rpc(hw, 'groups.state', room_id='linked'))['result']
            assert not any(action['kind'] == 'output_retry' for action in state['driver_status']['pending_actions'])
            assert (await rpc(hw, 'groups.disband', room_id='linked'))['result']['tombstone']
    try:
        with daemon(root, peer, pe, barrier=False) as (_, pd), daemon(root, home, he, barrier=False) as (_, hd):
            asyncio.run(exercise(hd, pd))
    finally:
        for server in (hm, pm): server.shutdown(); server.server_close()
