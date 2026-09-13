"""Actual Telegram polling ingress, SIGKILL, and uninstrumented owner recovery."""
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time
from urllib.parse import parse_qs

import pytest

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon


class BotAPI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        body = json.loads(raw) if 'application/json' in self.headers.get('Content-Type', '') else {
            k: values[0] for k, values in parse_qs(raw.decode()).items()
        }
        method = self.path.rsplit('/', 1)[-1]
        self.server.calls.append(method)
        if method == 'getUpdates':
            try:
                result = [self.server.updates.get(timeout=.2)]
            except queue.Empty:
                result = []
        elif method == 'getMe':
            result = {'id': 987654321, 'is_bot': True, 'first_name': 'Fixture', 'username': 'recovery_fixture_bot'}
        elif method == 'getWebhookInfo':
            result = {'url': '', 'pending_update_count': 0}
        elif method in ('sendMessage', 'editMessageText'):
            self.server.sent.append(body)
            result = {'message_id': len(self.server.sent) + 1000, 'date': int(time.time()),
                      'chat': {'id': int(body['chat_id']), 'type': 'private'}, 'text': body.get('text', '')}
        else:
            result = True
        payload = json.dumps({'ok': True, 'result': result}).encode()
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


def wait_for(predicate, detail, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    pytest.fail(detail())


@pytest.mark.linux_only
def test_telegram_fifo_unknown_and_current_authorization_survive_sigkill(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    bot = ThreadingHTTPServer(('127.0.0.1', 0), BotAPI)
    bot.calls, bot.sent, bot.updates = [], [], queue.Queue()
    for peer in (model, bot):
        threading.Thread(target=peer.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'local-wire-stub', 'base_url': base},
           'gateway': {'multiplex_profiles': False},
           'platforms': {'telegram': {'enabled': True, 'extra': {
               'base_url': f'http://127.0.0.1:{bot.server_port}/bot', 'dm_policy': 'allowlist'}}},
           'auxiliary': {'title_generation': {'enabled': False}},
           'platform_toolsets': {'telegram': []}, 'terminal': {'cwd': str(home)}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1',
               TELEGRAM_BOT_TOKEN='987654321:owned-loopback-fixture', TELEGRAM_ALLOWED_USERS='101,202,303')

    def rows():
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return {json.loads(payload)['text']: {'id': aid, 'request_id': rid, 'session': sid, 'status': status}
                    for aid, rid, sid, status, payload in db.execute(
                        'SELECT admission_id,request_id,target_session_id,status,payload_json FROM session_admissions')}

    def diagnostic():
        return repr(rows()) + '\n' + '\n'.join(p.read_text() for p in (home / 'logs').glob('*.log'))

    update_id = 0

    def send(uid, text, status):
        nonlocal update_id
        update_id += 1
        bot.updates.put({'update_id': update_id, 'message': {
            'message_id': update_id, 'date': int(time.time()),
            'chat': {'id': uid, 'type': 'private', 'first_name': 'Owned'},
            'from': {'id': uid, 'is_bot': False, 'first_name': 'Owned'}, 'text': text}})
        wait_for(lambda: rows().get(text, {}).get('status') == status, diagnostic)

    boots = []
    try:
        with daemon(root, home, env, barrier=True, fixture='native_telegram_recovery_daemon.py') as (proc, desc):
            # Warm the started route so first-message onboarding cannot obscure the model barrier.
            send(202, 'WARM_HISTORY', 'terminal')
            send(202, 'BLOCK_STARTED', 'started')
            assert model.blocked.wait(20), diagnostic()
            send(202, 'NEVER_REPLAY', 'queued')
            send(101, 'SAFE_FIRST', 'queued')
            send(101, 'SAFE_SECOND', 'queued')
            send(303, 'REVOKED_FIRST', 'queued')
            send(303, 'REVOKED_SECOND', 'queued')
            before = rows()
            assert before['SAFE_FIRST']['session'] == before['SAFE_SECOND']['session']
            assert before['BLOCK_STARTED']['session'] == before['NEVER_REPLAY']['session']
            boots.append({'pid': proc.pid, 'epoch': desc['authority_epoch']})
            proc.kill()
            assert proc.wait(timeout=10) == -9
        model.release.set()
        env['TELEGRAM_ALLOWED_USERS'] = '101,202'
        for _ in range(2):
            with daemon(root, home, env, barrier=False) as (proc, desc):
                wait_for(lambda: rows()['SAFE_SECOND']['status'] == 'terminal', diagnostic)
                after = rows()
                assert after['SAFE_FIRST']['status'] == 'terminal'
                assert after['BLOCK_STARTED']['status'] == 'unknown'
                assert after['NEVER_REPLAY']['status'] == 'queued'
                assert after['REVOKED_FIRST'] == before['REVOKED_FIRST']
                assert after['REVOKED_SECOND'] == before['REVOKED_SECOND']
                for text in before:
                    assert after[text]['id'] == before[text]['id']
                    assert after[text]['session'] == before[text]['session']
                boots.append({'pid': proc.pid, 'epoch': desc['authority_epoch']})
        # Re-authorizing later proves refusal retained executable rows, not just tombstones.
        env['TELEGRAM_ALLOWED_USERS'] = '101,202,303'
        with daemon(root, home, env, barrier=False) as (proc, desc):
            wait_for(lambda: rows()['REVOKED_SECOND']['status'] == 'terminal', diagnostic)
            assert rows()['REVOKED_FIRST']['status'] == 'terminal'
            assert rows()['NEVER_REPLAY']['status'] == 'queued'
            boots.append({'pid': proc.pid, 'epoch': desc['authority_epoch']})
        texts = [next(m['content'] for m in reversed(r['messages']) if m['role'] == 'user') for r in model.requests]
        markers = ['WARM_HISTORY', 'BLOCK_STARTED', 'SAFE_FIRST', 'SAFE_SECOND', 'REVOKED_FIRST', 'REVOKED_SECOND']
        observed = [next((marker for marker in markers if marker in text), text) for text in texts]
        assert observed == markers, observed
        assert len(set(b['pid'] for b in boots)) == len(boots)
        assert all(b['epoch'] > a['epoch'] for a, b in zip(boots, boots[1:]))
        assert 'getUpdates' in bot.calls and bot.sent
        print(json.dumps({'boots': boots, 'model_inputs': observed, 'final_rows': rows()}))
    finally:
        model.release.set()
        for peer in (model, bot):
            peer.shutdown()
            peer.server_close()
