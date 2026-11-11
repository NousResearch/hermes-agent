"""A Telegram chat's ``cd`` survives a gateway restart (#41128).

The terminal tool keeps each chat's working directory in a process-local record, so a restart sent
the chat's next command back to ``terminal.cwd``. Real daemon, loopback Bot API, and a model stub
that drives the real terminal tool: ``cd sub`` in one life, ``pwd`` in the next.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.machinery
import json
from pathlib import Path
import queue
import shutil
import threading
import time

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon
from tests.gateway.test_native_telegram_startup_recovery import BotAPI, wait_for


class Model(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        msgs = body.get('messages', [])
        last = msgs[-1] if msgs else {}
        user = next((m.get('content', '') for m in reversed(msgs) if m['role'] == 'user'), '')
        if isinstance(user, list):
            user = ' '.join(p.get('text', '') for p in user if isinstance(p, dict))
        if last.get('role') == 'tool':
            self.server.tool_outputs.append(last.get('content'))
            msg = {'role': 'assistant', 'content': 'DONE ' + str(last.get('content'))[:300]}
            finish = 'stop'
        else:
            cmd = self.server.sub_cmd if 'CD_SUB' in user else 'pwd'
            msg = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'c{len(self.server.tool_outputs)}', 'type': 'function',
                   'function': {'name': 'terminal', 'arguments': json.dumps({'command': cmd})}}]}
            finish = 'tool_calls'
        if body.get('stream'):
            delta = dict(msg)
            if 'tool_calls' in delta:
                delta['tool_calls'] = [dict(tc, index=0) for tc in delta['tool_calls']]
            payload = ('data: ' + json.dumps({'id': 'x', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})
                       + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            payload = json.dumps({'id': 'x', 'choices': [{'index': 0, 'message': msg, 'finish_reason': finish}],
                                  'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
            kind = 'application/json'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)



def _pwd_after_restart(tmp_path, *, remove_sub_before_restart):
    root = Path(__file__).resolve().parents[2]
    home, user, work = tmp_path / 'state', tmp_path / 'user', tmp_path / 'work'
    home.mkdir(mode=0o700)
    user.mkdir()
    (work / 'sub').mkdir(parents=True)
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.tool_outputs, model.sub_cmd = [], f'cd {work / "sub"} && pwd'
    bot = ThreadingHTTPServer(('127.0.0.1', 0), BotAPI)
    bot.calls, bot.sent, bot.updates = [], [], queue.Queue()
    for peer in (model, bot):
        threading.Thread(target=peer.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{model.server_port}/v1'
    cfg = {'model': {'provider': 'custom', 'default': 'stub', 'base_url': base},
           'gateway': {'multiplex_profiles': False},
           'platforms': {'telegram': {'enabled': True, 'extra': {
               'base_url': f'http://127.0.0.1:{bot.server_port}/bot', 'dm_policy': 'allowlist'}}},
           'auxiliary': {'title_generation': {'enabled': False}}, 'approvals': {'mode': 'off'},
           'platform_toolsets': {'telegram': ['terminal']}, 'terminal': {'cwd': str(work), 'backend': 'local'}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    env = child_env()
    env.update(HOME=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root), OPENAI_API_KEY='loopback-only',
               OPENAI_BASE_URL=base, PYTHONUNBUFFERED='1', TELEGRAM_BOT_TOKEN='987654321:owned-loopback-fixture',
               TELEGRAM_ALLOWED_USERS='202')
    update_id = 0

    def push(text):
        nonlocal update_id
        update_id += 1
        bot.updates.put({'update_id': update_id, 'message': {
            'message_id': update_id, 'date': int(time.time()),
            'chat': {'id': 202, 'type': 'private', 'first_name': 'Owned'},
            'from': {'id': 202, 'is_bot': False, 'first_name': 'Owned'}, 'text': text}})

    def diagnostic():
        return repr(model.tool_outputs) + '\n' + '\n'.join(
            p.read_text(encoding='utf-8')[-4000:] for p in (home / 'logs').glob('*.log'))

    def pwd(index):
        return json.loads(model.tool_outputs[index])['output'].strip()

    try:
        with daemon(root, home, env, barrier=False) as (proc, _desc):
            push('CD_SUB please')
            wait_for(lambda: len(model.tool_outputs) >= 1, diagnostic)
            push('where now')
            wait_for(lambda: len(model.tool_outputs) >= 2, diagnostic)
            assert pwd(1) == str(work / 'sub'), diagnostic()
            proc.terminate()
            proc.wait(timeout=60)
        if remove_sub_before_restart:
            shutil.rmtree(work / 'sub')
        with daemon(root, home, env, barrier=False) as (proc, _desc):
            push('where after restart')
            wait_for(lambda: len(model.tool_outputs) >= 3, diagnostic)
            return pwd(2), work
    finally:
        for peer in (model, bot):
            peer.shutdown()
            peer.server_close()


pytestmark = [
    pytest.mark.skipif(importlib.machinery.PathFinder.find_spec("telegram") is None,
                       reason="python-telegram-bot not installed (on-demand extra)"),
    pytest.mark.platforms("linux"),
]


def test_a_chats_cd_survives_a_gateway_restart(tmp_path):
    after, work = _pwd_after_restart(tmp_path, remove_sub_before_restart=False)
    assert after == str(work / 'sub')


def test_a_directory_gone_after_the_restart_falls_back_to_terminal_cwd(tmp_path):
    after, work = _pwd_after_restart(tmp_path, remove_sub_before_restart=True)
    assert after == str(work)
