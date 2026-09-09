"""Real serve + WebSocket + local HTTP inference fixture hook probe (no vendor calls)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml
from websockets.sync.client import connect

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--port', type=int, default=18000)
p.add_argument('--tool', choices=['write_file', 'terminal'], default='write_file')
p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
a = p.parse_args()
out = Path(a.output).absolute()
if out.exists() and any(out.iterdir()):
    raise SystemExit('Use a fresh output directory; prior evidence must not be reused.')
out.mkdir(parents=True, exist_ok=True)
home = out / 'home'
home.mkdir(exist_ok=True)
requests = []


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        requests.append(req)
        messages = req.get('messages', [])
        last_user = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')
        label = next((x for x in ['alpha', 'beta', 'default', 'unapproved'] if x in str(last_user)), 'default')
        tool_seen = any(m['role'] == 'tool' for m in messages)
        target = str(out / (label + '-protected.txt'))
        tool_args = ({'path': target, 'content': 'UNGUARDED'} if a.tool == 'write_file' else {
            'command': f'printf %s UNGUARDED > {shlex.quote(target)}'})
        delta = {'role': 'assistant', 'content': 'fixture complete'} if tool_seen or not req.get('tools') else {
            'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'call_probe', 'type': 'function', 'function': {
                'name': a.tool, 'arguments': json.dumps(tool_args)}}]}
        finish = 'stop' if 'content' in delta else 'tool_calls'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if req.get('stream') else 'application/json')
        self.end_headers()
        if req.get('stream'):
            for d, f in [(delta, None), ({}, finish)]:
                self.wfile.write(('data: ' + json.dumps({'id': 'fixture', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': d, 'finish_reason': f}]}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            delta.pop('index', None)
            for tc in delta.get('tool_calls', []):
                tc.pop('index', None)
            self.wfile.write(json.dumps({'id': 'fixture', 'object': 'chat.completion', 'choices': [{'index': 0, 'message': delta, 'finish_reason': finish}], 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}).encode())


provider = ThreadingHTTPServer(('127.0.0.1', a.port + 1), Provider)
threading.Thread(target=provider.serve_forever, daemon=True).start()
for label in ['default', 'alpha', 'beta', 'unapproved']:
    h = home if label == 'default' else home / 'profiles' / label
    h.mkdir(parents=True, exist_ok=True)
    hook = h / 'guard.py'
    hook.write_text('import json,sys\nfrom pathlib import Path\npayload=json.load(sys.stdin)\nwith Path(' + repr(str(out / (label + '-hooks.jsonl'))) + ').open("a") as f: f.write(json.dumps(payload)+"\\n")\nprint(json.dumps({"decision":"block","reason":' + repr('guard-' + label) + '}))\n')
    cfg = {'model': {'default': 'fixture-model', 'provider': 'custom', 'base_url': f'http://127.0.0.1:{a.port+1}/v1', 'api_key': 'fixture-key'},
           'hooks_auto_accept': label != 'unapproved', 'hooks': {'pre_tool_call': [{'command': shlex.join([sys.executable, str(hook)]), 'matcher': a.tool, 'fail_closed': True}]},
           'toolsets': ['file', 'terminal'], 'agent': {'max_turns': 3}, 'memory': {'memory_enabled': False, 'user_profile_enabled': False},
           'curator': {'enabled': False}, 'compression': {'enabled': False}, 'terminal': {'cwd': str(out)}}
    (h / 'config.yaml').write_text(yaml.safe_dump(cfg))
    (h / '.env').write_text('OPENAI_API_KEY=fixture-key\nOPENAI_BASE_URL=http://127.0.0.1:' + str(a.port+1) + '/v1\n')
env = {k: v for k, v in os.environ.items() if not (k.startswith('HERMES_') or k.endswith(('_API_KEY', '_TOKEN')))}
env.update(HOME=str(out / 'os-home'), HERMES_HOME=str(home), HERMES_DASHBOARD_SESSION_TOKEN='hooks-fixture-token', HERMES_IGNORE_RULES='1', PYTHONPATH=str(a.repo))
cmd = [sys.executable, '-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', str(a.port), '--skip-build']
log = (out / 'serve.log').open('w')
proc = subprocess.Popen(cmd, cwd=a.repo, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
events = []
results = []
try:
    deadline = time.monotonic() + 90
    while True:
        try:
            ws = connect(f'ws://127.0.0.1:{a.port}/api/ws?token=hooks-fixture-token', open_timeout=2)
            break
        except Exception:
            if time.monotonic() > deadline or proc.poll() is not None:
                raise RuntimeError('serve failed to become ready')
            time.sleep(.3)
    serial = 0
    def rpc(method, params):
        global serial
        serial += 1
        rid = str(serial)
        ws.send(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': method, 'params': params}))
        while True:
            event = json.loads(ws.recv(timeout=90))
            events.append(event)
            if str(event.get('id')) == rid:
                if 'error' in event:
                    raise RuntimeError(event)
                return event['result']
    for label in ['default', 'alpha', 'beta', 'alpha', 'unapproved']:
        hook_log = out / (label + '-hooks.jsonl')
        calls_before = len(hook_log.read_text().splitlines()) if hook_log.exists() else 0
        session = rpc('session.create', {'source': 'gui', 'profile': '' if label == 'default' else label, 'cwd': str(out)})
        sid = session['session_id']
        rpc('prompt.submit', {'session_id': sid, 'text': f'{label} probe: write the protected file'})
        deadline = time.monotonic() + 100
        completed = False
        while time.monotonic() < deadline:
            event = json.loads(ws.recv(timeout=100))
            events.append(event)
            data = event.get('params', {})
            if data.get('session_id') == sid and data.get('type') == 'message.complete':
                completed = data.get('payload', {}).get('status') == 'complete'
                break
        assert completed, f'{label}: no successful completion for {sid}'
        results.append({'profile': label, 'written': (out / (label + '-protected.txt')).exists(), 'hook_calls': len(hook_log.read_text().splitlines()) if hook_log.exists() else 0, 'hook_calls_before': calls_before})
        print(results[-1], flush=True)
    ws.close()
    assert all(not row['written'] and row['hook_calls'] - row['hook_calls_before'] == 1 for row in results if row['profile'] != 'unapproved'), 'consented profile hook did not protect the tool exactly once'
    assert results[-1]['written'] and results[-1]['hook_calls'] == 0, 'unapproved hook executed'
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    provider.shutdown()
    (out / 'events.json').write_text(json.dumps(events, indent=2))
    (out / 'requests.json').write_text(json.dumps(requests, indent=2))
    receipt = {'sha': subprocess.check_output(['git', '-C', str(a.repo), 'rev-parse', 'HEAD'], stdin=subprocess.DEVNULL, text=True).strip(), 'source_blob': subprocess.check_output(['git', '-C', str(a.repo), 'hash-object', 'tui_gateway/server.py'], stdin=subprocess.DEVNULL, text=True).strip(), 'tool': a.tool, 'process_exit': proc.returncode, 'command': cmd, 'results': results, 'provider_requests': len(requests)}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
