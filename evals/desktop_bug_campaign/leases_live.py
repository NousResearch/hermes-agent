"""Real WS/backend-process lease recovery probe; deterministic HTTP inference.

Run: .venv/bin/python evals/desktop_bug_campaign/leases_live.py --output DIR
Fault injection cancels an actual disconnected session's orphan timer and,
optionally, restores its real closed transport to model missed detachment.
Delegation uses the real registry/executor with an Event-blocked fixture runner.
No ownership predicate, lease registry, agent, or lifecycle function is replaced.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request


def backend(port):
    from hermes_cli.web_server import app, start_server
    from tui_gateway import server as gw

    work = {}

    @app.post('/lease-probe/{action}/{sid}')
    def control(action: str, sid: str):
        if action == 'remember-transport':
            work['transport:' + sid] = gw._sessions[sid]['transport']
        elif action == 'lose-detach':
            with gw._sessions_lock:
                rec = gw._sessions[sid]
                assert gw._ws_session_is_detached(rec)
                old_transport = work['transport:' + sid]
                assert gw._transport_is_dead(old_transport)
                gw._cancel_ws_orphan_reap(sid)
                rec['transport'] = old_transport
        elif action == 'lose-timer':
            with gw._sessions_lock:
                rec = gw._sessions[sid]
                assert gw._ws_session_is_detached(rec)
                assert sid in gw._pending_ws_reaps
                gw._cancel_ws_orphan_reap(sid)
        elif action == 'delegate':
            from tools.async_delegation import dispatch_async_delegation
            release, started, interrupted = threading.Event(), threading.Event(), threading.Event()
            work[sid] = (release, started, interrupted)
            def runner():
                started.set()
                release.wait(90)
                return {'summary': 'LEASE_DELEGATE_DONE', 'status': 'completed'}
            rec = gw._sessions[sid]
            handle = dispatch_async_delegation(goal='lease protection control', context=None,
                toolsets=[], role='leaf', model=None, session_key=rec['session_key'],
                origin_ui_session_id=sid, parent_session_id=rec['session_key'], runner=runner,
                interrupt_fn=interrupted.set)
            assert handle['status'] == 'dispatched', handle
            assert started.wait(10)
        elif action == 'finish-delegate':
            work[sid][0].set()
        elif action == 'sweep':
            gw._reap_idle_sessions()
        with gw._sessions_lock:
            rec = gw._sessions.get(sid)
            return {'resident': rec is not None,
                    'detached': gw._ws_session_is_detached(rec),
                    'pending': sid in gw._pending_ws_reaps,
                    'running': bool(rec and rec.get('running')),
                    'delegating': bool(rec and gw._session_has_active_delegations(sid, rec)),
                    'delegate_interrupted': sid in work and work[sid][2].is_set()}

    start_server(host='127.0.0.1', port=port, open_browser=False, headless=True)


def fixture(port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            text = 'LEASE_RECOVERY_REPLY' if 'RECOVERY_TURN_ONLY' in str(req.get('messages', [])) else 'LEASE_PROBE_REPLY'
            self.send_response(200)
            if req.get('stream'):
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                for delta, finish in [({'role': 'assistant', 'content': text}, None), ({}, 'stop')]:
                    event = {'id': 'probe', 'object': 'chat.completion.chunk', 'model': 'probe',
                             'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                    self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
            else:
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'id': 'probe', 'object': 'chat.completion', 'model': 'probe',
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13}}).encode())
    httpd = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class Client:
    counter = 0
    def __init__(self, port):
        from websockets.sync.client import connect
        self.ws = connect(f'ws://127.0.0.1:{port}/api/ws?token=lease-probe')
    def call(self, method, params):
        Client.counter += 1
        rid = Client.counter
        self.ws.send(json.dumps({'jsonrpc': '2.0', 'id': rid, 'method': method, 'params': params}))
        while True:
            frame = json.loads(self.ws.recv(timeout=90))
            if frame.get('id') == rid:
                return frame
    def close(self):
        self.ws.close()


def main(out, closed_transport=False):
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    home = Path(tempfile.mkdtemp(prefix='leases-live-'))
    (home / 'config.yaml').write_text('''model:
  default: probe
  provider: custom
  base_url: http://127.0.0.1:18042/v1
  api_key: probe-key
dashboard:
  turn_isolation: false
  ws_orphan_reap_grace_s: 3
  startup_orphan_sweep: false
memory:
  memory_enabled: false
  user_profile_enabled: false
''')
    env = {'PATH': os.environ.get('PATH', ''), 'HOME': str(home / 'user'), 'LANG': 'C.UTF-8'}
    (home / 'user').mkdir()
    env.update(HERMES_HOME=str(home), HERMES_DASHBOARD_SESSION_TOKEN='lease-probe',
               PYTHONPATH=str(repo), OPENAI_API_KEY='probe-key')
    processes = []
    logs = []
    service = fixture(18042)
    results = {'sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
               'home': str(home), 'fidelity': 'real backend subprocesses + WebSocket; deterministic local HTTP inference; lost-timer fault injection', 'checks': []}
    def check(name, ok, detail=None):
        results['checks'].append({'name': name, 'pass': bool(ok), 'detail': detail})
        print(json.dumps(results['checks'][-1]), flush=True)
    def control(port, action, sid):
        req = urllib.request.Request(f'http://127.0.0.1:{port}/lease-probe/{action}/{sid}', data=b'', method='POST', headers={'X-Hermes-Session-Token': 'lease-probe'})
        return json.load(urllib.request.urlopen(req, timeout=30))
    def wait_idle(port, sid):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = control(port, 'state', sid)
            if not state['running']:
                return state
            time.sleep(.1)
        raise TimeoutError('turn')
    try:
        for port in [18040, 18041]:
            log = (out / f'backend-{port}.log').open('w'); logs.append(log)
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--backend', str(port)], cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            processes.append(p)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    control(port, 'state', 'none'); break
                except Exception:
                    if p.poll() is not None:
                        raise RuntimeError(f'backend exited {p.returncode}')
                    time.sleep(.2)
            else:
                raise TimeoutError('backend readiness')
        a = Client(18040)
        created = a.call('session.create', {'source': 'desktop', 'close_on_disconnect': False})
        print('CREATE', created, flush=True)
        sid = created['result']['session_id']; stored = created['result']['stored_session_id']
        submitted = a.call('prompt.submit', {'session_id': sid, 'text': 'Reply LEASE_PROBE_REPLY'})
        check('initial prompt accepted', 'result' in submitted, submitted)
        wait_idle(18040, sid)
        b = Client(18041)
        resumed = b.call('session.resume', {'session_id': stored, 'source': 'desktop', 'lazy': True, 'omit_messages': True})
        print('RESUME', resumed, flush=True)
        sid_b = resumed['result']['session_id']
        refused = b.call('prompt.submit', {'session_id': sid_b, 'text': 'Reply LEASE_PROBE_REPLY'})
        check('foreign backend refused while owner attached', refused.get('error', {}).get('code') == 4090, refused)
        a.close()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if control(18040, 'state', sid)['detached']:
                break
            time.sleep(.02)
        lost = control(18040, 'lose-timer', sid)
        check('fault precondition resident detached without timer', lost['resident'] and lost['detached'] and not lost['pending'], lost)
        control(18040, 'sweep', sid)
        # Reconnect within the repaired grace: same runtime and fence survive.
        a = Client(18040)
        reattached = a.call('session.resume', {'session_id': stored, 'source': 'desktop', 'lazy': True, 'omit_messages': True})
        check('same backend reconnect retains runtime', reattached.get('result', {}).get('session_id') == sid, {'session_id': reattached.get('result', {}).get('session_id')})
        time.sleep(4)
        state = control(18040, 'state', sid)
        check('reconnect cancels repaired timer', state['resident'] and not state['detached'] and not state['pending'], state)
        refused = b.call('prompt.submit', {'session_id': sid_b, 'text': 'Still fenced'})
        check('foreign backend remains fenced after reconnect', refused.get('error', {}).get('code') == 4090, refused)
        control(18040, 'delegate', sid)
        control(18040, 'remember-transport', sid)
        a.close()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if control(18040, 'state', sid)['detached']:
                break
            time.sleep(.02)
        fault = control(18040, 'lose-detach' if closed_transport else 'lose-timer', sid)
        check('dead transport fault established', fault['resident'] and not fault['pending'] and (not fault['detached'] if closed_transport else fault['detached']), fault)
        control(18040, 'sweep', sid)
        time.sleep(4)
        state = control(18040, 'state', sid)
        check('real async registry work survives repaired reap', state['resident'] and state['delegating'] and not state['delegate_interrupted'], state)
        refused = b.call('prompt.submit', {'session_id': sid_b, 'text': 'Still fenced during delegation'})
        check('foreign backend fenced during detached delegation', refused.get('error', {}).get('code') == 4090, refused)
        control(18040, 'finish-delegate', sid)
        time.sleep(7)
        state = control(18040, 'state', sid)
        check('periodic maintenance recovers lost timer', not state['resident'], state)
        retry = b.call('prompt.submit', {'session_id': sid_b, 'text': 'RECOVERY_TURN_ONLY'})
        check('foreign backend submits after recovery', 'result' in retry, retry)
        wait_idle(18041, sid_b)
        import sqlite3
        with sqlite3.connect(home / 'state.db') as db:
            replies = db.execute("SELECT count(*) FROM messages WHERE session_id=? AND role='assistant' AND content LIKE ?", (stored, '%LEASE_RECOVERY_REPLY%')).fetchone()[0]
        check('post-recovery reply persisted to receiver database', replies >= 1, {'assistant_replies': replies})
        b.close()
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=10)
        for log in logs:
            log.close()
        service.shutdown()
        (out / 'result.json').write_text(json.dumps(results, indent=2))
    return 0 if all(x['pass'] for x in results['checks']) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--closed-transport', action='store_true')
    args = parser.parse_args()
    if args.backend:
        backend(args.backend)
    else:
        sys.exit(main(args.output, args.closed_transport))
