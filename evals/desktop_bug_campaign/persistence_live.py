"""Real serve/WS profile-rebuild probe; deterministic loopback model, no vendor inference."""
import argparse
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18020)
    parser.add_argument('--invalid-targets', action='store_true')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--failure', action='store_true', help='inject one model-config preparation failure')
    parser.add_argument('--observe', action='store_true', help='inspect ownership without injecting failure')
    args = parser.parse_args()
    observe = args.observe or args.failure
    args.out.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix='persistence-live-'))
    profile = home / 'profiles' / 'worker'
    profile.mkdir(parents=True)
    requests = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'data': [{'id': 'persistence-fixture', 'context_length': 128000}]}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append({'path': self.path, 'stream': body.get('stream'), 'messages': len(body.get('messages', []))})
            text = 'Fixture response ' + str(len(requests))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
            self.end_headers()
            if body.get('stream'):
                for delta, finish in [({'role': 'assistant', 'content': text}, None), ({}, 'stop')]:
                    chunk = {'id': 'fixture', 'object': 'chat.completion.chunk', 'model': 'persistence-fixture', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                    self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
            else:
                self.wfile.write(json.dumps({'id': 'fixture', 'object': 'chat.completion', 'model': 'persistence-fixture', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1000, 'completion_tokens': 10, 'total_tokens': 1010}}).encode())

    model = ThreadingHTTPServer(('127.0.0.1', args.port + 1), Model)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    config = {'model': {'default': 'persistence-fixture', 'provider': 'custom', 'base_url': f'http://127.0.0.1:{args.port+1}/v1'}, 'agent': {'max_turns': 1}, 'compression': {'enabled': False}, 'toolsets': [], 'platform_toolsets': {'gui': [], 'cli': []}, 'memory': {'memory_enabled': False, 'user_profile_enabled': False}}
    import yaml
    for p in (home, profile):
        (p / 'config.yaml').write_text(yaml.safe_dump(config))
        (p / '.env').write_text('OPENAI_API_KEY=local-fixture\n')
        (p / 'SOUL.md').write_text('You are a deterministic test assistant.\n')
        (p / '.no-bundled-skills').touch()
    (profile / 'profile.yaml').write_text('name: worker\nui_meta:\n  hermes-bots:\n    title: Worker\n')
    env = {k: v for k, v in os.environ.items() if not (k.startswith('HERMES_') or 'API_KEY' in k or 'TOKEN' in k or k in ('PYTHONPATH', 'PYTEST_PLUGINS'))}
    env.update(HERMES_HOME=str(home), HOME=str(home / 'os-home'), HERMES_DASHBOARD_SESSION_TOKEN='persistence-fixture-token', HERMES_IGNORE_RULES='1', PYTHONPATH=str(args.repo), OPENAI_API_KEY='local-fixture')
    log = (args.out / 'serve.log').open('w')
    cmd = [sys.executable, '-m', 'hermes_cli.main', 'serve', '--isolated', '--port', str(args.port)]
    if observe:
        cmd = [sys.executable, str(Path(__file__).with_name('rebuild_observer.py')),
               'serve', '--isolated', '--port', str(args.port)]
    proc = subprocess.Popen(cmd, cwd=args.repo, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    transcript = []
    result = {'home': str(home), 'command': cmd, 'sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo, text=True).strip()}
    try:
        for _ in range(180):
            if proc.poll() is not None:
                raise RuntimeError(f'serve exited {proc.returncode}')
            try:
                with socket.create_connection(('127.0.0.1', args.port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        from websockets.sync.client import connect
        with connect(f'ws://127.0.0.1:{args.port}/api/ws?token=persistence-fixture-token', max_size=20_000_000) as ws:
            counter = 0
            def receive():
                row = json.loads(ws.recv(timeout=120))
                transcript.append(row)
                with (args.out / 'wire-live.jsonl').open('a') as receipt:
                    receipt.write(json.dumps(row) + '\n')
                print(str(row)[:250], flush=True)
                return row
            def rpc(method, params, allow_error=False):
                nonlocal counter
                counter += 1
                ws.send(json.dumps({'jsonrpc': '2.0', 'id': counter, 'method': method, 'params': params}))
                while True:
                    row = receive()
                    if row.get('id') == counter:
                        if 'error' in row:
                            if allow_error:
                                return row
                            raise RuntimeError(row)
                        return row['result']
            def turn(sid, text):
                rpc('prompt.submit', {'session_id': sid, 'text': text})
                while True:
                    row = receive()
                    p = row.get('params', {})
                    if p.get('type') in ('turn.completed', 'turn.complete'):
                        return
                    if p.get('type') == 'session.info' and p.get('payload', {}).get('running') is False:
                        return
            created = rpc('session.create', {'profile': 'worker', 'source': 'desktop', 'title': 'Bot Chat', 'cwd': str(home)})
            result['created'] = created
            sid, key = created['session_id'], created['stored_session_id']
            turn(sid, 'PERSISTENCE_BEFORE')
            launch_before = (home / 'config.yaml').read_bytes()
            if observe:
                result['ownership_before'] = rpc('probe.ownership', {'session_id': sid})
            if args.failure:
                (home / 'fail-rebuild').touch()
            if args.compress:
                for i in range(12):
                    turn(sid, f'Historical note {i}: ' + ('The worker keeps the project history and decisions in its own profile. ' * 100))
                result['compression'] = rpc('session.compress', {'session_id': sid})
                with sqlite3.connect(profile / 'state.db') as db:
                    result['generations'] = db.execute('SELECT role, content, GROUP_CONCAT(active), COUNT(*) FROM messages WHERE session_id=? GROUP BY role, content HAVING COUNT(*)>1', (key,)).fetchall()
                    result['duplicate_active_groups'] = db.execute('SELECT COUNT(*) FROM (SELECT role, content FROM messages WHERE session_id=? AND active=1 GROUP BY role, content HAVING COUNT(*)>1)', (key,)).fetchone()[0]
            if args.reset:
                try:
                    result['reset'] = rpc('tools.configure', {'session_id': sid, 'action': 'enable', 'names': ['web']})
                except RuntimeError as exc:
                    if not args.failure or 'probe config read failure' not in str(exc):
                        raise
                    result['expected_failure'] = str(exc)
                result['launch_config_unchanged'] = (home / 'config.yaml').read_bytes() == launch_before
                sys.path.insert(0, str(args.repo))
                from hermes_cli.config import read_user_config_raw
                result['worker_tools'] = read_user_config_raw(profile / 'config.yaml')['platform_toolsets']['cli']
            else:
                (profile / 'SOUL.md').write_text('Capabilities changed for the worker.\n')
            if observe and args.reset:
                result['ownership_after_failure'] = rpc('probe.ownership', {'session_id': sid})
            turn(sid, 'PERSISTENCE_AFTER_REBUILD')
            if observe and not args.reset:
                result['ownership_after_failure'] = rpc('probe.ownership', {'session_id': sid})
            result['stores'] = {}
            for name, p in [('launch', home), ('worker', profile)]:
                with sqlite3.connect(p / 'state.db') as db:
                    result['stores'][name] = db.execute('SELECT role, content, active FROM messages WHERE session_id=? ORDER BY id', (key,)).fetchall()
            result['wrong_profile_writes'] = any('PERSISTENCE_AFTER_REBUILD' in str(row) for row in result['stores']['launch'])
            if args.invalid_targets:
                result['target_controls'] = {str(p): rpc('session.list', {'profile': p}) for p in (None, 'default', 'worker')}
                rpc('session.close', {'session_id': sid})
                before = (home / 'config.yaml').read_bytes()
                result['stale_session'] = rpc('tools.configure', {'session_id': sid, 'action': 'enable', 'names': ['terminal']}, allow_error=True)
                profile.rename(home / 'removed-worker')
                result['invalid_profiles'] = {name: {method: rpc(method, dict(params, profile=name), allow_error=True) for method, params in [('session.list', {}), ('config.get', {'key': 'full'}), ('config.set', {'key': 'busy', 'value': 'steer'})]} for name in ('worker', 'unknown')}
                result['invalid_config_unchanged'] = before == (home / 'config.yaml').read_bytes()
                result['global_tools'] = rpc('tools.configure', {'action': 'enable', 'names': ['terminal']})
                result['global_config_changed'] = before != (home / 'config.yaml').read_bytes()
            if observe:
                rpc('session.close', {'session_id': sid})
                result['ownership_after_close'] = rpc('probe.ownership', {'session_id': sid})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        model.shutdown()
        model.server_close()
        log.close()
        (args.out / 'wire.json').write_text(json.dumps(transcript, indent=2))
        result['model_requests'] = requests
        (args.out / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ('sha', 'home', 'wrong_profile_writes', 'duplicate_active_groups') if k in result}, indent=2))
    assert not result['wrong_profile_writes'], 'rebuild wrote the next turn to launch state.db'
    assert any('PERSISTENCE_AFTER_REBUILD' in str(row) for row in result['stores']['worker'])
    if args.invalid_targets:
        assert 'error' in result['stale_session'], 'stale session changed launch config'
        assert all('error' in reply for replies in result['invalid_profiles'].values() for reply in replies.values())
        assert result['invalid_config_unchanged']
        assert result['global_config_changed'] and not result['global_tools']['reset']
    if args.reset:
        assert result['launch_config_unchanged'], 'tools.configure changed launch config'
        assert 'web' in result['worker_tools'], 'tools.configure did not change worker config'
    if observe:
        before, after = result['ownership_before'], result['ownership_after_failure']
        assert before['owns_db'] and after['owns_db'], 'rebuild lost reachable DB owner'
        assert (before['agent'] == after['agent']) is args.failure, 'unexpected replacement outcome'
        assert str(profile / 'state.db') not in result['ownership_after_close']['refs'], 'teardown leaked profile DB ref'
    if args.compress:
        assert result['compression']['status'] == 'compressed'
        assert result['generations'], 'compaction must retain archived generations'
        assert result['duplicate_active_groups'] == 0



if __name__ == '__main__':
    main()
