"""Native ACP stdio + SQLite probe; local HTTP model fixture, never paid inference.

Run with the project Python from the checkout being measured. --output saves the
wire transcript and database measurements. No production predicates are patched.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    requests = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply({'object': 'list', 'data': [{'id': 'fixture-model', 'object': 'model'}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append({'path': self.path, 'body': body})
            if body.get('stream'):
                chunk = {'id': 'fixture-stream', 'object': 'chat.completion.chunk',
                         'created': 1, 'model': 'fixture-model',
                         'choices': [{'index': 0, 'delta': {'role': 'assistant',
                                      'content': 'Local fixture reply.'}, 'finish_reason': 'stop'}]}
                data = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.reply({'id': 'fixture-completion', 'object': 'chat.completion',
                        'created': 1, 'model': 'fixture-model',
                        'choices': [{'index': 0, 'message': {'role': 'assistant',
                                     'content': 'Local fixture reply.'}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})

        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    home = Path(tempfile.mkdtemp(prefix='hermes-acp-empty-'))
    hermes = home / '.hermes'
    hermes.mkdir()
    (hermes / 'config.yaml').write_text(
        'model:\n  provider: custom\n  default: fixture-model\n'
        f'  base_url: http://127.0.0.1:{server.server_port}/v1\n'
        '  api_key: local-fixture-key\n  api_mode: chat_completions\n'
        'mcp_servers: {}\nmemory:\n  memory_enabled: false\n  user_profile_enabled: false\n'
        'agent:\n  max_iterations: 1\n  disabled_toolsets: [all]\n'
    )
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(home), HERMES_HOME=str(hermes), PYTHONPATH=os.pathsep.join([str(repo), os.environ.get('PYTHONPATH', '')]),
               HERMES_ACP_SKIP_CONFIGURED_MCP='1', OPENAI_API_KEY='local-fixture-key',
               OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1')
    stderr = open(str(args.output) + '.stderr', 'w', encoding='utf-8')
    proc = subprocess.Popen([sys.executable, '-m', 'acp_adapter'], cwd=repo, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True)
    received = queue.Queue()
    wire = []

    def reader():
        for line in proc.stdout:
            received.put(line)
    threading.Thread(target=reader, daemon=True).start()

    def rpc(method, params):
        request_id = len(wire) + 1
        request = {'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params}
        wire.append({'sent': request})
        proc.stdin.write(json.dumps(request) + '\n')
        proc.stdin.flush()
        while True:
            response = json.loads(received.get(timeout=120))
            wire.append({'received': response})
            if response.get('id') == request_id:
                if 'error' in response:
                    raise RuntimeError(response)
                return response['result']

    def rows():
        with sqlite3.connect(hermes / 'state.db') as db:
            return db.execute('SELECT id, source, message_count FROM sessions ORDER BY id').fetchall()

    result = {'repo': str(repo), 'home': str(home)}
    try:
        rpc('initialize', {'protocolVersion': 1, 'clientCapabilities': {},
                           'clientInfo': {'name': 'local-acp-probe', 'version': '1'}})
        opened = rpc('session/new', {'cwd': str(home), 'mcpServers': []})
        sid = opened['sessionId']
        result['after_open'] = rows()
        result['model_requests_after_open'] = len(requests)
        result['prompt_response'] = rpc('session/prompt', {'sessionId': sid,
            'prompt': [{'type': 'text', 'text': 'Reply with a short greeting; do not use tools.'}]})
        result['after_prompt'] = rows()
        forked = rpc('session/fork', {'sessionId': sid, 'cwd': str(home), 'mcpServers': []})
        result['fork_id'] = forked['sessionId']
        result['after_fork'] = rows()
        with sqlite3.connect(hermes / 'state.db') as db:
            result['messages'] = db.execute('SELECT session_id, role, content FROM messages ORDER BY id').fetchall()
        for session_id in (sid, result['fork_id']):
            assert [(r[1], r[2]) for r in result['messages'] if r[0] == session_id] == [
                ('user', 'Reply with a short greeting; do not use tools.'),
                ('assistant', 'Local fixture reply.')]
        assert any(r[0] == sid and r[2] > 0 for r in result['after_prompt'])
        assert any(r[0] == result['fork_id'] and r[2] > 0 for r in result['after_fork'])
        legacy = rpc('session/new', {'cwd': str(home), 'mcpServers': []})['sessionId']
        with sqlite3.connect(hermes / 'state.db') as db:
            db.execute("INSERT OR IGNORE INTO sessions (id, source, started_at) VALUES (?, 'acp', 1)",
                       (legacy,))
        moved = home / 'moved'
        moved.mkdir()
        rpc('session/load', {'sessionId': legacy, 'cwd': str(moved), 'mcpServers': []})
        with sqlite3.connect(hermes / 'state.db') as db:
            result['existing_empty_metadata'] = db.execute(
                'SELECT model_config, message_count FROM sessions WHERE id = ?', (legacy,)).fetchone()
        assert json.loads(result['existing_empty_metadata'][0])['cwd'] == str(moved)
        assert result['existing_empty_metadata'][1] == 0
        assert result['after_open'] == [], 'session/new created an empty durable row'
    finally:
        proc.terminate()
        proc.wait(timeout=20)
        stderr.close()
        server.shutdown()
        result.update(wire=wire, model_requests=requests)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('wire', 'model_requests', 'messages')}, indent=2))


if __name__ == '__main__':
    main()
