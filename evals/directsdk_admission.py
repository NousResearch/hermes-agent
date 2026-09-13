"""Real native executable, synthetic HTTP peer; no vendor calls or credentials.

Run: .venv/bin/python evals/directsdk_admission.py /path/to/claude
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'plugins/model-providers/claude-subscription-directsdk-experimental'))
import directsdk

USAGE = {'input_tokens':101, 'output_tokens':37, 'cache_read_input_tokens':211, 'cache_creation_input_tokens':313}


def events(mode):
    blocks = [{'type':'text', 'text':'FIRST λ'}]
    if mode == 'thinking':
        blocks = [{'type':'thinking', 'thinking':'fixture thought', 'signature':'fixture-signature'}]
    if mode in ('tools', 'tool_max'):
        blocks += [{'type':'tool_use', 'id':'host_tool_1', 'name':'mcp__hermes__read_file', 'input':{'path':'fixture.txt'}}]
    yield {'type':'message_start', 'message':{'id':'msg_fixture', 'type':'message', 'role':'assistant', 'model':'claude-sonnet-4-6', 'content':[], 'usage':USAGE}}
    for i, block in enumerate(blocks):
        fields = {'text':('text',), 'thinking':('thinking','signature'), 'tool_use':('input',)}[block['type']]
        start = {**block, **{field:({} if field == 'input' else '') for field in fields}}
        yield {'type':'content_block_start', 'index':i, 'content_block':start}
        for field in fields:
            delta = {'type':field+'_delta', field:block[field]}
            if field == 'input':
                delta = {'type':'input_json_delta', 'partial_json':json.dumps(block['input'])}
            yield {'type':'content_block_delta', 'index':i, 'delta':delta}
        yield {'type':'content_block_stop', 'index':i}
    if mode != 'disconnect':
        stop = {'max':'max_tokens', 'tool_max':'max_tokens', 'context':'model_context_window_exceeded', 'tools':'tool_use', 'refusal':'refusal'}.get(mode, 'end_turn')
        yield {'type':'message_delta', 'delta':{'stop_reason':stop, 'stop_sequence':None}, 'usage':USAGE}
        yield {'type':'message_stop'}


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.server.requests += 1
        self.server.entered.set()
        if self.server.mode == 'error':
            self.send_response(429)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"type":"error","error":{"type":"rate_limit_error","message":"fixture"}}')
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        if self.server.mode == 'cancel':
            self.rfile.read(1)  # Cancellation must close this real upstream connection.
            self.server.disconnected.set()
            return
        payload = ''.join('event: '+e['type']+'\ndata: '+json.dumps(e, ensure_ascii=False)+'\n\n' for e in events(self.server.mode)).encode()
        self.wfile.write(payload)
        self.wfile.flush()


def run(binary):
    binary = str(Path(binary).absolute())
    before = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    rows = []
    for mode in ('final', 'tools', 'max', 'tool_max', 'context', 'thinking', 'refusal', 'error', 'disconnect', 'cancel'):
        with tempfile.TemporaryDirectory(prefix='directsdk-admission-eval-') as home:
            peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
            peer.mode, peer.requests = mode, 0
            peer.entered, peer.disconnected = threading.Event(), threading.Event()
            thread = threading.Thread(target=peer.serve_forever, daemon=True)
            thread.start()
            env = {'PATH':os.defpath, 'HOME':home, 'HERMES_HOME':home, 'XDG_CONFIG_HOME':home, 'CLAUDE_CONFIG_DIR':home,
                   'ANTHROPIC_API_KEY':'fixture-not-a-real-key', 'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}',
                   'DISABLE_TELEMETRY':'1', 'DISABLE_ERROR_REPORTING':'1'}
            client = directsdk.Client(command=binary, env=env, timeout=10)
            row = {'mode':mode}
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(client.create, model='claude-sonnet-4-6', messages=[{'role':'user','content':'Local protocol fixture'}],
                                         tools=[{'type':'function', 'function':{'name':'read_file', 'description':'Read a fixture', 'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}}], max_tokens=2400)
                    if mode == 'cancel':
                        assert peer.entered.wait(10)
                        start = time.monotonic()
                        client.cancel()
                    try:
                        result = future.result(timeout=20)
                    except (RuntimeError, TimeoutError) as exc:
                        assert mode in ('error', 'disconnect', 'cancel'), (mode, str(exc))
                        row['rejected'] = type(exc).__name__
                    else:
                        assert mode not in ('error', 'disconnect', 'cancel')
                        usage = result.usage.model_dump()
                        row.update(admission=usage['native_admission'], finish=result.choices[0].finish_reason)
                        assert usage['native_usage'] == USAGE
                        blocks = result.choices[0].message.reasoning_details[0]['messages'][0]['content']
                        if mode == 'thinking':
                            assert blocks[0]['signature'] == 'fixture-signature'
                        elif mode in ('tools', 'tool_max'):
                            assert result.choices[0].message.tool_calls[0].function.name == 'read_file'
                        else:
                            assert result.choices[0].message.content == 'FIRST λ'
                    if mode == 'cancel':
                        row['cancel_seconds'] = time.monotonic() - start
                        assert peer.disconnected.wait(2) and row['cancel_seconds'] < 3
                assert peer.requests == 1, (mode, peer.requests)
                row['upstream_requests'] = peer.requests
                rows.append(row)
                print(json.dumps(row), flush=True)
            finally:
                client.close()
                peer.shutdown()
                thread.join()
                peer.server_close()
    assert hashlib.sha256(Path(binary).read_bytes()).hexdigest() == before
    return {'native_version':subprocess.check_output([binary, '--version'], text=True, stdin=subprocess.DEVNULL).strip(),
            'native_sha256':before, 'synthetic_peer':True, 'cases':rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary')
    print(json.dumps(run(parser.parse_args().binary), indent=2))
