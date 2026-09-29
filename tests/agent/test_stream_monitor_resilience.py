'''The streaming monitor must survive its own display helpers failing.

128167: a dead stream sat in receiving status for ~610s until the turn-liveness
watchdog fired. The stale watchdog plus the 30s heartbeat are the only
dead-stream backstops on the chat path, and both run in _monitor_loop on a
daemon thread with no guard: one escaping exception (local-loader probe,
wait-notice render, heartbeat touch) silently disables all of them while the
worker stays parked in its socket read. The turn then hangs until the
turn-liveness watchdog fires instead of failing fast into the retry path.

Failing-first: with a wedged first stream and a monitor helper that always
raises, the stale kill must still fire and the call must still reconnect
within the stale budget (not the byte-read timeout). Pre-fix the watcher dies
on the first failing tick and recovery waits out the full read timeout.
'''
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import run_agent
from agent import chat_completion_helpers as helpers


class _SilentFirstRequest:
    '''Request 1: 200 + SSE headers, then no bytes (parked body read).
    Later requests: a normal completion.'''

    def __init__(self):
        self.completions = []
        self.release = threading.Event()
        hits = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_a):
                pass

            def do_POST(self):
                n = int(self.headers.get('content-length', 0))
                body = json.loads(self.rfile.read(n) or b'{}')
                if not self.path.endswith('/chat/completions'):
                    self.send_response(404)
                    self.send_header('content-length', '0')
                    self.end_headers()
                    return
                hits.completions.append(body)
                first = len(hits.completions) == 1
                self.send_response(200)
                self.send_header('content-type', 'text/event-stream')
                self.end_headers()
                if first:
                    hits.release.wait(timeout=60.0)
                    return
                try:
                    chunk = {'id': 'c1', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'm',
                             'choices': [{'index': 0, 'delta': {'content': 'reconnected'}, 'finish_reason': None}]}
                    self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
                    fin = {'id': 'c1', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'm',
                           'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
                    self.wfile.write(b'data: ' + json.dumps(fin).encode() + b'\n\n')
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.05), daemon=True).start()
        self.base_url = 'http://127.0.0.1:' + str(self.server.server_address[1]) + '/v1'

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


def test_monitor_survives_failing_display_helper_and_stale_kill_still_fires(
    monkeypatch, caplog,
):
    '''The local-loader probe runs inside the monitor tick from ~2s on a local
    endpoint. When it blows up every tick, the stale kill at 8s must still fire
    and the call must reconnect (not wait out the 30s byte-read timeout).'''
    import logging

    monkeypatch.setenv('HERMES_STREAM_STALE_TIMEOUT', '8')
    monkeypatch.setenv('HERMES_STREAM_RETRIES', '1')
    monkeypatch.setenv('HERMES_STREAM_READ_TIMEOUT', '30')

    def _boom(agent, api_kwargs):
        raise RuntimeError('simulated local-loader probe failure')

    monkeypatch.setattr(helpers, '_managed_local_load_notice', _boom)

    wire = _SilentFirstRequest()
    agent = run_agent.AIAgent(
        api_key='test-key', base_url=wire.base_url, model='m', provider='custom',
        platform='cli',
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )
    try:
        agent.api_mode = 'chat_completions'
        agent._interrupt_requested = False
        with caplog.at_level(logging.ERROR, logger='agent.chat_completion_stream_monitor'):
            started = time.time()
            response = agent._interruptible_streaming_api_call(
                {'model': 'm', 'messages': [{'role': 'user', 'content': 'hi'}]})
            elapsed = time.time() - started
    finally:
        wire.close()

    assert len(wire.completions) >= 2, (
        'stale kill never fired: requests=' + str(len(wire.completions)))
    assert response.choices[0].message.content == 'reconnected'
    assert elapsed < 20.0, (
        'recovery took ' + ('%.1f' % elapsed) + 's: the watcher died with the failing '
        'helper and the reader waited out the byte-read timeout instead of the stale budget')
    assert 'stays armed' in caplog.text, (
        'the surviving watcher must loudly log the suppressed tick failure')
