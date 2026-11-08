"""Owned probe: one real review-lane run completes end to end on a disposable home.

The fixture runs the real stack — a real ``gateway.run`` daemon, the real dispatcher, the real
worker client and the real managed child process — against a real board whose card sits in the
``review`` column. The loopback model peer answers the first turn with a ``kanban_comment`` tool
call and the second with ``kanban_complete``, so the reviewer really ends the run through the board
protocol, exactly as a live review does.

This is the acceptance the receipt lane could not meet while ``worker.persist`` was dying
(t_5ae5058d): the run must be booked ``completed``, the card must reach ``done``, and the worker
client must exit 0 — not ``Kanban execution unknown; retained by owner``.

Receipt keys:

  spawned       run ids the dispatcher launched (must be this card)
  run_outcome   the booked run outcome
  task_status   the card after the reviewer's completion
  client_exit   the worker client's classified exit
  lane          the claimed run's lane, so a silently-executed run cannot pass as a review
"""
import json
import os
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import time

from local_recovery_probe import daemon


class Model(BaseHTTPRequestHandler):
    """Two tool turns and a final answer; no reasoning, no network."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        has_result = any(m['role'] == 'tool' for m in messages)
        if os.environ.get('REVIEW_PROBE_MODE') == 'model-fails':
            # Negative control: a turn the model never completes must never book the card done.
            payload = json.dumps({'error': {'message': 'probe: model refused',
                                            'type': 'invalid_request_error'}}).encode()
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if not body.get('tools'):
            message, finish = {'role': 'assistant', 'content': 'REVIEW_DONE'}, 'stop'
        elif has_result:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': 'review-done',
                'type': 'function', 'function': {'name': 'kanban_complete',
                'arguments': json.dumps({'summary': 'Review probe: approved end to end'})}}]}
            finish = 'tool_calls'
        else:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': 'review-note',
                'type': 'function', 'function': {'name': 'kanban_comment',
                'arguments': json.dumps({'body': 'Review probe: reading the card'})}}]}
            finish = 'tool_calls'
        payload = json.dumps({'id': 'review', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'review', 'choices': [{'index': 0, 'delta': message,
                'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    root = Path(__file__).resolve().parents[3]
    home = Path(os.environ['HERMES_HOME'])
    home.mkdir(parents=True, exist_ok=True)
    Path(os.environ['HOME']).mkdir(parents=True, exist_ok=True)
    os.environ['HERMES_KANBAN_HOME'] = str(home)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    workspace = home / 'workspace'
    workspace.mkdir(exist_ok=True)
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'loop-model', 'base_url': url},
           'auxiliary': {'title_generation': {'enabled': False}},
           'platform_toolsets': {'cli': []}, 'terminal': {'backend': 'local'},
           'agent': {'api_max_retries': 1}, 'kanban': {'dispatch_in_gateway': False}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli import kanban_db_dispatch as dispatch
    dispatch._resolve_hermes_argv = lambda: [sys.executable, '-m', 'hermes_cli.main']
    os.environ['HERMES_KANBAN_CRASH_GRACE_SECONDS'] = '0'
    kb.create_board('reviewprobe')
    with closing(connect(board='reviewprobe')) as conn:
        tid = kb.create_task(conn, title='REVIEW_LANE_SENTINEL', body='Acceptance: one review run finishes',
                             assignee='default', workspace_kind='dir', workspace_path=str(workspace))
        assert kb.request_review(conn, tid, summary='Review probe handoff', reviewer='default'), 'review handoff refused'
        assert kb.get_task(conn, tid).status == 'review'
    env = dict(os.environ)
    clients, receipt = [], {}
    original_popen = subprocess.Popen

    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if 'hermes_cli.kanban_worker_client' in args[0]:
            clients.append(child)
        return child

    def wait_for(predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(.05)
        return False

    try:
        with daemon(root, home, env, barrier=False) as (_proc, _desc):
            subprocess.Popen = spawn
            try:
                with closing(connect(board='reviewprobe')) as conn:
                    launched = dispatch.dispatch_once(conn, board='reviewprobe', max_spawn=1)
                    receipt['spawned'] = [s[0] for s in launched.spawned]
                    receipt['run_id'] = kb.get_task(conn, tid).current_run_id
                receipt['ran'] = wait_for(lambda: bool(clients) and clients[0].poll() is not None)
                dispatch.reap_worker_zombies()
                with closing(connect(board='reviewprobe')) as conn:
                    dispatch.dispatch_once(conn, board='reviewprobe', max_spawn=0)
                    task = kb.get_task(conn, tid)
                    run = dict(conn.execute('SELECT * FROM task_runs WHERE id=?',
                                            (receipt['run_id'],)).fetchone())
                    receipt.update(task_status=task.status, failures=task.consecutive_failures,
                                   run_outcome=run['outcome'], run_error=(run['error'] or ''))
                if clients:
                    receipt['client_exit'] = dispatch._classify_worker_exit(clients[0].pid)
                    dispatch._recent_worker_exits.clear()
                receipt['review_requested'] = bool(peer.requests)
            finally:
                subprocess.Popen = original_popen
    finally:
        for child in clients:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        peer.shutdown()
        peer.server_close()
        (home / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
