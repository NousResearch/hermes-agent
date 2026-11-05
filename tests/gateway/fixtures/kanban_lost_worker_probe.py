"""Owned probe: a managed Kanban worker that dies at start, on a real daemon and a real board.

Reproduces the observed failure (t_925a816f) end to end: the card requires a skill nobody
installed, so the child raises ``ValueError: kanban_skills_unavailable`` while running its first
turn (``gateway/session_kanban.py``), sends ``managed_worker_failed`` and exits 1. Before raising,
the guard parks the card ``blocked``/``capability`` with the missing identifier in the reason, so
the first attempt is also the last one (t_5c02fad2 retried the same unprovisioned profile four
times). The dispatcher then books the attempt.

Everything here is real: a real ``gateway.run`` daemon in a disposable home, the real dispatcher
spawning the real worker client, the real managed child process, the real board database and the
real profile logs. The receipt says what the operator can see afterwards:

  cause        the ``worker_lost`` event's structured cause (step/exception/source/exit code)
  run_error    the booked run error, which must quote that cause
  errors_log   the profile's errors.log line, which must name it too
  block        the ``blocked`` event (kind + reason naming the missing skill), if the guard parked it
  claimable    whether a fresh claim succeeds — False once the card is parked for a human
"""
import asyncio
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from local_recovery_probe import daemon


class Model(BaseHTTPRequestHandler):
    """Answers any model call: this probe never reaches one, but agent construction wants a route."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        payload = json.dumps({'id': 'lost', 'choices': [{'index': 0, 'message': {'role': 'assistant',
            'content': 'UNREACHED'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
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
    kb.create_board('lost')
    # The card names a skill that is not installed: the child dies in its first turn, exactly as
    # the observed orca run did (`ValueError: kanban_skills_unavailable`).
    with closing(connect(board='lost')) as conn:
        tid = kb.create_task(conn, title='LOST_WORKER_SENTINEL', body='Acceptance: fail at start',
                             assignee='default', workspace_kind='dir', workspace_path=str(workspace),
                             skills=['absent-skill-t925'])
    env = dict(os.environ)
    clients = []
    original_popen = subprocess.Popen
    receipt = {}

    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if 'hermes_cli.kanban_worker_client' in args[0]:
            clients.append(child)
        return child

    def wait_for(predicate, timeout=90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(.05)
        return False

    def board_state(run_id=None):
        with closing(connect(board='lost')) as conn:
            task = kb.get_task(conn, tid)
            run = dict(conn.execute('SELECT * FROM task_runs WHERE id=?', (run_id or task.current_run_id,)).fetchone()) \
                if (run_id or task.current_run_id) else None
            lost = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_lost' "
                                "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
            blocked = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='blocked' "
                                   "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
            return task, run, lost, blocked

    try:
        with daemon(root, home, env, barrier=False) as (_proc, _desc):
            subprocess.Popen = spawn
            try:
                with closing(connect(board='lost')) as conn:
                    launched = dispatch.dispatch_once(conn, board='lost', max_spawn=1)
                    receipt['spawned'] = [s[0] for s in launched.spawned]
                    receipt['run_id'] = kb.get_task(conn, tid).current_run_id
                receipt['ran'] = wait_for(lambda: bool(clients) and clients[0].poll() is not None)
                dispatch.reap_worker_zombies()
                with closing(connect(board='lost')) as conn:
                    # The crash sweep books the dead attempt; one tick is what a dispatcher owes it.
                    dispatch.dispatch_once(conn, board='lost', max_spawn=0)
                task, run, lost, blocked = board_state(receipt.get('run_id'))
                receipt.update(task_status=task.status, claim_lock=task.claim_lock,
                               failures=task.consecutive_failures, run_outcome=run['outcome'] if run else None,
                               run_error=(run['error'] if run else '') or '',
                               last_failure_error=task.last_failure_error or '',
                               cause=(json.loads(lost[0])['cause'] if lost else None),
                               block=(json.loads(blocked[0]) if blocked else None))
                log = home / 'logs' / 'errors.log'
                if log.exists():
                    lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
                    receipt['errors_log'] = next((line for line in reversed(lines)
                                                  if 'Managed worker lost' in line), '')
                else:
                    receipt['errors_log'] = ''
                # Usable again, with no operator ``hermes sessions discard``: a fresh claim succeeds.
                with closing(connect(board='lost')) as conn:
                    again = kb.claim_task(conn, tid, claimer='probe:again')
                    receipt['claimable'] = again is not None and again.status == 'running'
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
