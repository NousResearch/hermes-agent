"""Owned probe: a live gateway names the managed turn it is running.

Observed defect (t_925a816f): ``hermes gateway restart`` refused with

    • managed run pid 3612
    • managed run pid 3612

— the GATEWAY's pid, once per managed turn, with no session or admission identity. An operator
could not tell two real worker turns from a leak, so ``--yes`` was made blind.

This probe runs the real stack (daemon, dispatcher, worker client, managed child) with a model
that holds the turn open, then asks the live control socket for ``active-work`` and records the
managed unit it answers with, plus the same unit as the restart guard would print it.
"""
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
from tests.gateway.test_normal_runtime_boot import control


class BlockingModel(BaseHTTPRequestHandler):
    """Holds the first model call open until the probe releases it."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        self.server.blocked.set()
        self.server.release.wait(120)
        payload = json.dumps({'id': 'held', 'choices': [{'index': 0, 'message': {'role': 'assistant',
            'content': 'HELD'}, 'finish_reason': 'stop'}],
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
    peer = ThreadingHTTPServer(('127.0.0.1', 0), BlockingModel)
    peer.blocked, peer.release = threading.Event(), threading.Event()
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
    kb.create_board('live')
    with closing(connect(board='live')) as conn:
        tid = kb.create_task(conn, title='ACTIVE_WORK_SENTINEL', body='Acceptance: hold a turn',
                             assignee='default', workspace_kind='dir', workspace_path=str(workspace))
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

    try:
        with daemon(root, home, env, barrier=False) as (_proc, _desc):
            subprocess.Popen = spawn
            try:
                with closing(connect(board='live')) as conn:
                    dispatch.dispatch_once(conn, board='live', max_spawn=1)
                assert wait_for(peer.blocked.is_set), 'the managed turn never reached the model'
                bound = {}
                def bound_pid():
                    with closing(connect(board='live')) as conn:
                        row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound' "
                                           "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
                        if row:
                            bound.update(json.loads(row[0]))
                        return bool(bound.get('pid'))
                assert wait_for(bound_pid, timeout=60), 'the managed worker never bound'
                # What the restart guard reads: the live control socket, not the state file.
                answer = control(home, 'active-work')
                from hermes_cli.update_cmd_drain_report import describe_active_work_unit
                units = [u for u in answer.get('active_work') or [] if isinstance(u, dict)]
                managed = [u for u in units if u.get('kind') == 'managed']
                receipt.update(gateway_pid=answer.get('pid'), active_agents=answer.get('active_agents'),
                               managed_units=managed, bound_pid=bound.get('pid'),
                               rendered=[describe_active_work_unit(u, home) for u in managed],
                               task_id=tid)
            finally:
                subprocess.Popen = original_popen
                peer.release.set()
    finally:
        peer.release.set()
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
