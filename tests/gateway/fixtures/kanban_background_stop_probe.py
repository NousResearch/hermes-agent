"""Real native Kanban Stop, background process and successor admission probe."""
import json
import os
from pathlib import Path
import sys

SOURCE = Path(sys.argv[2])

def child(persist):
    from contextlib import closing
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import shlex
    import threading
    import time
    from tests.waiting import wait_until
    sys.path.insert(0, str(SOURCE / 'tests/gateway/fixtures'))
    from local_recovery_probe import daemon, control
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect
    root = Path(os.environ['HERMES_HOME']).parent
    home = root / 'state'
    home.mkdir(mode=0o700)
    (root / 'user').mkdir()
    effects = root / 'project'
    effects.mkdir()
    effect, pidfile = effects / 'old-effect.txt', effects / 'background-pid.txt'
    gate = effects / 'allow-effect'
    code = ('import os,time,threading; from pathlib import Path; '
        f'Path({str(pidfile)!r}).write_text(str(os.getpid())); '
        f'gate=Path({str(gate)!r}); deadline=time.monotonic()+60; event=threading.Event()\n'
        'while not gate.exists() and time.monotonic()<deadline: event.wait(.02)\n'
        f'if gate.exists(): Path({str(effect)!r}).write_text("old consequential effect after Stop")')
    helper = effects / 'delayed_fixture_effect.py'
    helper.write_text(code)
    command = shlex.quote(sys.executable) + ' ' + shlex.quote(str(helper))
    barrier, release = threading.Event(), threading.Event()
    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.server.requests.append(body)
            tools = {t['function']['name'] for t in body.get('tools', [])}
            if not tools:
                message, finish = {'role': 'assistant', 'content': 'fixture'}, 'stop'
            else:
                if self.server.successor:
                    name, args = 'kanban_complete', {'summary': 'isolated successor remainder complete'}
                elif not self.server.started:
                    self.server.started = True
                    name, args = 'terminal', dict(command=command, background=True, persist_on_release=persist)
                else:
                    barrier.set()
                    release.wait(65)
                    name, args = 'kanban_complete', {'summary': 'old response released after Stop'}
                assert name in tools, (name, tools)
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0,
                    'id': 'fixture-' + str(len(self.server.requests)), 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(args)}}]}
                finish = 'tool_calls'
            envelope = {'id': 'fixture', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
            payload, kind = json.dumps(envelope).encode(), 'application/json'
            if body.get('stream'):
                envelope['choices'][0]['delta'] = envelope['choices'][0].pop('message')
                payload = ('data: ' + json.dumps(envelope) + '\n\ndata: [DONE]\n\n').encode()
                kind = 'text/event-stream'
            try:
                self.send_response(200); self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError): pass
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests, peer.started, peer.successor = [], False, False
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'loop-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['terminal', 'file', 'kanban']},
        'terminal': {'backend': 'local'}, 'agent': {'api_max_retries': 1, 'max_turns': 8},
        'kanban': {'dispatch_in_gateway': False}}))
    receipt = {'persist_on_release': persist}
    def wait_for(predicate, description, timeout=40):
        return wait_until(predicate, timeout=timeout, what=description)
    bgpid = None
    try:
        kb.create_board('audit')
        with closing(connect(board='audit')) as conn:
            tid = kb.create_task(conn, title='Isolated registered background Stop audit', assignee='default',
                body='Only run harmless local fixture tools; await instructions.', workspace_kind='dir',
                workspace_path=str(effects), board='audit')
            assert kb.get_task(conn, tid).status == 'ready'
            with daemon(SOURCE, home, dict(os.environ), barrier=False):
                launched = dispatch.dispatch_once(conn, board='audit', max_spawn=1)
                assert [x[0] for x in launched.spawned] == [tid], launched
                assert barrier.wait(60), 'background tool did not reach next model boundary'
                wait_for(pidfile.exists, 'background did not start')
                bgpid = int(pidfile.read_text())
                task = kb.get_task(conn, tid)
                runid, submitter = task.current_run_id, task.worker_pid
                bound = [e for e in kb.list_events(conn, tid) if e.kind == 'worker_bound'][-1].payload['pid']
                checkpoints = list(home.rglob('processes.json'))
                receipt['checkpoint_before_stop'] = {str(p.relative_to(home)): json.loads(p.read_text()) for p in checkpoints}
                latest = conn.execute('SELECT MAX(id) FROM task_events WHERE task_id=?', (tid,)).fetchone()[0]
                stopped = time.monotonic()
                assert kb.block_task(conn, tid, reason='operator Stop audit', kind='needs_input', expected_event_id=latest)
                wait_for(lambda: not kb._pid_alive(submitter) and not kb._pid_alive(bound), 'native worker survived Stop')
                receipt.update(task_id=tid, old_run_id=runid, submitter_pid=submitter, bound_worker_pid=bound,
                    background_pid=bgpid, workers_gone_seconds=round(time.monotonic()-stopped, 3),
                    background_alive_after_workers_gone=kb._pid_alive(bgpid), old_effect_at_worker_exit=effect.exists())
                assert not effect.exists(), 'effect occurred too early for useful Stop evidence'
                # Explicit authorized resume admits a successor through the real dispatcher.
                peer.successor = True
                assert kb.unblock_task(conn, tid)
                successor = dispatch.dispatch_once(conn, board='audit', max_spawn=1)
                assert [x[0] for x in successor.spawned] == [tid], successor
                wait_for(lambda: len([e for e in kb.list_events(conn, tid) if e.kind == 'owner_admitted']) == 2,
                    'successor never admitted')
                receipt['successor_admitted_before_old_effect'] = not effect.exists()
                receipt['background_alive_at_successor_admission'] = kb._pid_alive(bgpid)
                gate.touch()
                wait_for(lambda: not kb._pid_alive(bgpid), 'old background survives resumed execution', timeout=20)
                receipt['old_effect_after_stop'] = effect.exists()
                wait_for(lambda: kb.get_task(conn, tid).status == 'done', 'successor did not finish')
                release.set()
                receipt['task_status'] = kb.get_task(conn, tid).status
                receipt['events'] = [{'id': e.id, 'kind': e.kind, 'run_id': e.run_id, 'payload': e.payload}
                    for e in kb.list_events(conn, tid)]
    finally:
        release.set()
        peer.shutdown(); peer.server_close()
        (root / 'receipt.json').write_text(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    child(bool(int(sys.argv[1])))
