"""Upstream scheduled retirement preserves jobs; native session Stop needs no board resume."""
import json
import os
from pathlib import Path
import sys

SOURCE = Path(sys.argv[2])

def child(stop_later):
    from contextlib import closing
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import shlex
    import threading
    import time
    from tests.waiting import wait_until
    sys.path.insert(0, str(SOURCE / 'tests/gateway/fixtures'))
    from local_recovery_probe import daemon
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
        f'gate=Path({str(gate)!r}); deadline=time.monotonic()+60\n'
        'while not gate.exists() and time.monotonic()<deadline: threading.Event().wait(.02)\n'
        f'if gate.exists(): Path({str(effect)!r}).write_text("owned background effect")')
    helper = effects / 'delayed_fixture_effect.py'
    helper.write_text(code)
    command = shlex.quote(sys.executable) + ' ' + shlex.quote(str(helper))
    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.server.requests.append(body)
            tools = {t['function']['name'] for t in body.get('tools', [])}
            if not tools:
                message, finish = {'role': 'assistant', 'content': 'fixture'}, 'stop'
            else:
                name, args = None, None
                if self.server.successor:
                    if not self.server.successor_done:
                        self.server.successor_done = True
                        name, args = 'kanban_complete', {'summary': 'isolated successor remainder complete'}
                elif not self.server.started:
                    self.server.started = True
                    name, args = 'terminal', dict(command=command, background=True, persist_on_release=True)
                elif not self.server.scheduled:
                    self.server.scheduled = True
                    name, args = 'kanban_schedule', {'reason': 'planned stage waits for authorized next event'}
                message, finish = {'role': 'assistant', 'content': 'Planned stage complete; waiting.'}, 'stop'
                if name:
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
    peer.scheduled, peer.successor_done = False, False
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'loop-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['terminal', 'file', 'kanban']},
        'terminal': {'backend': 'local'}, 'tools': {'tool_search': False}, 'agent': {'api_max_retries': 1, 'max_turns': 8},
        'kanban': {'dispatch_in_gateway': False}}))
    receipt = {'persist_on_release': True, 'explicit_later_stop': stop_later}
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
                wait_for(pidfile.exists, 'background did not start')
                bgpid = int(pidfile.read_text())
                wait_for(lambda: kb.get_task(conn, tid).status == 'scheduled', 'native worker did not schedule task')
                events = kb.list_events(conn, tid)
                runid = [e for e in events if e.kind == 'owner_admitted'][-1].run_id
                submitter = [e for e in events if e.kind == 'spawned'][-1].payload['pid']
                bound = [e for e in events if e.kind == 'worker_bound'][-1].payload['pid']
                wait_for(lambda: not kb._pid_alive(submitter) and not kb._pid_alive(bound), 'planned workers never exited')
                wait_for(lambda: any(e.kind == 'worker_result' for e in kb.list_events(conn, tid)), 'planned run missing ordinary worker_result')
                receipt.update(task_id=tid, old_run_id=runid, submitter_pid=submitter, bound_worker_pid=bound,
                    background_pid=bgpid, background_alive_after_planned_worker_exit=kb._pid_alive(bgpid),
                    old_run_status_after_retirement=conn.execute('SELECT status FROM task_runs WHERE id=?',(runid,)).fetchone()[0])
                checkpoints = list(home.rglob('processes.json'))
                receipt['checkpoint_after_planned_worker_exit'] = {str(p.relative_to(home)): json.loads(p.read_text()) for p in checkpoints}
                assert receipt['background_alive_after_planned_worker_exit'], 'ordinary planned retirement killed persisted job'
                assert not effect.exists(), 'gate not opened yet'
                if stop_later:
                    # Upstream deliberately refuses scheduled -> blocked. Keep that contract;
                    # its existing session Stop still cleans physical work without resuming it.
                    assert not kb.block_task(conn, tid, reason='scheduled block remains unsupported', kind='needs_input')
                    owner = [event for event in kb.list_events(conn, tid) if event.kind == 'owner_admitted'][-1]
                    async def stop_owner():
                        from hermes_cli.gateway_client import connect_gateway
                        from hermes_cli.gateway_runtime import discover_gateway_endpoint
                        endpoint = discover_gateway_endpoint(home)
                        assert endpoint.state == 'ready'
                        async with connect_gateway(endpoint=endpoint.endpoint) as client:
                            snapshot = await client.rpc('session.resume', session_id=owner.payload['session_id'])
                            await client.rpc('session.interrupt', session_id=owner.payload['session_id'],
                                             execution_generation=snapshot['execution_generation'])
                    import asyncio
                    asyncio.run(stop_owner())
                    receipt['stop_surface'] = 'session.interrupt'
                    assert kb.get_task(conn, tid).status == 'scheduled'
                    wait_for(lambda: not kb._pid_alive(bgpid), 'explicit idle Stop failed before any resume')
                    receipt['background_alive_after_explicit_stop'] = kb._pid_alive(bgpid)
                    gate.touch()
                    receipt['no_resume_stop_proof'] = not effect.exists() and len([e for e in kb.list_events(conn, tid) if e.kind == 'owner_admitted']) == 1
                    assert receipt['no_resume_stop_proof']
                    peer.successor = True
                    assert kb.unblock_task(conn, tid)
                    successor = dispatch.dispatch_once(conn, board='audit', max_spawn=1)
                    assert [x[0] for x in successor.spawned] == [tid], successor
                    wait_for(lambda: len([e for e in kb.list_events(conn, tid) if e.kind == 'owner_admitted']) == 2,
                        'successor never admitted')
                    receipt['background_alive_at_successor_admission'] = kb._pid_alive(bgpid)
                    receipt['old_run_status_after_explicit_stop'] = conn.execute('SELECT status FROM task_runs WHERE id=?',(runid,)).fetchone()[0]
                gate.touch()
                if not stop_later:
                    wait_for(effect.exists, 'background effect did not occur after gate', timeout=10)
                receipt['effect_after_gate'] = effect.exists()
                wait_for(lambda: not kb._pid_alive(bgpid), 'finite background did not exit')
                if stop_later:
                    wait_for(lambda: kb.get_task(conn, tid).status == 'done', 'successor did not finish')
                receipt['task_status'] = kb.get_task(conn, tid).status
                receipt['events'] = [{'id': e.id, 'kind': e.kind, 'run_id': e.run_id, 'payload': e.payload}
                    for e in kb.list_events(conn, tid)]
    finally:
        peer.shutdown(); peer.server_close()
        (root / 'receipt.json').write_text(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    child(bool(int(sys.argv[1])))
