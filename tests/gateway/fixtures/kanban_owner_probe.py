"""Actual dispatcher argv, ordinary daemon, loopback tool round, disposable board."""
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

from local_recovery_probe import daemon, websocket, rpc


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
        if messages and not self.server.blocked.is_set():
            self.server.blocked.set()
            assert self.server.release.wait(65)
        if messages and self.server.mode in {'rate_limit', 'billing'}:
            payload = json.dumps({'error': {'message': 'Rate limit exceeded' if self.server.mode == 'rate_limit' else 'Insufficient quota',
                'type': 'rate_limit_error' if self.server.mode == 'rate_limit' else 'insufficient_quota',
                'code': 'rate_limit_exceeded' if self.server.mode == 'rate_limit' else 'insufficient_quota'}}).encode()
            self.send_response(429)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        has_result = any(m['role'] == 'tool' for m in messages)
        message = {'role': 'assistant', 'content': 'OWNED_COMPLETE'}
        finish = 'stop'
        if messages and not body.get('tools'):
            message['content'] = json.dumps({'verdict': 'done', 'reason': 'Owned acceptance satisfied'})
        elif messages and (not has_result or messages[-1]['role'] == 'user'):
            continuation = has_result
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': 'owned-complete',
                'type': 'function', 'function': {'name': 'kanban_complete' if continuation else 'kanban_comment', 'arguments': json.dumps({'summary': 'Owned probe complete'} if continuation else {'body': 'Owned first turn'})}}]}
            finish = 'tool_calls'
        payload = json.dumps({'id': 'owned', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'owned', 'choices': [{'index': 0, 'delta': message, 'finish_reason': finish}]}) + '\n\ndata: [DONE]\n\n').encode()
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
    peer.mode = os.environ.get('KANBAN_PROBE_MODE', 'complete')
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    os.environ.update(OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    workspace = home / 'workspace'
    workspace.mkdir(exist_ok=True)
    skill = home / 'skills' / 'owned-skill'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text('---\nname: owned-skill\ndescription: Owned fixture\n---\nKANBAN_SKILL_SENTINEL\n', encoding='utf-8')
    hook = home / 'hook.py'
    hook.write_text('from pathlib import Path\nPath(' + repr(str(home / 'hook-effect')) + ').write_text("accepted")\n', encoding='utf-8')
    cfg = {'gateway': {'multiplex_profiles': False}, 'model': {'provider': 'custom', 'default': 'loop-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': ['terminal', 'file']},
        'terminal': {'backend': 'local'}, 'agent': {'api_max_retries': 1}, 'kanban': {'dispatch_in_gateway': False},
        'hooks': {'post_tool_call': [{'command': f'{sys.executable} {hook}'}]}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli import kanban_db_dispatch as dispatch
    # Pin the installed launcher to this checkout for base/fixed argv comparisons.
    dispatch._resolve_hermes_argv = lambda: [sys.executable, '-m', 'hermes_cli.main']
    os.environ['HERMES_KANBAN_CRASH_GRACE_SECONDS'] = '0'
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='KANBAN_TASK_SENTINEL', body='Acceptance: finish owned card', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'],
            goal_mode=peer.mode == 'complete', goal_max_turns=2,
            max_runtime_seconds=3600 if peer.mode == 'timeout' else None)
    env = dict(os.environ)
    receipt, clients = {}, []
    original_popen = subprocess.Popen
    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        if 'hermes_cli.kanban_worker_client' in args[0]:
            clients.append(child)
            if peer.mode == 'timeout':
                assert peer.blocked.wait(55), 'managed worker did not reach loopback'
                with closing(connect(board='owned')) as snapshot:
                    receipt['managed_pid'] = kb.get_task(snapshot, tid).worker_pid
        return child
    def wait_for(predicate):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        raise AssertionError(('deadline', receipt))
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            subprocess.Popen = spawn
            try:
                with closing(connect(board='owned')) as conn:
                    launched = dispatch.dispatch_once(conn, board='owned', max_spawn=1)
                    assert [s[0] for s in launched.spawned] == [tid], launched
                    task = kb.get_task(conn, tid)
                    params = dict(task_id=tid, board='owned', run_id=task.current_run_id, claim_lock=task.claim_lock)
                    assert peer.blocked.wait(55), 'model never reached'
                    active = kb.get_task(conn, tid)
                    receipt.update(client_pid=clients[0].pid, stored_pid=active.worker_pid, run_id=task.current_run_id)
                    if peer.mode == 'timeout':
                        receipt['pid_preserved'] = active.worker_pid == receipt['managed_pid'] != clients[0].pid
                        # Deterministically elapse this owned attempt; keep real timeout/kill logic.
                        conn.execute('UPDATE task_runs SET started_at=? WHERE id=?', (int(time.time()) - 3601, task.current_run_id))
                        conn.commit()
                        swept = dispatch.dispatch_once(conn, board='owned', max_spawn=0)
                        receipt.update(timed_out=swept.timed_out, managed_dead=not kb._pid_alive(receipt['managed_pid']))
                        assert receipt['pid_preserved'] and receipt['managed_dead'], receipt
                    elif peer.mode == 'crash':
                        import signal
                        os.kill(active.worker_pid, signal.SIGKILL)
                    peer.release.set()
                    if peer.mode != 'timeout':
                        wait_for(lambda: not kb._pid_alive(active.worker_pid))
                    wait_for(lambda: not kb._pid_alive(clients[0].pid))
                    dispatch.reap_worker_zombies()
                    receipt['client_exit'] = dispatch._classify_worker_exit(clients[0].pid)
                    dispatch._recent_worker_exits.clear()
                    swept = dispatch.dispatch_once(conn, board='owned', max_spawn=0)
                    task = kb.get_task(conn, tid)
                    run = dict(conn.execute('SELECT * FROM task_runs WHERE id=?', (params['run_id'],)).fetchone())
                    from gateway.session_kanban import worker_exit_code
                    assert worker_exit_code(kb.kanban_db_path(board='owned'), params | {'claim_lock': 'foreign'}) == 1
                    receipt.update(task_status=task.status, failures=task.consecutive_failures, run_outcome=run['outcome'],
                        run_metadata=kb._json_dict(run['metadata']), rate_limited=swept.rate_limited,
                        cooldown=dispatch.check_respawn_guard(conn, tid))
                    if peer.mode in {'rate_limit', 'billing'}:
                        assert task.status == 'ready' and task.consecutive_failures == 0 and run['outcome'] == 'rate_limited', receipt
                        assert receipt['client_exit'] == ('rate_limited', kb.KANBAN_RATE_LIMIT_EXIT_CODE), receipt
                        assert receipt['cooldown'] == 'rate_limit_cooldown', receipt
                    elif peer.mode == 'crash':
                        assert task.status == 'ready' and task.consecutive_failures == 1 and run['outcome'] == 'crashed', receipt
                    elif peer.mode == 'timeout':
                        assert task.status == 'ready' and task.consecutive_failures == 1 and run['outcome'] == 'timed_out', receipt
            finally:
                subprocess.Popen = original_popen
            if peer.mode != 'complete':
                return
            assert receipt['client_exit'] == ('clean_exit', 0), receipt
            async def retry():
                async with websocket(home, desc) as ws:
                    first = await rpc(ws, 'kanban.run', **params)
                    second = await rpc(ws, 'kanban.run', **params)
                    assert 'result' in first and 'result' in second, (first, second)
                    return first['result']['session_id'], first['result']['session_id'] == second['result']['session_id']
            sid, same = asyncio.run(retry())
            import sqlite3
            with closing(sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
                receipt['source'] = db.execute('SELECT source FROM sessions WHERE id=?', (sid,)).fetchone()[0]
                receipt['admissions'] = db.execute('SELECT count(*) FROM session_admissions').fetchone()[0]
            with closing(connect(board='owned')) as conn:
                receipt['task_status'] = kb.get_task(conn, tid).status
            wire = json.dumps(peer.requests)
            turns = [r for r in peer.requests if r.get('tools')]
            prefixes = [json.dumps([m for m in r['messages'] if m['role'] in {'system', 'developer'}]) for r in turns]
            receipt.update(goal_continuation=len(turns) >= 4, stable_prefix=len(set(prefixes)) == 1,
                retry_same_session=same, tools='kanban_complete' in wire,
                task_context='KANBAN_TASK_SENTINEL' in wire, skill_context='KANBAN_SKILL_SENTINEL' in wire,
                hook_effect=(home / 'hook-effect').exists())
    finally:
        peer.release.set()
        for child in clients:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        peer.shutdown()
        peer.server_close()
        (home / 'model-requests.json').write_text(json.dumps(peer.requests, indent=2), encoding='utf-8')
        (home / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
