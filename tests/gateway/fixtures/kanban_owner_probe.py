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

from local_recovery_probe import daemon, websocket, rpc


class Model(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        if messages:
            self.server.requests.append(body)
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
        'terminal': {'backend': 'local'}, 'kanban': {'dispatch_in_gateway': False},
        'hooks': {'post_tool_call': [{'command': f'{sys.executable} {hook}'}]}}
    (home / 'config.yaml').write_text(json.dumps(cfg), encoding='utf-8')
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from hermes_cli import kanban_db_dispatch as dispatch
    # Pin the installed launcher to this checkout for base/fixed argv comparisons.
    dispatch._resolve_hermes_argv = lambda: [sys.executable, '-m', 'hermes_cli.main']
    _worker_argv = dispatch._worker_argv
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title='KANBAN_TASK_SENTINEL', body='Acceptance: finish owned card', assignee='default',
            workspace_kind='dir', workspace_path=str(workspace), skills=['owned-skill'], goal_mode=True, goal_max_turns=2)
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, tid)
    env = dict(os.environ)
    worker_env = env | {'HERMES_KANBAN_TASK': tid, 'HERMES_KANBAN_BOARD': 'owned',
        'HERMES_KANBAN_RUN_ID': str(task.current_run_id), 'HERMES_KANBAN_CLAIM_LOCK': task.claim_lock,
        'HERMES_KANBAN_DB': str(kb.kanban_db_path(board='owned'))}
    params = dict(task_id=tid, board='owned', run_id=task.current_run_id, claim_lock=task.claim_lock)
    receipt = {}
    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            command = _worker_argv(task, 'default', str(home))
            if os.environ.get('KANBAN_PROBE_TRACE'):
                command = ['strace', '-f', '-e', 'trace=openat', '-o', str(home / 'client.strace'), *command]
            result = subprocess.run(command, cwd=workspace, env=worker_env,
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=80)
            receipt.update(worker_rc=result.returncode, stdout=result.stdout, stderr=result.stderr)
            assert result.returncode == 0, receipt
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
        peer.shutdown()
        peer.server_close()
        (home / 'model-requests.json').write_text(json.dumps(peer.requests, indent=2), encoding='utf-8')
        (home / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
