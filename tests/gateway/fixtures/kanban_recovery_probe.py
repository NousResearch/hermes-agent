"""Disposable ordinary owner, actual dispatcher, custom paths and process death."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time

from kanban_owner_probe import Model as BaseModel
from local_recovery_probe import daemon, websocket, rpc


class Model(BaseModel):
    def do_POST(self):
        try:
            super().do_POST()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    root = Path(__file__).resolve().parents[3]
    home = Path(os.environ['HERMES_HOME'])
    home.mkdir(parents=True)
    user = Path(os.environ['HOME'])
    user.mkdir(parents=True)
    mode = os.environ['KANBAN_RECOVERY_MODE']
    owner = home / 'profiles/assigned' if mode == 'cross_profile' else home
    owner.mkdir(parents=True, exist_ok=True)
    os.environ['HERMES_KANBAN_HOME'] = str(home / 'shared')
    board_path = home / 'custom.sqlite' if mode == 'custom' else home / 'shared/kanban/boards/owned/kanban.db'
    os.environ['HERMES_KANBAN_DB'] = str(board_path)
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests, peer.mode = [], 'complete'
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    workspace = home / 'workspace'
    workspace.mkdir()
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'frozen-model', 'base_url': url},
           'auxiliary': {'title_generation': {'enabled': False}},
           'platform_toolsets': {'cli': []}, 'kanban': {'dispatch_in_gateway': False}}
    (owner / 'config.yaml').write_text(json.dumps(cfg))
    env = dict(os.environ, HERMES_HOME=str(owner), OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    # Already-running owner deliberately has no dispatcher-specific board overrides.
    for key in list(env):
        if key.startswith('HERMES_KANBAN_'):
            env.pop(key)
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.kanban_db_connect import connect
    os.environ['HERMES_KANBAN_CRASH_GRACE_SECONDS'] = '0'
    clients, receipt = [], {'mode': mode}
    original_popen = subprocess.Popen
    def spawn(*args, **kwargs):
        if 'hermes_cli.kanban_worker_client' in args[0]:
            # Trace only the transport client's tree, not the independently started owner.
            args = (['strace', '-f', '-e', 'trace=openat', '-o', str(home / 'client.strace'), *args[0]], *args[1:])
            child = original_popen(*args, **kwargs)
            clients.append(child)
            return child
        return original_popen(*args, **kwargs)
    def wait_for(predicate):
        until = time.monotonic() + 50
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(.05)
        raise AssertionError(('deadline', receipt, [p.poll() for p in clients],
                              list((home / 'shared').rglob('*.log'))))
    def admission():
        with closing(sqlite3.connect((owner / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT * FROM session_admissions').fetchone()
            return dict(row) if row else None
    async def retry(desc, params, *, resolve=False):
        async with websocket(owner, desc) as ws:
            answer = await rpc(ws, 'kanban.run', **params)
            assert 'result' in answer, answer
            for changed in ({'board': 'foreign'}, {'claim_lock': 'forged'}, {'db': str(home / 'other.sqlite')}, {'run_id': params['run_id'] + 1}):
                bad = await rpc(ws, 'kanban.run', **(params | changed))
                assert 'error' in bad, bad
            if resolve:
                value = answer['result']['receipt']
                rejected = await rpc(ws, 'prompt.resolve_unknown', session_id=answer['result']['session_id'],
                                     admission_id=value['admission_id'], execution_generation=value['execution_generation'] + 1)
                assert 'error' in rejected, rejected
                result = await rpc(ws, 'prompt.resolve_unknown', session_id=answer['result']['session_id'],
                                   admission_id=value['admission_id'], execution_generation=value['execution_generation'])
                assert 'result' in result, result
            return answer['result']
    # A dispatcher iterating a slug under a HERMES_KANBAN_DB pin resolves through the pin (the
    # gateway's embedded dispatcher does the same); an unfenced explicit board= would bypass it.
    pinned = kb.pin_first_board_resolution()
    pinned.__enter__()
    try:
        # Board identity is board.json (#43243): create it explicitly; under the pin its DB is board_path.
        kb.create_board('owned')
        with closing(connect(board='owned')) as conn:
            tid = kb.create_task(conn, title='RECOVERY_TASK', assignee='assigned' if mode == 'cross_profile' else 'default',
                                 workspace_kind='dir', workspace_path=str(workspace), goal_mode=True, goal_max_turns=2)
            with daemon(root, owner, env, barrier=False) as (proc, desc):
                subprocess.Popen = spawn
                try:
                    launched = dispatch.dispatch_once(conn, board='owned', max_spawn=1)
                finally:
                    subprocess.Popen = original_popen
                assert [s[0] for s in launched.spawned] == [tid], launched
                task = kb.get_task(conn, tid)
                params = dict(board='owned', task_id=tid, run_id=task.current_run_id, claim_lock=task.claim_lock,
                              db=str(board_path.resolve()))
                wait_for(lambda: peer.blocked.is_set() or any(p.poll() is not None for p in clients))
                assert peer.blocked.is_set(), 'Owner failed to discover dispatcher board'
                first = admission()
                receipt.update(session_id=first['target_session_id'], run_id=task.current_run_id, claim_lock=task.claim_lock)
                if mode == 'restart':
                    proc.kill(); proc.wait(timeout=10)
                    # Model effects are ambiguous: terminate the exact owned executing interpreter.
                    bound = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='worker_bound' "
                                         "ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
                    os.kill(json.loads(bound[0])['pid'], signal.SIGKILL)
                    dispatch._recent_worker_exits.clear()
                    swept = dispatch.dispatch_once(conn, board='owned', max_spawn=1)
                    assert not swept.spawned and kb.get_task(conn, tid).current_run_id == task.current_run_id
                else:
                    peer.release.set()
                    wait_for(lambda: admission()['status'] == 'terminal')
                    wait_for(lambda: kb.get_task(conn, tid).status == 'done')
                    assert asyncio.run(retry(desc, params))['session_id'] == first['target_session_id']
            if mode == 'restart':
                cfg['model']['default'] = 'changed-default'
                (owner / 'config.yaml').write_text(json.dumps(cfg))
                with daemon(root, owner, env, barrier=False) as (proc, desc):
                    answer = asyncio.run(retry(desc, params))
                    assert answer['receipt']['status'] == 'unknown', answer
                    assert answer['session_id'] == first['target_session_id']
                    with closing(sqlite3.connect((owner / 'state.db').as_uri() + '?mode=ro', uri=True)) as db:
                        frozen = json.loads(db.execute('SELECT value FROM state_meta WHERE key=?',
                            ('gateway.local_policy.v1:' + first['target_session_id'],)).fetchone()[0])
                        assert frozen['policy']['model'] == 'frozen-model', frozen
                        private = json.loads(frozen['policy']['kanban_json'])
                        assert private['db'] == str(board_path.resolve()) and private['claim_lock'] == task.claim_lock
                    # The attempt's interpreter and its spawned worker are both gone: an ordinary crash,
                    # booked once and re-queued once (same as a killed one-shot worker), never duplicated.
                    wait_for(lambda: all(p.poll() is not None for p in clients))
                    assert len(peer.requests) == 1, peer.requests  # the restart never replays the lost call
                    conn.execute('UPDATE tasks SET claim_expires=1,last_heartbeat_at=1,max_runtime_seconds=1 WHERE id=?', (tid,))
                    conn.execute('UPDATE task_runs SET started_at=1 WHERE id=?', (task.current_run_id,))
                    conn.commit()
                    dispatch._recent_worker_exits.clear()
                    dispatch.dispatch_once(conn, board='owned', max_spawn=0, stale_timeout_seconds=1)
                    same = kb.get_task(conn, tid)
                    run = conn.execute('SELECT outcome FROM task_runs WHERE id=?', (task.current_run_id,)).fetchone()
                    # 'running' when the owner's own dispatcher already re-queued it onto a fresh run.
                    assert same.status in {'ready', 'running'} and (same.consecutive_failures, run[0]) == (1, 'crashed'), (same, run[0])
                    assert same.current_run_id != task.current_run_id or same.status == 'ready', same
                    dispatch.dispatch_once(conn, board='owned', max_spawn=0, stale_timeout_seconds=1)
                    assert kb.get_task(conn, tid).consecutive_failures == 1
                receipt['crash_requeued_once'] = True
            receipt.update(inference_calls=len(peer.requests), frozen_model=peer.requests[0]['model'])
            assert peer.requests[0]['model'] == 'frozen-model'
            wait_for(lambda: all(p.poll() is not None for p in clients))
            trace = (home / 'client.strace').read_text()
            writes = [line for line in trace.splitlines() if 'state.db' in line and ('O_RDWR' in line or 'O_WRONLY' in line)]
            assert not writes, writes
            receipt['client_canonical_writable_opens'] = len(writes)
    finally:
        pinned.__exit__(None, None, None)
        peer.release.set()
        for child in clients:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        peer.shutdown(); peer.server_close()
        (home / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        (home / 'model-requests.json').write_text(json.dumps(peer.requests, indent=2))
        print(json.dumps(receipt))


if __name__ == '__main__':
    main()
