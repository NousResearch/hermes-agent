"""Real worker/Git/Stop/Finish receipt for #144287; no provider traffic leaves loopback."""
import asyncio
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_managed_worker_construct import Primary, _reply, _serve
from tests.waiting import async_wait_until


# Observer only: run Git directly inside the production worker's turn, not in a terminal-tool
# subprocess. No stdin= and no _git_out(), so call-site DEVNULL cannot mask a broken worker claim.
WITNESS = '''import json, os, pathlib, subprocess, sys, threading
if "agent.managed_worker" in sys.orig_argv:
    output = pathlib.Path(os.environ["HERMES_HOME"]) / "git-witness.jsonl"
    def emit(kind, **fields):
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(kind=kind, pid=os.getpid(), **fields)) + "\\n")
    def observe(frame, event, arg):
        name = frame.f_code.co_name
        obj = frame.f_locals.get("self")
        if event == "return" and type(obj).__name__ == "WorkerControls":
            if name in {"stop", "read"}:
                emit("control_" + name, stopped=obj.stopped.is_set(), finish=obj.finish.is_set())
        if event != "call" or name != "run_worker_turns":
            return
        parent = frame.f_back
        while parent and parent.f_code.co_name != "execute":
            parent = parent.f_back
        if parent is None or "controls" not in parent.f_locals:
            return
        controls = parent.f_locals["controls"]
        from tests.waiting import wait_until
        def parked():
            current = sys._current_frames().get(controls.reader.ident)
            while current:
                if current.f_code.co_name == "read_frame":
                    return True
                current = current.f_back
            return False
        wait_until(parked, timeout=15, what="production control reader parked in read_frame")
        emit("git_start", parked=parked(), worker_file=parent.f_code.co_filename)
        log = output.with_name("git-child-" + str(os.getpid()) + ".log")
        with log.open("w", encoding="utf-8") as out:
            proc = subprocess.Popen(["git", "-C", os.environ["HERMES_HOME"], "rev-parse", "--show-toplevel"],
                                    stdout=out, stderr=subprocess.STDOUT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                import psutil
                process = psutil.Process(proc.pid)
                for child in process.children(recursive=True):
                    try: child.kill()
                    except psutil.NoSuchProcess: pass
                proc.kill()
                proc.wait(timeout=5)
                emit("git_done", outcome="timeout")
                return
        emit("git_done", outcome="returned", code=proc.returncode, output=log.read_text().strip(),
             parked=parked(), stopped=controls.stopped.is_set(), finish=controls.finish.is_set())
    sys.setprofile(observe)
    threading.setprofile(observe)
'''


class Model(Primary):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if any(m.get('content') == 'WAIT_FOR_STOP' for m in body.get('messages', [])):
            self.server.blocked.set()
            self.server.release.wait(60)
        _reply(self, body, 'WORKER_GIT_OK')


@pytest.mark.platforms('windows')
def test_real_managed_worker_inherited_git_and_controls(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user, site = (tmp_path / name for name in ('state', 'user', 'observer'))
    for directory in (home, user, site):
        directory.mkdir()
    import subprocess
    subprocess.run(['git', 'init', str(home)], capture_output=True, check=True)
    (site / 'sitecustomize.py').write_text(WITNESS)
    peer = _serve(Model)
    peer.blocked, peer.release = threading.Event(), threading.Event()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    config = {'gateway': {'multiplex_profiles': False, 'managed_workers': True},
              'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': url},
              'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(config))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               PYTHONPATH=os.pathsep.join([str(site), str(root)]),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

    def events():
        path = home / 'git-witness.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def query(sql, args=()):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql, args).fetchall()

    async def exercise(desc, owner):
        async with websocket(home, desc) as ws:
            for name, text in [('normal', 'RETURN_NORMALLY'), ('stop', 'WAIT_FOR_STOP')]:
                created = await rpc(ws, 'session.create', request_id=name, source='cli', cwd=str(home),
                                    model='primary-model', provider='custom', base_url=url,
                                    api_key='loopback-only', toolsets=[], ignore_rules=True)
                assert 'result' in created, created
                sid = created['result']['session_id']
                submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id=name, text=text)
                assert 'result' in submitted, submitted
                await async_wait_until(lambda: len([e for e in events() if e['kind'] == 'git_done'])
                                       >= (1 if name == 'normal' else 2), timeout=90, what='Git in real worker returned')
                result = [e for e in events() if e['kind'] == 'git_done'][-1]
                assert result['outcome'] == 'returned' and result['code'] == 0, result
                assert Path(result['output']).resolve() == home.resolve(), result
                assert result['parked'] and not result['finish'] and not result['stopped'], result
                assert result['pid'] != owner.pid
                if name == 'stop':
                    assert await asyncio.to_thread(peer.blocked.wait, 30), (home / 'restart.log').read_text()
                    [(generation,)] = query('SELECT generation FROM session_admissions WHERE request_id=?', (name,))
                    stopped = await rpc(ws, 'session.interrupt', session_id=sid, execution_generation=generation)
                    assert 'result' in stopped, stopped
                    await async_wait_until(lambda: any(e['kind'] == 'control_stop' and e['pid'] == result['pid']
                                                       and e['stopped'] for e in events()),
                                           timeout=20, what='real worker consumed Stop')
                    peer.release.set()
                await async_wait_until(lambda: query('SELECT status FROM session_admissions WHERE request_id=?',
                                                     (name,)) == [('terminal',)], timeout=90, what='worker settled')
                assert query('SELECT status FROM worker_executions WHERE session_id=?', (sid,)) == [('terminal',)]
                await async_wait_until(lambda: any(e['kind'] == 'control_read' and e['pid'] == result['pid']
                                                   and e['finish'] for e in events()),
                                       timeout=20, what='real worker consumed Finish')
                if name == 'normal':
                    [(admission,)] = query('SELECT admission_id FROM session_admissions WHERE request_id=?', (name,))
                    receipt = await rpc(ws, 'prompt.receipt', session_id=sid, admission_id=admission, include_result=True)
                    assert receipt['result']['result']['final_response'] == 'WORKER_GIT_OK', receipt
            print(json.dumps({'owner_pid': owner.pid, 'worker_events': events(), 'provider': 'loopback-only'}))

    try:
        with daemon(root, home, env, barrier=False) as (owner, desc):
            asyncio.run(exercise(desc, owner))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
