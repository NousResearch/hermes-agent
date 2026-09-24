"""Ordinary daemon + local authenticated clients + actual terminal completion."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading

from automation_peer import Model
from local_recovery_probe import daemon, websocket, rpc


def probe(base, source):
    restart = source == 'restart'
    watch = source == 'watch'
    source = 'gui' if restart or watch else source
    root = Path(__file__).resolve().parents[3]
    home, user = base / 'state', base / 'user'
    home.mkdir(); user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.watch = watch
    model.blocked, model.release = threading.Event(), threading.Event()
    model.gate = home / 'terminal-gate'
    threading.Thread(target=model.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{model.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'local-fixture', 'base_url': url},
        'terminal': {'cwd': str(home)}, 'approvals': {'mode': 'off'},
        'auxiliary': {'title_generation': {'enabled': False}}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
        OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def rows():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute('SELECT * FROM session_admissions ORDER BY seq')]

    async def wait(predicate):
        async with asyncio.timeout(30):
            while not predicate():
                await asyncio.sleep(.05)

    async def run(desc):
        async with websocket(home, desc) as ws:
            result = await rpc(ws, 'session.create', request_id='automation', source=source,
                cwd=str(home), toolsets=['terminal'])
            assert 'result' in result, result
            sid = result['result']['session_id']
            async def submit(identity, text):
                reply = await rpc(ws, 'prompt.submit', session_id=sid, input_id=identity, text=text)
                assert 'result' in reply, reply
            await submit('spawn', 'SPAWN_AUTOMATION')
            await wait(lambda: rows()[0]['status'] == 'terminal')
            assert 'REAL_TERMINAL_COMPLETION' in json.dumps(model.requests)
            if restart:
                model.gate.touch()
                await wait(lambda: any(r['principal_id'].startswith('automation:') for r in rows()))
                assert rows()[-1]['status'] == 'queued', rows()
                return {'session_id': sid, 'committed': rows()}
            await submit('held', 'HOLD_AUTOMATION')
            assert await asyncio.to_thread(model.blocked.wait, 20)
            await submit('follower', 'HUMAN_FIFO_FOLLOWER')
        # Last observer closes while owner remains in a model call.
        model.gate.touch()
        await asyncio.sleep(8)
        ledger = rows()
        automatic = [r for r in ledger if r['principal_id'].startswith('automation:')]
        assert automatic, {'missing_automation': ledger}
        assert len(automatic) == 1 and automatic[0]['status'] == 'queued', automatic
        if watch:
            assert json.loads(automatic[0]['request_id'])[0] == 'watch_match', automatic
        assert next(r for r in ledger if r['request_id'] == 'held')['status'] == 'started'
        assert automatic[0]['seq'] > next(r for r in ledger if r['request_id'] == 'follower')['seq']
        model.release.set()
        await wait(lambda: all(r['status'] == 'terminal' for r in rows()))
        assert all(r['outcome'] == 'completed' for r in rows()), rows()
        texts = [next((m.get('content', '') for m in reversed(r['messages']) if m['role'] == 'user'), '')
                 for r in model.requests if r.get('messages')]
        assert 'REAL_TERMINAL_COMPLETION' in str(texts[-1]), texts
        assert sum('REAL_TERMINAL_COMPLETION' in str(text) for text in texts) == 1, texts
        with sqlite3.connect(home / 'state.db') as db:
            notices = db.execute("SELECT display_kind FROM messages WHERE role='user' AND content LIKE ?",
                                 ('%REAL_TERMINAL_COMPLETION%',)).fetchall()
        assert notices == [('internal_notification',)], notices
        prefixes = [json.dumps([m for m in r['messages'] if m['role'] in ('system', 'developer')], sort_keys=True)
                    for r in model.requests if r.get('messages')]
        assert len(set(prefixes)) == 1, 'automation changed the cached system prefix'
        assert 'HUMAN_FIFO_FOLLOWER' in str(texts[-2]), texts
        async with websocket(home, desc) as ws:
            resumed = await rpc(ws, 'session.resume', session_id=sid)
            assert 'result' in resumed, resumed
        return {'source': source, 'ledger': rows(), 'model_inputs': texts,
                'observer_closed_before_completion': True}

    try:
        with daemon(root, home, env, barrier=restart, fixture='local_automation_daemon.py') as (proc, desc):
            try:
                receipt = asyncio.run(run(desc))
                receipt['daemon_pid'] = proc.pid
                if not restart:
                    print(json.dumps(receipt))
            except BaseException:
                print((home / ('first.log' if restart else 'restart.log')).read_text()[-15000:], file=sys.stderr)
                raise
        if restart:
            pids = [receipt['daemon_pid']]
            epochs = [desc['authority_epoch']]
            async def recovered(desc):
                await wait(lambda: all(r['status'] == 'terminal' for r in rows()))
                # Leave ordinary startup/watchers running before checking zero replay.
                await asyncio.sleep(8)
                assert len(rows()) == len(receipt['committed']), rows()
                assert rows()[-1]['outcome'] == 'completed', rows()
                async with websocket(home, desc) as ws:
                    result = await rpc(ws, 'session.resume', session_id=receipt['session_id'])
                    assert 'result' in result, result
            for _ in range(2):
                with daemon(root, home, env, barrier=False) as (proc, desc):
                    pids.append(proc.pid); epochs.append(desc['authority_epoch'])
                    asyncio.run(recovered(desc))
            texts = [next((m.get('content', '') for m in reversed(r['messages']) if m['role'] == 'user'), '')
                     for r in model.requests if r.get('messages')]
            completion_inputs = [t for t in texts if 'REAL_TERMINAL_COMPLETION' in str(t)]
            assert len(completion_inputs) == 1, texts
            assert len(set(pids)) == 3 and epochs == sorted(set(epochs))
            print(json.dumps({'pids': pids, 'epochs': epochs, 'ledger': rows(),
                              'model_inputs': texts, 'lost_producer_ack': True}))
    finally:
        model.gate.touch(); model.release.set()
        model.shutdown(); model.server_close()


if __name__ == '__main__':
    probe(Path(sys.argv[1]), sys.argv[2])
