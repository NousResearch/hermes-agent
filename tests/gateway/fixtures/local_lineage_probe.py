"""Separate ordinary daemon restarts after the production physical-target handoff."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

from tests.gateway.fixtures.local_recovery_probe import Model, daemon, rpc, websocket


def probe(tmp_path):
    root = Path(__file__).resolve().parents[3]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'original', 'base_url': url},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')
    sessions, params, targets, pids, epochs = {}, {}, {}, [], []
    route_targets = {}
    def db_rows():
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return dict(db.execute('SELECT request_id,status FROM session_admissions'))
    async def terminal(request_id):
        async with asyncio.timeout(30):
            while db_rows().get(request_id) != 'terminal':
                await asyncio.sleep(.03)
    async def first(desc, proc):
        async with websocket(home, desc) as ws:
            for name in ('safe', 'reset', 'unknown', 'foreign'):
                cwd = home / name
                cwd.mkdir()
                params[name] = dict(request_id=name, source='gui', cwd=str(cwd), model='frozen-' + name, toolsets=[])
                created = await rpc(ws, 'session.create', **params[name])
                assert 'result' in created, created
                sessions[name] = created['result']['session_id']
            for name, text in [('safe', 'WARM_HISTORY'), ('reset', 'RESET_OLD')]:
                await rpc(ws, 'prompt.submit', session_id=sessions[name], input_id='warm-' + name, text=text)
                await terminal('warm-' + name)
            await rpc(ws, 'prompt.submit', session_id=sessions['unknown'], input_id='started', text='BLOCK_STARTED')
            assert await asyncio.to_thread(peer.blocked.wait, 20)
            await rpc(ws, 'prompt.submit', session_id=sessions['unknown'], input_id='follower', text='NEVER_REPLAY')
            await rpc(ws, 'prompt.submit', session_id=sessions['reset'], input_id='reset', text='RECOVER_RESET')
            await rpc(ws, 'prompt.submit', session_id=sessions['foreign'], input_id='foreign', text='FOREIGN_QUEUE')
            await ws.send(json.dumps({'id': 'crash', 'method': 'prompt.submit', 'params': {
                'session_id': sessions['safe'], 'input_id': 'safe', 'text': 'RECOVER_SAFE'}}))
            # Wait for the actual publication, not just admission before the fixture ran.
            async with asyncio.timeout(20):
                while True:
                    with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
                        ended = dict(db.execute('SELECT id,end_reason FROM sessions'))
                    if all(ended.get(sessions[n]) == reason for n, reason in
                           [('safe', 'compression'), ('reset', 'session_reset'), ('unknown', 'compression')]):
                        break
                    assert proc.poll() is None
                    await asyncio.sleep(.03)
            proc.kill()
            await asyncio.to_thread(proc.wait, 10)
    async def recovered(desc):
        async with websocket(home, desc) as ws:
            for name in ('safe', 'reset'):
                snapshot = await rpc(ws, 'session.resume', session_id=sessions[name])
                assert 'result' in snapshot, snapshot
                await terminal(name)
                snapshot = await rpc(ws, 'session.resume', session_id=sessions[name])
                assert 'result' in snapshot, snapshot
                history = json.dumps(snapshot['result']['messages'])
                assert any(m['role'] == 'assistant' and m.get('content', '').startswith('RECOVERY_ACK_')
                           and m['content'].endswith('RECOVER_' + name.upper())
                           for m in snapshot['result']['messages']), history
                assert ('RECOVERY_ACK_WARM_HISTORY' in history) == (name == 'safe'), history
                assert 'RESET_OLD' not in history, history
                same = await rpc(ws, 'session.create', **params[name])
                assert same['result']['session_id'] == sessions[name], same
                assert same['result']['info']['model'] == params[name]['model'], same
                assert same['result']['info']['source'] == params[name]['source'], same
                retry = await rpc(ws, 'prompt.submit', session_id=sessions[name], input_id=name,
                                  text='RECOVER_' + name.upper())
                assert retry['result']['status'] == 'terminal', retry
            unknown = await rpc(ws, 'session.resume', session_id=sessions['unknown'])
            assert [p['status'] for p in unknown['result']['pending']] == ['unknown', 'queued'], unknown
            for name, reason in [('foreign', 'profile_mismatch'), ('fork', 'storage_unavailable')]:
                refusal = await rpc(ws, 'session.resume', session_id=sessions[name])
                assert refusal['error']['message'] == reason, refusal
    try:
        with daemon(root, home, env, barrier=True, fixture='local_lineage_daemon.py') as (proc, desc):
            pids.append(proc.pid); epochs.append(desc['authority_epoch'])
            asyncio.run(first(desc, proc))
        peer.release.set()
        with sqlite3.connect(home / 'state.db') as db:
            for name, sid in sessions.items():
                receipt = json.loads(db.execute('SELECT value FROM state_meta WHERE key=?',
                                    ('gateway.local_policy.v1:' + sid,)).fetchone()[0])
                targets[name] = receipt['entry']['session_id']
                if name != 'foreign':
                    route = json.loads(db.execute("SELECT entry_json FROM gateway_routing WHERE scope='' AND session_key=?",
                                                  (receipt['route'],)).fetchone()[0])
                    route_targets[name] = route['session_id']
                    row = db.execute('SELECT cwd FROM sessions WHERE id=?', (targets[name],)).fetchone()
                    assert row[0] == params[name]['cwd'], row
                else:
                    receipt['profile_id'] = str(user)
                    db.execute('UPDATE state_meta SET value=? WHERE key=?',
                               (json.dumps(receipt), 'gateway.local_policy.v1:' + sid))
            # A same-origin fork and a copied receipt do not gain the root's authority.
            sessions['fork'] = 'unrelated-fork'
            db.execute("""INSERT INTO sessions(id,source,parent_session_id,model_config,chat_id,user_id,session_key,started_at)
                SELECT ?,source,id,'{"_branched_from":"fixture-parent"}',chat_id,user_id,session_key,started_at FROM sessions WHERE id=?""",
                (sessions['fork'], targets['safe']))
            db.execute('INSERT INTO state_meta(key,value) SELECT ?,value FROM state_meta WHERE key=?',
                       ('gateway.local_policy.v1:' + sessions['fork'], 'gateway.local_policy.v1:' + sessions['safe']))
            from gateway.session_admission import admission_fingerprint
            payload = {'text': 'FORK_QUEUE'}
            digest = admission_fingerprint(canonical_target=sessions['fork'], payload={'input': payload, 'intent': 'queue'})
            db.execute('''INSERT INTO session_admissions(admission_id,request_id,principal_id,target_session_id,
                lineage_json,payload_json,payload_digest,intent,status,owner_epoch)
                SELECT 'fork-admission','fork',principal_id,?,?,?,?,intent,'queued',owner_epoch
                FROM session_admissions WHERE request_id='safe' ''',
                (sessions['fork'], json.dumps([sessions['fork']]), json.dumps(payload), digest))
        cfg['model']['default'] = 'changed-default'
        cfg['platform_toolsets']['cli'] = ['terminal']
        (home / 'config.yaml').write_text(json.dumps(cfg))
        for _ in range(2):
            with daemon(root, home, env, barrier=False) as (proc, desc):
                pids.append(proc.pid); epochs.append(desc['authority_epoch'])
                asyncio.run(recovered(desc))
                assert db_rows() == {'warm-safe': 'terminal', 'warm-reset': 'terminal', 'started': 'unknown',
                                     'follower': 'queued', 'reset': 'terminal', 'foreign': 'queued', 'fork': 'queued', 'safe': 'terminal'}
        assert all(route_targets[name] == targets[name] for name in ('safe', 'reset', 'unknown')), route_targets
        assert all(targets[name] != sessions[name] for name in ('safe', 'reset', 'unknown')), targets
        raw_texts = [next((m.get('content') for m in reversed(r['messages']) if m['role'] == 'user'), '') for r in peer.requests]
        # The production restart note may precede the newly admitted text; retain it in the receipt.
        texts = [text.rsplit('\n\n', 1)[-1] for text in raw_texts]
        assert sorted(texts) == sorted(['WARM_HISTORY', 'RESET_OLD', 'BLOCK_STARTED', 'RECOVER_SAFE', 'RECOVER_RESET']), texts
        for name in ('safe', 'reset'):
            request = next(r for r in peer.requests if any(isinstance(m.get('content'), str) and m['content'].endswith('RECOVER_' + name.upper()) for m in r['messages']))
            assert request['model'] == params[name]['model'] and not request.get('tools')
            assert params[name]['cwd'] in json.dumps(request['messages']), request['messages']
        return {'pids': pids, 'epochs': epochs, 'logical_ids': sessions, 'physical_targets': targets,
                'inference_texts': texts, 'raw_inference_texts': raw_texts, 'stable_receipts': True, 'policy_cwd_history_retained': True,
                'unknown_not_replayed': True, 'foreign_and_fork_refused': True}
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
