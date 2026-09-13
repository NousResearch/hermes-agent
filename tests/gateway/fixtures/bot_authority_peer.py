"""Authenticated production WebSockets and loopback-only inference."""
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


def probe(base):
    root = Path(__file__).resolve().parents[3]
    home, user = base / 'state', base / 'user'
    home.mkdir(); user.mkdir()
    model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    model.requests = []
    model.blocked, model.release = threading.Event(), threading.Event()
    threading.Thread(target=model.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{model.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'local-fixture', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)

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
            created = await rpc(ws, 'session.create', request_id='bot', source='gui', toolsets=[], cwd=str(home))
            sid = created['result']['session_id']
            from tools.bot_relay import enqueue_envelope, relay_root
            roster = await rpc(ws, 'bot_relay.roster.sync', agents=[{'profile': 'remote', 'handle': 'remote', 'connection_id': 'peer'}])
            assert roster.get('result', {}).get('count') == 1, roster
            envelope = enqueue_envelope(home, target={'profile': 'remote', 'handle': 'remote', 'connection_id': 'peer'},
                message='OUTBOUND', sender_profile='default', sender_handle='sender')
            drained = await rpc(ws, 'bot_relay.outbox.drain')
            assert drained['result']['envelopes'][0]['id'] == envelope['id'], drained
            reply = await rpc(ws, 'bot_relay.reply', id=envelope['id'], reply='REMOTE_ACK')
            assert reply.get('result', {}).get('ok'), reply
            assert json.loads((relay_root(home) / 'replies' / (envelope['id'] + '.json')).read_text())['reply'] == 'REMOTE_ACK'
            renamed = await rpc(ws, 'session.mutate', session_id=sid, request_id='name',
                expected_revision=created['result']['revision'], operation='rename', payload={'title': 'Bot Chat'})
            assert 'result' in renamed, renamed
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='warm', text='WARM_BOT')
            await wait(lambda: rows()[0]['status'] == 'terminal')
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='hold', text='HOLD_AUTOMATION')
            assert await asyncio.to_thread(model.blocked.wait, 20)
            from tools.bot_mode_dm import _admit_live_dm
            from tools.bot_live_delivery import _locked, _write, _read
            legacy_owner = dict(profile_home=str(home), session_id=sid, lease_id='departed', live_session_id='old-ui')
            with _locked(home) as mailbox:
                for key, status, text in [('b' * 32, 'queued', 'LEGACY_QUEUED_ONCE'),
                                          ('c' * 32, 'claimed', 'LEGACY_CLAIMED_NEVER')]:
                    _write(mailbox / f'{key}.json', dict(delivery_id=key, id=key, owner=legacy_owner,
                        **legacy_owner, message=text, status=status, created_at=1, sequence=1,
                        author={'id': 'bot:legacy', 'name': 'Legacy', 'is_bot': True}))
            dm = home / 'local-dm.txt'
            dm.write_text('[Message from @local-sender] LOCAL_DM_ONCE')
            local = await asyncio.to_thread(_admit_live_dm, home, str(dm))
            assert local is not None and local['status'] == 'queued', local
            same = await asyncio.to_thread(_admit_live_dm, home, str(dm))
            assert same['admission_id'] == local['admission_id'], same
            with _locked(home) as mailbox:
                migrated = _read(mailbox / ('b' * 32 + '.json'))
                unknown = _read(mailbox / ('c' * 32 + '.json'))
            assert migrated.get('admission_id') and unknown['status'] == 'ambiguous', (migrated, unknown)
            params = {'id': 'a' * 32, 'profile': 'default', 'message': '[Message from @sender] BOT_DM_ONCE',
                      'author': {'id': 'bot:peer/sender', 'name': 'Sender', 'is_bot': True}}
            wrong = await rpc(ws, 'bot_relay.deliver', **{**params, 'profile': 'wrong-profile'})
            assert wrong['error']['message'] == 'profile_mismatch', wrong
            # Discard an actual response at the socket boundary before reconnecting.
            await ws.send(json.dumps({'id': 'lost-ack', 'method': 'bot_relay.deliver', 'params': params}))
            await wait(lambda: any('BOT_DM_ONCE' in r['payload_json'] for r in rows()))
        async with websocket(home, desc) as observer:
            assert 'result' in await rpc(observer, 'session.resume', session_id=sid)
            retry = await rpc(observer, 'bot_relay.deliver', **params)
            assert 'result' in retry, retry
            assert retry['result']['status'] == 'queued', retry
            conflict = await rpc(observer, 'bot_relay.deliver', **{**params, 'message': 'CHANGED'})
            assert conflict['error']['message'] == 'admission_conflict', conflict
            bot = [r for r in rows() if 'BOT_DM_ONCE' in r['payload_json']]
            assert len(bot) == 1 and bot[0]['target_session_id'] == sid and bot[0]['status'] == 'queued', bot
            changed = await rpc(observer, 'bot_relay.deliver', **{**params, 'author': {'id': 'other'}})
            assert changed['error']['message'] == 'admission_conflict', changed
            assert json.loads(bot[0]['payload_json'])['local_automation_v1']['turn_author'] == params['author']
            legacy = next(r for r in rows() if 'LEGACY_QUEUED_ONCE' in r['payload_json'])
            assert json.loads(legacy['payload_json'])['local_automation_v1']['turn_author']['id'] == 'bot:legacy'
            model.release.set()
            await wait(lambda: all(r['status'] == 'terminal' for r in rows()))
            settled = await rpc(observer, 'bot_relay.deliver', **params)
            assert settled['result']['status'] == 'settled', settled
            assert settled['result']['reply'] == 'AUTOMATION_ACK', settled
        texts = [next((m.get('content', '') for m in reversed(r['messages']) if m['role'] == 'user'), '')
                 for r in model.requests if r.get('messages')]
        assert sum('BOT_DM_ONCE' in str(t) for t in texts) == 1, texts
        assert sum('LOCAL_DM_ONCE' in str(t) for t in texts) == 1, texts
        assert sum('LEGACY_QUEUED_ONCE' in str(t) for t in texts) == 1, texts
        assert not any('LEGACY_CLAIMED_NEVER' in str(t) for t in texts), texts
        print(json.dumps({'ledger': rows(), 'inputs': texts, 'lost_ack': True, 'same_target': sid}))
        return sid

    try:
        with daemon(root, home, env, barrier=False) as (proc, desc):
            try:
                sid = asyncio.run(run(desc))
                print(json.dumps({'ordinary_daemon_pid': proc.pid}))
            except BaseException:
                print((home / 'restart.log').read_text()[-10000:], file=sys.stderr)
                raise
        # Model the exact post-settlement/pre-mailbox-publication crash boundary.
        # Keep the actual admission/result, remove only the derivative reply receipt.
        from tools.bot_live_delivery import _locked, _read, _write
        with _locked(home) as mailbox:
            path = mailbox / ('a' * 32 + '.json')
            record = _read(path)
            record.update(status='canonical', reply='')
            _write(path, record)
            owner = dict(profile_home=str(home), session_id=sid, lease_id='departed', live_session_id='old')
            _write(mailbox / ('d' * 32 + '.json'), dict(delivery_id='d' * 32, id='d' * 32,
                owner=owner, **owner, status='queued', message='STARTUP_LEGACY_ONCE', created_at=2, sequence=2))
        async def restarted(desc):
            def read_delivery_result(home, key):
                return _read(home / 'runtime' / 'bot_live_delivery' / (key + '.json'))
            await wait(lambda: read_delivery_result(home, 'a' * 32)['status'] == 'settled')
            recovered = read_delivery_result(home, 'a' * 32)
            assert recovered['admission_id'] == record['admission_id']
            assert recovered['reply'] == 'AUTOMATION_ACK', recovered
            await wait(lambda: read_delivery_result(home, 'd' * 32)['status'] == 'settled')
            assert sum('STARTUP_LEGACY_ONCE' in r['payload_json'] for r in rows()) == 1
            print(json.dumps({'restart_receipt': recovered, 'startup_migration': True}))
        with daemon(root, home, env, barrier=False) as (proc, desc):
            asyncio.run(restarted(desc))
            print(json.dumps({'restart_daemon_pid': proc.pid}))
    finally:
        model.release.set(); model.shutdown(); model.server_close()


if __name__ == '__main__':
    probe(Path(sys.argv[1]))
