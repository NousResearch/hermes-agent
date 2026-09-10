"""Actual classic CLI persistence followed by native owner migration (loopback)."""
import asyncio
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket


def probe(base, legacy_root, owner_root=None):
    fixture_root = Path(__file__).resolve().parents[3]
    root = owner_root or fixture_root
    home, user = base / 'state', base / 'user'
    home.mkdir(parents=True); user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'legacy-model', 'base_url': url},
        'gateway': {'multiplex_profiles': False}, 'platform_toolsets': {'cli': []},
        'auxiliary': {'title_generation': {'enabled': False}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(legacy_root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url)
    def cli(checkout, *args):
        result = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'chat', *args],
            cwd=checkout, env={**env, 'PYTHONPATH': str(checkout)}, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=120)
        (home / ('cli-' + str(len(list(home.glob('cli-*')))) + '.log')).write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout
    def query(sql, args=()):
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute(sql, args)]
    async def wait_settled():
        async with asyncio.timeout(45):
            while any(r['status'] != 'terminal' for r in query('SELECT status FROM session_admissions')):
                await asyncio.sleep(.05)
    async def run(desc):
        async with websocket(home, desc) as ws:
            first = await rpc(ws, 'session.resume', session_id=sid)
            assert first.get('result', {}).get('session_id') == sid, first
            assert 'OLD_HISTORY' in json.dumps(first['result']['messages']), first
            params = {'id': 'a' * 32, 'profile': 'default', 'message': 'BOT_NEW_INPUT'}
            result = await rpc(ws, 'bot_relay.deliver', **params)
            assert 'result' in result, result
            await wait_settled()
            retry = await rpc(ws, 'bot_relay.deliver', **params)
            assert retry['result']['admission_id'] == result['result']['admission_id'], retry
            assert retry['result']['status'] == 'settled', retry
            return first, retry
    try:
        cli(legacy_root, '-c', 'Bot Chat', '--create-if-missing', '--toolsets', '', '-q', 'OLD_HISTORY')
        old = query("SELECT * FROM sessions WHERE title='Bot Chat'")[0]
        sid = old['id']
        assert not old['chat_id'] and not old['session_key'], old
        assert not query("SELECT key FROM state_meta WHERE key LIKE 'gateway.local_policy.v1:%'")
        subprocess.run([sys.executable, str(fixture_root / 'tests/gateway/fixtures/legacy_bot_mailbox.py'), str(home), sid],
            cwd=legacy_root, env=env, stdin=subprocess.DEVNULL, check=True, timeout=30)
        env['PYTHONPATH'] = str(root)
        with daemon(root, home, env, barrier=False) as (_, desc):
            cli(root, '--resume', sid, '-q', 'CLI_NEW_INPUT')
            first, retry = asyncio.run(run(desc))
        async def restarted(desc):
            async with websocket(home, desc) as ws:
                result = await rpc(ws, 'session.resume', session_id=sid)
                assert result.get('result', {}).get('session_id') == sid, result
                assert 'CLI_NEW_INPUT' in json.dumps(result['result']['messages']), result
                compressed = await rpc(ws, 'session.mutate', session_id=sid, request_id='compress-legacy',
                    expected_revision=result['result']['revision'], expected_generation=result['result']['execution_generation'],
                    operation='compress', payload={})
                assert 'result' in compressed, compressed
                after = await rpc(ws, 'session.resume', session_id=sid)
                assert after.get('result', {}).get('session_id') == sid, after
                assert compressed['result']['previous_target_session_id'] == sid, compressed
                return after
        with daemon(root, home, env, barrier=False) as (_, desc):
            resumed = asyncio.run(restarted(desc))
        mailbox = home / 'runtime/bot_live_delivery'
        queued = json.loads((mailbox / ('b' * 32 + '.json')).read_text())
        claimed = json.loads((mailbox / ('c' * 32 + '.json')).read_text())
        foreign = json.loads((mailbox / ('d' * 32 + '.json')).read_text())
        assert queued.get('admission_id') and queued['status'] == 'settled', queued
        assert claimed['status'] == 'ambiguous' and foreign['status'] == 'claimed'
        inputs = [str(next((m.get('content') for m in reversed(r['messages']) if m['role'] == 'user'), '')) for r in peer.requests]
        assert sum(text == 'LEGACY_QUEUED_ONCE' for text in inputs) == 1, inputs
        assert not any('LEGACY_CLAIMED_NEVER' in text or 'UNRELATED_NEVER' in text for text in inputs)
        current = query('SELECT * FROM sessions WHERE id=?', (sid,))[0]
        assert current['title'] == old['title'] and current['parent_session_id'] == old['parent_session_id']
        new = [r for r in peer.requests if next((m.get('content') for m in reversed(r['messages']) if m['role'] == 'user'), '') in {'BOT_NEW_INPUT', 'CLI_NEW_INPUT'}]
        assert new and all('OLD_HISTORY' in json.dumps(r) for r in new)
        return {'historical_id': sid, 'legacy_row': old, 'first_resume': first, 'bot_retry': retry,
                'restart_resume': resumed, 'requests': peer.requests, 'admissions': query('SELECT * FROM session_admissions')}
    finally:
        peer.release.set(); peer.shutdown(); peer.server_close()


if __name__ == '__main__':
    output = probe(Path(sys.argv[1]), Path(sys.argv[2]))
    (Path(sys.argv[1]) / 'receipt.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({'historical_id': output['historical_id'], 'admissions': len(output['admissions'])}))
