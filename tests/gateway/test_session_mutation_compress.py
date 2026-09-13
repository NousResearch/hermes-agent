"""Real summary preparation commits only against the unchanged local transcript."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_session_mutation_model import CatalogModel


class SummaryModel(CatalogModel):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        summary = any('You are a summarization agent creating a context checkpoint.' in str(m.get('content', ''))
                      for m in body.get('messages', []))
        text = 'SUMMARY_RETAINED_FACTS' if summary else 'ORDINARY_REPLY ' + ('historical detail ' * 700)
        payload = json.dumps({'id': 'fixture', 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}}).encode()
        kind = 'application/json'
        if body.get('stream'):
            payload = ('data: ' + json.dumps({'id': 'fixture', 'choices': [{'index': 0,
                'delta': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.parametrize(('mode', 'in_place'), [
    ('normal', True), ('normal', False), ('managed', True), ('managed', False), ('safe', None),
])
def test_compress_preserves_configured_history_and_admission_owner(tmp_path, mode, in_place):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), SummaryModel)
    peer.requests = []
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False, 'managed_workers': mode == 'managed'},
        'model': {'provider': 'custom', 'default': 'original', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False},
                      'compression': {'provider': 'custom', 'model': 'summary', 'base_url': url}},
        'platform_toolsets': {'cli': []},
        'compression': {'protect_first_n': 1, 'protect_last_n': 2,
                        'min_tail_user_messages': 8, 'in_place': in_place if in_place is not None else False}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 1000000, 'output': 4000}}
        for name in ('original', 'summary')}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
        OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    async def run(desc):
        async with websocket(home, desc) as ws:
            async def call(method, **params):
                response = await rpc(ws, method, **params)
                assert 'result' in response, response
                return response['result']
            sid = (await call('session.create', source='cli', request_id='owner', toolsets=[],
                              model='original', provider='custom', base_url=url,
                              safe_mode=mode == 'safe'))['session_id']
            physical = sid
            if in_place is True:
                initial = await call('session.resume', session_id=sid)
                physical = (await call('session.mutate', session_id=sid, request_id='reset-first',
                    operation='reset', payload={}, expected_revision=initial['revision'],
                    expected_generation=initial['execution_generation']))['target_session_id']
                assert physical != sid
            accepted = {}
            async def turn(text):
                accepted[text] = await call('prompt.submit', session_id=sid, input_id=text, text=text)
                async with asyncio.timeout(30):
                    while True:
                        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                            row = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (text,)).fetchone()
                        if row and row[0] == 'terminal':
                            return await call('session.resume', session_id=sid)
                        await asyncio.sleep(.03)
            history_count = 16
            for i in range(history_count):
                before = await turn('HISTORY_' + str(i))
            params = dict(session_id=sid, request_id='compress', operation='compress', payload={},
                expected_revision=before['revision'], expected_generation=before['execution_generation'])
            # Profile edits cannot replace the session's frozen compression policy.
            changed = dict(cfg, compression={'in_place': not bool(in_place), 'min_tail_user_messages': 1})
            (home / 'config.yaml').write_text(json.dumps(changed))
            result = await call('session.mutate', **params)
            assert result['revision'] == before['revision'] + 1
            assert result['execution_generation'] == before['execution_generation'] + 1
            assert result['previous_target_session_id'] == physical
            effective_in_place = in_place is not False
            assert (result['target_session_id'] == physical) is effective_in_place
            compressed = await call('session.resume', session_id=sid)
            assert 'SUMMARY_RETAINED_FACTS' in json.dumps(compressed['messages'])
            retained = 1 if mode == 'safe' else cfg['compression']['min_tail_user_messages']
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                tail_users = [row[0] for row in db.execute(
                    "SELECT content FROM messages WHERE session_id=? AND active=1 AND role='user'",
                    (result['target_session_id'],))]
            for i in range(history_count - retained, history_count):
                assert f'HISTORY_{i}' in tail_users, tail_users
            for text, receipt in accepted.items():
                retried = await call('prompt.submit', session_id=sid, input_id=text, text=text)
                assert retried['admission_id'] == receipt['admission_id']
            await turn('AFTER_COMPRESSION')
            assert 'SUMMARY_RETAINED_FACTS' in json.dumps(peer.requests[-1]['messages'])
            for i in range(history_count - retained, history_count):
                assert f'HISTORY_{i}' in json.dumps(peer.requests[-1]['messages'])
            count = len(peer.requests)
            assert await call('session.mutate', **params) == result
            assert len(peer.requests) == count
            # Later mutations must target the current transcript, not a closed ancestor.
            after = await call('session.resume', session_id=sid)
            branch = await call('session.mutate', session_id=sid, request_id='branch-after-compress',
                operation='branch', payload={}, expected_revision=after['revision'],
                expected_generation=after['execution_generation'])
            child = await call('session.resume', session_id=branch['branched_session_id'])
            assert 'SUMMARY_RETAINED_FACTS' in json.dumps(child['messages'])
            with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
                assert db.execute('SELECT end_reason FROM sessions WHERE id=?', (physical,)).fetchone()[0] == (
                    None if effective_in_place else 'compression')
                assert db.execute('SELECT DISTINCT target_session_id FROM session_admissions').fetchall() == [(sid,)]
                assert db.execute("SELECT session_id FROM messages WHERE role='user' AND content='AFTER_COMPRESSION' AND session_id!=?",
                                  (branch['branched_session_id'],)).fetchall() == [(result['target_session_id'],)]
                archived = db.execute('SELECT count(*) FROM messages WHERE session_id=? AND active=0 AND compacted=1', (physical,)).fetchone()[0]
                assert (archived > 0) is effective_in_place
                if mode != 'normal':
                    assert not db.execute("SELECT execution_id FROM worker_executions WHERE status!='terminal'").fetchall()
                    journals = list((home / 'worker-outboxes').rglob('pending.json'))
                    assert journals and all(json.loads(p.read_text())['pending'] == [] for p in journals)
                assert db.execute('SELECT count(*) FROM messages WHERE session_id=?', (physical,)).fetchone()[0] > 0
                receipt = json.loads(db.execute('SELECT value FROM state_meta WHERE key=?',
                    ('gateway.local_policy.v1:' + sid,)).fetchone()[0])
                assert receipt['entry']['session_id'] == result['target_session_id']
            print(json.dumps({'mode': mode, 'configured_in_place': in_place, 'logical': sid,
                              'physical': result['target_session_id'], 'retained_users': retained,
                              'archived_rows': archived, 'admissions': len(accepted), 'mutation': result}))
            return params, result
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            params, result = asyncio.run(run(desc))
        async def retry(desc):
            async with websocket(home, desc) as ws:
                assert (await rpc(ws, 'session.mutate', **params)).get('result') == result
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(retry(desc))
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
