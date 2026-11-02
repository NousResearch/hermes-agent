"""#134889 NF4: deleting a compressed, branched conversation leaves its branch a working conversation."""
import asyncio
from contextlib import closing
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket
from tests.gateway.test_session_mutation_compress import SummaryModel
from tests.waiting import async_wait_until

KEY = 'launch-only-key-0123456789abcdef'


class AuthModel(SummaryModel):
    def do_POST(self):
        self.server.auth.append(self.headers.get('Authorization'))
        super().do_POST()


def test_branch_keeps_its_launch_key_after_the_compressed_root_is_deleted(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), AuthModel)
    peer.requests, peer.auth = [], []
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': False},
        'model': {'provider': 'custom', 'default': 'original', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False},
                      'compression': {'provider': 'custom', 'model': 'summary', 'base_url': url}},
        'platform_toolsets': {'cli': []},
        'compression': {'protect_first_n': 1, 'protect_last_n': 2, 'min_tail_user_messages': 1, 'in_place': False}}))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 1000000, 'output': 4000}}
        for name in ('original', 'summary')}}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='profile-key', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def db():
        return closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True))

    async def run(desc):
        async with websocket(home, desc) as ws:
            async def call(method, **params):
                response = await rpc(ws, method, **params)
                assert 'result' in response, response
                return response['result']

            async def turn(sid, text):
                await call('prompt.submit', session_id=sid, input_id=text, text=text)

                def outcome():
                    with db() as conn:
                        row = conn.execute("SELECT outcome FROM session_admissions WHERE request_id=? "
                                           "AND status='terminal'", (text,)).fetchone()
                    return row and row[0]
                await async_wait_until(outcome, timeout=30, what=f'{text} terminal')
                sent = [auth for auth, body in zip(peer.auth, peer.requests) if text in json.dumps(body['messages'][-1:])]
                return outcome(), sent

            async def mutate(sid, operation):
                state = await call('session.resume', session_id=sid)
                return await call('session.mutate', session_id=sid, request_id=operation + sid, operation=operation,
                                  payload={}, expected_revision=state['revision'],
                                  expected_generation=state['execution_generation'])

            sid = (await call('session.create', source='cli', request_id='owner', toolsets=[], model='original',
                              provider='custom', base_url=url, api_key=KEY))['session_id']
            for i in range(5):
                await turn(sid, f'ROOT_{i}')
            continuation = (await mutate(sid, 'compress'))['target_session_id']
            branch = (await mutate(sid, 'branch'))['branched_session_id']
            assert await turn(branch, 'BEFORE_DELETE') == ('completed', ['Bearer ' + KEY])
            assert set((await mutate(sid, 'delete'))['deleted_ids']) == {sid, continuation}
            with db() as conn:
                # Independent conversation: unparented, its fork marker kept as provenance (prune does the same).
                parent, config = conn.execute('SELECT parent_session_id,model_config FROM sessions WHERE id=?',
                                              (branch,)).fetchone()
            assert parent is None and json.loads(config)['_branched_from'] == continuation
            assert await turn(branch, 'AFTER_DELETE') == ('completed', ['Bearer ' + KEY])
            compressed = await mutate(branch, 'compress')
            assert set((await mutate(branch, 'delete'))['deleted_ids']) == {branch, compressed['target_session_id']}
            with db() as conn:
                assert conn.execute('SELECT count(*) FROM sessions').fetchone() == (0,)
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(run(desc))
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)
