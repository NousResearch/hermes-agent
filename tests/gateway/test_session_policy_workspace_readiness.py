"""A removed workspace (a deleted worktree) never blocks a local chat: reads, moves and turns all
carry on, the turn working in the checkout the folder belonged to (N87, then Teknium's ruling that a
missing worktree must not refuse the session)."""
import pytest


@pytest.mark.asyncio
async def test_removed_workspace_turn_runs_in_the_nearest_checkout_without_rewriting_the_policy(tmp_path, monkeypatch):
    from agent.runtime_cwd import scoped_session_cwd
    from gateway.session_policy import policy_for_source, policy_scope
    from hermes_state_local import local_receipt
    from tests.gateway.test_session_policy_verbs import _mutation, _owner
    from tools.terminal_scope import terminal_env

    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    gone = tmp_path / 'project-a'
    target = tmp_path / 'project-b'
    target.mkdir()
    try:
        gone.rmdir()  # git worktree remove / deleted scratch folder
        resumed = await owner.dispatch({'id': 1, 'method': 'session.resume',
                                        'params': {'session_id': ref.session_id}})
        assert 'result' in resumed, resumed
        policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
        with policy_scope(policy, authority=authority):
            assert scoped_session_cwd() == str(tmp_path)
            assert terminal_env('TERMINAL_CWD') == str(tmp_path)
        assert local_receipt(db, ref.session_id)['policy']['cwd'] == str(gone)  # frozen policy untouched
        gone.mkdir()  # the folder coming back is used again
        policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
        assert policy.cwd == str(gone)
        moved = await owner.dispatch(_mutation(db, ref, 'mv', 'workspace', {'cwd': str(target)}))
        assert moved['result']['cwd'] == str(target), moved
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.platforms("linux")
def test_managed_turn_in_removed_workspace_runs(tmp_path):
    """Safe-mode and config-only sessions run their turns in a managed worker; a removed folder used to
    refuse every one of them. Real gateway, real worker, loopback model."""
    import asyncio
    import json
    import sqlite3
    import threading
    from http.server import ThreadingHTTPServer
    from pathlib import Path

    from tests.gateway.fixtures.local_recovery_probe import Model, child_env, daemon, rpc, websocket
    from tests.waiting import async_wait_until

    root = Path(__file__).resolve().parents[2]
    home, user, folder = tmp_path / 'state', tmp_path / 'user', tmp_path / 'project'
    home.mkdir(mode=0o700)
    user.mkdir()
    folder.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests, peer.blocked, peer.release = [], threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False}, 'model': {'provider': 'custom', 'default': 'm', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}))
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    def admission(name):
        with sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True) as db:
            return db.execute('SELECT status, outcome FROM session_admissions WHERE request_id=?', (name,)).fetchone()

    async def drive(desc):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='safe', source='cli', cwd=str(folder),
                                toolsets=[], safe_mode=True,  # a bypass launch names its own route
                                model='m', provider='custom', base_url=url, api_key='loopback-only')
            assert 'result' in created, created
            sid = created['result']['session_id']
            folder.rmdir()
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='gone', text='RUNS_ANYWAY')
            await async_wait_until(lambda: (admission('gone') or ('',))[0] in ('terminal', 'unknown'),
                                   timeout=60, what='turn in a removed folder settled')
            assert admission('gone') == ('terminal', 'completed')

    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            asyncio.run(drive(desc))
    finally:
        peer.shutdown()
    assert 'RUNS_ANYWAY' in json.dumps(peer.requests)
