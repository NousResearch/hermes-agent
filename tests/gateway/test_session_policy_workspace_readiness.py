"""A removed workspace keeps a local chat readable and movable; only a turn refuses (N87)."""
import pytest


@pytest.mark.asyncio
async def test_removed_workspace_reads_and_moves_but_refuses_a_turn(tmp_path, monkeypatch):
    from gateway.session_policy import policy_for_source, policy_scope
    from hermes_state_runtime import RuntimeStoreError
    from tests.gateway.test_session_policy_verbs import _mutation, _owner

    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    target = tmp_path / 'project-b'
    target.mkdir()
    try:
        (tmp_path / 'project-a').rmdir()  # git worktree remove / deleted scratch folder
        resumed = await owner.dispatch({'id': 1, 'method': 'session.resume',
                                        'params': {'session_id': ref.session_id}})
        assert 'result' in resumed, resumed
        policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
        with pytest.raises(RuntimeStoreError, match='workspace_unavailable'):
            with policy_scope(policy, authority=authority):
                pass
        moved = await owner.dispatch(_mutation(db, ref, 'mv', 'workspace', {'cwd': str(target)}))
        assert moved['result']['cwd'] == str(target), moved
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_turn_in_removed_workspace_names_the_folder_instead_of_failing_generically(tmp_path, monkeypatch):
    """A turn in a session whose folder is gone settles as a failed, never-run turn that names the
    folder, with its own reason code; it used to escape as an exception every client showed as a
    generic "model request failed" / "something went wrong"."""
    from types import SimpleNamespace

    from agent.error_surface import build_error_surface_from_result
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_results import execution_result
    from tests.gateway.test_session_policy_verbs import _owner

    store, _db, authority, owner, ref = _owner(tmp_path, monkeypatch, None)
    authority.runner.session_authority = authority
    folder = tmp_path / 'project-a'
    try:
        folder.rmdir()
        source = authority.sessions[ref.session_id].source
        ctx = SimpleNamespace(source=source, session_id=ref.session_id, session_key=None)
        token = execution_result.set({})
        try:
            result = TurnRunner(authority.runner, ctx).run_sync()
            captured = execution_result.get()
        finally:
            execution_result.reset(token)
        assert result['failed'] and not result['completed'] and result['api_calls'] == 0, result
        assert result['failure_reason'] == 'workspace_unavailable' and result['failure_retryable'] is False
        assert str(folder) in result['final_response'], result['final_response']
        assert captured['result'] == result  # what the classic CLI viewer reads off the receipt
        surface = build_error_surface_from_result(result)
        assert (surface['layer'], surface['code'], surface['retryable']) == ('gateway', 'workspace_unavailable', False)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.platforms("linux")
def test_managed_turn_in_removed_workspace_settles_failed_and_never_parks_the_queue(tmp_path):
    """Safe-mode and config-only sessions run their turns in a managed worker. A removed folder used
    to crash that worker in its policy scope, which the owner records as a LOST execution: the turn
    settled ``unknown`` and every later message stayed queued behind it even after the folder came
    back. Real gateway, real worker, loopback model."""
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
            sent = await rpc(ws, 'prompt.submit', session_id=sid, input_id='gone', text='NEVER_RUN')
            await async_wait_until(lambda: (admission('gone') or ('',))[0] in ('terminal', 'unknown'),
                                   timeout=60, what='refused turn settled')
            assert admission('gone') == ('terminal', 'failed')
            receipt = await rpc(ws, 'prompt.receipt', session_id=sid, admission_id=sent['result']['admission_id'],
                                include_result=True)
            result = receipt['result']['result']
            assert result['failure_reason'] == 'workspace_unavailable' and result['api_calls'] == 0, result
            assert str(folder) in result['final_response']
            folder.mkdir()
            await rpc(ws, 'prompt.submit', session_id=sid, input_id='back', text='FOLDER_BACK')
            await async_wait_until(lambda: (admission('back') or ('',))[0] == 'terminal',
                                   timeout=60, what='turn after the folder came back ran')
            assert admission('back') == ('terminal', 'completed')

    try:
        with daemon(root, home, env, barrier=False) as (_proc, desc):
            asyncio.run(drive(desc))
    finally:
        peer.shutdown()
    sent_texts = json.dumps(peer.requests)
    assert 'NEVER_RUN' not in sent_texts and 'FOLDER_BACK' in sent_texts
