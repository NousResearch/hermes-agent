"""Explicit model changes cross the real provider wire, not profile defaults."""
import asyncio
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import threading

from tests.waiting import async_wait_until

import pytest
from tests.gateway.fixtures.local_recovery_probe import Model, daemon, rpc, websocket


class CatalogModel(Model):
    def do_GET(self):
        payload = json.dumps({'data': [{'id': name} for name in ('original', 'switched')]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_model_commit_republishes_session_info(monkeypatch):
    """A canonical picker switch must repaint the intro/header from its committed policy."""
    from types import SimpleNamespace

    from gateway.session_contract import SessionRef
    from gateway.session_mutations import _project_committed
    from gateway import session_local

    published = []
    authority = SimpleNamespace(
        sessions={"session": SimpleNamespace(event_stream=SimpleNamespace(
            publish=lambda sid, payload, *, event_type: published.append((sid, payload, event_type))))},
        runner=SimpleNamespace(_set_session_reasoning_override=lambda *_args: None),
        _schedule=lambda _ref: None,
    )
    monkeypatch.setattr(session_local, "publish_local_policy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(session_local, "local_session_info", lambda *_args: {"model": "switched"})

    _project_committed(authority, SessionRef("profile", "session"), "model", {})

    assert published == [("session", {"model": "switched"}, "session.info")]


@pytest.mark.platforms("linux")
def test_model_receipt_changes_next_wire_and_branch_keeps_independent_history(tmp_path):
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), CatalogModel)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'original', 'base_url': url},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 100000, 'output': 4000}}
        for name in ('original', 'switched')}}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1')

    async def run(desc):
        async with websocket(home, desc) as ws:
            async def call(method, **params):
                response = await rpc(ws, method, **params)
                assert 'result' in response, response
                return response['result']
            async def turn(sid, text):
                await call('prompt.submit', session_id=sid, input_id=text, text=text)
                def terminal():
                    with sqlite3.connect(home / 'state.db') as db:
                        status = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (text,)).fetchone()
                    return status and status[0] == 'terminal'
                await async_wait_until(terminal, timeout=30, what=f'{text} terminal')
                return await call('session.resume', session_id=sid)
            sid = (await call('session.create', source='cli', request_id='owner', toolsets=[]))['session_id']
            before = await turn(sid, 'ORIGINAL_HISTORY')
            params = dict(session_id=sid, request_id='switch', operation='model', payload={'model': 'switched'},
                expected_revision=before['revision'], expected_generation=before['execution_generation'])
            from hermes_cli.gateway_chat_view import GatewayChatView
            from acp_adapter.gateway_server import GatewayACPAgent
            class Client:
                async def rpc(self, method, _timeout=None, **kwargs):  # the client budget is never on the wire
                    return await call(method, **kwargs)
            client = Client()
            view = GatewayChatView(client, before)
            await view.command('/model switched')
            result = await call('session.info', session_id=sid)
            assert result['model'] == 'switched'
            after = await turn(sid, 'AFTER_SWITCH')
            assert peer.requests[-1]['model'] == 'switched'
            assert peer.requests[0]['model'] == 'original'
            agent = GatewayACPAgent()
            agent._gateway = client
            agent._snapshots[sid] = after
            fork = await agent.fork_session(cwd=(await call('session.info', session_id=sid))['cwd'], session_id=sid)
            child = fork.session_id
            branched = await call('session.resume', session_id=child)
            # Same logical history; each copy has its own physical ``row_id`` (rewind target).
            def logical(rows):
                return [{k: v for k, v in row.items() if k != 'row_id'} for row in rows]
            assert logical(branched['messages']) == logical(after['messages'])
            assert {r['row_id'] for r in branched['messages']}.isdisjoint(r['row_id'] for r in after['messages'])
            await turn(child, 'ONLY_BRANCH')
            parent = await call('session.resume', session_id=sid)
            assert 'ONLY_BRANCH' not in json.dumps(parent['messages'])
            assert peer.requests[-1]['model'] == 'switched'
            assert 'ORIGINAL_HISTORY' in json.dumps(peer.requests[-1]['messages'])
            return params, result
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            params, _result = asyncio.run(run(desc))
        async def retry(desc):
            async with websocket(home, desc) as ws:
                info = await rpc(ws, 'session.info', session_id=params['session_id'])
                assert info['result']['model'] == 'switched'
        with daemon(root, home, env, barrier=False) as (_, desc):
            asyncio.run(retry(desc))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


class RecordingModel(CatalogModel):
    """A loopback endpoint that records each inference request's model and bearer."""
    def do_POST(self):
        self.server.bearers.append(self.headers.get('Authorization'))
        super().do_POST()


def _loopback():
    peer = ThreadingHTTPServer(('127.0.0.1', 0), RecordingModel)
    peer.requests, peer.bearers = [], []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    return peer, thread, f'http://127.0.0.1:{peer.server_port}/v1'


@pytest.mark.platforms("linux")
def test_desktop_pick_resolves_a_provider_added_after_the_session_and_freezes_only_it(tmp_path):
    """#139740: a session created before ``providers.loop`` existed picks ``loop`` the way the Desktop
    picker sends it (``{model, provider}``). The pick resolves against the owning profile's CURRENT
    ``providers:`` (as CLI/TUI ``/model`` do), the next turn reaches that endpoint with its
    ``key_env`` key, the receipt freezes only that one entry plus the pointer (never the key
    value), nothing else in the frozen config moves, and another profile's same-named provider
    is never read."""
    from hermes_state_local import POLICY_PREFIX
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    origin, origin_thread, origin_url = _loopback()
    picked, picked_thread, picked_url = _loopback()
    foreign, foreign_thread, foreign_url = _loopback()
    secret = 'sk-loop-only-7c41e0'
    cfg = {'gateway': {'multiplex_profiles': False},
           'model': {'provider': 'custom', 'default': 'original', 'base_url': origin_url},
           'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []},
           'agent': {'max_turns': 7}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (home / '.env').write_text(f'LOOP_KEY={secret}\n')
    # Another profile on this host declares the same slug at another endpoint: never read.
    other = home / 'profiles' / 'other'
    other.mkdir(parents=True)
    (other / 'config.yaml').write_text(json.dumps({'providers': {'loop': {
        'name': 'Foreign', 'base_url': foreign_url, 'key_env': 'FOREIGN_KEY', 'models': {'switched': {}}}}}))
    (home / 'models_dev_cache.json').write_text(json.dumps({'custom': {'id': 'custom', 'models': {
        name: {'id': name, 'name': name, 'limit': {'context': 100000, 'output': 4000}}
        for name in ('original', 'switched')}}}))
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ', 'TIRITH_ENABLED') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=origin_url, PYTHONUNBUFFERED='1',
               FOREIGN_KEY='sk-foreign-never')

    def receipt(sid):
        with sqlite3.connect(home / 'state.db') as db:
            (value,), = db.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchall()
        return json.loads(value)['policy']

    async def run(desc):
        async with websocket(home, desc) as ws:
            async def call(method, **params):
                response = await rpc(ws, method, **params)
                assert 'result' in response, response
                return response['result']
            async def turn(sid, text):
                await call('prompt.submit', session_id=sid, input_id=text, text=text)
                def terminal():
                    with sqlite3.connect(home / 'state.db') as db:
                        status = db.execute('SELECT status FROM session_admissions WHERE request_id=?', (text,)).fetchone()
                    return status and status[0] == 'terminal'
                await async_wait_until(terminal, timeout=30, what=f'{text} terminal')
                return await call('session.resume', session_id=sid)
            sid = (await call('session.create', source='gui', request_id='owner', toolsets=[]))['session_id']
            before = await turn(sid, 'BEFORE_PICK')
            frozen_before = json.loads(receipt(sid)['config_json'])
            # The user adds the endpoint while the session exists (the Desktop menu then lists it).
            (home / 'config.yaml').write_text(json.dumps({**cfg, 'providers': {'loop': {
                'name': 'Loopback', 'base_url': picked_url, 'key_env': 'LOOP_KEY', 'api_mode': 'chat_completions',
                'models': {'switched': {'context_length': 100000}}}}}))
            picked_result = await rpc(ws, 'session.mutate', session_id=sid, request_id='pick', operation='model',
                                      payload={'model': 'switched', 'provider': 'loop'},
                                      expected_revision=before['revision'],
                                      expected_generation=before['execution_generation'])
            assert 'result' in picked_result, picked_result
            assert picked_result['result']['model'] == 'switched'
            await turn(sid, 'AFTER_PICK')
            return sid, frozen_before
    try:
        with daemon(root, home, env, barrier=False) as (_, desc):
            sid, frozen_before = asyncio.run(run(desc))
        assert [r['model'] for r in origin.requests] == ['original']
        assert [r['model'] for r in picked.requests] == ['switched'], 'the next turn did not reach the pick'
        assert picked.bearers == [f'Bearer {secret}']
        assert foreign.requests == [] and foreign.bearers == [], "another profile's provider was read"
        frozen = json.loads(receipt(sid)['config_json'])
        assert frozen['providers'] == {'loop': {
            'name': 'Loopback', 'base_url': picked_url, 'key_env': 'LOOP_KEY', 'api_mode': 'chat_completions',
            'models': {'switched': {'context_length': 100000}}}}
        assert frozen['model']['provider'] == 'loop' and frozen['model']['key_env'] == 'LOOP_KEY'
        assert frozen['model']['default'] == 'switched'
        # Nothing else in the frozen config moved.
        assert {k: v for k, v in frozen.items() if k not in ('model', 'providers')} == \
            {k: v for k, v in frozen_before.items() if k not in ('model', 'providers')}
        assert 'providers' not in frozen_before
        # The key value never lands in the session store.
        raw = b''.join(path.read_bytes() for path in home.glob('state.db*'))
        assert secret.encode() not in raw and b'sk-foreign-never' not in raw
    finally:
        for peer, thread in ((origin, origin_thread), (picked, picked_thread), (foreign, foreign_thread)):
            peer.release.set()
            peer.shutdown()
            peer.server_close()
            thread.join(timeout=5)


@pytest.mark.asyncio
async def test_input_queued_while_a_model_receipt_commits_still_runs(tmp_path, monkeypatch):
    """A drain that runs while the model receipt commits off-loop sees the new stored policy beside
    the old live one and pauses; publishing the policy must wake it, or the input stays queued
    until another submit or a restart."""
    from types import SimpleNamespace
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Submission
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import model_switch
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='switched', target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    executed = []

    async def handle(event):
        executed.append(event.text)
        return 'ok'
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _handle_message=handle,
                             _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='cli', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    # A contended receipt transaction: the commit thread holds it open while the loop runs.
    entered, release = threading.Event(), threading.Event()
    db._conn.create_function('hold_receipt', 1, lambda s: (entered.set(), release.wait(s))[1] and 0)
    db._execute_write(lambda c: c.execute("CREATE TRIGGER hold_receipt AFTER INSERT ON state_meta "
                                          "WHEN NEW.key LIKE 'gateway.mutation.v1.%' BEGIN SELECT hold_receipt(30); END"))
    snap = db.get_session(ref.session_id)
    try:
        mutation = asyncio.create_task(owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'switch', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': 'switched'}}}))
        await asyncio.to_thread(entered.wait, 5)
        # The follower is submitted while the receipt transaction is still open: hold it until
        # that submit is parked behind the session's mutation lock.
        submit = asyncio.create_task(authority.submit(owner.actor, Submission('during-switch', ref, {'text': 'FOLLOWER'}, 'queue')))
        lock = authority.sessions[ref.session_id].mutation_lock
        await async_wait_until(lambda: getattr(lock, '_waiters', None), timeout=10, what='the follower submit queued')
        release.set()
        queued = await submit
        assert (await mutation)['result']['model'] == 'switched'
        await async_wait_until(
            lambda: rt.get_session_admission(db, admission_id=queued.admission_id)['status'] == 'terminal',
            timeout=10, what='the queued follower ran after the policy was published')
        assert executed == ['FOLLOWER']
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_provider_change_drops_the_launch_key_and_keeps_config_secrets(tmp_path, monkeypatch):
    """A session created with an explicit --api-key AND secret-bearing config can change provider:
    the launch key must not cross providers, the frozen config secrets stay bound."""
    from dataclasses import asdict
    from types import SimpleNamespace
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import bind_launch_key, build_policy, restore_policy
    from hermes_state import SessionDB
    from hermes_cli import model_switch
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='b', target_provider='anthropic', provider_changed=True,
        base_url='https://api.anthropic.com'))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=db, runner=SimpleNamespace(
            _resolve_session_agent_runtime=lambda **k: (None, {'api_key': 'sk-launch'})))
        private = {}
        config = {'model': {'provider': 'openrouter', 'default': 'a'},
                  'providers': {'mine': {'api_key': 'sk-config-secret', 'base_url': 'http://127.0.0.1:9/v1'}}}
        policy = build_policy({'cwd': str(tmp_path), 'model': 'a', 'api_key': 'sk-launch'}, config, private_secrets=private)
        policy = bind_launch_key(authority, 'sid', policy, 'sk-launch', config_secrets=private)
        assert policy.credential_ref and policy.config_secret_ref
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}
        switched = restore_policy((await prepare_model(authority, SimpleNamespace(source=None, route='r'),
                                                       {'model': 'b', 'provider': 'anthropic'}, prepared))['policy'])
        assert switched.credential_ref is None, 'the launch key crossed into another provider'
        assert switched.config(authority)['providers']['mine']['api_key'] == 'sk-config-secret'


@pytest.mark.asyncio
async def test_model_pick_reads_current_provider_config_when_legacy_policy_lacks_it(tmp_path, monkeypatch):
    """Regression for #139740: a Desktop pick can outlive a policy written without ``providers``."""
    from dataclasses import asdict
    from types import SimpleNamespace
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import build_policy, restore_policy
    from hermes_state import SessionDB
    from hermes_cli import model_switch

    (tmp_path / 'config.yaml').write_text(
        'providers:\n'
        '  inkstone:\n'
        '    api: https://example.invalid/v1\n'
        '    key_env: INKSTONE_API_KEY\n'
        '    models: {glm-5.3: {context_length: 1048576}}\n')
    seen = {}

    def switch(**kwargs):
        seen.update(kwargs)
        return model_switch.ModelSwitchResult(
            success=True, new_model='glm-5.3', target_provider='inkstone',
            base_url='https://example.invalid/v1', api_mode='chat_completions')

    monkeypatch.setattr(model_switch, 'switch_model', switch)
    monkeypatch.setenv('INKSTONE_API_KEY', 'inkstone-secret')
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=db)
        runner = SimpleNamespace(session_authority=authority,
                                 _resolve_session_agent_runtime=lambda **k: (None, {}))
        authority.runner = runner
        policy = build_policy({'cwd': str(tmp_path), 'model': 'old'}, {
            'model': {'provider': 'deepseek', 'default': 'old'},
            'providers': {'stale': {'base_url': 'https://stale.example.invalid/v1'}}})
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}

        committed = await prepare_model(authority, SimpleNamespace(source=None, route='r'), {
            'model': 'glm-5.3', 'provider': 'inkstone'}, prepared)

    assert seen['user_providers']['inkstone']['key_env'] == 'INKSTONE_API_KEY'
    policy = restore_policy(committed['policy'])
    model = policy.config()['model']
    assert model['provider'] == 'inkstone'
    assert model['key_env'] == 'INKSTONE_API_KEY'
    _, runtime = _resolve_policy_agent_runtime(runner, policy)
    assert runtime['base_url'].rstrip('/') == 'https://example.invalid/v1'
    assert runtime['api_key'] == 'inkstone-secret'


@pytest.mark.asyncio
async def test_model_pick_freezes_legacy_custom_provider_for_the_next_runtime(tmp_path, monkeypatch):
    """A legacy ``custom_providers`` declaration must also survive a Desktop model pick."""
    from dataclasses import asdict
    from types import SimpleNamespace
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import build_policy, restore_policy
    from hermes_state import SessionDB
    from hermes_cli import model_switch

    (tmp_path / 'config.yaml').write_text(
        'custom_providers:\n'
        '  - name: inkstone\n'
        '    base_url: https://example.invalid/v1\n'
        '    key_env: INKSTONE_API_KEY\n', encoding='utf-8')
    seen = {}

    def switch(**kwargs):
        seen.update(kwargs)
        return model_switch.ModelSwitchResult(
            success=True, new_model='glm-5.3', target_provider='inkstone',
            base_url='https://example.invalid/v1', api_mode='chat_completions')

    monkeypatch.setattr(model_switch, 'switch_model', switch)
    monkeypatch.setenv('INKSTONE_API_KEY', 'inkstone-secret')
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=db)
        runner = SimpleNamespace(session_authority=authority,
                                 _resolve_session_agent_runtime=lambda **k: (None, {}))
        authority.runner = runner
        policy = build_policy({'cwd': str(tmp_path), 'model': 'old'}, {
            'model': {'provider': 'deepseek', 'default': 'old'}})
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}
        committed = await prepare_model(authority, SimpleNamespace(source=None, route='r'), {
            'model': 'glm-5.3', 'provider': 'inkstone'}, prepared)

    assert seen['custom_providers'][0]['name'] == 'inkstone'
    _, runtime = _resolve_policy_agent_runtime(runner, restore_policy(committed['policy']))
    assert runtime['base_url'].rstrip('/') == 'https://example.invalid/v1'
    assert runtime['api_key'] == 'inkstone-secret'


@pytest.mark.asyncio
async def test_model_pick_resolves_the_profiles_current_entry_over_a_frozen_one(tmp_path, monkeypatch):
    """A pick is an explicit user action: a slug the session froze earlier resolves to the profile's
    CURRENT entry (as CLI/TUI ``/model`` do), and only that entry is re-frozen. Another frozen entry
    the profile no longer declares stays as it was."""
    from dataclasses import asdict
    from types import SimpleNamespace
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import build_policy, restore_policy
    from hermes_state import SessionDB
    from hermes_cli import model_switch

    (tmp_path / 'config.yaml').write_text(
        'providers:\n'
        '  inkstone:\n'
        '    api: https://current.example.invalid/v1\n'
        '    key_env: CURRENT_KEY\n', encoding='utf-8')
    seen = {}

    def switch(**kwargs):
        seen.update(kwargs)
        return model_switch.ModelSwitchResult(
            success=True, new_model='new', target_provider='inkstone',
            base_url=kwargs['user_providers']['inkstone']['api'], api_mode='chat_completions')

    monkeypatch.setattr(model_switch, 'switch_model', switch)
    monkeypatch.setenv('CURRENT_KEY', 'current-secret')
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=db)
        runner = SimpleNamespace(session_authority=authority,
                                 _resolve_session_agent_runtime=lambda **k: (None, {}))
        authority.runner = runner
        policy = build_policy({'cwd': str(tmp_path), 'model': 'old'}, {
            'model': {'provider': 'deepseek', 'default': 'old'},
            'providers': {'inkstone': {'api': 'https://frozen.example.invalid/v1', 'key_env': 'FROZEN_KEY'},
                          'kept': {'api': 'https://kept.example.invalid/v1', 'key_env': 'KEPT_KEY'}}})
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}
        committed = await prepare_model(authority, SimpleNamespace(source=None, route='r'), {
            'model': 'new', 'provider': 'inkstone'}, prepared)

    assert seen['user_providers']['inkstone']['key_env'] == 'CURRENT_KEY'
    switched = restore_policy(committed['policy'])
    frozen = switched.config()
    assert frozen['providers'] == {
        'inkstone': {'api': 'https://current.example.invalid/v1', 'key_env': 'CURRENT_KEY'},
        'kept': {'api': 'https://kept.example.invalid/v1', 'key_env': 'KEPT_KEY'}}
    assert frozen['model']['key_env'] == 'CURRENT_KEY'
    _, runtime = _resolve_policy_agent_runtime(runner, switched)
    assert runtime['base_url'].rstrip('/') == 'https://current.example.invalid/v1'
    assert runtime['api_key'] == 'current-secret'


@pytest.mark.asyncio
async def test_committed_model_selection_is_the_next_resolved_runtime(tmp_path, monkeypatch):
    """custom -> custom on a new origin: the committed policy and the next resolved runtime agree on
    endpoint, protocol and credentials. The creation request's launch URL, the old endpoint's launch
    key, its key_env pointer, api_mode and context pin must not survive; a same-route re-pick keeps them."""
    from dataclasses import asdict
    from types import SimpleNamespace
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_mutation_model import prepare_model
    from gateway.session_policy import bind_launch_key, build_policy, restore_policy
    from hermes_state import SessionDB
    from hermes_cli import model_switch
    old_url, new_url = 'http://127.0.0.1:9/v1', 'http://127.0.0.2:9/v1'
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(instance_id='i', epoch=1, profile_id='p', db=db)
        runner = SimpleNamespace(session_authority=authority,
                                 _resolve_session_agent_runtime=lambda **k: (None, {'api_key': 'sk-endpoint-a'}))
        authority.runner = runner
        config = {'model': {'provider': 'custom', 'default': 'a', 'base_url': old_url, 'api_mode': 'chat_completions',
                            'context_length': 4096, 'key_env': 'ENDPOINT_A_KEY'}}
        private = {}
        policy = build_policy({'cwd': str(tmp_path), 'model': 'a', 'provider': 'custom', 'base_url': old_url,
                               'api_key': 'sk-endpoint-a'}, config, private_secrets=private)
        policy = bind_launch_key(authority, 'sid', policy, 'sk-endpoint-a', config_secrets=private)
        prepared = {'snapshot': {'receipt': {'session_id': 'sid', 'policy': asdict(policy)}}}

        async def switch(url, mode):
            monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
                success=True, new_model='b', target_provider='custom', base_url=url, api_mode=mode))
            switched = restore_policy((await prepare_model(authority, SimpleNamespace(source=None, route='r'),
                                                           {'model': 'b'}, prepared))['policy'])
            return switched, _resolve_policy_agent_runtime(runner, switched)[1]

        moved, runtime = await switch(new_url, 'anthropic_messages')
        committed = moved.config()['model']
        assert (committed['base_url'], committed['api_mode']) == (new_url, 'anthropic_messages')
        assert 'context_length' not in committed and 'key_env' not in committed
        assert moved.credential_ref is None, "endpoint A's launch key crossed to another origin"
        assert (runtime['base_url'].rstrip('/'), runtime['api_mode']) == (new_url, 'anthropic_messages')
        assert runtime['api_key'] != 'sk-endpoint-a'
        same, runtime = await switch(old_url, 'chat_completions')
        assert same.credential_ref is not None and same.config()['model']['key_env'] == 'ENDPOINT_A_KEY'
        assert (runtime['base_url'].rstrip('/'), runtime['api_key']) == (old_url, 'sk-endpoint-a')


@pytest.mark.asyncio
async def test_guarded_model_target_commits_only_with_its_confirmation_token(tmp_path, monkeypatch):
    """The shared selection guards run before the commit: an unconfirmed (or wrongly confirmed)
    guarded target writes nothing; its own token applies it exactly once; a token for another
    target or a superseded selection gets a fresh decision."""
    from types import SimpleNamespace
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import model_cost_guard, model_switch
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k['raw_input'], target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    monkeypatch.setattr(model_cost_guard, 'expensive_model_warning', lambda model, **k: (
        model_cost_guard.ExpensiveModelWarning(model, 'custom', None, None, 'test', f'{model} IS EXPENSIVE')
        if model.startswith('pricey') else None))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='cli', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    calls = iter(range(100))

    async def mutate(payload, retry=None):
        snap = db.get_session(ref.session_id)
        params = retry or {'session_id': ref.session_id, 'request_id': f'switch-{next(calls)}', 'operation': 'model',
                           'payload': payload, 'expected_revision': snap['runtime_revision'],
                           'expected_generation': snap['runtime_generation']}
        response = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': params})
        assert 'result' in response, response
        return response['result'], params

    def state():
        row = db.get_session(ref.session_id)
        return row['runtime_revision'], row['runtime_generation'], row['model'], authority.sessions[ref.session_id]
    try:
        before = state()
        refused, _ = await mutate({'model': 'pricey'})
        assert refused['status'] == 'confirmation_required' and 'pricey IS EXPENSIVE' in refused['confirm_message']
        assert 'model' not in refused, 'a refusal must not read as an applied switch'
        wrong, _ = await mutate({'model': 'pricey', 'confirm': '0' * 64})
        other, _ = await mutate({'model': 'pricey-xl', 'confirm': refused['confirm']})
        assert wrong['status'] == other['status'] == 'confirmation_required'
        assert other['confirm'] != refused['confirm'], 'a token must not carry to another target'
        assert state() == before, 'an unconfirmed selection changed revision/generation/policy'
        confirmed = {'model': 'pricey', 'confirm': refused['confirm']}
        applied, params = await mutate(confirmed)
        assert applied['model'] == 'pricey' and applied['execution_generation'] == before[1] + 1
        assert (await mutate(confirmed, params))[0] == applied  # exact retry: applied once
        assert db.get_session(ref.session_id)['runtime_generation'] == before[1] + 1
        stale, _ = await mutate(confirmed)  # same target, but the selection it replaced is gone
        assert stale['status'] == 'confirmation_required' and stale['confirm'] != refused['confirm']
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_large_context_guard_reads_the_persisted_anchor_with_no_resident_agent(tmp_path, monkeypatch):
    """The cached agent is evicted by every applied switch and absent after an owner restart, so the
    large-context guard must measure from the session's persisted usage anchor; reading only a live
    agent let the switch right after another one apply with no prompt."""
    from types import SimpleNamespace
    from agent.usage_anchor import USAGE_ANCHOR_MODEL_CONFIG_KEY, capture_usage_anchor
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import model_selection_guards, model_switch
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    monkeypatch.setattr(model_selection_guards, '_context_cache_threshold', lambda: 1000)
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k['raw_input'], target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='cli', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    try:
        db.append_message(ref.session_id, role='user', content='hello')
        db.append_message(ref.session_id, role='assistant', content='hi')
        messages = db.get_messages_as_conversation(ref.session_id)
        db.patch_session_model_config(ref.session_id, {
            USAGE_ANCHOR_MODEL_CONFIG_KEY: capture_usage_anchor(250_000, 10, messages)})
        snap = db.get_session(ref.session_id)
        response = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'switch', 'operation': 'model',
            'payload': {'model': 'other-model'}, 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation']}})
        result = response['result']
        assert result['status'] == 'confirmation_required', result
        assert 'LARGE CONTEXT' in result['confirm_message']
        assert db.get_session(ref.session_id)['model'] == snap['model']
    finally:
        await owner.close()
        store.close_all_db_handles()
