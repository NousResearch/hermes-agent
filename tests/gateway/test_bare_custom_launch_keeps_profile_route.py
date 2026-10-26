"""A bare ``custom`` launch pick stays on the profile's own route whenever that route can serve it."""
import pytest

LOCAL, RELAY = 'http://127.0.0.1:8000/v1', 'https://relay.example.com/v1'


def _launch(tmp_path, config, launch_key=None, **pick):
    """session.create's route steps, then the runtime the first turn resolves from the frozen policy."""
    from gateway.session_local_route import resolve_launch_route
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import route_runtime
    params = {'cwd': str(tmp_path), 'toolsets': [], 'provider': 'custom', **pick}
    params = resolve_launch_route({**params, **({'api_key': launch_key} if launch_key else {})}, config)
    policy = build_policy(params, config, profile_terminal=False)
    frozen, model = policy.effective()
    runtime, _ = route_runtime(policy, frozen, model, launch_key, fallback=False)
    return frozen['model'], runtime


BARE_LOCAL = {'model': {'provider': 'custom', 'base_url': LOCAL, 'default': 'qwen3-32b', 'key_env': 'LOCAL_KEY',
                        'context_length': 32768},
              'providers': {'relay': {'api': RELAY, 'key_env': 'RELAY_KEY', 'models': ['qwen3-32b']}}}


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv('LOCAL_KEY', 'local-key')
    monkeypatch.setenv('RELAY_KEY', 'relay-key')
    monkeypatch.delenv('CUSTOM_BASE_URL', raising=False)


@pytest.mark.parametrize('pick, key', [
    ({'model': 'qwen3-32b'}, 'local-key'),                             # --provider custom -m / a Desktop pick
    ({'model': 'qwen3-32b', 'base_url': LOCAL}, 'local-key'),          # ... with the profile's own --base-url
    ({'model': 'qwen3-32b', 'launch_key': 'sk-launch'}, 'sk-launch'),  # ... with --api-key
], ids=['no-url', 'own-url', 'launch-key'])
def test_a_bare_custom_pick_on_a_bare_custom_profile_stays_on_its_endpoint(tmp_path, pick, key):
    model_cfg, runtime = _launch(tmp_path, BARE_LOCAL, **pick)
    assert (runtime['base_url'].rstrip('/'), runtime['api_key'], model_cfg.get('context_length')) == (
        LOCAL, key, 32768)


@pytest.mark.parametrize('pick', [{}, {'base_url': LOCAL}], ids=['no-url', 'own-url'])
def test_an_entry_on_the_profiles_own_url_does_not_take_over_its_pins(tmp_path, pick):
    config = {**BARE_LOCAL, 'providers': {'box': {'api': LOCAL, 'key_env': 'RELAY_KEY', 'default_model': 'qwen3-32b'}}}
    model_cfg, runtime = _launch(tmp_path, config, model='qwen3-32b', **pick)
    assert (runtime['api_key'], model_cfg.get('context_length')) == ('local-key', 32768)


def test_a_model_only_another_entry_lists_leaves_the_profiles_endpoint(tmp_path):
    config = {**BARE_LOCAL, 'providers': {'relay': {'api': RELAY, 'key_env': 'RELAY_KEY', 'models': ['relay-only']}}}
    _cfg, runtime = _launch(tmp_path, config, model='relay-only')
    assert (runtime['base_url'].rstrip('/'), runtime['api_key']) == (RELAY, 'relay-key')


ENDPOINTLESS = {'model': {'provider': 'custom', 'default': 'qwen3-32b'},
                'providers': {'box': {'api': LOCAL, 'key_env': 'LOCAL_KEY', 'models': ['qwen3-32b']}}}


def test_an_endpointless_bare_custom_profile_still_finds_the_entry_serving_the_model(tmp_path):
    _cfg, runtime = _launch(tmp_path, ENDPOINTLESS, model='qwen3-32b')
    assert (runtime['base_url'].rstrip('/'), runtime['api_key']) == (LOCAL, 'local-key')


def test_a_launch_key_on_an_endpointless_bare_custom_profile_gets_the_model_only_refusal(tmp_path):
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError) as refused:
        _launch(tmp_path, ENDPOINTLESS, 'sk-launch', model='qwen3-32b')
    assert (refused.value.reason, refused.value.field) == ('invalid_params', 'api_key')


SHADOWED = {'model': {'provider': 'custom:local', 'default': 'local-only', 'base_url': LOCAL, 'key_env': 'LOCAL_KEY',
                      'context_length': 32768},
            'providers': {'relay': {'api': RELAY, 'default_model': 'relay-only', 'key_env': 'RELAY_KEY'},
                          'local': {'api': LOCAL, 'default_model': 'local-only', 'key_env': 'LOCAL_KEY'}},
            # Same name as providers.local, which runtime resolution gives the slug to.
            'custom_providers': [{'name': 'local', 'base_url': RELAY, 'model': 'relay-only', 'key_env': 'RELAY_KEY'}]}


@pytest.mark.parametrize('pick', [{'base_url': RELAY}, {}], ids=['relay-url', 'no-url'])
def test_a_shadowed_legacy_entry_does_not_win_the_profile_preference(tmp_path, pick):
    model_cfg, runtime = _launch(tmp_path, SHADOWED, model='relay-only', **pick)
    assert (runtime['base_url'].rstrip('/'), runtime['api_key'], model_cfg.get('context_length')) == (
        RELAY, 'relay-key', None)


def test_an_endpointless_providers_block_does_not_hide_the_legacy_entry_of_that_name(tmp_path):
    config = {'model': {'provider': 'openrouter', 'default': 'anthropic/claude-sonnet-4'},
              # Tuning only: runtime resolution skips it and runs the legacy entry.
              'providers': {'bedrock': {'stale_timeout_seconds': 600}},
              'custom_providers': [{'name': 'bedrock', 'base_url': RELAY, 'model': 'relay-only', 'key_env': 'RELAY_KEY'}]}
    _cfg, runtime = _launch(tmp_path, config, model='relay-only')
    assert (runtime['base_url'].rstrip('/'), runtime['api_key']) == (RELAY, 'relay-key')


def test_a_bare_custom_pick_naming_another_endpoint_uses_that_entrys_own_key(tmp_path):
    _cfg, runtime = _launch(tmp_path, BARE_LOCAL, model='qwen3-32b', base_url=RELAY)
    assert (runtime['base_url'].rstrip('/'), runtime['api_key']) == (RELAY, 'relay-key')


def test_a_named_profile_keeps_its_entry_when_it_serves_the_model(tmp_path):
    config = {'model': {'provider': 'custom:local', 'default': 'qwen3-32b'},
              'providers': {'relay': {'api': RELAY, 'key_env': 'RELAY_KEY', 'models': ['qwen3-32b']},
                            'local': {'api': LOCAL, 'key_env': 'LOCAL_KEY', 'default_model': 'qwen3-32b'}}}
    _cfg, runtime = _launch(tmp_path, config, model='qwen3-32b')
    assert (runtime['base_url'].rstrip('/'), runtime['api_key']) == (LOCAL, 'local-key')


def test_a_launch_key_is_refused_for_an_entry_only_the_model_chose(tmp_path):
    """Only another entry lists the model: the gateway would guess its host, so the user's key is not sent there."""
    from hermes_state_runtime import RuntimeStoreError
    config = {'model': {'provider': 'custom:local', 'default': 'qwen3-32b'},
              'providers': {'relay': {'api': RELAY, 'key_env': 'RELAY_KEY', 'models': ['relay-only']},
                            'local': {'api': LOCAL, 'key_env': 'LOCAL_KEY', 'default_model': 'qwen3-32b'}}}
    with pytest.raises(RuntimeStoreError) as refused:
        _launch(tmp_path, config, model='relay-only', launch_key='sk-launch')
    assert (refused.value.reason, refused.value.field) == ('invalid_params', 'api_key')


def test_a_kanban_custom_provider_override_stays_on_the_profiles_endpoint(tmp_path, monkeypatch):
    from contextlib import closing
    from types import SimpleNamespace
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='t', assignee='default', workspace_kind='dir',
                                 workspace_path=str(tmp_path), model_override='qwen3-32b', provider_override='custom')
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), native_owner=True,
                                 actor=Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport'))
    policy, _secrets = build_kanban_policy(connection, dict(board='owned', task_id=task.id, run_id=task.current_run_id,
                                                            claim_lock=task.claim_lock), BARE_LOCAL)
    model_cfg = policy.effective()[0]['model']
    assert (model_cfg['base_url'], model_cfg.get('key_env'), model_cfg.get('context_length')) == (
        LOCAL, 'LOCAL_KEY', 32768)


@pytest.mark.asyncio
async def test_a_session_create_with_a_bare_custom_pick_stays_on_the_profiles_endpoint(monkeypatch):
    """The real ``session.create`` (``stage_local_session``) and the turn's runtime resolver."""
    import json
    from gateway import run
    from hermes_constants import get_hermes_home
    from tests.gateway.test_session_follow_profile_config import _owner, _route
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    (home / 'config.yaml').write_text(json.dumps(BARE_LOCAL))
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        response = await owner.dispatch({'id': 'c', 'method': 'session.create', 'params': dict(
            request_id='c', source='cli', cwd=str(home), toolsets=[], provider='custom', model='qwen3-32b')})
        assert 'result' in response, response
        live = authority.sessions[response['result']['session_id']]
        assert _route(runner, live) == ('qwen3-32b', LOCAL, 'local-key')
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_adopting_a_legacy_custom_row_recovers_the_entry_serving_its_model(monkeypatch):
    """Real ``adopt_legacy_session``: the row recorded bare ``custom``, no endpoint, and a model only another entry lists."""
    import json
    from gateway import run
    from gateway.session_local_migration import resolve_local_target
    from hermes_constants import get_hermes_home
    from tests.gateway.test_session_follow_profile_config import _owner, _route
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'base_url': LOCAL, 'default': 'qwen3-32b', 'key_env': 'LOCAL_KEY'},
        'providers': {'relay': {'api': RELAY, 'key_env': 'RELAY_KEY', 'models': ['saved-model']}}}))
    store, runner, authority, owner = _owner(home)
    runner._peek_session_state = lambda key: None
    try:
        authority.db.create_session('old-tui', cwd=str(home), source='tui', model='saved-model',
                                    model_config={'provider': 'custom'})
        authority.db.end_session('old-tui', 'tui_close')
        authority._native_legacy_transports = {owner.actor.transport_id: owner.actor}
        live = authority.sessions[resolve_local_target(authority, owner.actor, 'old-tui').session_id]
        assert _route(runner, live) == ('saved-model', RELAY, 'relay-key')
    finally:
        await owner.close()
        store.close_all_db_handles()
