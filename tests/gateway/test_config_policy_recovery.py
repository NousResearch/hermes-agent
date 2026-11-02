"""Config credentials borrow only their frozen profile source across restart."""
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest


def test_config_credentials_recover_only_exact_source(tmp_path, monkeypatch):
    from gateway.session_policy import build_policy, bind_launch_key
    from hermes_state_runtime import RuntimeStoreError
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    cfg = {'model': {'provider': 'custom', 'api_key': 'owned-config-token',
                     'base_url': 'http://127.0.0.1:1234/v1'}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    def owner(path=home, profile='owned', instance='new'):
        return SimpleNamespace(db=SimpleNamespace(db_path=path / 'state.db'),
                               profile_id=profile, instance_id=instance, epoch=2)
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    policy = bind_launch_key(owner(instance='old'), 'session', policy, None, config_secrets=private)
    assert policy.config(owner())['model']['api_key'] == cfg['model']['api_key']
    assert cfg['model']['api_key'] not in json.dumps(asdict(policy))
    for wrong, mismatched in ((owner(profile='wrong'), 'profile'), (owner(path=tmp_path), 'home')):
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
            policy.config(wrong)
        assert refused.value.field == mismatched
    # A whole-policy stamp that predates the current snapshot (a profile edit rewrote the frozen
    # config after the reference froze -- a repointed base_url, keys unchanged) degrades to
    # per-entry verification instead of stranding the session: the recovered values are still
    # fingerprint-gated against the live config.yaml.
    stale = replace(policy, model='other-policy')
    assert stale.config(owner())['model']['api_key'] == cfg['model']['api_key']
    cfg['model']['api_key'] = 'changed-unrelated-token'
    (home / 'config.yaml').write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
        stale.config(owner())
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
        policy.config(owner())



@pytest.mark.asyncio
async def test_config_creation_retry_does_not_rebind_current_credentials(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway import run
    config = {'model': {'api_key': 'original-private'}, 'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    def runner():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        return SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    first = await initialize_session_authority(runner(), profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create'}), 'socket')
    params = dict(request_id='retry', cwd=str(tmp_path), model='frozen')
    ref = create_local_session(first, actor, params)
    original = first.runner.adapters[Platform.LOCAL].policies[ref.session_id]
    cold = await initialize_session_authority(runner(), profile_id='fixture', instance_id='cold')
    config['model']['api_key'] = 'changed-private'
    assert create_local_session(cold, actor, params) == ref
    assert cold.runner.adapters[Platform.LOCAL].policies[ref.session_id] == original
    assert not getattr(cold, '_local_config_secrets', {})
    assert not getattr(cold, '_local_launch_keys', {})


def test_dropped_top_level_setting_does_not_wedge_the_session(tmp_path, monkeypatch):
    """A profile that stops defining a frozen top-level setting (an env-routed key the user moved to
    .env, where ``hermes config set`` writes it) leaves the session running on what its frozen tree
    holds instead of failing every read of the chat with ``launch_credentials_unavailable``. A
    ROUTE credential that disappeared still refuses, and a settings-only read never needs either."""
    from gateway.session_policy import build_policy, bind_launch_key
    from hermes_state_runtime import RuntimeStoreError
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    cfg = {'API_SERVER_KEY': 'owned-api-server-token',
           'model': {'provider': 'custom', 'api_key': 'route-token', 'base_url': 'http://127.0.0.1:1234/v1'}}
    config_path = home / 'config.yaml'
    config_path.write_text(json.dumps(cfg))

    def owner():
        # A fresh authority per read, like a restarted owner: no in-memory secret cache.
        return SimpleNamespace(db=SimpleNamespace(db_path=home / 'state.db'),
                               profile_id='owned', instance_id='new', epoch=2)
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    assert set(private) == {('API_SERVER_KEY',), ('model', 'api_key')}
    policy = bind_launch_key(owner(), 'session', policy, None, config_secrets=private)
    assert cfg['API_SERVER_KEY'] not in json.dumps(asdict(policy))
    assert policy.config(owner())['API_SERVER_KEY'] == cfg['API_SERVER_KEY']

    del cfg['API_SERVER_KEY']  # the user moves the env-routed key to .env
    config_path.write_text(json.dumps(cfg))
    config = policy.config(owner())
    assert config['model']['api_key'] == 'route-token'
    assert config.get('API_SERVER_KEY') is None

    del cfg['model']['api_key']  # the session's own route credential is gone: still refused
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
        policy.config(owner())
    settings = policy.config(owner(), secrets=False)  # a display/model-option read needs no secret
    assert settings['model']['provider'] == 'custom' and settings['model']['api_key'] is None


@pytest.mark.parametrize('edit', ['change', 'remove'])
def test_an_unrelated_secret_edit_keeps_the_route_and_a_gone_route_key_is_named(tmp_path, monkeypatch, edit):
    """CFG-1: editing ANY config secret (here an MCP server's env token) used to strand every older
    session after a restart. Only the session's own route credential may refuse, and that refusal
    names the credential; the unrelated secret is simply no longer supplied, never swapped."""
    from gateway.session_policy import build_policy, bind_launch_key
    from hermes_state_runtime import RuntimeStoreError
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    cfg = {'model': {'provider': 'custom', 'api_key': 'route-token', 'base_url': 'http://127.0.0.1:1234/v1'},
           'mcp_servers': {'docs': {'command': 'docs-mcp', 'env': {'DOCS_TOKEN': 'old-mcp-token'}}}}
    config_path = home / 'config.yaml'
    config_path.write_text(json.dumps(cfg))

    def owner():  # a fresh authority per read, like a restarted owner
        return SimpleNamespace(db=SimpleNamespace(db_path=home / 'state.db'),
                               profile_id='owned', instance_id='new', epoch=2)
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    policy = bind_launch_key(owner(), 'session', policy, None, config_secrets=private)

    if edit == 'change':
        cfg['mcp_servers']['docs']['env']['DOCS_TOKEN'] = 'rotated-mcp-token'
    else:
        del cfg['mcp_servers']['docs']['env']
    config_path.write_text(json.dumps(cfg))
    config = policy.config(owner())
    assert config['model']['api_key'] == 'route-token'
    assert config['mcp_servers']['docs']['env'] is None  # neither the old nor a swapped-in token

    if edit == 'change':
        cfg['model']['api_key'] = 'another-route-token'
    else:
        del cfg['model']['api_key']
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
        policy.config(owner())
    assert refused.value.field == 'model.api_key'
    assert 'route-token' not in str(refused.value.detail) and 'another' not in str(refused.value.detail)


def _bound_session(tmp_path, monkeypatch, cfg):
    """A session frozen on *cfg* as the gateway loads it (``${VAR}`` expanded), plus a
    fresh-authority factory (a restarted owner, no secret cache)."""
    from gateway.run import _load_gateway_config
    from gateway.session_policy import build_policy, bind_launch_key
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text(json.dumps(cfg))
    cfg = _load_gateway_config(home / 'config.yaml')

    def owner():
        return SimpleNamespace(db=SimpleNamespace(db_path=home / 'state.db'),
                               profile_id='owned', instance_id='new', epoch=2)
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    owner.running = owner()  # the owner that created the session: it holds the frozen secrets in memory
    return bind_launch_key(owner.running, 'session', policy, None, config_secrets=private), owner, home / 'config.yaml'


def test_an_unchanged_custom_providers_key_survives_a_restart(tmp_path, monkeypatch):
    """A route key saved under a list (``custom_providers[0].api_key``) is found again after a
    restart; it used to read as removed and refuse every turn of the session."""
    from hermes_state_runtime import RuntimeStoreError
    cfg = {'model': {'provider': 'custom:local', 'default': 'm'},
           'custom_providers': [{'name': 'Local', 'base_url': 'http://127.0.0.1:1234/v1', 'api_key': 'cp-token'}]}
    policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
    assert policy.config(owner())['custom_providers'][0]['api_key'] == 'cp-token'

    cfg['custom_providers'][0]['api_key'] = 'rotated-token'
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError) as refused:
        policy.config(owner())
    assert refused.value.field == 'custom_providers.0.api_key'
    assert 'changed in' in refused.value.detail


def test_only_the_entry_the_runtime_selects_is_the_route(tmp_path, monkeypatch):
    """Two ``custom_providers`` entries share a name: the runtime routes through the FIRST, so
    rotating the unused second key leaves the session running (that key is simply not supplied);
    rotating the selected one refuses and names it."""
    from hermes_state_runtime import RuntimeStoreError
    cfg = {'model': {'provider': 'custom:shared', 'default': 'm'},
           'custom_providers': [
               {'name': 'Shared', 'base_url': 'http://127.0.0.1:1111/v1', 'api_key': 'first-token'},
               {'name': 'Shared', 'base_url': 'http://127.0.0.1:2222/v1', 'api_key': 'second-token'}]}
    policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)

    cfg['custom_providers'][1]['api_key'] = 'rotated-second'
    config_path.write_text(json.dumps(cfg))
    config = policy.config(owner())
    assert config['custom_providers'][0]['api_key'] == 'first-token'
    assert config['custom_providers'][1]['api_key'] is None

    cfg['custom_providers'][0]['api_key'] = 'rotated-first'
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError) as refused:
        policy.config(owner())
    assert refused.value.field == 'custom_providers.0.api_key'


def test_a_custom_prefixed_providers_key_is_the_route(tmp_path, monkeypatch):
    """``providers['custom:foo']`` is what ``model.provider: custom:foo`` resolves to at runtime; a
    rotated key there refuses by name instead of being silently dropped."""
    from hermes_state_runtime import RuntimeStoreError
    cfg = {'model': {'provider': 'custom:foo', 'default': 'm'},
           'providers': {'custom:foo': {'base_url': 'http://127.0.0.1:1234/v1', 'api_key': 'foo-token'}}}
    policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
    assert policy.config(owner())['providers']['custom:foo']['api_key'] == 'foo-token'

    cfg['providers']['custom:foo']['api_key'] = 'rotated-foo'
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError) as refused:
        policy.config(owner())
    assert refused.value.field == 'providers.custom:foo.api_key'


def test_a_bare_custom_key_selected_by_url_is_the_route(tmp_path, monkeypatch):
    """A bare ``model.provider: custom`` with a ``base_url`` authenticates with the
    ``custom_providers`` entry owning that URL. Rotating that key and recovering cold used to skip
    the frozen value as an unrelated secret, so the pool seeded the rotated key from live config;
    it now refuses by name, like a named route."""
    from hermes_state_runtime import RuntimeStoreError
    url = 'http://127.0.0.1:1234/v1'
    cfg = {'model': {'provider': 'custom', 'base_url': url, 'default': 'm'},
           'custom_providers': [
               {'name': 'Other', 'base_url': 'http://127.0.0.1:9999/v1', 'api_key': 'other-token'},
               {'name': 'Local', 'base_url': url + '/', 'api_key': 'url-token'}]}
    policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
    assert policy.config(owner())['custom_providers'][1]['api_key'] == 'url-token'

    cfg['custom_providers'][0]['api_key'] = 'rotated-other'  # not the route: simply not supplied
    config_path.write_text(json.dumps(cfg))
    assert policy.config(owner())['custom_providers'][0]['api_key'] is None

    cfg['custom_providers'][1]['api_key'] = 'rotated-url-token'
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError) as refused:
        policy.config(owner())
    assert refused.value.field == 'custom_providers.1.api_key'
    assert 'rotated' not in str(refused.value.detail)


def _policy_runtime(policy, owner):
    """The real gateway resolution a frozen session's turn runs (primary, then its fallback chain)."""
    from gateway.run_turn_prepare import _resolve_policy_agent_runtime
    from gateway.session_policy import policy_scope
    authority = owner()
    runner = SimpleNamespace(session_authority=authority)
    with policy_scope(policy, authority=authority):
        return _resolve_policy_agent_runtime(runner, policy)


def _send(model, runtime):
    from openai import OpenAI
    OpenAI(api_key=runtime['api_key'], base_url=runtime['base_url'], max_retries=0).chat.completions.create(
        model=model, messages=[{'role': 'user', 'content': 'hi'}])


@pytest.mark.parametrize('form', ['literal', 'reference', 'named'])
def test_a_changed_fallback_secret_removes_that_fallback(tmp_path, monkeypatch, caplog, form):
    """The primary has no credential, so the turn walks the fallback chain. A changed frozen fallback
    key used to leave the entry with ``api_key: null``, which means 'resolve another credential': the
    fallback then sent the provider's OTHER account key (or the rotated key re-seeded from live
    config). A changed fallback secret now removes that fallback; no credential is substituted."""
    from hermes_cli.auth import AuthError
    from tests.fakes.fake_llm_provider import FakeLLMServer
    for name in ('ANTHROPIC_API_KEY', 'ANTHROPIC_TOKEN', 'CLAUDE_CODE_OAUTH_TOKEN'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('BACKUP_PROBE_KEY', 'sk-original-fallback')
    with FakeLLMServer() as primary, FakeLLMServer() as backup:
        cfg = {'model': {'provider': 'anthropic', 'default': 'primary-model', 'base_url': primary.base_url},
               'providers': {'box': {'base_url': backup.base_url, 'api_key': 'sk-different-account'}},
               'fallback_providers': [{'provider': 'custom:box', 'model': 'backup-model'}]}
        if form == 'literal':
            cfg['fallback_providers'][0]['api_key'] = 'sk-original-fallback'
        elif form == 'reference':
            cfg['fallback_providers'][0]['api_key'] = '${BACKUP_PROBE_KEY}'
        else:
            cfg['providers']['box']['api_key'] = 'sk-original-fallback'
        policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
        model, runtime = _policy_runtime(policy, owner)  # unchanged: the frozen fallback serves
        assert (model, runtime['api_key']) == ('backup-model', 'sk-original-fallback')

        if form == 'literal':
            cfg['fallback_providers'][0]['api_key'] = 'sk-rotated-fallback'
        elif form == 'reference':
            monkeypatch.setenv('BACKUP_PROBE_KEY', 'sk-rotated-fallback')
        else:
            cfg['providers']['box']['api_key'] = 'sk-rotated-fallback'
        config_path.write_text(json.dumps(cfg))
        assert policy.config(owner())['fallback_providers'] == []
        try:
            model, runtime = _policy_runtime(policy, owner)
        except AuthError:
            pass  # the primary's own failure: no fallback left to try
        else:
            _send(model, runtime)
        sent = [r['auth'] for r in primary.requests + backup.requests]
        assert not any('sk-different-account' in a or 'sk-rotated' in a for a in sent), sent
        assert backup.requests == []
    assert any('fallback' in r.getMessage() and 'removed' in r.getMessage() for r in caplog.records)


def test_a_changed_auxiliary_secret_returns_that_task_to_the_primary_route(tmp_path, monkeypatch, caplog):
    """A frozen ``auxiliary.compression`` key rotated in config.yaml used to be dropped while the
    auxiliary client read live YAML and sent the ROTATED key. The task's override now leaves the
    session's effective policy: compression runs on the session's primary route with its own key."""
    from agent.auxiliary_client import get_text_auxiliary_client, reset_runtime_main, set_runtime_main
    from gateway.session_policy import policy_scope
    from tests.fakes.fake_llm_provider import FakeLLMServer
    with FakeLLMServer() as main, FakeLLMServer() as aux:
        cfg = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': main.base_url,
                         'api_key': 'sk-main-route'},
               'auxiliary': {'compression': {'provider': 'custom', 'model': 'aux-model',
                                             'base_url': aux.base_url, 'api_key': 'sk-original-aux'}}}
        policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)

        def compress():
            authority = owner()
            with policy_scope(policy, authority=authority):
                _model, runtime = _policy_runtime(policy, owner)
                token = set_runtime_main(runtime['provider'], 'primary-model', base_url=runtime['base_url'],
                                         api_key=runtime['api_key'], api_mode=runtime.get('api_mode') or '')
                try:
                    client, model = get_text_auxiliary_client('compression')
                    client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': 'hi'}])
                finally:
                    reset_runtime_main(token)
        compress()  # unchanged: the frozen auxiliary route serves with its frozen key
        assert [r['auth'] for r in aux.requests] == ['Bearer sk-original-aux']

        cfg['auxiliary']['compression']['api_key'] = 'sk-rotated-aux'
        config_path.write_text(json.dumps(cfg))
        compress()
        sent = [r['auth'] for r in main.requests + aux.requests]
        assert not any('sk-rotated-aux' in a for a in sent), sent
        assert len(aux.requests) == 1  # nothing more reached the auxiliary endpoint
        assert main.requests[-1]['auth'] == 'Bearer sk-main-route'
    assert any('auxiliary' in r.getMessage() and 'primary route' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('phase', ['warm', 'cold'])
def test_a_rotated_bare_custom_key_is_never_sent(tmp_path, monkeypatch, phase):
    """A bare ``custom`` route whose key comes from the URL-matched ``custom_providers`` entry. After
    that key is rotated in config.yaml, neither the running owner (warm: its in-process secret cache)
    nor a restarted one (cold) sends the rotated key: the credential pool seeds from the session's
    frozen config, never live config, and cold recovery refuses the changed route key by name."""
    from hermes_state_runtime import RuntimeStoreError
    from tests.fakes.fake_llm_provider import FakeLLMServer
    with FakeLLMServer() as llm:
        cfg = {'model': {'provider': 'custom', 'default': 'primary-model', 'base_url': llm.base_url},
               'custom_providers': [{'name': 'box', 'base_url': llm.base_url, 'api_key': 'sk-original-route'}]}
        policy, owner, config_path = _bound_session(tmp_path, monkeypatch, cfg)
        running = owner.running  # warm: the in-process owner still running the session
        _send(*_policy_runtime(policy, lambda: running))  # before the rotation: the frozen key
        assert [r['auth'] for r in llm.requests] == ['Bearer sk-original-route']

        cfg['custom_providers'][0]['api_key'] = 'sk-rotated-route'
        config_path.write_text(json.dumps(cfg))
        authority = running if phase == 'warm' else owner()
        try:
            _send(*_policy_runtime(policy, lambda: authority))
        except RuntimeStoreError as refused:
            assert phase == 'cold' and refused.field == 'custom_providers.0.api_key'
        sent = [r['auth'] for r in llm.requests]
        assert not any('sk-rotated-route' in a for a in sent), sent
