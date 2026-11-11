"""A new local session expands ``config.yaml`` ``${VAR}`` refs against the profile's CURRENT ``.env``
(the source tool env already refreshes from), not the daemon's start-time ``os.environ`` (CFG-4)."""
from types import SimpleNamespace

import pytest


async def _owner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('HERMES_HOME', str(home))  # a single-profile owner: this IS the launch home
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
    return owner, AuthorityConnection(owner, object(), {'user_id': 'owner'})


def _created_model(owner, actor, home, request_id):
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    ref = create_local_session(owner, actor, {'request_id': request_id, 'source': 'cli', 'cwd': str(home),
                                              'toolsets': []})
    return restore_policy(local_receipt(owner.db, ref.session_id)['policy']).config()['model']


@pytest.mark.asyncio
async def test_new_session_expands_config_refs_against_current_profile_env(tmp_path, monkeypatch):
    from agent.secret_scope import current_secret_scope
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'platform_toolsets: {cli: []}\n'
        'model: {provider: custom, default: m, base_url: "${CFG4_BASE_URL}", api_mode: "${CFG4_INJECTED}"}\n')
    # Gateway-start values: CFG4_BASE_URL came from .env at boot; CFG4_INJECTED only from the
    # process env (systemd / ``op run``), which a single-profile owner must keep honouring.
    monkeypatch.setenv('CFG4_BASE_URL', 'http://boot.invalid/v1')
    monkeypatch.setenv('CFG4_INJECTED', 'chat_completions')
    (home / '.env').write_text('CFG4_BASE_URL=http://rotated.invalid/v1\n')
    owner, connection = await _owner(home, monkeypatch)
    try:
        first = _created_model(owner, connection.actor, home, 'first')
        assert first['base_url'] == 'http://rotated.invalid/v1'
        assert first['api_mode'] == 'chat_completions'
        (home / '.env').write_text('CFG4_BASE_URL=http://rotated-again.invalid/v1\n')
        assert _created_model(owner, connection.actor, home, 'second')['base_url'] == \
            'http://rotated-again.invalid/v1'
        assert current_secret_scope() is None  # the read's scope never outlives the build
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('model', [None, 'alias-env'])
async def test_exact_create_retry_after_an_env_rotation_returns_the_original_session(tmp_path, monkeypatch, model):
    """A session committed before this change (config expanded against the daemon's start env) and
    retried after the profile's ``.env`` rotated: the exact retry returns the original session and its
    frozen policy, never ``invalid_params`` or a re-expanded route. Also for a ``-m`` alias launch
    whose endpoint is a ``${VAR}`` ref (the resolved route rides the frozen request)."""
    from gateway import session_local
    from gateway.run import _load_gateway_config
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_cli import model_switch
    from hermes_state_local import local_receipt
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'platform_toolsets: {cli: []}\n'
        'model: {provider: custom, default: m, base_url: "${CFG4_BASE_URL}"}\n'
        'model_aliases:\n'
        '  alias-env: {model: model-alias, provider: custom, base_url: "${CFG4_BASE_URL}"}\n')
    monkeypatch.setattr(model_switch, 'DIRECT_ALIASES', {})
    monkeypatch.setenv('CFG4_BASE_URL', 'http://boot.invalid/v1')
    (home / '.env').write_text('CFG4_BASE_URL=http://boot.invalid/v1\n')
    owner, connection = await _owner(home, monkeypatch)
    params = {'request_id': 'retry', 'source': 'cli', 'cwd': str(home), 'toolsets': [],
              **({'model': model} if model else {})}
    try:
        with monkeypatch.context() as old:  # the pre-change create: the daemon's start env
            old.setattr(session_local, 'load_profile_config', lambda home=None: _load_gateway_config())
            ref = create_local_session(owner, connection.actor, dict(params))
        saved = local_receipt(owner.db, ref.session_id)['policy']
        assert restore_policy(saved).config()['model']['base_url'] == 'http://boot.invalid/v1'
        (home / '.env').write_text('CFG4_BASE_URL=http://rotated.invalid/v1\n')
        model_switch.DIRECT_ALIASES.clear()
        assert create_local_session(owner, connection.actor, dict(params)) == ref
        assert local_receipt(owner.db, ref.session_id)['policy'] == saved
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_new_alias_session_uses_the_current_profile_env(tmp_path, monkeypatch):
    """A ``-m`` alias whose endpoint is a ``${VAR}`` ref resolves from the same fresh config snapshot
    as the rest of the policy: not the daemon's start env, and not an alias cache that only notices
    ``config.yaml`` edits."""
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_cli import model_switch
    from hermes_state_local import local_receipt
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'platform_toolsets: {cli: []}\n'
        'model: {provider: custom, default: m}\n'
        'model_aliases:\n'
        '  alias-env: {model: model-alias, provider: custom, base_url: "${CFG4_BASE_URL}"}\n')
    monkeypatch.setattr(model_switch, 'DIRECT_ALIASES', {})
    monkeypatch.setattr(model_switch, '_DIRECT_ALIAS_LOADED', None)
    monkeypatch.setattr(model_switch, '_DIRECT_ALIAS_IDENTITY', None)
    monkeypatch.setenv('CFG4_BASE_URL', 'http://boot.invalid/v1')
    (home / '.env').write_text('CFG4_BASE_URL=http://rotated.invalid/v1\n')
    owner, connection = await _owner(home, monkeypatch)

    def created(request_id):
        ref = create_local_session(owner, connection.actor, {
            'request_id': request_id, 'source': 'cli', 'cwd': str(home), 'toolsets': [], 'model': 'alias-env'})
        policy = restore_policy(local_receipt(owner.db, ref.session_id)['policy'])
        return policy.model, policy.config()['model']['base_url']
    try:
        assert created('first') == ('model-alias', 'http://rotated.invalid/v1')
        (home / '.env').write_text('CFG4_BASE_URL=http://rotated-again.invalid/v1\n')
        assert created('second') == ('model-alias', 'http://rotated-again.invalid/v1')
    finally:
        await connection.close()
        owner.db.close()


def test_bound_scope_is_kept(tmp_path, monkeypatch):
    """A served profile's owner scope (multiplex) is already the right source: never rebuilt from
    another home."""
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from gateway.session_local import load_profile_config
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text('model: {base_url: "${CFG4_BASE_URL}"}\n')
    (home / '.env').write_text('CFG4_BASE_URL=http://file.invalid/v1\n')
    token = set_secret_scope({'CFG4_BASE_URL': 'http://scope.invalid/v1'}, profile_home=str(home))
    try:
        assert load_profile_config(home)['model']['base_url'] == 'http://scope.invalid/v1'
    finally:
        reset_secret_scope(token)
