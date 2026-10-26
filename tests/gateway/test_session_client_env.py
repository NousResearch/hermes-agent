"""LC-8: a shell-launched client's allowlisted ``HERMES_*`` behaviour values ride ``session.create``
(``client_env``) and apply to that session only; an auto-started gateway stops inheriting them from
whichever client started it."""
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

_CONFIG = 'platform_toolsets: {cli: []}\nmodel: {provider: custom, default: m, base_url: http://x.invalid/v1}\n'


async def _owner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('HERMES_HOME', str(home))
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
    return owner, AuthorityConnection(owner, object(), {'user_id': 'owner'})


def _home(tmp_path, config=_CONFIG):
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(config)
    return home


def _create(owner, actor, home, request_id, **extra):
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    # A shell client attached through the gateway's own ticket (the gateway's proof it is on this host).
    ref = create_local_session(owner, actor, {'request_id': request_id, 'source': 'cli', 'cwd': str(home),
                                              'toolsets': [], **extra}, client_local=True)
    return ref, restore_policy(local_receipt(owner.db, ref.session_id)['policy'])


def _turn_prompt(owner, policy):
    """The gateway ephemeral prompt an in-process turn of *policy* resolves (inside its policy scope)."""
    from gateway.run_config_loaders import GatewayConfigLoadersMixin
    from gateway.session_policy import policy_scope
    with policy_scope(policy, authority=owner):
        return GatewayConfigLoadersMixin._load_ephemeral_system_prompt()


@pytest.mark.asyncio
async def test_each_session_runs_its_own_clients_ephemeral_prompt(tmp_path, monkeypatch):
    home = _home(tmp_path)
    # The first client started the daemon, so its export is the daemon's environment.
    monkeypatch.setenv('HERMES_EPHEMERAL_SYSTEM_PROMPT', 'FIRST')
    owner, connection = await _owner(home, monkeypatch)
    try:
        _, first = _create(owner, connection.actor, home, 'a', client_env={'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'FIRST'})
        _, second = _create(owner, connection.actor, home, 'b', client_env={'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'SECOND'})
        _, bare = _create(owner, connection.actor, home, 'c', client_env={})  # a shell that exported nothing
        assert _turn_prompt(owner, first) == 'FIRST'
        assert _turn_prompt(owner, second) == 'SECOND'
        assert _turn_prompt(owner, bare) == ''
        # The managed worker's owner-side inputs read the same per-session value.
        from gateway.run_config_loaders import GatewayConfigLoadersMixin
        from gateway.session_worker_construct import _ephemeral_prompt
        owner.runner._get_system_prompt_for_channel = (
            lambda platform, chat_id: GatewayConfigLoadersMixin._load_ephemeral_system_prompt())
        assert _ephemeral_prompt(owner, second, 'chat') == 'SECOND'
        assert _ephemeral_prompt(owner, bare, 'chat') is None
        # Messaging/cron (no local policy) and a session whose client sent nothing (Desktop, or one
        # created before this change) keep the gateway's own environment.
        assert GatewayConfigLoadersMixin._load_ephemeral_system_prompt() == 'FIRST'
        _, desktop = _create(owner, connection.actor, home, 'd')
        assert _turn_prompt(owner, desktop) == 'FIRST'
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_each_session_reads_its_own_clients_prefill_file(tmp_path, monkeypatch):
    from gateway.session_worker_construct import session_prefill
    home = _home(tmp_path)
    (home / 'one.json').write_text(json.dumps([{'role': 'user', 'content': 'ONE'}]))
    (home / 'two.json').write_text(json.dumps([{'role': 'user', 'content': 'TWO'}]))
    monkeypatch.setenv('HERMES_PREFILL_MESSAGES_FILE', str(home / 'one.json'))
    owner, connection = await _owner(home, monkeypatch)
    try:
        ref, policy = _create(owner, connection.actor, home, 'p2',
                              client_env={'HERMES_PREFILL_MESSAGES_FILE': str(home / 'two.json')})
        assert session_prefill(owner, policy, ref.session_id) == [{'role': 'user', 'content': 'TWO'}]
        ref, policy = _create(owner, connection.actor, home, 'p0', client_env={})
        assert session_prefill(owner, policy, ref.session_id) is None
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_each_session_resolves_its_own_clients_provider_request(tmp_path, monkeypatch):
    from gateway.session_policy import policy_scope
    from hermes_cli.runtime_provider import resolve_requested_provider
    home = _home(tmp_path, 'platform_toolsets: {cli: []}\nmodel: {default: m}\n')
    monkeypatch.setenv('HERMES_INFERENCE_PROVIDER', 'nous')
    owner, connection = await _owner(home, monkeypatch)
    try:
        _, mine = _create(owner, connection.actor, home, 'r1', client_env={'HERMES_INFERENCE_PROVIDER': 'openrouter'})
        _, bare = _create(owner, connection.actor, home, 'r2', client_env={})
        with policy_scope(mine, authority=owner):
            assert resolve_requested_provider() == 'openrouter'
        with policy_scope(bare, authority=owner):
            assert resolve_requested_provider() == 'auto'
        assert resolve_requested_provider() == 'nous'  # the gateway's own (messaging/cron)
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_client_env_is_filtered_validated_and_never_echoed(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        _, policy = _create(owner, connection.actor, home, 'f1', client_env={
            'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'P', 'OPENAI_API_KEY': 'sk-leak', 'HERMES_YOLO_MODE': '1',
            'NODE_CREDS': 'c', 'CONDA_DSN': 'postgres://u:p@h/db', 'NVM_DIR': '/client/.nvm'})
        assert policy.client_env() == {'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'P', 'NVM_DIR': '/client/.nvm'}
        assert 'client_env' not in json.loads(policy.request_json)  # launch_request stays value-free
        for bad in ('PATH=/x', {'PATH': 1}, {1: 'x'}):
            with pytest.raises(RuntimeStoreError) as exc:
                _create(owner, connection.actor, home, f'bad-{bad!r}', client_env=bad)
            assert exc.value.reason == 'invalid_params'
    finally:
        await connection.close()
        owner.db.close()


def _strip_new_policy_fields(db_path, session_id):
    """Rewrite a receipt into the exact shape a gateway before this change committed."""
    from hermes_state_local import POLICY_PREFIX
    conn = sqlite3.connect(db_path)
    try:
        key = POLICY_PREFIX + session_id
        receipt = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
        assert receipt['policy'].pop('client_env_json') is None
        conn.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), key))
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_exact_retry_of_a_session_created_before_client_env(tmp_path, monkeypatch):
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    params = {'request_id': 'old', 'source': 'cli', 'cwd': str(home), 'toolsets': []}
    try:
        ref = create_local_session(owner, connection.actor, dict(params))
        _strip_new_policy_fields(owner.db.db_path, ref.session_id)
        saved = local_receipt(owner.db, ref.session_id)['policy']
        assert create_local_session(owner, connection.actor, dict(params)) == ref
        assert local_receipt(owner.db, ref.session_id)['policy'] == saved
        # A retry is the same request: a client environment it never carried is a different one.
        with pytest.raises(RuntimeStoreError):
            create_local_session(owner, connection.actor, {**params, 'client_env': {}})
        new = {**params, 'request_id': 'new', 'client_env': {'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'A'}}
        ref = create_local_session(owner, connection.actor, dict(new))
        assert create_local_session(owner, connection.actor, dict(new)) == ref
        with pytest.raises(RuntimeStoreError):
            create_local_session(owner, connection.actor, {**new, 'client_env': {'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'B'}})
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_exact_retry_is_judged_by_the_committed_map_not_the_current_allowlist(tmp_path, monkeypatch):
    from gateway.session_local import create_local_session
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path, _CONFIG + 'terminal: {client_env_passthrough: [GEM_HOME]}\n')
    owner, connection = await _owner(home, monkeypatch)
    env = {'GEM_HOME': '/gems', 'BUNDLE_PATH': '/bundle', 'PATH': '/client/bin'}
    params = {'request_id': 'g', 'source': 'cli', 'cwd': str(home), 'toolsets': [], 'client_env': env}
    try:
        ref = create_local_session(owner, connection.actor, dict(params), client_local=True)
        saved = local_receipt(owner.db, ref.session_id)['policy']
        assert json.loads(saved['client_env_json']) == {'GEM_HOME': '/gems', 'PATH': '/client/bin'}
        # The profile stops listing GEM_HOME and starts listing BUNDLE_PATH: both directions of the
        # change leave the exact retry resolving to its committed session, unchanged.
        (home / 'config.yaml').write_text(_CONFIG + 'terminal: {client_env_passthrough: [BUNDLE_PATH]}\n')
        assert create_local_session(owner, connection.actor, dict(params)) == ref
        assert local_receipt(owner.db, ref.session_id)['policy'] == saved
        # A committed value changed or missing is a different request.
        for changed in ({**env, 'GEM_HOME': '/other'}, {'PATH': '/client/bin'}):
            with pytest.raises(RuntimeStoreError):
                create_local_session(owner, connection.actor, {**params, 'client_env': changed})
        # A fresh create takes the current rules.
        _, fresh = _create(owner, connection.actor, home, 'g2', client_env=env)
        assert fresh.client_env() == {'BUNDLE_PATH': '/bundle', 'PATH': '/client/bin'}
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_exact_retry_compares_the_complete_map_under_its_committed_rules(tmp_path, monkeypatch):
    """The retry, normalised by the rules committed with its session (allowlist, secret filter, locality),
    must equal the committed map: a name those rules keep added, a stored one dropped or a value changed
    is a different request; a name they would have dropped is ignored, whatever the rules are now."""
    from gateway.session_local import create_local_session
    from hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path, _CONFIG + 'terminal: {client_env_passthrough: [GEM_HOME]}\n')
    owner, connection = await _owner(home, monkeypatch)
    base = {'source': 'cli', 'cwd': str(home), 'toolsets': []}

    def create(request_id, env, local=True):
        return create_local_session(owner, connection.actor, {**base, 'request_id': request_id, 'client_env': env},
                                    client_local=local)
    env = {'GEM_HOME': '/gems', 'PATH': '/client/bin'}
    try:
        ref, empty = create('x', env), create('e', {})
        remote = create('r', {'PATH': '/a', 'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'P'}, local=False)
        (home / 'config.yaml').write_text(_CONFIG + 'terminal: {client_env_passthrough: [BUNDLE_PATH]}\n')
        assert create('x', {**env, 'BUNDLE_PATH': '/b', 'NODE_AUTH_TOKEN': 't', 'HERMES_YOLO_MODE': '1'}) == ref
        for added in ({'VIRTUAL_ENV': '/venv'}, {'NVM_DIR': '/nvm'}, {'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'P'}):
            with pytest.raises(RuntimeStoreError):
                create('x', {**env, **added})
        assert create('e', {'BUNDLE_PATH': '/b'}) == empty  # not allowlisted when 'e' committed
        for sent in ({'PATH': '/client/bin'}, {'GEM_HOME': '/g'}, 'PATH=/x'):
            with pytest.raises(RuntimeStoreError):
                create('e', sent)
        # Committed without the gateway's proof of locality: host values were dropped then, so they are now.
        assert create('r', {'PATH': '/b', 'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'P'}) == remote
        with pytest.raises(RuntimeStoreError):
            create('r', {'PATH': '/a', 'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'Q'})
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_overlapping_retries_across_an_allowlist_change_agree_with_sequential_ones(tmp_path, monkeypatch):
    """Two creates of one request_id staged on either side of an allowlist edit: the transactional commit
    judges the loser by the winner's committed rules, exactly as a sequential retry would be."""
    from gateway.session_local import stage_local_session
    from hermes_state_local import commit_local_session, local_receipt
    from hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path, _CONFIG + 'terminal: {client_env_passthrough: [GEM_HOME]}\n')
    owner, connection = await _owner(home, monkeypatch)
    env = {'GEM_HOME': '/gems', 'BUNDLE_PATH': '/bundle', 'PATH': '/client/bin'}

    def stage(sent):
        return stage_local_session(owner, connection.actor, {'request_id': 'o', 'source': 'cli', 'cwd': str(home),
                                                             'toolsets': [], 'client_env': sent}, client_local=True)
    try:
        first = stage(env)
        (home / 'config.yaml').write_text(_CONFIG + 'terminal: {client_env_passthrough: [BUNDLE_PATH]}\n')
        same, added = stage(env), stage({**env, 'VIRTUAL_ENV': '/venv'})
        assert all(isinstance(receipt, dict) for receipt in (first, same, added))  # all staged before the row
        commit_local_session(owner.db, epoch=owner.epoch, receipt=first)
        saved = local_receipt(owner.db, first['session_id'])
        assert commit_local_session(owner.db, epoch=owner.epoch, receipt=same)['policy'] == saved['policy']
        with pytest.raises(RuntimeStoreError):
            commit_local_session(owner.db, epoch=owner.epoch, receipt=added)
        assert local_receipt(owner.db, first['session_id']) == saved
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_a_copied_staged_receipt_is_never_judged_by_its_own_filtered_map(tmp_path, monkeypatch):
    """An overlapping create's own ``client_env`` is owner memory on its staged receipt. A plain copy has lost
    it, and its filtered map (an allowlist edit dropped GEM_HOME) equals the winner's: the commit refuses it."""
    from gateway.session_local import stage_local_session
    from hermes_state_local import commit_local_session
    from hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path, _CONFIG + 'terminal: {client_env_passthrough: [GEM_HOME]}\n')
    owner, connection = await _owner(home, monkeypatch)

    def stage(sent):
        return stage_local_session(owner, connection.actor, {'request_id': 'c', 'source': 'cli', 'cwd': str(home),
                                                             'toolsets': [], 'client_env': sent}, client_local=True)
    try:
        first = stage({'PATH': '/client/bin'})
        (home / 'config.yaml').write_text(_CONFIG)
        added = stage({'PATH': '/client/bin', 'GEM_HOME': '/gems'})
        commit_local_session(owner.db, epoch=owner.epoch, receipt=first)
        with pytest.raises(RuntimeStoreError):
            commit_local_session(owner.db, epoch=owner.epoch, receipt=added)
        with pytest.raises(RuntimeStoreError):
            commit_local_session(owner.db, epoch=owner.epoch, receipt=dict(added))
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_configured_toolchain_names_a_local_client_omitted_never_leak_from_the_daemon(tmp_path, monkeypatch):
    """Client A exported a configured extra (``terminal.client_env_passthrough``) and auto-started the
    gateway, seeding the daemon's environment; client B's session, which did not export it, runs its tools
    without it. PATH is the documented exception (a client that sent none keeps the gateway's)."""
    import sys
    from gateway.session_policy import policy_scope
    from tools.code_execution_env import _build_child_env
    from tools.environments.local import _make_run_env
    from tools.process_registry import ProcessRegistry
    home = _home(tmp_path, _CONFIG + 'terminal: {client_env_passthrough: [GEM_HOME]}\n')
    monkeypatch.setenv('GEM_HOME', '/alpha')
    monkeypatch.setenv('PATH', '/daemon/bin:/usr/bin:/bin')
    owner, connection = await _owner(home, monkeypatch)

    def seen(policy):
        with policy_scope(policy, authority=owner):
            envs = (_make_run_env({}), ProcessRegistry._spawn_env({}), _build_child_env(
                rpc_endpoint='s', rpc_token='t', tmpdir=str(tmp_path), child_python=sys.executable))
            backend = _make_run_env({'GEM_HOME': '/configured'})['GEM_HOME']  # terminal.env still wins
        return [env.get('GEM_HOME') for env in envs], envs[0]['PATH'].split(':')[0], backend
    try:
        _, mine = _create(owner, connection.actor, home, 'a', client_env={'GEM_HOME': '/a', 'PATH': '/client/bin'})
        _, other = _create(owner, connection.actor, home, 'b', client_env={'PATH': '/client/bin'})
        _, bare = _create(owner, connection.actor, home, 'c', client_env={})
        assert seen(mine) == (['/a'] * 3, '/client/bin', '/configured')
        assert seen(other) == ([None] * 3, '/client/bin', '/configured')
        assert seen(bare) == ([None] * 3, '/daemon/bin', '/configured')
        assert _make_run_env({})['GEM_HOME'] == '/alpha'  # messaging/cron/Desktop: the gateway's own
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_host_values_apply_only_over_a_gateway_proven_local_attach(tmp_path, monkeypatch):
    """Toolchain and path-valued values are the client's only when the gateway itself proves the attach is
    on its host: the connection redeemed a ticket minted over the gateway's private control socket
    (``run_api``: ``native_bootstrap``). Any other attach keeps the gateway's values and is told so;
    non-path behaviour values apply for any authenticated client. A client-sent flag never counts."""
    from gateway import session_policy
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import local_session_info
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_policy import policy_scope, restore_policy
    from gateway.session_worker_construct import session_prefill
    from hermes_cli.runtime_provider import resolve_requested_provider
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError
    from tools.environments.local import _make_run_env
    home = _home(tmp_path, 'platform_toolsets: {cli: []}\nmodel: {default: m}\n')  # no configured provider
    repo = tmp_path / 'repo'
    (repo / '.hermes' / 'plugins' / 'mine').mkdir(parents=True)
    for name in ('gateway', 'client'):
        (home / f'{name}.json').write_text(json.dumps([{'role': 'user', 'content': name}]))
    monkeypatch.setenv('HERMES_PREFILL_MESSAGES_FILE', str(home / 'gateway.json'))
    monkeypatch.setenv('PATH', '/daemon/bin:/usr/bin:/bin')
    monkeypatch.delenv('NVM_DIR', raising=False)
    owner, remote = await _owner(home, monkeypatch)  # an attach without the gateway's own ticket
    local = AuthorityConnection(owner, object(), {'user_id': 'owner', 'native_bootstrap': True})
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: restore_policy(
        local_receipt(owner.db, source.chat_id)['policy']))
    sent = {'PATH': '/client/bin:/usr/bin', 'NVM_DIR': '/client/nvm', 'HERMES_ENABLE_PROJECT_PLUGINS': '1',
            'HERMES_PREFILL_MESSAGES_FILE': str(home / 'client.json'),
            'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'P', 'HERMES_INFERENCE_PROVIDER': 'openrouter'}

    async def create(connection, request_id):
        snapshot = await connection.create(None, {'request_id': request_id, 'source': 'cli', 'cwd': str(repo),
                                                  'toolsets': [], 'client_env': dict(sent)})
        ref = restore_local_session(owner, snapshot['session_id'])
        policy = restore_policy(local_receipt(owner.db, ref.session_id)['policy'])
        with policy_scope(policy, authority=owner):
            run = _make_run_env({})
            seen = (run['PATH'].split(':')[0], run.get('NVM_DIR'), _turn_prompt(owner, policy),
                    resolve_requested_provider())
        return policy, seen, session_prefill(owner, policy, ref.session_id), local_session_info(owner, ref)
    try:
        policy, seen, prefill, info = await create(local, 'here')
        assert policy.client_env() == sent
        assert seen == ('/client/bin', '/client/nvm', 'P', 'openrouter')
        assert prefill == [{'role': 'user', 'content': 'client'}]
        assert 'project plugins' in info['config_warning']
        policy, seen, prefill, info = await create(remote, 'there')
        assert policy.client_env() == {'HERMES_EPHEMERAL_SYSTEM_PROMPT': 'P', 'HERMES_INFERENCE_PROVIDER': 'openrouter'}
        assert seen == ('/daemon/bin', None, 'P', 'openrouter')
        assert prefill == [{'role': 'user', 'content': 'gateway'}]
        assert 'not attached to this gateway locally' in info['config_warning']
        assert 'project plugins' not in info['config_warning']
        with pytest.raises(RuntimeStoreError):  # a client cannot claim locality for itself
            await remote.create(None, {'request_id': 'claim', 'source': 'cli', 'cwd': str(repo), 'toolsets': [],
                                       'client_env': dict(sent), 'client_local': True})
    finally:
        await local.close()
        await remote.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_runtime_describe_lists_client_env(tmp_path, monkeypatch):
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        described = await connection.describe(None, {})
        assert 'client_env' in described['session_create']['parameters']
    finally:
        await connection.close()
        owner.db.close()


def test_auto_started_gateway_does_not_inherit_client_behaviour_env(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_start
    home = tmp_path / 'home'
    home.mkdir()
    captured = {}

    def popen(command, **kwargs):
        captured.update(kwargs['env'])
        return SimpleNamespace(pid=1)
    monkeypatch.setattr(gateway_runtime_start.subprocess, 'Popen', popen)
    for name in ('HERMES_EPHEMERAL_SYSTEM_PROMPT', 'HERMES_PREFILL_MESSAGES_FILE', 'HERMES_INFERENCE_PROVIDER',
                 'HERMES_ENABLE_PROJECT_PLUGINS', 'HERMES_TUI_CLIENT_ENV'):
        monkeypatch.setenv(name, 'from-the-first-client')
    monkeypatch.setenv('PATH', '/client/bin:/usr/bin')
    gateway_runtime_start.spawn_unmanaged_gateway(home, deadline=time.monotonic() + 30)
    leaked = sorted(name for name, value in captured.items() if value == 'from-the-first-client')
    assert leaked == []
    assert captured['PATH']  # the daemon still needs a PATH of its own
