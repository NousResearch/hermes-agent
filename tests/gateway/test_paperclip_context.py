"""Private Paperclip launch context: never a durable client environment."""
import asyncio
import json
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from persistence_assembly import close_stores_under


def context(tag='a', **extra):
    return {'PAPERCLIP_API_URL': 'http://127.0.0.1:3100', 'PAPERCLIP_API_KEY': 'fixture-key-' + tag,
            'PAPERCLIP_AGENT_ID': tag, 'PAPERCLIP_COMPANY_ID': 'company', 'PAPERCLIP_RUN_ID': 'run-' + tag, **extra}


def test_bound_child_builders_and_no_daemon_inheritance(monkeypatch, tmp_path):
    from agent.paperclip_context import paperclip_scope, child_environ
    from tools.environments.local import _make_run_env
    monkeypatch.setattr('pm.environments._payload_manifest', lambda root: None)
    monkeypatch.setattr('tools.environments.local._user_tools_first', lambda p: p)
    monkeypatch.setattr('tools.environments.local._prepend_hermes_bin_dir', lambda p: p)
    from tools.code_execution_env import _build_child_env
    monkeypatch.setenv('PAPERCLIP_API_KEY', 'daemon-token')
    monkeypatch.setenv('PAPERCLIP_AGENT_JWT_SECRET', 'server-secret')
    for values in (context(), {}, context('b')):
        with paperclip_scope(values):
            envs = [child_environ(), _make_run_env({}), _build_child_env(
                rpc_endpoint='sock', rpc_token='fixture', tmpdir=str(tmp_path), child_python=sys.executable)]
            for env in envs:
                assert {k: v for k, v in env.items() if k.startswith('PAPERCLIP_')} == values
                result = subprocess.run([sys.executable, '-c',
                    'import os; assert os.environ.get("PAPERCLIP_AGENT_ID", "") == ' + repr(values.get('PAPERCLIP_AGENT_ID', ''))],
                    env=env, capture_output=True, check=False)
                assert result.returncode == 0
    assert not any(k.startswith('PAPERCLIP_') for k in child_environ())


def test_opaque_run_key_is_redacted_from_real_child_output_even_after_turn(monkeypatch, tmp_path):
    from agent.paperclip_context import paperclip_scope
    from agent.redact import clear_vault_redaction_values, redact_sensitive_text, redact_terminal_output
    from tools.code_execution_env import _build_child_env
    from tools.environments.local import _make_run_env
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'redaction-home'))
    monkeypatch.setattr('agent.redact._redact_enabled', lambda: False)
    monkeypatch.setattr('pm.environments._payload_manifest', lambda root: None)
    monkeypatch.setattr('tools.environments.local._user_tools_first', lambda p: p)
    monkeypatch.setattr('tools.environments.local._prepend_hermes_bin_dir', lambda p: p)
    opaque = 'opaque-paperclip-private-value-without-a-known-token-prefix'
    try:
        with paperclip_scope(context(PAPERCLIP_API_KEY=opaque)):
            envs = [_make_run_env({}), _build_child_env(rpc_endpoint='unused', rpc_token='fixture',
                    tmpdir=str(tmp_path), child_python=sys.executable)]
            for env in envs:
                child = subprocess.run([sys.executable, '-c', 'import os;print(os.environ["PAPERCLIP_API_KEY"])'],
                                       env=env, capture_output=True, text=True, check=True)
                assert child.stdout.strip() == opaque
                assert opaque not in redact_sensitive_text(child.stdout)
                assert opaque not in redact_terminal_output(child.stdout, 'printenv PAPERCLIP_API_KEY')
        # Background/log readers may drain after the authorizing turn has ended.
        assert opaque not in redact_sensitive_text(opaque)
    finally:
        clear_vault_redaction_values()


def test_collect_exact_allowlist_and_client_env_cannot_smuggle():
    from agent.paperclip_context import collect_context
    from agent.session_client_env import collect_client_env
    values = context(PAPERCLIP_AGENT_JWT_SECRET='server', PAPERCLIP_DB_PASSWORD='db')
    assert collect_context(values) == context()
    assert collect_client_env(values, {'terminal': {'client_env_passthrough': list(values)}}) == {}


@pytest.mark.asyncio
async def test_concurrent_scopes_are_isolated():
    from agent.paperclip_context import paperclip_scope, child_environ
    async def read(values):
        with paperclip_scope(values):
            await asyncio.sleep(0)
            return await asyncio.to_thread(lambda: {k: v for k, v in child_environ().items() if k.startswith('PAPERCLIP_')})
    assert await asyncio.gather(*(read(v) for v in [context(), context('b'), {}])) == [context(), context('b'), {}]


def test_validate_context_rejects_unknown_names_and_oversize_and_partial():
    from agent.paperclip_context import validate_context
    with pytest.raises(ValueError):
        validate_context({'PAPERCLIP_AGENT_JWT_SECRET': 'x'})  # not an allowlisted name
    with pytest.raises(ValueError):
        validate_context({'PAPERCLIP_API_KEY': 'x' * (64 * 1024 + 1)})  # oversize
    with pytest.raises(ValueError):
        validate_context({'PAPERCLIP_API_KEY': 'only-one-of-the-required-names'})  # partial, non-empty
    assert validate_context({}) == {}  # an empty context is a legitimate "no run bound" value


# --- RPC/Policy wiring: session.create -> LocalSessionPolicy -> policy_scope (MON-1024, phase 2) ---

_CONFIG = 'platform_toolsets: {cli: []}\nmodel: {provider: custom, default: m, base_url: http://x.invalid/v1}\n'


def _home(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(_CONFIG)
    return home


async def _owner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('HERMES_HOME', str(home))
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
    return owner, AuthorityConnection(owner, object(), {'user_id': 'owner'})


def _create(owner, actor, home, request_id, **extra):
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    ref = create_local_session(owner, actor, {'request_id': request_id, 'source': 'cli', 'cwd': str(home),
                                              'toolsets': [], **extra})
    return ref, restore_policy(owner.persistence.bindings.receipt(ref.session_id).policy)


@pytest.mark.asyncio
async def test_session_create_binds_paperclip_context_for_the_turns_tools(tmp_path, monkeypatch):
    from gateway.session_policy import policy_scope
    from tools.code_execution_env import _build_child_env
    from tools.environments.local import _make_run_env
    monkeypatch.setattr('pm.environments._payload_manifest', lambda root: None)
    monkeypatch.setattr('tools.environments.local._user_tools_first', lambda p: p)
    monkeypatch.setattr('tools.environments.local._prepend_hermes_bin_dir', lambda p: p)
    monkeypatch.delenv('PAPERCLIP_API_KEY', raising=False)
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)

    def seen(policy):
        with policy_scope(policy, authority=owner):
            envs = (_make_run_env({}), _build_child_env(
                rpc_endpoint='s', rpc_token='t', tmpdir=str(tmp_path), child_python=sys.executable))
            return [{k: v for k, v in env.items() if k.startswith('PAPERCLIP_')} for env in envs]
    try:
        _, mine = _create(owner, connection.actor, home, 'a', paperclip_context=context('x'))
        _, bare = _create(owner, connection.actor, home, 'b')  # a wake-free launch (TUI, Desktop, ACP...)
        assert mine.paperclip_context(owner) == context('x')
        assert 'paperclip_context' not in json.loads(mine.request_json)  # never echoed to a viewer
        assert bare.paperclip_context(owner) is None
        assert seen(mine) == [context('x')] * 2
        assert seen(bare) == [{}] * 2
        # Outside any turn's policy scope, the gateway's own process never carries it either.
        from agent.paperclip_context import child_environ
        assert not any(k.startswith('PAPERCLIP_') for k in child_environ())
    finally:
        await connection.close()
        close_stores_under(owner.persistence.scope.profile_dir)


@pytest.mark.asyncio
async def test_paperclip_context_values_never_reach_the_durable_policy_or_receipt(tmp_path, monkeypatch):
    """MON-1027 point (a): the committed policy (``asdict(policy)``, what ``gateway.session_local``'s
    ``stage_session_prefill`` puts in ``NewSessionBinding.policy`` and the sqlite receipt persists) must
    carry no PAPERCLIP_* value -- only the opaque ``paperclip_context_ref``, same indirection as
    ``credential_ref`` for a launch API key."""
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        secret = context('durable-secret')
        ref, policy = _create(owner, connection.actor, home, 'durable', paperclip_context=secret)
        committed = asdict(policy)
        blob = json.dumps(committed)
        for value in secret.values():
            assert value not in blob
        assert committed['paperclip_context_ref'] is not None
        assert '#' in committed['paperclip_context_ref']
        # The exact dict the sqlite receipt persists (gateway/session_local.py: 'policy': asdict(policy)).
        receipt = owner.persistence.bindings.receipt(ref.session_id)
        receipt_blob = json.dumps(receipt.policy)
        for value in secret.values():
            assert value not in receipt_blob
        assert policy.paperclip_context(owner) == secret  # still resolvable through the live authority
    finally:
        await connection.close()
        close_stores_under(owner.persistence.scope.profile_dir)


def test_paperclip_context_is_revoked_when_the_authority_restarts():
    """A restarted authority's in-memory registry is empty (``bind_paperclip_context`` never persists the
    values): a session that carried a context then simply runs its turns with none, the same fail-closed
    disposition ``bind_launch_key``'s docstring already promises for its API key."""
    from gateway.session_policy import LocalSessionPolicy
    from agent.paperclip_context import child_environ
    stale_ref = 'fixture:0:some-retired-session#sha256:deadbeefdeadbeef'
    policy = replace(LocalSessionPolicy('cli', 'cli', '/tmp', None, (), '{}', '{}', '{}'),
                      paperclip_context_ref=stale_ref)
    restarted_authority = SimpleNamespace(instance_id='fixture', epoch=0)  # no _local_paperclip_contexts
    assert policy.paperclip_context(restarted_authority) is None
    with policy.paperclip_scope(restarted_authority):
        assert not any(k.startswith('PAPERCLIP_') for k in child_environ())


@pytest.mark.asyncio
async def test_paperclip_context_is_validated_and_an_exact_retry_must_match_it(tmp_path, monkeypatch):
    from persistence.core.errors import AuthorityRefusal
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        for bad in ({'PAPERCLIP_AGENT_JWT_SECRET': 'x'}, {'PAPERCLIP_API_KEY': 'partial-only'}):
            with pytest.raises(AuthorityRefusal) as exc:
                _create(owner, connection.actor, home, f'bad-{bad!r}', paperclip_context=bad)
            assert exc.value.reason == 'invalid_params'
        # A retry of the very same request_id must carry the identity that request originally froze:
        # a different wake's context (even a legitimate one) is a conflicting request, not a silent swap.
        from gateway.session_local import create_local_session
        params = {'request_id': 'same', 'source': 'cli', 'cwd': str(home), 'toolsets': [],
                  'paperclip_context': context('x')}
        ref = create_local_session(owner, connection.actor, dict(params))
        assert create_local_session(owner, connection.actor, dict(params)) == ref  # a true retry
        with pytest.raises(AuthorityRefusal):
            create_local_session(owner, connection.actor, {**params, 'paperclip_context': context('y')})
        with pytest.raises(AuthorityRefusal):
            create_local_session(owner, connection.actor, {k: v for k, v in params.items()
                                                            if k != 'paperclip_context'})
    finally:
        await connection.close()
        close_stores_under(owner.persistence.scope.profile_dir)


@pytest.mark.asyncio
async def test_runtime_describe_lists_paperclip_context(tmp_path, monkeypatch):
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        described = await connection.describe(None, {})
        assert 'paperclip_context' in described['session_create']['parameters']
    finally:
        await connection.close()
        close_stores_under(owner.persistence.scope.profile_dir)


def test_policy_identity_always_ignores_the_opaque_paperclip_ref():
    """MON-1027 point (a), the identity side: ``paperclip_context_ref`` is an opaque per-bind reference
    (session slot + fingerprint), never the context's content -- it must never change a policy's identity
    hash, exactly like ``credential_ref``/``config_secret_ref`` already don't for a launch API key."""
    from gateway.session_policy import LocalSessionPolicy
    from gateway.session_policy_credentials import policy_identity
    base = LocalSessionPolicy('cli', 'cli', '/tmp', None, (), '{}', '{}', '{}')
    ref = 'fixture:0:some-session#sha256:deadbeefdeadbeef'
    assert policy_identity(base) == policy_identity(replace(base, paperclip_context_ref=ref))


def test_client_side_paperclip_context_policy_collects_this_processs_own_env(monkeypatch):
    from hermes_cli.gateway_chat import paperclip_context_policy
    assert paperclip_context_policy([]) == {}  # a gateway that predates the field
    for name, value in context('z').items():
        monkeypatch.setenv(name, value)
    assert paperclip_context_policy(['paperclip_context']) == {'paperclip_context': context('z')}


def test_owner_key_retry_requires_matching_paperclip_context():
    """MON-1027 round-1 follow-up finding: the owner-key path (``bind_launch_key``'s ``owners[ref]``,
    an overlapping alias expansion reusing the session slot's launch key) must judge a retry's Paperclip
    identity exactly like the sequential (``same_create_retry`` at commit) and the sqlite-commit paths
    already do -- not silently drop it because ``owners[ref]`` never carried ``paperclip_context_ref``.
    ``owners[ref]`` is captured off *policy* (gateway/session_policy.py), so ``bind_paperclip_context``
    must already have frozen the ref on it before ``bind_launch_key`` runs -- why
    gateway/session_local.py binds the context first now, not after.
    """
    from persistence.core.errors import AuthorityRefusal
    from gateway.session_policy import LocalSessionPolicy, bind_launch_key, bind_paperclip_context
    authority = SimpleNamespace(instance_id='fixture', epoch=0)
    base = LocalSessionPolicy('cli', 'cli', '/tmp', None, (), '{}', '{}', '{}')
    sid, ref = 'sid-owner', 'fixture:0:sid-owner'

    policy_x = bind_paperclip_context(authority, sid, base, context('x'))
    bind_launch_key(authority, sid, policy_x, 'k1', owner_key=True,
                     client_request={'paperclip_context': context('x')})
    assert authority._local_launch_keys[ref] == 'k1'

    # The same identity retried, a different key offered: the held one is reused, not replaced.
    policy_x2 = bind_paperclip_context(authority, sid, base, context('x'))
    bind_launch_key(authority, sid, policy_x2, 'k2', owner_key=True,
                     client_request={'paperclip_context': context('x')})
    assert authority._local_launch_keys[ref] == 'k1'

    # A different identity, or the slot's identity omitted entirely, is a conflicting request --
    # never a silent inheritance of the held key (the pre-fix behaviour: ``owners[ref]`` lacked the
    # ref, so ``committed.get('paperclip_context_ref')`` read None regardless of what was really bound).
    policy_y = bind_paperclip_context(authority, sid, base, context('y'))
    with pytest.raises(AuthorityRefusal):
        bind_launch_key(authority, sid, policy_y, 'k3', owner_key=True,
                         client_request={'paperclip_context': context('y')})
    with pytest.raises(AuthorityRefusal):
        bind_launch_key(authority, sid, policy_y, 'k3', owner_key=True, client_request={})

