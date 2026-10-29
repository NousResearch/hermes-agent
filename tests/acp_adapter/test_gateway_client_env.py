"""`hermes acp` sends its allowlisted launch environment as ``session.create``'s ``client_env``, as the
classic CLI and the TUI do, so the shared gateway binds it to that ACP session's tools only.

Regression for #142671 (the ``GatewayACPAgent.new_session`` omission). The allowlist is unchanged:
other toolchain names still need ``terminal.client_env_passthrough``, and credential-like names are
never sent.
"""
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent

_EDITOR_PATH = os.pathsep.join(['/editor/bin', '/usr/bin', '/bin'])
_CONFIG = ('platform_toolsets: {cli: [], acp: []}\n'
           'model: {provider: custom, default: m, base_url: http://x.invalid/v1}\n'
           'terminal: {client_env_passthrough: [DEMO_VAR, DEMO_TOKEN]}\n')
_LAUNCH = ('HERMES_SAFE_MODE', 'HERMES_IGNORE_USER_CONFIG', 'HERMES_MODEL', 'HERMES_INFERENCE_MODEL',
           'HERMES_INFERENCE_PROVIDER', 'HERMES_YOLO_MODE', 'HERMES_IGNORE_RULES', 'HERMES_ACCEPT_HOOKS')


def _agent(created, parameters):
    async def rpc(method, **params):
        if method == 'runtime.describe':
            return {'session_create': {'sources': ['acp'], 'parameters': parameters},
                    'capabilities': ['acp-editor-policy-v1']}
        if method == 'config.set':
            return {'key': params['key'], 'value': params['value'], 'scope': 'session'}
        assert method == 'session.create', method
        created.append(params)
        return {'session_id': f's{len(created)}', 'info': {'cwd': params['cwd']}}

    agent = GatewayACPAgent()
    agent._gateway = AsyncMock()
    agent._gateway.rpc.side_effect = rpc
    agent._session_fields = AsyncMock(return_value={})
    return agent


@pytest.fixture
def editor_launch(tmp_path, monkeypatch):
    """The environment an ACP client spawned `hermes acp` with, over a profile that allowlists DEMO_VAR."""
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(_CONFIG)
    monkeypatch.setenv('HERMES_HOME', str(home))
    for name in _LAUNCH:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PATH', _EDITOR_PATH)
    monkeypatch.setenv('DEMO_TOKEN', 'listed-but-a-secret')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-not-forwarded')
    return home


async def _new_sessions(tmp_path, monkeypatch, parameters, values):
    created = []
    for value in values:
        monkeypatch.setenv('DEMO_VAR', value)
        await _agent(created, parameters).new_session(cwd=str(tmp_path))
    return created


@pytest.mark.asyncio
async def test_each_acp_session_sends_its_own_launch_env(editor_launch, tmp_path, monkeypatch):
    created = await _new_sessions(tmp_path, monkeypatch, ['cwd', 'source', 'editor', 'client_env'], ['a', 'b'])
    assert [request['client_env'].get('DEMO_VAR') for request in created] == ['a', 'b']
    for request in created:
        env = request['client_env']
        assert env['PATH'] == _EDITOR_PATH
        assert 'DEMO_TOKEN' not in env and 'OPENAI_API_KEY' not in env
    # A gateway that predates the field is sent nothing new.
    old = await _new_sessions(tmp_path, monkeypatch, ['cwd', 'source', 'editor'], ['a'])
    assert 'client_env' not in old[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', ['HERMES_SAFE_MODE', 'HERMES_IGNORE_USER_CONFIG'])
async def test_bypass_acp_launch_reads_no_profile_allowlist(editor_launch, tmp_path, monkeypatch, flag):
    monkeypatch.setenv(flag, '1')
    monkeypatch.setenv('HERMES_MODEL', 'explicit-model')
    monkeypatch.setenv('HERMES_INFERENCE_PROVIDER', 'openrouter')
    created = await _new_sessions(tmp_path, monkeypatch, ['cwd', 'source', 'editor', 'model', 'provider',
                                                          'safe_mode', 'ignore_user_config', 'client_env'], ['a'])
    env = created[0]['client_env']
    assert env['PATH'] == _EDITOR_PATH
    assert 'DEMO_VAR' not in env


async def _owner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
    return owner, AuthorityConnection(owner, object(), {'user_id': 'owner'})


@pytest.mark.asyncio
async def test_gateway_runs_each_acp_sessions_tools_with_its_own_launch_env(editor_launch, tmp_path, monkeypatch):
    """The real owner freezes each ACP session's map and applies it to that session's terminal
    subprocess only; the gateway's own environment and a later launch do not change it."""
    from gateway.session_local import create_local_session
    from gateway.session_policy import policy_scope, restore_policy
    from hermes_state_local import local_receipt
    from tools.environments.local import _make_run_env
    requests = await _new_sessions(tmp_path, monkeypatch, ['cwd', 'source', 'editor', 'client_env'], ['a', 'b'])
    # The gateway daemon's own environment, which the editor did not launch.
    monkeypatch.setenv('DEMO_VAR', 'gateway')
    monkeypatch.setenv('PATH', os.pathsep.join(['/daemon/bin', '/usr/bin', '/bin']))
    monkeypatch.delenv('DEMO_TOKEN')
    owner, connection = await _owner(editor_launch, monkeypatch)

    def frozen(session_id):
        return restore_policy(local_receipt(owner.db, session_id)['policy'])

    def terminal_sees(policy):
        with policy_scope(policy, authority=owner):
            env = _make_run_env({})
        probe = "import os; print('|'.join(os.environ.get(k, 'unset') for k in ('DEMO_VAR', 'DEMO_TOKEN', 'PATH')))"
        out = subprocess.run([sys.executable, '-c', probe],
                             env=env, capture_output=True, text=True, timeout=30, check=True)
        value, token, path = out.stdout.strip().split('|')
        dirs = path.split(os.pathsep)
        return value, token, '/editor/bin' in dirs, '/daemon/bin' in dirs
    try:
        ids = [create_local_session(owner, connection.actor, {**request, 'toolsets': []}, client_local=True).session_id
               for request in requests]
        assert [terminal_sees(frozen(sid)) for sid in ids] == [('a', 'unset', True, False), ('b', 'unset', True, False)]
        assert os.environ['DEMO_VAR'] == 'gateway' and _make_run_env({})['DEMO_VAR'] == 'gateway'
        monkeypatch.setenv('DEMO_VAR', 'later-launch')
        assert terminal_sees(frozen(ids[0]))[0] == 'a'  # a resume restores the frozen map
    finally:
        await connection.close()
        owner.db.close()
