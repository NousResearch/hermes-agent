"""LC-8 / T3 client side: the classic CLI and the Ink TUI launcher send this shell's allowlisted
environment as ``session.create``'s ``client_env``, and only to a gateway that lists the field."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _shell(monkeypatch):
    monkeypatch.setenv('PATH', '/client/venv/bin:/usr/bin')
    monkeypatch.setenv('VIRTUAL_ENV', '/client/venv')
    monkeypatch.setenv('HERMES_EPHEMERAL_SYSTEM_PROMPT', 'mine')
    monkeypatch.setenv('NODE_AUTH_TOKEN', 'npm-secret')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-secret')


def _expected(env):
    assert env['PATH'] == '/client/venv/bin:/usr/bin'
    assert env['VIRTUAL_ENV'] == '/client/venv'
    assert env['HERMES_EPHEMERAL_SYSTEM_PROMPT'] == 'mine'
    assert 'NODE_AUTH_TOKEN' not in env and 'OPENAI_API_KEY' not in env


class _Connection:
    def __init__(self, parameters):
        self.parameters, self.calls, self.final = parameters, [], False

    def call(self, method, **params):
        self.calls.append((method, params))
        if method == 'runtime.describe':
            return {'session_create': {'sources': ['cli'], 'parameters': self.parameters}}
        return {'stored_session_id': 'sid'}

    def state(self, sid):
        return SimpleNamespace(adopt_snapshot=lambda snapshot: None)


@pytest.mark.parametrize('takes', [True, False])
def test_classic_cli_create_sends_client_env_when_the_gateway_takes_it(monkeypatch, takes):
    from hermes_cli.gateway_session_store import GatewaySessionStore
    _shell(monkeypatch)
    connection = _Connection(['cwd', 'source', 'row_cwd', 'client_env'] if takes else ['cwd', 'source', 'row_cwd'])
    GatewaySessionStore(connection).create({})
    created = [params for method, params in connection.calls if method == 'session.create']
    assert len(created) == 1
    if takes:
        _expected(created[0]['client_env'])
    else:
        assert 'client_env' not in created[0]


def test_classic_cli_new_sends_this_shells_client_env(monkeypatch):
    from hermes_cli import gateway_chat_commands
    _shell(monkeypatch)
    sent = {}

    async def rpc(method, **params):
        if method == 'session.info':
            return {'launch_request': {'source': 'cli'}, 'cwd': '/tmp'}
        if method == 'runtime.describe':
            return {'session_create': {'parameters': ['source', 'cwd', 'client_env']}}
        sent.update(params)
        return {'stored_session_id': 'new'}

    async def adopt(snapshot):
        return None
    view = SimpleNamespace(client=SimpleNamespace(rpc=rpc), session_id='old', adopt=adopt, launch_api_key=None)
    asyncio.run(gateway_chat_commands._new(view, ''))
    _expected(sent['client_env'])


def test_client_env_policy_reads_the_profiles_extra_names(monkeypatch, tmp_path):
    from hermes_cli.gateway_chat import client_env_policy
    _shell(monkeypatch)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('terminal: {client_env_passthrough: [MY_TOOLCHAIN_HOME]}\n')
    monkeypatch.setenv('MY_TOOLCHAIN_HOME', '/opt/tc')
    assert client_env_policy(['source']) == {}
    env = client_env_policy(['client_env'])['client_env']
    _expected(env)
    assert env['MY_TOOLCHAIN_HOME'] == '/opt/tc'


def test_client_env_policy_reads_the_effective_config_and_bypass_reads_none(monkeypatch, tmp_path):
    """The gateway takes extra names from the effective config (managed layer included); a bypass launch's
    owner freezes code defaults, so the client then reads no profile either."""
    from hermes_cli.gateway_chat import client_env_policy
    _shell(monkeypatch)
    home, managed = tmp_path / 'home', tmp_path / 'managed'
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(managed))
    (managed / 'config.yaml').write_text('terminal: {client_env_passthrough: [GEM_HOME]}\n')
    monkeypatch.setenv('GEM_HOME', '/gems')
    for name in ('HERMES_SAFE_MODE', 'HERMES_IGNORE_USER_CONFIG'):
        monkeypatch.delenv(name, raising=False)
    env = client_env_policy(['client_env'])['client_env']
    _expected(env)
    assert env['GEM_HOME'] == '/gems'
    (home / 'config.yaml').write_text('terminal: {client_env_passthrough: [MY_TOOLCHAIN_HOME]}\n')
    monkeypatch.setenv('MY_TOOLCHAIN_HOME', '/opt/tc')
    monkeypatch.setenv('HERMES_IGNORE_USER_CONFIG', '1')
    bypass = client_env_policy(['client_env'])['client_env']
    _expected(bypass)
    assert 'GEM_HOME' not in bypass and 'MY_TOOLCHAIN_HOME' not in bypass


def test_tui_launcher_hands_the_shell_env_to_ink(monkeypatch):
    from hermes_cli import main as main_mod, main_tui_launch
    _shell(monkeypatch)
    monkeypatch.setenv('HERMES_PYTHON', sys.executable)
    monkeypatch.setattr(main_mod, '_has_any_provider_configured', lambda: True)
    monkeypatch.setattr(main_mod, '_oneshot_cleanup_done', False)
    captured = {}
    monkeypatch.setattr(main_tui_launch, '_make_tui_argv', lambda tui_dir, tui_dev: (['node', 'dist/entry.js'], Path('.')))

    def fake_call(argv, cwd=None, env=None):
        captured.update(env)
        return 1
    monkeypatch.setattr(main_mod.subprocess, 'call', fake_call)
    with pytest.raises(SystemExit):
        main_mod._launch_tui()
    _expected(json.loads(captured['HERMES_TUI_CLIENT_ENV']))


def test_tui_bootstrap_keeps_the_client_env_out_of_a_daemon_it_starts():
    text = (Path(__file__).resolve().parents[2] / 'ui-tui' / 'scripts' / 'gateway_bootstrap.py').read_text()
    assert 'HERMES_TUI_CLIENT_ENV' in text
