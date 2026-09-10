"""Worker overrides must target the endpoint the real migration registers."""

from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER, KANBAN_ENV_KEYS, delegated_child_context, non_dispatcher_owned_context,
)
from agent.transports.codex_app_server import CodexAppServerClient, check_codex_binary
from hermes_cli.codex_runtime_plugin_migration import migrate


@pytest.fixture
def invocation(monkeypatch, tmp_path):
    home = tmp_path / 'profile'
    home.mkdir()
    (home / 'config.yaml').write_text('toolsets: []\n')
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(home))
    for key in (*KANBAN_ENV_KEYS, DELEGATED_CHILD_ENV_MARKER, 'HERMES_KANBAN_DB', 'HERMES_KANBAN_BOARD'):
        monkeypatch.delenv(key, raising=False)
    worker_env = {
        'HERMES_KANBAN_TASK': '11111111-1111-4111-8111-111111111111',
        'HERMES_KANBAN_RUN_ID': '42',
        'HERMES_KANBAN_CLAIM_LOCK': (tmp_path / 'claim.lock').as_posix(),
        'HERMES_KANBAN_GOAL_MODE': '1',
        'HERMES_KANBAN_GOAL_MAX_TURNS': '3',
        'HERMES_KANBAN_DB': (tmp_path / 'board with spaces' / 'kanban.db').as_posix(),
        'HERMES_KANBAN_BOARD': 'test-board',
    }

    def build(mode, with_user_server):
        if mode != 'ordinary':
            for key, value in worker_env.items():
                monkeypatch.setenv(key, value)
        user_servers = {'hermes-mcp': {
            'command': sys.executable, 'args': ['-c', 'raise SystemExit(0)'],
            'env': {'USER_MARKER': 'unchanged'},
        }} if with_user_server else {}
        codex_home = tmp_path / 'codex'
        report = migrate({'mcp_servers': user_servers}, codex_home=codex_home,
                         discover_plugins=False, expose_hermes_tools=True)
        assert not report.errors
        config_path = codex_home / 'config.toml'
        original = config_path.read_bytes()
        servers = tomllib.loads(original.decode())['mcp_servers']
        managed_name, = [name for name, cfg in servers.items()
                        if cfg.get('args') == ['-m', 'agent.transports.hermes_tools_mcp_server']]
        captured = {}

        class Process:
            def __init__(self, command, **kwargs):
                captured.update(command=list(command), env=kwargs['env'].copy())
                self.stdin = self.stdout = self.stderr = None

        with ExitStack() as stack, monkeypatch.context() as patch:
            if mode == 'delegated':
                stack.enter_context(delegated_child_context())
            elif mode == 'non-worker':
                stack.enter_context(non_dispatcher_owned_context())
            patch.setattr(subprocess, 'Popen', Process)
            client = CodexAppServerClient(codex_bin='record-only', codex_home=str(codex_home))
            client._closed = True
            client._reader.join(2)
            client._stderr_reader.join(2)
            assert not client._reader.is_alive() and not client._stderr_reader.is_alive()
        assert config_path.read_bytes() == original
        return SimpleNamespace(**captured, servers=servers, managed_name=managed_name,
                               worker_env=worker_env, config_path=config_path, home=home)

    return build


@pytest.mark.parametrize('mode', ['ordinary', 'worker', 'delegated', 'non-worker'])
@pytest.mark.parametrize('with_user_server', [False, True])
def test_worker_scope_matches_migrated_endpoint(invocation, mode, with_user_server):
    case = invocation(mode, with_user_server)
    overrides = {}
    for index, argument in enumerate(case.command):
        if argument == '-c' and case.command[index + 1].startswith('mcp_servers.'):
            for name, config in tomllib.loads(case.command[index + 1])['mcp_servers'].items():
                overrides.setdefault(name, {}).update(config.get('env', {}))
    assert set(overrides) == ({case.managed_name} if mode == 'worker' else set())
    assert not any(key in case.env for key in KANBAN_ENV_KEYS)
    if mode != 'ordinary':
        assert case.env[DELEGATED_CHILD_ENV_MARKER] == '1'
    if with_user_server:
        assert 'hermes-mcp' not in overrides
        assert case.servers['hermes-mcp']['env'] == {'USER_MARKER': 'unchanged'}

    endpoint = case.servers[case.managed_name]
    endpoint_env = {key: case.env[key] for key in endpoint.get('env_vars', []) if key in case.env}
    endpoint_env.update(endpoint.get('env', {}))
    endpoint_env.update(overrides.get(case.managed_name, {}))
    if mode == 'worker':
        assert all(endpoint_env[key] == value for key, value in case.worker_env.items())
        assert endpoint_env[DELEGATED_CHILD_ENV_MARKER] == ''
    else:
        assert not any(key in endpoint_env for key in KANBAN_ENV_KEYS)

    # Evaluate the tool consumer in a fresh process, not the parent's ContextVars.
    env = {**case.env, 'HERMES_HOME': str(case.home)}
    for key in (*KANBAN_ENV_KEYS, DELEGATED_CHILD_ENV_MARKER):
        env.pop(key, None)
    env.update(endpoint_env)
    script = (
        'import json; from tools.kanban_tools import _check_kanban_mode, _default_task_id; '
        'print(json.dumps({"available": _check_kanban_mode(), "task": _default_task_id(None)}))'
    )
    result = subprocess.run([sys.executable, '-c', script], env=env, cwd=Path(__file__).resolve().parents[3],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        'available': mode == 'worker', 'task': case.worker_env['HERMES_KANBAN_TASK'] if mode == 'worker' else None,
    }


@pytest.fixture(scope='module')
def codex_cli():
    binary = shutil.which('codex')
    if binary is None:
        pytest.skip('Codex CLI is not installed; producer/consumer tests still run')
    available, version = check_codex_binary(binary)
    if not available:
        pytest.skip(version)
    return binary


@pytest.mark.parametrize('mode', ['ordinary', 'worker'])
@pytest.mark.parametrize('with_user_server', [False, True])
@pytest.mark.parametrize('command', [('mcp', 'list', '--json'), ('app-server',)])
def test_native_codex_accepts_migrated_worker_config(invocation, codex_cli, mode, with_user_server, command):
    case = invocation(mode, with_user_server)
    # Immediate EOF: load configuration, but never initialize a model turn or invoke a tool.
    result = subprocess.run([codex_cli, *command, *case.command[2:]], input='', env=case.env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    if command[0] == 'mcp':
        assert {server['name'] for server in json.loads(result.stdout)} == set(case.servers)
