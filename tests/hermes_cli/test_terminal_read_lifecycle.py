"""Attachment reads reuse normal execution without crossing resource generations."""
import subprocess
import threading
from types import SimpleNamespace

import pytest

from hermes_cli.terminal_workspace_fs import TerminalWorkspaceFs, TerminalWorkspaceFsError
from tools.environments.base import BaseEnvironment
from tools.environments.vercel_sandbox import VercelSandboxEnvironment


def vercel_env(monkeypatch, run_command):
    env = VercelSandboxEnvironment.__new__(VercelSandboxEnvironment)
    BaseEnvironment.__init__(env, cwd='/workspace', timeout=30)
    env._workspace_root = '/workspace'
    env._requested_cwd = '/workspace'
    env._task_id = 'test'
    env._lock = threading.RLock()
    env._sync_manager = None
    env._persistent = False
    def lost():
        raise ConnectionError('sandbox was deleted')
    env._sandbox = SimpleNamespace(refresh=lost, run_command=run_command)
    creates = []
    def create():
        creates.append(True)
        raise RuntimeError('intercepted sandbox creation')
    monkeypatch.setattr(env, '_create_sandbox', create)
    return env, creates


def test_deleted_vercel_read_uses_existing_backend_recovery(monkeypatch):
    def lost(*args, **kwargs):
        raise ConnectionError('sandbox was deleted')
    env, creates = vercel_env(monkeypatch, lost)
    from hermes_cli.ssh_workspace_fs import SshWorkspaceFsError
    with pytest.raises(SshWorkspaceFsError):
        TerminalWorkspaceFs(env).inspect_file('/workspace/file')
    assert creates == [True], 'file read bypassed normal backend recovery'
    with pytest.raises(RuntimeError, match='intercepted sandbox creation'):
        env.execute('true')
    assert creates == [True, True]


@pytest.mark.parametrize('replace_during_read', [False, True])
def test_reader_rejects_recreated_resource_on_same_environment(monkeypatch, replace_during_read):
    env = None
    def run(*args, **kwargs):
        if replace_during_read:
            env._mark_recreated()
        return SimpleNamespace(stdout='replacement bytes', stderr='', exit_code=0)
    env, _ = vercel_env(monkeypatch, run)
    env._sandbox.refresh = lambda: None
    env._sandbox.status = None
    reader = TerminalWorkspaceFs(env)
    if not replace_during_read:
        env._mark_recreated()
    with pytest.raises(TerminalWorkspaceFsError):
        reader._execute('printf replacement')


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('snapshot_ready', [False, True])
def test_finished_turn_read_uses_ssh_sync_and_shell_snapshot(tmp_path, monkeypatch, snapshot_ready):
    from agent.chat_completion_helpers import cleanup_task_resources
    from hermes_cli.terminal_workspace_fs import get_terminal_workspace_fs
    from tools import terminal_tool
    from tools.environments.ssh import SSHEnvironment
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope

    class ConnectionOnly(SSHEnvironment):
        _hermes_backend_name = 'test_remote'
        def __init__(self):
            BaseEnvironment.__init__(self, cwd=str(tmp_path), timeout=30)
            self._snapshot_path = str(tmp_path / 'snapshot.sh')
            self._snapshot_ready = snapshot_ready
            self.cleaned = 0
            self._sync_manager = SimpleNamespace(sync=lambda: syncs.append(True))
        def _build_ssh_command(self, **kwargs):
            return ['fixture-ssh']

        def cleanup(self):
            self.cleaned += 1
            # Like a connection-only plugin, remove shell state but leave workspace.
            from pathlib import Path
            Path(self._snapshot_path).unlink(missing_ok=True)

    syncs = []
    logins = []
    def local_ssh(argv, stdin_data, **kwargs):
        import shlex
        logins.append('-l' in argv)
        return subprocess.Popen(['bash', '-c', shlex.split(argv[-1])[0]],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    monkeypatch.setattr('tools.environments.ssh._popen_bash', local_ssh)
    env = ConnectionOnly()
    (tmp_path / 'file').write_bytes(b'finished artifact\x00\xff')
    token = set_terminal_scope({'TERMINAL_ENV': 'test_remote'})
    key = terminal_tool._qualify_task_key('session:finished')
    monkeypatch.setitem(terminal_tool._active_environments, key, env)
    monkeypatch.setattr(terminal_tool, 'ensure_task_env', lambda *a, **k: pytest.fail('created environment'))
    try:
        cleanup_task_resources(SimpleNamespace(verbose_logging=False), key)
        assert env.cleaned == 1
        assert key not in terminal_tool._active_environments
        reader = get_terminal_workspace_fs({'id': 'finished'})
        assert reader.read_bytes(str(tmp_path / 'file'), max_bytes=100)[0] == b'finished artifact\x00\xff'
        from pathlib import Path
        assert syncs, 'lazy read bypassed the ordinary SSH sync hook'
        assert logins and all(login == (not snapshot_ready) for login in logins)
        assert Path(env._snapshot_path).exists() == snapshot_ready
        from tools.terminal_tool_lifecycle import cleanup_vm
        cleanup_vm(key)
        with pytest.raises(TerminalWorkspaceFsError):
            reader._execute('true')
        with pytest.raises(TerminalWorkspaceFsError):
            get_terminal_workspace_fs({'id': 'finished'})
    finally:
        reset_terminal_scope(token)


def test_rpc_token_rejects_same_object_after_resource_recreation(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from tui_gateway.terminal_files import handle_file_request
    from tools import terminal_tool
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    env, _ = vercel_env(monkeypatch, lambda *a, **k: SimpleNamespace(output=lambda: 'ok', exit_code=0))
    token = set_terminal_scope({'TERMINAL_ENV': 'test_remote'})
    key = terminal_tool._qualify_task_key('session:durable')
    monkeypatch.setitem(terminal_tool._active_environments, key, env)
    session = {'session_key': 'durable', 'profile_home': str(tmp_path)}
    server = SimpleNamespace(_sessions={'live': session}, _session_profile_runtime_scope=lambda s: nullcontext())
    frame = {'sid': 'live', 'session_key': 'durable', 'profile_home': str(tmp_path),
             'backend': 'test_remote', 'operation': 'open'}
    try:
        opened = handle_file_request(server, frame)
        env._mark_recreated()
        fresh = handle_file_request(server, frame)
        assert env._recreated_notice_pending is True
        assert fresh['env_token'] != opened['env_token']
        for operation in ('inspect', 'read'):
            with pytest.raises(TerminalWorkspaceFsError):
                handle_file_request(server, {**frame, 'operation': operation, 'env_token': opened['env_token'],
                                            'path': '/workspace/file', 'offset': 0, 'length': 1})
    finally:
        reset_terminal_scope(token)
