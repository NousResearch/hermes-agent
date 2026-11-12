"""A Docker worker is told the workspace path its sandbox sees, not the host path (REA-1329).

The host path of a ``dir``/``scratch``/``worktree`` workspace does not exist inside the
container; a shell command writing there lands in the container's own filesystem while the
mounted workspace stays empty.
"""
import json
from types import SimpleNamespace

import pytest


def _docker_scope(**extra):
    return {'TERMINAL_ENV': 'docker', 'TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE': 'true',
            'TERMINAL_DOCKER_VOLUMES': '[]', **extra}


def test_container_path_matches_the_mount_rule(tmp_path):
    from tools.environments.docker import host_cwd_container_path
    ws = tmp_path / 'ws'
    ws.mkdir()
    assert host_cwd_container_path([], str(ws), True) == '/workspace'
    # A user volume owns /workspace: the cwd gets the first free fallback mount.
    assert host_cwd_container_path([f'{tmp_path}/other:/workspace'], str(ws), True) == '/host-cwd'
    # A user volume already binds this very directory: that container path is reused.
    assert host_cwd_container_path([f'{ws}:/srv/proj:rw'], str(ws), True) == '/srv/proj'
    assert host_cwd_container_path([], str(ws), False) is None
    assert host_cwd_container_path([], str(tmp_path / 'missing'), True) is None


def test_prediction_agrees_with_the_container_mounts(tmp_path):
    """The predictor and DockerEnvironment._mount_args share one rule."""
    from tools.environments.docker import DockerEnvironment, host_cwd_container_path
    ws = tmp_path / 'ws'
    ws.mkdir()
    for volumes in ([], [f'{tmp_path}/other:/workspace'], [f'{ws}:/srv/proj']):
        env = DockerEnvironment.__new__(DockerEnvironment)
        env._persistent = False
        volume_args, _writable = env._mount_args(volumes, str(ws), True, 'default')
        assert env.host_cwd_mount == host_cwd_container_path(volumes, str(ws), True)
        assert env.host_cwd == str(ws)
        if not volumes or volumes[0].endswith(':/workspace'):
            assert f'{ws}:{env.host_cwd_mount}' in volume_args
    env = DockerEnvironment.__new__(DockerEnvironment)
    env._persistent = False
    env._mount_args([], str(ws), False, 'default')
    assert env.host_cwd is None and env.host_cwd_mount is None


def test_sandbox_workspace_path_only_for_a_mounted_docker_cwd(tmp_path):
    from gateway.session_kanban import sandbox_workspace_path
    ws = tmp_path / 'ws'
    ws.mkdir()
    assert sandbox_workspace_path(_docker_scope(), str(ws)) == '/workspace'
    assert sandbox_workspace_path(_docker_scope(TERMINAL_DOCKER_VOLUMES=json.dumps([f'{tmp_path}/o:/workspace'])),
                                  str(ws)) == '/host-cwd'
    assert sandbox_workspace_path({'TERMINAL_ENV': 'local'}, str(ws)) is None
    assert sandbox_workspace_path({'TERMINAL_ENV': 'ssh'}, str(ws)) is None
    assert sandbox_workspace_path(_docker_scope(TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE='false'), str(ws)) is None
    assert sandbox_workspace_path(_docker_scope(TERMINAL_DOCKER_SHARED_CONTAINER_KEY='team'), str(ws)) is None
    assert sandbox_workspace_path(_docker_scope(TERMINAL_DOCKER_VOLUMES='not json'), str(ws)) is None
    assert sandbox_workspace_path(_docker_scope(), None) is None


@pytest.mark.parametrize('kind', ['scratch', 'dir', 'worktree'])
def test_worker_header_names_the_sandbox_path(kind):
    from hermes_cli.kanban_db_context import _ctx_header
    task = SimpleNamespace(id='t_1', title='T', assignee='default', status='running', tenant=None,
                           workspace_kind=kind, workspace_path='/home/u/ws', max_runtime_seconds=None,
                           branch_name=None, body='')
    lines: list[str] = []
    _ctx_header(lines, task, '/workspace')
    assert f'Workspace: {kind} @ /workspace' in lines
    assert any(line.startswith('Host path: /home/u/ws (mounted at /workspace;') for line in lines)
    plain: list[str] = []
    _ctx_header(plain, task)
    assert f'Workspace: {kind} @ /home/u/ws' in plain
    assert not any(line.startswith('Host path:') for line in plain)


def _claimed(tmp_path, monkeypatch, terminal_yaml):
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))
    (tmp_path / 'config.yaml').write_text(terminal_yaml)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('sbx')
    with closing(connect(board='sbx')) as conn:
        task_id = kb.create_task(conn, title='Write artifact', body='create verification.txt', assignee='default',
                                 workspace_kind='dir', workspace_path=str(workspace),
                                 model_override='m', provider_override='custom')
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    return workspace, task


@pytest.mark.parametrize('backend, expected', [('docker', '/workspace'), ('local', None)])
def test_kanban_policy_context_carries_the_sandbox_path(tmp_path, monkeypatch, backend, expected):
    from gateway.session_contract import SessionActor
    from gateway.session_kanban import build_kanban_policy
    workspace, task = _claimed(
        tmp_path, monkeypatch, f'terminal:\n  backend: {backend}\n  docker_mount_cwd_to_workspace: true\n')
    actor = SessionActor('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='sbx', task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    config = {'model': {'default': 'm', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal', 'file']}}
    policy, _secrets = build_kanban_policy(connection, params, config)
    context = json.loads(policy.kanban_json)
    assert policy.cwd == str(workspace) and context['workspace'] == str(workspace)
    if expected:
        assert f'Workspace: dir @ {expected}' in context['context']
        assert f'Host path: {workspace}' in context['context']
    else:
        assert f'Workspace: dir @ {workspace}' in context['context']
        assert 'Host path:' not in context['context']
