"""#140665 (#134889 CLI-39): a gateway-owned session's client cwd reaches the Docker terminal backend.

Before #106742 every ``hermes chat`` was its own process with its own ``TERMINAL_CWD`` and its own
container. Now the turn runs in the gateway: the client's cwd (or ``--in DIR``) is frozen into the
session's policy, but never reached ``resolve_task_overrides(task_id)['cwd']``, the per-session mount
source. So ``container_persistent: false`` mounted no ``/workspace`` at all, and ``true`` shared ONE
``profile:<name>`` container bound to whichever session's cwd created it first.
"""
import json
from dataclasses import asdict

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    from tools import terminal_tool
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.delenv('TERMINAL_ENV', raising=False)
    monkeypatch.delenv('TERMINAL_CWD', raising=False)
    monkeypatch.delenv('TERMINAL_CONTAINER_PERSISTENT', raising=False)
    monkeypatch.delenv('TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE', raising=False)
    saved = dict(terminal_tool._task_env_overrides), dict(terminal_tool._session_cwd)
    yield home
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(saved[0])
    terminal_tool._session_cwd.clear()
    terminal_tool._session_cwd.update(saved[1])


def _policy(home, cwd, persistent, mount=True):
    from gateway.session_policy import build_policy, restore_policy
    (home / 'config.yaml').write_text(json.dumps({'terminal': {
        'backend': 'docker', 'docker_mount_cwd_to_workspace': mount,
        'container_persistent': persistent}}), encoding='utf-8')
    return restore_policy(asdict(build_policy({'source': 'cli', 'cwd': str(cwd), 'model': 'm'}, {})))


def _turn(policy, session_id):
    """The gateway turn's own scope: frozen policy, session key, and the turn's workspace binding."""
    from contextlib import ExitStack
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_policy import bind_turn_workspace, policy_scope
    stack = ExitStack()
    stack.enter_context(policy_scope(policy))
    tokens = set_session_vars(session_key=f'agent:main:local:dm:{session_id}')
    stack.callback(clear_session_vars, tokens)
    bind_turn_workspace(policy, session_id)
    return stack


def _plan(session_id):
    from tools.terminal_tool import _get_env_config, _resolve_container_task_id, _resolve_task_host_cwd
    return _resolve_container_task_id(session_id), _resolve_task_host_cwd(_get_env_config(), session_id)


def test_per_session_container_mounts_each_sessions_own_client_cwd(home, tmp_path):
    ws1, ws2 = tmp_path / 'ws1', tmp_path / 'ws2'
    ws1.mkdir(), ws2.mkdir()
    plans = {}
    for sid, ws in (('local-a', ws1), ('local-b', ws2)):
        with _turn(_policy(home, ws, persistent=False), sid):
            plans[sid] = _plan(sid)
    # Each session: its own container, mounting its own client cwd at /workspace (None on main).
    assert plans == {'local-a': ('local-a', str(ws1)), 'local-b': ('local-b', str(ws2))}


def test_persistent_container_is_keyed_per_workspace_not_per_profile(home, tmp_path):
    ws1, ws2 = tmp_path / 'ws1', tmp_path / 'ws2'
    ws1.mkdir(), ws2.mkdir()
    plans = {}
    for sid, ws in (('local-a', ws1), ('local-b', ws2), ('local-c', ws1)):
        with _turn(_policy(home, ws, persistent=True), sid):
            plans[sid] = _plan(sid)
    # Two workspaces: two containers, each bound to its own folder (main: one "default" for all).
    assert plans['local-a'][0] != plans['local-b'][0]
    assert (plans['local-a'][1], plans['local-b'][1]) == (str(ws1), str(ws2))
    # Same workspace: still ONE persistent container, shared across sessions as before.
    assert plans['local-c'] == plans['local-a']


def test_binding_keeps_a_cd_and_follows_a_workspace_move(home, tmp_path):
    from tools.terminal_tool import get_session_cwd, record_session_cwd
    ws1, ws2 = tmp_path / 'ws1', tmp_path / 'ws2'
    ws1.mkdir(), ws2.mkdir()
    policy = _policy(home, ws1, persistent=False)
    with _turn(policy, 'local-a'):
        assert get_session_cwd('local-a') == str(ws1)
        record_session_cwd('local-a', '/workspace/sub')  # the agent's own `cd`
    with _turn(policy, 'local-a'):  # the next turn re-binds nothing that is already bound
        assert get_session_cwd('local-a') == '/workspace/sub'
    with _turn(_policy(home, ws2, persistent=False), 'local-a'):  # moved to another project
        assert _plan('local-a') == ('local-a', str(ws2))


@pytest.mark.parametrize('terminal', [{'backend': 'local'}, {'backend': 'ssh', 'cwd': '/srv'},
                                      {'backend': 'docker', 'docker_mount_cwd_to_workspace': False}])
def test_backends_that_do_not_mount_the_host_cwd_register_nothing(home, tmp_path, terminal):
    from gateway.session_policy import bind_turn_workspace, build_policy, policy_scope, restore_policy
    from tools import terminal_tool
    (home / 'config.yaml').write_text(json.dumps({'terminal': terminal}), encoding='utf-8')
    policy = restore_policy(asdict(build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'm'}, {})))
    with policy_scope(policy):
        bind_turn_workspace(policy, 'local-x')
        assert terminal_tool.resolve_task_overrides('local-x') == {}


def test_delegated_child_shares_its_parents_persistent_workspace_container(home, tmp_path):
    # Review probe (@kshitijk4poor W1, @dokterdok 1): parent `default:ws:<h>`, child `default`.
    from tools.terminal_tool import register_container_alias
    ws1, ws2 = tmp_path / 'ws1', tmp_path / 'ws2'
    ws1.mkdir(), ws2.mkdir()
    keys = {}
    for sid, ws in (('local-a', ws1), ('local-b', ws2)):
        with _turn(_policy(home, ws, persistent=True), sid):
            register_container_alias(f'{sid}-child', sid)  # delegate_tool_child_run.seed_workspace
            keys[sid] = (_plan(sid)[0], _plan(f'{sid}-child')[0])
    assert keys['local-a'][1] == keys['local-a'][0]
    assert keys['local-b'][1] == keys['local-b'][0]
    assert keys['local-a'][0] != keys['local-b'][0]  # children of two workspaces never share one


def test_workspace_container_key_is_idempotent_and_lazy_setup_creates_the_sessions_own(home, tmp_path, monkeypatch):
    # Review probe (@kshitijk4poor W2, @dokterdok 2): get_active_env(effective) -> the profile container.
    from tools import terminal_tool, terminal_tool_lifecycle
    from tools.terminal_tool import _resolve_container_task_id
    ws1 = tmp_path / 'ws1'
    ws1.mkdir()
    other = object()
    created = []
    monkeypatch.setattr(terminal_tool_lifecycle, '_create_configured_env',
                        lambda *a, **kw: created.append(kw['task_id']) or object())
    monkeypatch.setattr(terminal_tool, '_start_cleanup_thread', lambda: None)
    saved = dict(terminal_tool._active_environments)
    terminal_tool._active_environments['default'] = other  # a messaging/cron turn's profile container
    try:
        with _turn(_policy(home, ws1, persistent=True), 'local-a'):
            eff = _resolve_container_task_id('local-a')
            assert ':ws:' in eff and _resolve_container_task_id(eff) == eff
            assert terminal_tool_lifecycle.get_active_env(eff) is not other
            env = terminal_tool_lifecycle.ensure_task_env('local-a')
            assert created == [eff] and env is not other
            assert terminal_tool_lifecycle.get_active_env('local-a') is env
    finally:
        for key in [k for k in terminal_tool._active_environments if k not in saved]:
            terminal_tool._active_environments.pop(key, None)
        terminal_tool._active_environments.update(saved)


def test_switching_away_from_mounted_docker_drops_the_session_workspace_override(home, tmp_path):
    # Review probe (@kshitijk4poor W3, @dokterdok 3, Greptile): Docker -> SSH kept the old host cwd.
    from gateway.session_policy import bind_turn_workspace, build_policy, policy_scope, restore_policy
    from tools import terminal_tool
    ws1 = tmp_path / 'ws1'
    ws1.mkdir()
    with _turn(_policy(home, ws1, persistent=True), 'local-a'):
        assert terminal_tool.resolve_task_overrides('local-a')['cwd'] == str(ws1)
    (home / 'config.yaml').write_text(json.dumps({'terminal': {'backend': 'ssh', 'cwd': '/srv/project'}}))
    ssh = restore_policy(asdict(build_policy({'source': 'cli', 'cwd': str(ws1), 'model': 'm'}, {})))
    with policy_scope(ssh):
        bind_turn_workspace(ssh, 'local-a')
        assert terminal_tool.resolve_task_overrides('local-a') == {}
        assert terminal_tool.get_session_cwd('local-a') != str(ws1)


def test_unbind_keeps_an_override_the_session_workspace_binding_did_not_register(home, tmp_path):
    from gateway.session_policy import unbind_turn_workspace
    from tools import terminal_tool
    terminal_tool.register_task_env_overrides('acp-1', {'cwd': str(tmp_path), 'cwd_source': 'session'})
    unbind_turn_workspace('acp-1')
    assert terminal_tool.resolve_task_overrides('acp-1')['cwd'] == str(tmp_path)


def test_deleting_a_session_releases_its_workspace_binding(home, tmp_path):
    # Review (@kshitijk4poor W3): the bindings were never cleared on session close.
    from types import SimpleNamespace
    from gateway.session_mutations import retire_live_sessions
    from tools import terminal_tool
    ws1 = tmp_path / 'ws1'
    ws1.mkdir()
    with _turn(_policy(home, ws1, persistent=True), 'local-a'):
        pass
    assert terminal_tool.resolve_task_overrides('local-a')['cwd'] == str(ws1)
    authority = SimpleNamespace(runner=SimpleNamespace(), sessions={}, profile_id='fixture', instance_id='i', epoch=1)
    retire_live_sessions(authority, ['local-a'])
    assert terminal_tool.resolve_task_overrides('local-a') == {}


def test_idle_reaper_stops_workspace_containers_and_keeps_the_profile_one_running(monkeypatch):
    # Review (@kshitijk4poor W4): one RUNNING container per (profile, workspace), never reclaimed.
    import time
    from tools import terminal_tool, terminal_tool_lifecycle
    calls = []

    class Env:
        def __init__(self, name):
            self.name = name

        def cleanup(self):
            calls.append(('cleanup', self.name))  # persist mode: a no-op that leaves it running

        def stop_idle(self):
            calls.append(('stop', self.name))

    monkeypatch.setattr(terminal_tool, '_active_environments', {})
    monkeypatch.setattr(terminal_tool, '_last_activity', {})
    old = time.time() - 3600
    for key in ('default', 'default:ws:aaaa', 'profile:beta:ws:bbbb'):
        terminal_tool._active_environments[key] = Env(key)
        terminal_tool._last_activity[key] = old
    terminal_tool_lifecycle._cleanup_inactive_envs(lifetime_seconds=300)
    assert sorted(calls) == [('cleanup', 'default'), ('stop', 'default:ws:aaaa'),
                             ('stop', 'profile:beta:ws:bbbb')]
    assert terminal_tool._active_environments == {}


@pytest.mark.parametrize('persistent', [False, True])
def test_side_agent_runs_in_its_sessions_workspace(home, tmp_path, persistent):
    # Review (@kshitijk4poor S2, @dokterdok 4): /bg, /btw and preview restart never bound the workspace.
    from types import SimpleNamespace
    from gateway.session_tui_side import _side_scope
    from tools import terminal_tool
    ws1 = tmp_path / 'ws1'
    ws1.mkdir()
    policy = _policy(home, ws1, persistent=persistent)
    with _turn(policy, 'local-a'):
        session_key = _plan('local-a')[0]
    authority = SimpleNamespace(runner=SimpleNamespace(), profile_id='fixture')
    with _side_scope(authority, policy, 'bg_123456'):
        key, mount = _plan('bg_123456')
    assert mount == str(ws1)
    if persistent:
        assert key == session_key  # the session's own workspace container, not the profile's
    assert terminal_tool.resolve_task_overrides('bg_123456') == {}  # released with the side task


def test_managed_worker_turn_runs_in_its_sessions_workspace(home, tmp_path, monkeypatch):
    # Review (@kshitijk4poor S2): out-of-process (managed worker) turns never bound the workspace.
    from gateway import session_kanban
    from gateway.session_policy import policy_scope
    ws1 = tmp_path / 'ws1'
    ws1.mkdir()
    policy = _policy(home, ws1, persistent=True)
    seen = {}

    class Agent:
        def run_conversation(self, text, **kwargs):
            seen['task_id'] = kwargs.get('task_id')
            seen['plan'] = _plan(kwargs.get('task_id'))
            return {}

    frame = {'policy': asdict(policy), 'text': 'hi', 'route': 'agent:main:local:dm:worker-s1',
             'scope': {'session_id': 'worker-s1'}}
    with policy_scope(policy):  # agent/managed_worker.py: policy_scope(policy), no session key bound
        session_kanban.run_worker_turns(Agent(), frame, [])
    assert seen['task_id'] == 'worker-s1'
    assert ':ws:' in seen['plan'][0]
    assert seen['plan'][1] == str(ws1)
