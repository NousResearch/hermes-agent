import json
from types import SimpleNamespace

import pytest

from acp_adapter import session as acp_session
from hermes_cli.config import validate_config_structure
from hermes_state import SessionDB
from tools import code_execution_tool as code_tool
from tools import file_tools
from tools import terminal_tool
from tools import terminal_tool_backends
from tools.environments import ssh as ssh_env


ROUTE = {
    "backend": "ssh",
    "host": "workspace.example",
    "user": "developer",
    "port": 2222,
    "key": "~/.ssh/workspace",
    "sync": False,
}


def _agent():
    return SimpleNamespace(model="test-model", _session_db=None, _session_db_created=False)


def test_acp_workspace_route_is_captured_persisted_and_restored_fail_closed(
    tmp_path, monkeypatch
):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"acp": {"workspace": ROUTE}},
    )
    session_ids = []
    try:
        manager = acp_session.SessionManager(agent_factory=_agent, db=db)
        state = manager.create_session(cwd="/workspace/project")
        session_ids.append(state.session_id)

        assert state.workspace_route == ROUTE
        assert terminal_tool.resolve_task_overrides(state.session_id) == {
            "cwd": "/workspace/project",
            "env_type": "ssh",
            "ssh_host": "workspace.example",
            "ssh_user": "developer",
            "ssh_port": 2222,
            "ssh_key": "~/.ssh/workspace",
            "ssh_sync": False,
        }
        state.history.append({"role": "user", "content": "hello"})
        manager.save_session(state.session_id)
        row = db.get_session(state.session_id)
        assert row is not None
        persisted = json.loads(row["model_config"])
        assert persisted["workspace_route"] == ROUTE

        restored_manager = acp_session.SessionManager(agent_factory=_agent, db=db)
        restored = restored_manager.get_session(state.session_id)
        assert restored is not None and restored.workspace_route == ROUTE

        db.update_session_meta(
            state.session_id,
            json.dumps(
                {
                    "cwd": "/workspace/project",
                    "workspace_route": {
                        "backend": "ssh",
                        "host": "workspace.example",
                    },
                }
            ),
            "test-model",
        )
        terminal_tool.clear_task_env_overrides(state.session_id)
        corrupt_manager = acp_session.SessionManager(agent_factory=_agent, db=db)
        with pytest.raises(ValueError, match="persisted ACP workspace route"):
            corrupt_manager.get_session(state.session_id)

        issues = validate_config_structure(
            {"acp": {"workspace": {"backend": "ssh", "host": "workspace.example"}}}
        )
        assert any("acp.workspace.user" in issue.message for issue in issues)

        def fail_registration(_task_id, _overrides):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(
            terminal_tool, "register_task_env_overrides", fail_registration
        )
        with pytest.raises(RuntimeError, match="register ACP SSH workspace route"):
            manager.create_session(cwd="/workspace/other")
    finally:
        for session_id in session_ids:
            terminal_tool.clear_task_env_overrides(session_id)
        db.close()


def test_delegated_acp_route_reaches_all_tools_and_sync_false_needs_no_scp(
    monkeypatch,
):
    parent_id, child_id = "acp-parent", "delegated-child"
    base_config = {
        "env_type": "local",
        "cwd": "/srv/hermes",
        "timeout": 180,
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    captures = {}

    class DummyEnv:
        cwd = "/workspace/project"

    def fake_configured(config, env_type, **kwargs):
        captures["file"] = (config, env_type, kwargs)
        return DummyEnv()

    def fake_backend(**kwargs):
        captures["code"] = kwargs
        return DummyEnv()

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: dict(base_config))
    monkeypatch.setattr(terminal_tool, "_create_configured_env", fake_configured)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        "tools.terminal_scope.enforce_no_refusal",
        lambda: None,
    )
    monkeypatch.setattr(terminal_tool_backends, "_create_environment", fake_backend)

    acp_session._register_task_cwd(parent_id, "/workspace/project", ROUTE)
    terminal_tool.register_container_alias(child_id, parent_id)
    try:
        assert terminal_tool._resolve_container_task_id(child_id) == parent_id
        plan = terminal_tool._plan_execution(
            "pwd", task_id=child_id, timeout=None, background=False, _host_local=False
        )
        assert plan.env_type == "ssh"
        assert plan.cwd == "/workspace/project"
        assert terminal_tool_backends._ssh_config_from_config(plan.config) == {
            "host": "workspace.example",
            "user": "developer",
            "port": 2222,
            "key": "~/.ssh/workspace",
            "persistent": False,
            "sync": False,
        }

        host_plan = terminal_tool._plan_execution(
            "pwd", task_id=parent_id, timeout=None, background=False, _host_local=True
        )
        assert host_plan.env_type == "local"
        assert host_plan.cwd == "/srv/hermes"

        file_env_type, _ = file_tools._create_terminal_env_for_file_ops(
            child_id, parent_id
        )
        assert file_env_type == "ssh"
        assert captures["file"][0]["ssh_host"] == "workspace.example"

        _, code_env_type = code_tool._get_or_create_env(child_id)
        assert code_env_type == "ssh"
        assert captures["code"]["ssh_config"]["sync"] is False
    finally:
        with terminal_tool._env_lock:
            terminal_tool._active_environments.clear()
            terminal_tool._last_activity.clear()
        file_tools.clear_file_ops_cache()
        terminal_tool.clear_task_env_overrides(child_id)
        terminal_tool.clear_task_env_overrides(parent_id)

    sync_calls = {"dirs": 0, "manager": 0}
    monkeypatch.setattr(
        ssh_env.shutil,
        "which",
        lambda command: "/usr/bin/ssh" if command == "ssh" else None,
    )
    monkeypatch.setattr(ssh_env.SSHEnvironment, "_establish_connection", lambda self: None)
    monkeypatch.setattr(
        ssh_env.SSHEnvironment, "_detect_remote_home", lambda self: "/home/developer"
    )
    monkeypatch.setattr(
        ssh_env.SSHEnvironment,
        "_ensure_remote_dirs",
        lambda self: sync_calls.__setitem__("dirs", sync_calls["dirs"] + 1),
    )
    monkeypatch.setattr(ssh_env.SSHEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        ssh_env,
        "FileSyncManager",
        lambda **_kwargs: sync_calls.__setitem__(
            "manager", sync_calls["manager"] + 1
        ),
    )

    env = ssh_env.SSHEnvironment(
        host="workspace.example",
        user="developer",
        cwd="/workspace/project",
        sync=False,
    )
    env._before_execute()

    assert sync_calls == {"dirs": 0, "manager": 0}
    assert env._sync_manager is None
