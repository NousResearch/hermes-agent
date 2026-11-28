"""Deferred profile resolution must not forward the launch terminal policy."""
from contextlib import closing
from pathlib import Path
import os
import subprocess
from types import SimpleNamespace


def test_disappearing_assignee_scrubs_terminal_policy_before_task_pins(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as dispatch
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    from hermes_cli.kanban_db_connect import connect
    from tools import process_registry
    import hermes_constants

    root = tmp_path / "hermes-data"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("terminal:\n  backend: docker\n")
    (root / "config.yaml").write_text("{}\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", str(root))
    for key, value in {
        "HOME": str(tmp_path), "HERMES_HOME": str(root), "HERMES_KANBAN_HOME": str(root),
        "HERMES_RUNTIME_DIR": str(tmp_path / "runtime"), "TERMINAL_ENV": "docker",
        "TERMINAL_HOME_MODE": "profile", "TERMINAL_DOCKER_IMAGE": "launch-image",
        "terminal_docker_image": "lowercase-image", "terminal_home_mode": "profile",
        "TERMINAL_DOCKER_FORWARD_ENV": '["LAUNCH_SECRET"]', "TERMINAL_CWD": "launch-cwd",
        "TERMINAL_TIMEOUT": "17", "TERMINAL_MAX_FOREGROUND_TIMEOUT": "19",
        "TERMINAL_UNRELATED": "preserved",
    }.items():
        monkeypatch.setenv(key, value)

    captured = {}
    def popen(argv, **kwargs):
        captured.update(env=dict(kwargs["env"]), cwd=kwargs["cwd"])
        return SimpleNamespace(pid=424242)

    monkeypatch.setattr(dispatch, "_restart_safe_worker_argv", lambda task, argv: argv)
    monkeypatch.setattr(process_registry, "systemd_user_bus_env", lambda env: env)
    monkeypatch.setattr(dispatch, "_open_worker_log", lambda task, board: open(os.devnull, "wb"))
    monkeypatch.setattr(subprocess, "Popen", popen)
    parked = profile.with_name("parked")
    def boundary(task, workspace, board=None):
        assert profile.exists()
        profile.rename(parked)
        return dispatch._default_spawn(task, workspace, board=board)

    kb.create_board("owned")
    with closing(connect(board="owned")) as conn:
        task_id = kb.create_task(conn, title="race", assignee="worker", workspace_kind="dir",
                                 workspace_path=str(workspace), max_runtime_seconds=900)
        result = dispatch.dispatch_once(conn, spawn_fn=boundary, board="owned", reconcile_orphans=False)
    assert result.spawned and result.spawned[0][0] == task_id
    assert parked.exists() and not profile.exists()
    env = captured["env"]
    assert env["HERMES_PROFILE"] == "worker"
    assert env["HERMES_KANBAN_BOARD"] == "owned"
    assert env["HERMES_KANBAN_TASK"] == task_id
    assert env["HERMES_KANBAN_RUN_ID"] and env["HERMES_KANBAN_CLAIM_LOCK"]
    assert env["HERMES_SESSION_SOURCE"] == "kanban"
    assert captured["cwd"] == env["TERMINAL_CWD"] == str(workspace)
    assert env["TERMINAL_TIMEOUT"] == env["TERMINAL_MAX_FOREGROUND_TIMEOUT"] == "870"
    assert env["TERMINAL_UNRELATED"] == "preserved"
    assert os.environ["TERMINAL_HOME_MODE"] == "profile"
    assert not {k for k in env if k.upper() in {
        "TERMINAL_ENV", "TERMINAL_HOME_MODE", "TERMINAL_DOCKER_IMAGE", "TERMINAL_DOCKER_FORWARD_ENV"
    }}
    pins = {"TERMINAL_CWD", "TERMINAL_TIMEOUT"}
    residue = {key for key in env if key.upper() in set(TERMINAL_CONFIG_ENV_MAP.values()) - pins}
    assert not residue
