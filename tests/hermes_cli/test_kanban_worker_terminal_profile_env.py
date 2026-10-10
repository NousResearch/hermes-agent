"""Residual terminal-policy isolation at Kanban worker spawn boundaries."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.kanban_db import Task
from hermes_cli import kanban_db_dispatch as kbd


def _spawn(monkeypatch, workspace, assignee):
    captured = {}

    def capture(_cmd, **kwargs):
        captured.update(kwargs["env"])
        return SimpleNamespace(pid=4245)

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda _home: None)
    task = Task(
        id="t_policy", title="policy", body=None, assignee=assignee,
        status="running", priority=0, created_by=None, created_at=0,
        started_at=None, completed_at=None, workspace_kind="shared",
        workspace_path=str(workspace), claim_lock=None, claim_expires=None, tenant=None,
        max_runtime_seconds=900,
    )
    with monkeypatch.context() as spawn:
        spawn.setattr(subprocess, "Popen", capture)
        assert kbd._default_spawn(task, str(workspace)) == 4245
    return captured


@pytest.mark.parametrize("worker_mode", ["real", None])
def test_worker_home_policy_does_not_inherit_launch_bridge(monkeypatch, tmp_path, worker_mode):
    """Routed workers reload their own home policy; same-profile workers retain theirs."""
    from gateway.run import _bridge_terminal_config_to_env

    root = tmp_path / ".hermes"
    worker = root / "profiles" / "worker"
    worker.mkdir(parents=True)
    config = {"terminal": {"home_mode": worker_mode} if worker_mode else {}}
    (worker / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (root / "config.yaml").write_text("terminal:\n  home_mode: profile\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _bridge_terminal_config_to_env({"home_mode": "profile"})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parents[2])!r})\n"
        "from hermes_cli.env_loader import load_hermes_dotenv\n"
        "load_hermes_dotenv(hermes_home=os.environ['HERMES_HOME'])\n"
        "from hermes_cli.config import apply_terminal_config_to_env\n"
        "apply_terminal_config_to_env()\n"
        "from hermes_constants import get_subprocess_home\n"
        "print(json.dumps({'mode': os.getenv('TERMINAL_HOME_MODE', 'auto'), "
        "'home': get_subprocess_home()}))\n", encoding="utf-8",
    )
    for assignee in ("default", "worker", "default"):
        env = _spawn(monkeypatch, workspace, assignee)
        if assignee == "default":
            assert env["TERMINAL_HOME_MODE"] == "profile"
        else:
            assert "TERMINAL_HOME_MODE" not in env
            child = subprocess.run(
                [sys.executable, str(probe)], env=env, cwd=Path(__file__).parents[2],
                capture_output=True, text=True, check=True, timeout=30,
            )
            effective = json.loads(child.stdout)
            assert effective["mode"] == (worker_mode or "auto")
            if worker_mode == "real":
                assert effective["home"] is None
        assert os.environ["TERMINAL_HOME_MODE"] == "profile"


def test_deferred_profile_cleanup_precedes_workspace_and_runtime_pins(monkeypatch, tmp_path):
    """Even unresolved selectors lose launch policy before task-owned overrides are applied."""
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP

    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in TERMINAL_CONFIG_ENV_MAP.values():
        monkeypatch.setenv(name, "launch-policy")
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
    monkeypatch.setenv("TERMINAL_UNRELATED", "preserved")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env = _spawn(monkeypatch, workspace, "not-created-yet")
    pins = {"TERMINAL_CWD", "TERMINAL_TIMEOUT"}
    assert not (set(TERMINAL_CONFIG_ENV_MAP.values()) - pins) & env.keys()
    assert env["TERMINAL_UNRELATED"] == "preserved"
    assert env["HERMES_PROFILE"] == "not-created-yet"
    assert env["TERMINAL_CWD"] == str(workspace)
    assert env["TERMINAL_TIMEOUT"] == str(900 - kbd.KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS)
    assert env["TERMINAL_MAX_FOREGROUND_TIMEOUT"] == env["TERMINAL_TIMEOUT"]
    assert os.environ["TERMINAL_HOME_MODE"] == "profile"
