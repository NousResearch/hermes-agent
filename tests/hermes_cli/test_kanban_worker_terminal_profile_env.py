"""Profile-scoped terminal config for dispatcher-spawned Kanban workers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import yaml

from hermes_cli import kanban_db as kb


def test_spawn_rebinds_terminal_env_to_assignee_profile(monkeypatch, tmp_path):
    """Gateway terminal env must not override the assigned worker profile.

    The long-lived gateway mirrors its own ``terminal.*`` config into
    ``TERMINAL_*`` variables. A Kanban worker switches ``HERMES_HOME`` to the
    assignee profile, so those inherited variables must be rebound as well.
    Profile omissions resolve to that profile's defaults instead of leaking
    values from the gateway profile.
    """
    hermes_root = tmp_path / ".hermes"
    worker_home = hermes_root / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    (worker_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "terminal": {
                    "backend": "docker",
                    "timeout": 321,
                    "docker_image": "worker/image:latest",
                    "docker_forward_env": ["WORKER_TOKEN"],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_root))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "17")
    monkeypatch.setenv("TERMINAL_HOME_MODE", "project")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "gateway/image:latest")
    monkeypatch.setenv("TERMINAL_DOCKER_FORWARD_ENV", "[]")
    monkeypatch.setenv(
        "TERMINAL_DOCKER_ENV",
        json.dumps({"GATEWAY_ONLY": "must-not-leak"}),
    )
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)

    captured: dict[str, object] = {}
    real_popen = subprocess.Popen

    class FakeProc:
        pid = 4245

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = kb.Task(
        id="t_profile_env",
        title="profile env",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="shared",
        workspace_path=str(workspace),
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )

    assert kb._default_spawn(task, str(workspace)) == 4245

    assert cast(list[str], captured["cmd"])[:3] == ["hermes", "-p", "worker"]
    env = cast(dict[str, str], captured["env"])
    assert env["HERMES_HOME"] == str(worker_home)
    assert "TERMINAL_ENV" not in env
    assert "TERMINAL_TIMEOUT" not in env
    assert "TERMINAL_HOME_MODE" not in env
    assert "TERMINAL_DOCKER_IMAGE" not in env
    assert "TERMINAL_DOCKER_FORWARD_ENV" not in env
    assert "TERMINAL_DOCKER_ENV" not in env
    assert env["TERMINAL_CWD"] == str(workspace)

    # Exercise the child-side fallback bridge with the exact environment that
    # _default_spawn produced. This closes the loop from dispatcher isolation
    # through assignee config loading to terminal_tool's effective settings.
    monkeypatch.setattr(subprocess, "Popen", real_popen)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import os

from hermes_cli.env_loader import load_hermes_dotenv

load_hermes_dotenv(hermes_home=os.environ["HERMES_HOME"])

from tools.terminal_tool import _get_env_config

cfg = _get_env_config()
print(json.dumps({
    "env_type": cfg["env_type"],
    "docker_forward_env": cfg["docker_forward_env"],
    "docker_image": cfg["docker_image"],
    "docker_env": cfg["docker_env"],
    "timeout": cfg["timeout"],
}))
""",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    effective = json.loads(child.stdout)
    assert effective == {
        "env_type": "docker",
        "docker_forward_env": ["WORKER_TOKEN"],
        "docker_image": "worker/image:latest",
        "docker_env": {},
        "timeout": 321,
    }


def test_spawn_clears_gateway_terminal_env_when_profile_resolution_is_deferred(
    monkeypatch,
    tmp_path,
):
    """A missing assignee directory must not preserve the gateway bridge."""
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_DOCKER_FORWARD_ENV", '["GATEWAY_TOKEN"]')
    monkeypatch.setenv(
        "TERMINAL_DOCKER_ENV",
        json.dumps({"GATEWAY_ONLY": "must-not-leak"}),
    )
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)

    captured: dict[str, object] = {}

    class FakeProc:
        pid = 4246

    def fake_popen(_cmd, *args, **kwargs):
        captured["env"] = dict(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = kb.Task(
        id="t_deferred_profile_env",
        title="deferred profile env",
        body=None,
        assignee="not-created-yet",
        status="running",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="shared",
        workspace_path=str(workspace),
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )

    assert kb._default_spawn(task, str(workspace)) == 4246

    env = cast(dict[str, str], captured["env"])
    assert env["HERMES_PROFILE"] == "not-created-yet"
    assert env["HERMES_HOME"] == str(hermes_root)
    assert "TERMINAL_ENV" not in env
    assert "TERMINAL_DOCKER_FORWARD_ENV" not in env
    assert "TERMINAL_DOCKER_ENV" not in env
    assert env["TERMINAL_CWD"] == str(workspace)
