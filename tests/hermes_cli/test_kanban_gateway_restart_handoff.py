"""Managed-gateway isolation for dispatcher-owned Kanban workers."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def worker_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, kb.Task]:
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    workspace = tmp_path / "candidate-worktree"
    workspace.mkdir()
    task = kb.Task(
        id="t_candidate_restart",
        title="activate candidate",
        body=None,
        assignee="coder",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=1,
        completed_at=None,
        workspace_kind="worktree",
        workspace_path=str(workspace),
        claim_lock="host:dispatcher",
        claim_expires=999,
        tenant=None,
        branch_name="wt/t_candidate_restart",
        current_run_id=23,
    )
    return workspace, task


@pytest.mark.linux_only
def test_managed_gateway_worker_is_spawned_in_restart_safe_scope(
    worker_setup: tuple[Path, kb.Task], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, task = worker_setup
    captured_cmd: list[str] = []
    captured_env: dict[str, str] = {}
    captured_cwd: str | None = None

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        nonlocal captured_cwd
        captured_cmd.extend(cmd)
        captured_env.update(kwargs.get("env") or {})
        captured_cwd = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-profile")
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("tools.process_registry._worker_memory_max_bytes", lambda: 536_870_912)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

    assert kbd._default_spawn(task, str(workspace)) == 4242
    assert captured_cmd[:4] == ["/usr/bin/systemd-run", "--user", "--scope", "--quiet"]
    unit_index = captured_cmd.index("--unit")
    assert captured_cmd[unit_index + 1] == "hermes-worker-kanban-t_candidate_restart-run-23"
    assert "MemoryMax=536870912" in captured_cmd
    separator = captured_cmd.index("--")
    assert captured_cmd[separator + 1 : separator + 4] == ["hermes", "-p", "coder"]
    assert captured_cwd == str(workspace)
    assert captured_env["HERMES_KANBAN_TASK"] == task.id
    assert captured_env["HERMES_KANBAN_RUN_ID"] == "23"
    assert "ANTHROPIC_API_KEY" not in captured_env


@pytest.mark.linux_only
def test_managed_gateway_worker_spawn_fails_closed_without_scope(
    worker_setup: tuple[Path, kb.Task], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, task = worker_setup
    popen_calls: list[list[str]] = []
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: popen_calls.append(list(cmd)))
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: False)

    with pytest.raises(RuntimeError, match="restart-safe systemd scope"):
        kbd._default_spawn(task, str(workspace))
    assert popen_calls == []


@pytest.mark.linux_only
def test_managed_gateway_scope_builder_fails_closed_if_binary_disappears(
    worker_setup: tuple[Path, kb.Task], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, task = worker_setup
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("unsafe direct spawn"))

    with pytest.raises(RuntimeError, match="restart-safe systemd scope"):
        kbd._default_spawn(task, str(workspace))


def test_standalone_dispatcher_keeps_direct_worker_spawn(
    worker_setup: tuple[Path, kb.Task], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, task = worker_setup
    captured_cmd: list[str] = []

    class FakeProc:
        pid = 4243

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kwargs: captured_cmd.extend(cmd) or FakeProc())
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(
        "tools.process_registry._systemd_run_user_scope_available",
        lambda: pytest.fail("scope probe must not run outside managed gateway"),
    )

    assert kbd._default_spawn(task, str(workspace)) == 4243
    assert captured_cmd[:3] == ["hermes", "-p", "coder"]


@pytest.mark.linux_only
def test_real_user_systemd_scope_preserves_worker_context(
    worker_setup: tuple[Path, kb.Task], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import process_registry

    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("systemd-run --user --scope is unavailable on this host")

    workspace, task = worker_setup
    receipt = workspace / "worker-receipt.json"
    script = (
        "import json, os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
        "'pid': os.getpid(), 'cwd': os.getcwd(), "
        "'task': os.environ.get('HERMES_KANBAN_TASK'), "
        "'run': os.environ.get('HERMES_KANBAN_RUN_ID'), "
        "'cgroup': pathlib.Path('/proc/self/cgroup').read_text()})); time.sleep(0.5)"
    )
    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: [sys.executable, "-c", script, str(receipt)])
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)

    pid = kbd._default_spawn(task, str(workspace))
    deadline = time.monotonic() + 5
    while not receipt.exists() and time.monotonic() < deadline:
        time.sleep(0.05)

    assert receipt.exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["pid"] == pid
    assert payload["cwd"] == str(workspace)
    assert payload["task"] == task.id
    assert payload["run"] == "23"
    assert ".scope" in payload["cgroup"]
    assert "hermes-gateway.service" not in payload["cgroup"]
