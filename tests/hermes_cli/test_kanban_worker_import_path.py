"""Worker startup must use the dispatcher's install, even from a Hermes worktree."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd


@pytest.mark.parametrize("shadow", ["hermes_cli", "hermes_bootstrap"])
def test_worker_startup_ignores_workspace_packages(tmp_path, monkeypatch, shadow):
    home = tmp_path / "home"
    home.mkdir()
    home.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    monkeypatch.delenv("INVOCATION_ID", raising=False)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if shadow == "hermes_cli":
        package = workspace / shadow
        package.mkdir()
        target = package / "__init__.py"
    else:
        target = workspace / f"{shadow}.py"
    target.write_text("raise RuntimeError('workspace package imported')\n", encoding="utf-8")

    task = kb.Task(
        id="t_import_path", title="worker import path", body=None, assignee="default",
        status="running", priority=0, created_by="test", created_at=1,
        started_at=None, completed_at=None, workspace_kind="dir",
        workspace_path=str(workspace), claim_lock="test-claim", claim_expires=None,
        tenant=None, current_run_id=None,
    )
    # No run ID: real worker startup stops before contacting an owner or changing the board.
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    processes = []
    popen = subprocess.Popen

    def spawn(command, **kwargs):
        process = popen(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    log_path = tmp_path / "worker.log"
    with log_path.open("ab") as log:
        monkeypatch.setattr(kbd, "_open_worker_log", lambda *_: log)
        try:
            pid = kbd._default_spawn(task, str(workspace))
            assert pid == processes[0].pid
            assert processes[0].wait(timeout=20) == 1
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=20)

    output = log_path.read_text(encoding="utf-8")
    assert "Kanban owner refused: 'HERMES_KANBAN_RUN_ID'" in output, output
    assert "workspace package imported" not in output
