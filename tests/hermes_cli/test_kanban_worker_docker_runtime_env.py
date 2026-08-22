"""Dispatcher → worker regression coverage for #91568."""
from __future__ import annotations

import subprocess

from hermes_cli.kanban_runtime import (
    KANBAN_TERMINAL_RUNTIME_ENV,
    decode_kanban_terminal_runtime,
)


def _make_task(kb, *, assignee: str = "w", workspace_kind: str = "dir"):
    return kb.Task(
        id="t_runtime",
        title="docker runtime",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind=workspace_kind,
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=1,
    )


def test_default_spawn_emits_task_scoped_runtime(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(_make_task(kb), str(workspace))

    env = captured["env"]
    runtime = decode_kanban_terminal_runtime(
        env[KANBAN_TERMINAL_RUNTIME_ENV],
        expected_task_id="t_runtime",
        expected_workspace=workspace,
    )
    assert runtime["mounts"][0]["target"] == "/workspace"
    assert env["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert env["TERMINAL_CWD"] == str(workspace)
