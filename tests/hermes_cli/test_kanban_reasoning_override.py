"""Tests for per-task Kanban reasoning overrides."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_create_task_stores_reasoning_override(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="needs deeper thought",
            assignee="dante",
            reasoning_override="high",
        )
        task = kb.get_task(conn, tid)

    assert task.reasoning_override == "high"


def test_run_slash_create_reasoning_override_is_visible_in_json_and_show(kanban_home):
    out = kc.run_slash(
        "create 'needs deeper thought' --assignee dante --reasoning high --json"
    )
    payload = json.loads(out)

    assert payload["reasoning_override"] == "high"

    show = kc.run_slash(f"show {payload['id']}")
    assert "reasoning: high" in show


def test_default_spawn_passes_reasoning_override_to_worker_cli(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "dante"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("toolsets:\n  - hermes-cli\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = kb.Task(
        id="t_reasoning",
        title="reasoning task",
        body=None,
        assignee="dante",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
        reasoning_override="high",
    )

    pid = kb._default_spawn(task, str(workspace))

    assert pid == 4242
    assert "--reasoning" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--reasoning") + 1] == "high"


def test_kanban_create_tool_accepts_reasoning_override(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_PROFILE", "wren")

    from tools import kanban_tools

    result = json.loads(
        kanban_tools._handle_create(
            {
                "title": "review carefully",
                "assignee": "dante",
                "reasoning_override": "high",
            }
        )
    )

    assert result["ok"] is True
    with kb.connect() as conn:
        task = kb.get_task(conn, result["task_id"])
    assert task.reasoning_override == "high"
