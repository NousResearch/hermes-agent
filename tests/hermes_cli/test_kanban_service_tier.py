"""Kanban-scoped service-tier override resolution and worker propagation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    connection = kb.connect()
    yield connection
    connection.close()


def _spawn_and_capture(monkeypatch, tmp_path, task):
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured = {}

    class FakeProc:
        pid = 4246

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    kb._default_spawn(task, str(workspace))
    return captured["cmd"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        (" fast ", "fast"),
        ("priority", "fast"),
        ("normal", "normal"),
        ("off", "normal"),
        ("invalid-tier", None),
    ],
)
def test_kanban_service_tier_config_normalizes(raw, expected):
    assert kb.kanban_service_tier_config({"service_tier": raw}) == expected


@pytest.mark.parametrize("assignee", ["default", "worker"])
def test_spawn_passes_normal_service_tier_override(
    monkeypatch, tmp_path, conn, assignee
):
    task_id = kb.create_task(conn, title="t", assignee=assignee)
    task = kb.get_task(conn, task_id)
    monkeypatch.setattr(kb, "kanban_service_tier_config", lambda: "normal")

    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)

    index = cmd.index("--service-tier")
    assert cmd[index + 1] == "normal"
    assert index > cmd.index("chat")
    assert cmd[cmd.index("-p") + 1] == assignee


def test_spawn_passes_fast_service_tier_override(monkeypatch, tmp_path, conn):
    task_id = kb.create_task(conn, title="t", assignee="worker")
    task = kb.get_task(conn, task_id)
    monkeypatch.setattr(kb, "kanban_service_tier_config", lambda: "fast")

    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)

    index = cmd.index("--service-tier")
    assert cmd[index + 1] == "fast"


def test_spawn_omits_service_tier_when_unset(monkeypatch, tmp_path, conn):
    task_id = kb.create_task(conn, title="t", assignee="worker")
    task = kb.get_task(conn, task_id)
    monkeypatch.setattr(kb, "kanban_service_tier_config", lambda: None)

    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)

    assert "--service-tier" not in cmd


def test_spawn_reads_service_tier_from_config(monkeypatch, tmp_path, conn):
    from hermes_cli import config as config_mod

    task_id = kb.create_task(conn, title="t", assignee="worker")
    task = kb.get_task(conn, task_id)
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"kanban": {"service_tier": "normal"}},
    )

    cmd = _spawn_and_capture(monkeypatch, tmp_path, task)

    index = cmd.index("--service-tier")
    assert cmd[index + 1] == "normal"


def test_worker_cli_accepts_service_tier_flag():
    from hermes_cli._parser import build_top_level_parser

    parser = build_top_level_parser()[0]
    args = parser.parse_args(
        ["--cli", "chat", "-q", "hi", "--service-tier", "normal"]
    )
    assert args.service_tier == "normal"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--cli", "--service-tier", "normal", "chat", "-q", "hi"], "normal"),
        (["--service-tier", "fast", "-z", "hi"], "fast"),
    ],
)
def test_top_level_service_tier_flag_is_preserved(argv, expected):
    from hermes_cli._parser import build_top_level_parser

    parser = build_top_level_parser()[0]
    args = parser.parse_args(argv)

    assert args.service_tier == expected
