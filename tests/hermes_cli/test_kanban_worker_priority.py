"""Focused tests for child-only Kanban worker priority wrappers."""

from __future__ import annotations

import logging
import subprocess

import pytest

from hermes_cli import kanban_db as kb


BASE_CMD = [
    "/venv/bin/hermes",
    "-p",
    "worker",
    "--cli",
    "--accept-hooks",
    "chat",
    "-q",
    "work kanban task t_priority",
]


def _task() -> kb.Task:
    return kb.Task(
        id="t_priority",
        title="priority test",
        body=None,
        assignee="worker",
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
        current_run_id=1,
    )


def _prepare_spawn(monkeypatch, tmp_path, kanban_cfg, resolver):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(kb, "_IS_WINDOWS", False)
    monkeypatch.setattr(kb, "_IS_LINUX", True)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["/venv/bin/hermes"])
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(kb, "_safe_which_no_cwd", resolver)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": dict(kanban_cfg)},
    )
    return workspace


def _capture_popen(monkeypatch):
    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return captured


def test_default_priority_config_preserves_worker_argv(monkeypatch, tmp_path):
    def unexpected_lookup(command):
        pytest.fail(f"default config must not resolve priority command {command!r}")

    workspace = _prepare_spawn(monkeypatch, tmp_path, {}, unexpected_lookup)
    captured = _capture_popen(monkeypatch)

    pid = kb._default_spawn(_task(), str(workspace))

    assert pid == 4242
    assert captured["cmd"] == BASE_CMD
    assert captured["kwargs"]["start_new_session"] is True
    assert "shell" not in captured["kwargs"]


def test_nice_and_idle_wrap_only_worker_argv(monkeypatch, tmp_path):
    resolved = {"nice": "/usr/bin/nice", "ionice": "/usr/bin/ionice"}
    workspace = _prepare_spawn(
        monkeypatch,
        tmp_path,
        {"worker_nice": 15, "worker_ionice_class": "idle"},
        resolved.get,
    )
    captured = _capture_popen(monkeypatch)

    pid = kb._default_spawn(_task(), str(workspace))

    assert pid == 4242
    assert captured["cmd"] == [
        "/usr/bin/nice",
        "-n",
        "15",
        "/usr/bin/ionice",
        "-c",
        "3",
        *BASE_CMD,
    ]
    assert "shell" not in captured["kwargs"]


@pytest.mark.parametrize(
    "kanban_cfg",
    [
        {"worker_nice": 20, "worker_ionice_class": "realtime"},
        {"worker_nice": True, "worker_ionice_class": 3},
    ],
)
def test_invalid_priority_values_warn_and_do_not_wrap(
    monkeypatch, tmp_path, caplog, kanban_cfg
):
    def unexpected_lookup(command):
        pytest.fail(f"invalid config must not resolve priority command {command!r}")

    workspace = _prepare_spawn(
        monkeypatch, tmp_path, kanban_cfg, unexpected_lookup
    )
    captured = _capture_popen(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        pid = kb._default_spawn(_task(), str(workspace))

    assert pid == 4242
    assert captured["cmd"] == BASE_CMD
    messages = [record.getMessage() for record in caplog.records]
    assert any("kanban.worker_nice" in message for message in messages)
    assert any("kanban.worker_ionice_class" in message for message in messages)


def test_missing_priority_executable_falls_back_to_normal_worker(
    monkeypatch, tmp_path, caplog
):
    resolved = {"nice": "/usr/bin/nice", "ionice": None}
    workspace = _prepare_spawn(
        monkeypatch,
        tmp_path,
        {"worker_nice": 15, "worker_ionice_class": "idle"},
        resolved.get,
    )
    captured = _capture_popen(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        pid = kb._default_spawn(_task(), str(workspace))

    assert pid == 4242
    assert captured["cmd"] == BASE_CMD
    assert any("ionice" in record.getMessage() for record in caplog.records)


def test_non_linux_idle_falls_back_without_lookup(monkeypatch, tmp_path, caplog):
    def unexpected_lookup(command):
        pytest.fail(f"non-Linux idle mode must not resolve {command!r}")

    workspace = _prepare_spawn(
        monkeypatch,
        tmp_path,
        {"worker_ionice_class": "idle"},
        unexpected_lookup,
    )
    monkeypatch.setattr(kb, "_IS_LINUX", False)
    captured = _capture_popen(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        kb._default_spawn(_task(), str(workspace))

    assert captured["cmd"] == BASE_CMD
    assert any("requires Linux" in record.getMessage() for record in caplog.records)


def test_windows_falls_back_without_resolving_wrappers(monkeypatch, tmp_path):
    def unexpected_lookup(command):
        pytest.fail(f"Windows must not resolve priority command {command!r}")

    workspace = _prepare_spawn(
        monkeypatch,
        tmp_path,
        {"worker_nice": 15, "worker_ionice_class": "idle"},
        unexpected_lookup,
    )
    captured = _capture_popen(monkeypatch)
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    pid = kb._default_spawn(_task(), str(workspace))

    assert pid == 4242
    assert captured["cmd"] == BASE_CMD
    assert captured["kwargs"]["creationflags"] == 0x08000000


def test_wrapper_launch_race_retries_unwrapped(monkeypatch, tmp_path, caplog):
    workspace = _prepare_spawn(
        monkeypatch,
        tmp_path,
        {"worker_nice": 15},
        lambda command: "/usr/bin/nice" if command == "nice" else None,
    )
    calls = []

    class FakeProc:
        pid = 4343

    def fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        if len(calls) == 1:
            raise FileNotFoundError("wrapper disappeared")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        pid = kb._default_spawn(_task(), str(workspace))

    assert pid == 4343
    assert calls == [["/usr/bin/nice", "-n", "15", *BASE_CMD], BASE_CMD]
    assert any("retrying without" in record.getMessage() for record in caplog.records)
