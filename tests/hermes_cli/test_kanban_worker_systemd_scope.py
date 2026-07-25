"""Focused lifecycle and fallback coverage for Kanban worker scopes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _task(task_id="task-1", run_id=7):
    return SimpleNamespace(id=task_id, current_run_id=run_id)


def test_scope_unit_identity_is_canonical_board_and_run_bound(monkeypatch):
    monkeypatch.setattr(
        kb,
        "kanban_db_path",
        lambda board=None: Path("/srv/hermes") / (board or "default") / "kanban.db",
    )

    first = kb._systemd_scope_unit_name("task with text", 11, board="alpha")
    same = kb._systemd_scope_unit_name("task with text", 11, board="alpha")
    other_run = kb._systemd_scope_unit_name("task with text", 12, board="alpha")
    other_board = kb._systemd_scope_unit_name("task with text", 11, board="beta")

    assert first == same
    assert first != other_run
    assert first != other_board
    assert kb._SYSTEMD_WORKER_SCOPE_RE.fullmatch(first)
    assert "task with text" not in first


@pytest.mark.parametrize(
    ("platform", "cgroup", "run_id", "runner", "ready"),
    [
        ("darwin", "/user.slice/user-1000.slice/user@1000.service/hermes.service", 1, "/usr/bin/systemd-run", True),
        ("linux", "/system.slice/hermes.service", 1, "/usr/bin/systemd-run", True),
        ("linux", "/user.slice/user-1000.slice/user@1000.service", 1, "/usr/bin/systemd-run", True),
        ("linux", "/user.slice/user-1000.slice/user@1000.service/app.slice/app.scope", 1, "/usr/bin/systemd-run", True),
        ("linux", "/user.slice/user-1000.slice/user@1000.service/hermes.service", None, "/usr/bin/systemd-run", True),
        ("linux", "/user.slice/user-1000.slice/user@1000.service/hermes.service", 1, None, True),
        ("linux", "/user.slice/user-1000.slice/user@1000.service/hermes.service", 1, "/usr/bin/systemd-run", False),
    ],
)
def test_scope_capability_fail_open(
    monkeypatch, platform, cgroup, run_id, runner, ready
):
    monkeypatch.setattr(kb.sys, "platform", platform)
    target = kb._SystemdUserManagerTarget(
        os.getuid(), Path("/run/user") / str(os.getuid()), Path("/run/user") / str(os.getuid()) / "bus"
    )
    monkeypatch.setattr(
        kb,
        "_systemd_user_manager_target_for_cgroup",
        lambda _: target
        if cgroup.endswith("hermes.service") and "/system.slice/" not in cgroup
        else None,
    )
    monkeypatch.setattr(kb.shutil, "which", lambda name: runner if name == "systemd-run" else "/usr/bin/systemctl")

    original = ["/usr/bin/hermes", "chat", "-q", "work"]
    scoped, unit, manager = kb._systemd_scope_argv(
        original,
        _task(run_id=run_id),
        cgroup_path=cgroup,
        user_manager_ready=ready,
    )

    assert scoped is original
    assert unit is None
    assert manager is None


def test_scope_verification_failure_stops_exact_unit(monkeypatch):
    unit = "hermes-kanban-worker-0123456789abcdef0123456789abcdef.scope"
    target = kb._SystemdUserManagerTarget(
        os.getuid(), Path("/run/user") / str(os.getuid()), Path("/run/user") / str(os.getuid()) / "bus"
    )

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("systemd-run", timeout)
            return 0

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    proc = FakeProc()
    monkeypatch.setattr(kb, "_SYSTEMD_SCOPE_VERIFY_TIMEOUT", 0.01)
    monkeypatch.setattr(
        kb,
        "_systemd_scope_properties",
        lambda *args, **kwargs: {
            "LoadState": "loaded",
            "ActiveState": "active",
            "ControlGroup": "/expected.scope",
        },
    )
    monkeypatch.setattr(kb, "_process_cgroup_path", lambda pid: "/wrong.scope")
    with pytest.raises(RuntimeError, match="could not verify systemd scope"):
        kb._verify_systemd_scope_pid(proc, unit, target)

    stop_calls = []
    monkeypatch.setattr(kb.shutil, "which", lambda name: "/usr/bin/systemctl")
    monkeypatch.setattr(
        kb.subprocess,
        "run",
        lambda argv, **kwargs: stop_calls.append((argv, kwargs)) or SimpleNamespace(returncode=0, stdout=""),
    )
    kb._cleanup_systemd_scope_launch(proc, unit, target)
    assert stop_calls[0][0] == ["/usr/bin/systemctl", "--user", "stop", unit]
    assert getattr(proc, "terminated", False)


def test_default_spawn_receipt_reaches_spawned_event(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    target = kb._SystemdUserManagerTarget(
        os.getuid(), Path("/run/user") / str(os.getuid()), Path("/run/user") / str(os.getuid()) / "bus"
    )
    captured = {}

    class FakeProc:
        pid = 6789

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return FakeProc()

    def fake_scope_argv(cmd, task, **kwargs):
        unit = kb._systemd_scope_unit_name(task.id, task.current_run_id)
        captured["unit"] = unit
        return ["/usr/bin/systemd-run", "--user", "--scope", "--unit=" + unit, "--", *cmd], unit, target

    monkeypatch.setattr(kb, "_systemd_scope_argv", fake_scope_argv)
    monkeypatch.setattr(kb, "_verify_systemd_scope_pid", lambda *args: None)
    monkeypatch.setattr(kb.subprocess, "Popen", fake_popen)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="scoped", assignee="worker")
        result = kb.dispatch_once(conn, spawn_fn=kb._default_spawn)
        event = next(event for event in kb.list_events(conn, task_id) if event.kind == "spawned")

    assert result.spawned and result.spawned[0][0] == task_id
    assert event.payload == {
        "pid": 6789,
        "launch_mode": "systemd-user-scope",
        "scope_unit": captured["unit"],
        "verification_status": "verified",
    }
    assert isinstance(kb._WorkerLaunchPid(6789), int)
    assert captured["cmd"][0] == "/usr/bin/systemd-run"


def test_native_systemd_scope_lifecycle_when_user_manager_is_available(kanban_home):
    """Exercise the real unit/control-group boundary, not just argv shape."""
    if not sys.platform.startswith("linux"):
        pytest.skip("native systemd user scopes require Linux")
    cgroup = kb._current_cgroup_path()
    target = kb._systemd_user_manager_target_for_cgroup(cgroup)
    if target is None:
        pytest.skip("dispatcher is not hosted in its own authenticated user-service cgroup")
    runner = kb.shutil.which("systemd-run")
    if runner is None:
        pytest.skip("systemd-run is unavailable")
    version = kb._systemd_run_version(runner)
    if version is None or version < kb._SYSTEMD_SCOPE_MIN_VERSION:
        pytest.skip("systemd-run lacks the required transient-scope support")
    if not kb._systemd_user_manager_reachable(target):
        pytest.skip("authenticated systemd user manager is unreachable")

    scoped, unit, manager = kb._systemd_scope_argv(
        ["/bin/sleep", "30"],
        _task("native-lifecycle", 1),
        cgroup_path=cgroup,
        manager_target=target,
        systemd_run=runner,
        user_manager_ready=True,
    )
    assert unit is not None and manager is target
    native_env = dict(os.environ)
    native_env.update(kb._systemd_user_manager_environment(target))
    proc = subprocess.Popen(
        scoped,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=native_env,
    )
    try:
        kb._verify_systemd_scope_pid(proc, unit, manager)
    finally:
        kb._cleanup_systemd_scope_launch(proc, unit, manager)
    assert proc.poll() is not None
    props = kb._systemd_scope_properties(unit, manager)
    assert props is None or props.get("LoadState") == "not-found"
