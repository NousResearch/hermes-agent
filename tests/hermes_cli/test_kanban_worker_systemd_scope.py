"""Focused lifecycle and fallback coverage for Kanban worker scopes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

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
    monkeypatch.setattr(kb, "_systemd_scope_process_ids", lambda path: (4242,))
    monkeypatch.setattr(kb, "_process_cgroup_path", lambda pid: "/expected.scope")
    monkeypatch.setattr(kb, "_process_command_argv", lambda pid: ("/wrong",))
    with pytest.raises(RuntimeError, match="could not verify systemd scope"):
        kb._verify_systemd_scope_worker_pid(proc, unit, target, ["/bin/sleep", "30"])

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


def test_scope_verification_returns_matching_worker_not_launcher(monkeypatch):
    unit = "hermes-kanban-worker-0123456789abcdef0123456789abcdef.scope"
    target = kb._SystemdUserManagerTarget(
        os.getuid(),
        Path("/run/user") / str(os.getuid()),
        Path("/run/user") / str(os.getuid()) / "bus",
    )
    proc = SimpleNamespace(pid=111, poll=lambda: None)
    monkeypatch.setattr(
        kb,
        "_systemd_scope_properties",
        lambda *args, **kwargs: {
            "LoadState": "loaded",
            "ActiveState": "active",
            "ControlGroup": "/expected.scope",
        },
    )
    monkeypatch.setattr(kb, "_systemd_scope_process_ids", lambda path: (111, 222))
    monkeypatch.setattr(kb, "_process_cgroup_path", lambda pid: "/expected.scope")
    monkeypatch.setattr(
        kb,
        "_process_command_argv",
        lambda pid: ("/usr/bin/systemd-run",) if pid == 111 else ("/bin/sleep", "30"),
    )

    assert (
        kb._verify_systemd_scope_worker_pid(
            cast(subprocess.Popen, proc),
            unit,
            target,
            ["/bin/sleep", "30"],
        )
        == 222
    )


def test_process_command_matches_shebang_interpreter(monkeypatch):
    monkeypatch.setattr(
        kb,
        "_process_command_argv",
        lambda pid: ("/usr/bin/python3", "/usr/local/bin/hermes", "chat", "-q", "work"),
    )

    assert getattr(kb, "_process_command_matches")(
        222,
        ["/usr/local/bin/hermes", "chat", "-q", "work"],
    )


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
    monkeypatch.setattr(kb, "_verify_systemd_scope_worker_pid", lambda *args: 2468)
    monkeypatch.setattr(kb.subprocess, "Popen", fake_popen)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="scoped", assignee="worker")
        result = kb.dispatch_once(conn, spawn_fn=kb._default_spawn)
        event = next(event for event in kb.list_events(conn, task_id) if event.kind == "spawned")

    assert result.spawned and result.spawned[0][0] == task_id
    assert event.payload == {
        "pid": 2468,
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
    assert unit is not None
    assert manager is target
    assert manager is not None
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
        worker_pid = kb._verify_systemd_scope_worker_pid(
            proc,
            unit,
            manager,
            ["/bin/sleep", "30"],
        )
        props = kb._systemd_scope_properties(unit, manager)
        assert props is not None
        control_group = props["ControlGroup"]
        assert worker_pid in kb._systemd_scope_process_ids(control_group)
        assert kb._process_cgroup_path(worker_pid) == control_group

        with kb.connect() as conn:
            task_id = kb.create_task(
                conn,
                title="native scope timeout",
                assignee="worker",
                max_runtime_seconds=1,
            )
            claimed = kb.claim_task(conn, task_id)
            assert claimed is not None
            kb._set_worker_pid(conn, task_id, worker_pid)
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (int(kb.time.time()) - 2, claimed.current_run_id),
            )
            conn.commit()
            assert kb.enforce_max_runtime(conn) == [task_id]
            timed_out_task = kb.get_task(conn, task_id)
            assert timed_out_task is not None
            assert timed_out_task.status == "ready"
        proc.wait(timeout=kb._SYSTEMD_SCOPE_CLEANUP_TIMEOUT)
    finally:
        kb._cleanup_systemd_scope_launch(proc, unit, manager)
    assert proc.poll() is not None
    props = kb._systemd_scope_properties(unit, manager)
    assert props is None or props.get("LoadState") == "not-found"
