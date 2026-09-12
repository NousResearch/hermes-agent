"""Windows gateway pause/resume and launcher identity remain live after PM cutover."""

from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd



# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_pause_windows_gateways_for_update_stops_profile_and_unmapped_pids(
    monkeypatch,
    tmp_path,
    capsys,
):
    import gateway.status as status_mod
    import hermes_cli.gateway as gateway_mod

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    profile_proc = SimpleNamespace(profile="work", path=profile_home, pid=101)

    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [101, 202])
    monkeypatch.setattr(
        gateway_mod, "find_windows_gateway_services", lambda **_k: []
    )
    monkeypatch.setattr(
        gateway_mod,
        "find_profile_gateway_processes",
        lambda **_k: [profile_proc],
    )
    monkeypatch.setattr(gateway_mod, "_get_restart_drain_timeout", lambda: 0.1)
    waited_for = []

    def fake_wait(pids, *, timeout):
        waited_for.extend(pids)
        return set()

    monkeypatch.setattr(cli_main, "_wait_for_windows_update_gateway_exit", fake_wait)
    monkeypatch.setattr(
        gateway_mod,
        "_capture_gateway_argv",
        lambda pid: ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"]
        if pid == 202
        else None,
    )

    terminated = []
    monkeypatch.setattr(
        status_mod,
        "terminate_pid",
        lambda pid, force=False, **kwargs: terminated.append((pid, force)),
    )

    token = cli_main._pause_windows_gateways_for_update()

    assert token == {
        "resume_needed": True,
        "profiles": {"work": 101},
        "unmapped_pids": [202],
        "unmapped": [
            {
                "pid": 202,
                "argv": ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"],
            }
        ],
    }
    assert waited_for == [101]
    assert terminated == [(202, True)]

    marker = json.loads(
        (profile_home / ".gateway-planned-stop.json").read_text(encoding="utf-8")
    )
    assert marker["target_pid"] == 101
    assert marker["stopper_pid"] == os.getpid()

    captured = capsys.readouterr().out
    assert "Paused gateway profile(s): work" in captured
    assert "without profile mapping" in captured
    # An unmapped PID whose argv we captured is respawnable, so we must NOT
    # tell the user to restart it manually.
    assert "Restart manually after update" not in captured


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_pause_and_resume_windows_gateway_service(
    monkeypatch,
    tmp_path,
):
    """A real Windows service is stopped before venv mutation and restarted
    afterward instead of spawning a competing detached gateway."""
    import hermes_cli.gateway as gateway_mod
    import hermes_cli.update_cmd as update_cmd
    import hermes_cli.update_cmd_windows as update_cmd_windows

    profile_home = tmp_path / "profiles" / "default"
    profile_home.mkdir(parents=True)
    profile_proc = SimpleNamespace(profile="default", path=profile_home, pid=101)
    service = SimpleNamespace(
        name="HermesGateway",
        profile="default",
        service_pid=11,
        service_create_time=11.0,
        gateway_pid=101,
        gateway_create_time=101.0,
        descendant_pids=frozenset({11, 22, 101}),
        descendant_identities=((22, 22.0), (101, 101.0)),
    )
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(
        gateway_mod, "find_profile_gateway_processes", lambda **_k: [profile_proc]
    )
    monkeypatch.setattr(
        gateway_mod,
        "find_windows_gateway_services",
        lambda **_k: [service],
        raising=False,
    )
    monkeypatch.setattr(gateway_mod, "_get_restart_drain_timeout", lambda: 0.1)

    stopped = []
    started = []
    monkeypatch.setattr(
        update_cmd,
        "_stop_windows_gateway_service",
        lambda name, **_kwargs: stopped.append(name),
        raising=False,
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_stop_windows_gateway_service",
        lambda name, **_kwargs: stopped.append(name),
        raising=False,
    )
    monkeypatch.setattr(
        update_cmd,
        "_start_windows_gateway_service",
        lambda name: started.append(name),
        raising=False,
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_start_windows_gateway_service",
        lambda name: started.append(name),
        raising=False,
    )
    monkeypatch.setattr(cli_main, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        cli_main,
        "_cold_start_windows_gateway_after_update",
        lambda: (_ for _ in ()).throw(AssertionError("service resume must not cold-start")),
    )

    token = cli_main._pause_windows_gateways_for_update()
    assert token == {
        "resume_needed": True,
        "profiles": {},
        "unmapped_pids": [],
        "unmapped": [],
        "services": ["HermesGateway"],
        "expected_services": ["HermesGateway"],
        "restarted_services": [],
        "service_profiles": {"HermesGateway": "default"},
    }
    assert stopped == ["HermesGateway"]

    cli_main._resume_windows_gateways_after_update(token)
    assert started == ["HermesGateway"]


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_pause_windows_gateway_service_failure_restores_every_attempted_service(
    monkeypatch,
):
    """A service that times out after accepting stop is restarted too."""
    import hermes_cli.gateway as gateway_mod
    import hermes_cli.update_cmd as update_cmd
    import hermes_cli.update_cmd_windows as update_cmd_windows

    services = [
        SimpleNamespace(name="HermesGateway", service_pid=11, service_create_time=11.0, gateway_pid=101, gateway_create_time=101.0, descendant_identities=()),
        SimpleNamespace(name="HermesGatewayPicasso", service_pid=22, service_create_time=22.0, gateway_pid=202, gateway_create_time=202.0, descendant_identities=()),
    ]
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(
        gateway_mod, "find_windows_gateway_services", lambda **_k: services
    )

    def fake_stop(name, **_kwargs):
        if name == "HermesGatewayPicasso":
            raise RuntimeError("simulated stop timeout")

    restarted = []
    monkeypatch.setattr(update_cmd, "_stop_windows_gateway_service", fake_stop)
    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateway_service", fake_stop)
    monkeypatch.setattr(
        update_cmd,
        "_restore_windows_gateway_service",
        lambda name: restarted.append(name),
        raising=False,
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_restore_windows_gateway_service",
        lambda name: restarted.append(name),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="HermesGatewayPicasso"):
        cli_main._pause_windows_gateways_for_update()

    assert restarted == ["HermesGatewayPicasso", "HermesGateway"]


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_pause_windows_gateway_service_surfaces_rollback_start_failure(
    monkeypatch,
):
    import hermes_cli.gateway as gateway_mod
    import hermes_cli.update_cmd as update_cmd
    import hermes_cli.update_cmd_windows as update_cmd_windows

    services = [
        SimpleNamespace(name="HermesGateway", service_pid=11, service_create_time=11.0, gateway_pid=101, gateway_create_time=101.0, descendant_identities=()),
        SimpleNamespace(name="HermesGatewayPicasso", service_pid=22, service_create_time=22.0, gateway_pid=202, gateway_create_time=202.0, descendant_identities=()),
    ]
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(
        gateway_mod, "find_windows_gateway_services", lambda **_k: services
    )

    def fake_stop(name, **_kwargs):
        if name == "HermesGatewayPicasso":
            raise RuntimeError("simulated stop timeout")

    def fake_start(name):
        if name == "HermesGateway":
            raise RuntimeError("simulated rollback start failure")

    monkeypatch.setattr(update_cmd, "_stop_windows_gateway_service", fake_stop)
    monkeypatch.setattr(update_cmd_windows, "_stop_windows_gateway_service", fake_stop)
    monkeypatch.setattr(
        update_cmd,
        "_restore_windows_gateway_service",
        fake_start,
        raising=False,
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_restore_windows_gateway_service",
        fake_start,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="rollback failures: HermesGateway"):
        cli_main._pause_windows_gateways_for_update()


def test_restore_windows_gateway_service_waits_out_stop_pending(monkeypatch):
    import hermes_cli.update_cmd as update_cmd
    import hermes_cli.update_cmd_windows as update_cmd_windows

    statuses = iter(["stop_pending", "stopped"])
    service = SimpleNamespace(status=lambda: next(statuses))
    fake_psutil = SimpleNamespace(win_service_get=lambda _name: service)
    restarted = []
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(update_cmd._time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        update_cmd,
        "_start_windows_gateway_service",
        lambda name: restarted.append(name),
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_start_windows_gateway_service",
        lambda name: restarted.append(name),
    )

    update_cmd._restore_windows_gateway_service("HermesGateway")

    assert restarted == ["HermesGateway"]


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_pause_windows_gateways_aborts_when_service_discovery_is_indeterminate(
    monkeypatch,
):
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(
        gateway_mod,
        "find_windows_gateway_services",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("SCM scan indeterminate")),
    )
    monkeypatch.setattr(
        gateway_mod,
        "find_gateway_pids",
        lambda **_k: (_ for _ in ()).throw(
            AssertionError("ordinary gateway teardown must not begin")
        ),
    )

    with pytest.raises(RuntimeError, match="SCM scan indeterminate"):
        cli_main._pause_windows_gateways_for_update()


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_pause_windows_gateways_aborts_when_gateway_pid_discovery_is_indeterminate(
    monkeypatch,
):
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "find_windows_gateway_services", lambda **_k: [])
    monkeypatch.setattr(
        gateway_mod,
        "find_gateway_pids",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("PID discovery failed")),
    )

    with pytest.raises(RuntimeError, match="PID discovery failed"):
        cli_main._pause_windows_gateways_for_update()


def test_stop_windows_gateway_service_waits_for_original_descendants(
    monkeypatch,
):
    """SCM STOPPED is insufficient while the original process identity lives."""
    import hermes_cli.update_cmd as update_cmd

    service = SimpleNamespace(status=lambda: "stopped")
    fake_psutil = SimpleNamespace(
        win_service_get=lambda _name: service,
        Process=lambda pid: SimpleNamespace(create_time=lambda: 12.5),
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(
        update_cmd.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="process tree"):
        update_cmd._stop_windows_gateway_service(
            "HermesGateway",
            expected_processes=((123, 12.5),),
            timeout=0,
        )


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_resume_windows_gateway_service_failure_stays_retryable(
    monkeypatch,
):
    import hermes_cli.update_cmd as update_cmd
    import hermes_cli.update_cmd_windows as update_cmd_windows

    token = {
        "resume_needed": True,
        "profiles": {},
        "unmapped": [],
        "services": ["HermesGateway"],
    }
    monkeypatch.setattr(cli_main, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        update_cmd,
        "_start_windows_gateway_service",
        lambda _name: (_ for _ in ()).throw(RuntimeError("simulated start failure")),
    )
    monkeypatch.setattr(
        update_cmd_windows,
        "_start_windows_gateway_service",
        lambda _name: (_ for _ in ()).throw(RuntimeError("simulated start failure")),
    )

    with pytest.raises(RuntimeError, match="HermesGateway"):
        cli_main._resume_windows_gateways_after_update(token)

    assert token["resume_needed"] is True
    assert token["services"] == ["HermesGateway"]


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_resume_windows_gateway_launcher_refresh_failure_stays_retryable(
    monkeypatch,
):
    token = {
        "resume_needed": True,
        "profiles": {},
        "unmapped": [],
        "services": ["HermesGateway"],
    }
    monkeypatch.setattr(
        cli_main,
        "_refresh_windows_gateway_launchers",
        lambda: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )

    with pytest.raises(RuntimeError, match="refresh failed"):
        cli_main._resume_windows_gateways_after_update(token)

    assert token["resume_needed"] is True
    assert token["services"] == ["HermesGateway"]


def _fake_psutil_tree(tree, venv_exe, worker_exe, dead=None):
    """Build a psutil stand-in where ``tree`` maps worker pid -> parent pid.

    Parents whose pid is even are venv-side (``venv_exe``); odd parents are
    unrelated ancestors (``worker_exe``) that must NOT be returned. Pids in
    ``dead`` (a live reference — later additions count) are uninspectable:
    construction raises, exactly like psutil.NoSuchProcess for an exited
    process.
    """

    dead_set = dead if dead is not None else set()

    class FakeProc:
        def __init__(self, pid):
            self.pid = pid
            if pid in dead_set:
                raise ValueError(f"process {pid} has exited")
            if pid not in tree and pid not in tree.values():
                raise ValueError(f"no such pid {pid}")

        def parent(self):
            ppid = tree.get(self.pid)
            return FakeProc(ppid) if ppid else None

        def parents(self):
            return []

        def exe(self):
            # Parents of workers are the launchers under test.
            return venv_exe if self.pid % 2 == 0 else worker_exe

    mod = types.SimpleNamespace(Process=FakeProc)
    return mod


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_venv_launcher_ancestors_returns_venv_side_parent(monkeypatch):
    """The worker's venv-side parent is included in the gateway pause."""
    venv_exe = str(cli_main.PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
    worker_exe = r"C:\Users\x\AppData\Roaming\uv\python\cpython-3.11\python.exe"

    # worker 200 -> launcher 100 (even == venv-side)
    fake = _fake_psutil_tree({200: 100}, venv_exe, worker_exe)
    monkeypatch.setitem(sys.modules, "psutil", fake)

    assert cli_main._venv_launcher_ancestors([200]) == [100]


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_venv_launcher_ancestors_ignores_non_venv_parents(monkeypatch):
    """A Scheduled Task's cmd.exe / an operator shell is not a venv holder."""
    venv_exe = str(cli_main.PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
    worker_exe = r"C:\Windows\System32\cmd.exe"

    # worker 200 -> parent 101 (odd == NOT venv-side)
    fake = _fake_psutil_tree({200: 101}, venv_exe, worker_exe)
    monkeypatch.setitem(sys.modules, "psutil", fake)

    assert cli_main._venv_launcher_ancestors([200]) == []


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_venv_launcher_ancestors_is_empty_without_pids():
    """No mapped gateways means nothing to walk up from."""
    assert cli_main._venv_launcher_ancestors([]) == []


# Windows lifecycle dispatch must run on its real host.
@pytest.mark.platforms("windows")
def test_pause_stops_launcher_after_worker_drain(
    monkeypatch,
    tmp_path,
):
    """Capture the launcher identity while its worker is still inspectable."""
    import hermes_cli.gateway as gateway_mod
    import gateway.status as status_mod

    venv_exe = str(cli_main.PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
    worker_exe = r"C:\Users\x\AppData\Roaming\uv\python\cpython-3.11\python.exe"

    profile_home = tmp_path / "profiles" / "default"
    profile_home.mkdir(parents=True)
    # The PID file records the WORKER (even-numbered parent 400 is its launcher).
    worker_pid, launcher_pid = 500, 400
    profile_proc = SimpleNamespace(
        profile="default", path=profile_home, pid=worker_pid
    )

    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [worker_pid])
    monkeypatch.setattr(
        gateway_mod, "find_windows_gateway_services", lambda **_k: []
    )
    monkeypatch.setattr(
        gateway_mod, "find_profile_gateway_processes", lambda **_k: [profile_proc]
    )
    monkeypatch.setattr(gateway_mod, "_get_restart_drain_timeout", lambda: 0.1)
    # Graceful drain succeeds: the worker exits, leaving zero survivors — and
    # an exited worker is UNINSPECTABLE afterwards, exactly like the real
    # process table. Resolving the launcher after this point is impossible,
    # so the pause must snapshot launcher ancestors before draining. This is
    # precisely the case that used to leave the launcher alive and abort.
    drained_dead: set[int] = set()

    def _drain_marks_workers_dead(pids, *, timeout):
        drained_dead.update(int(p) for p in pids)
        return set()

    monkeypatch.setattr(
        cli_main,
        "_wait_for_windows_update_gateway_exit",
        _drain_marks_workers_dead,
    )

    fake = _fake_psutil_tree(
        {worker_pid: launcher_pid}, venv_exe, worker_exe, dead=drained_dead
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)

    terminated = []
    monkeypatch.setattr(
        status_mod,
        "terminate_pid",
        lambda pid, force=False, **kwargs: terminated.append(int(pid)),
    )

    cli_main._pause_windows_gateways_for_update()

    assert terminated == [launcher_pid]


def test_stop_service_refuses_pid_reuse_before_sc_stop(monkeypatch):
    import hermes_cli.update_cmd as update_cmd

    fake_psutil = SimpleNamespace(
        win_service_get=lambda _name: SimpleNamespace(
            status=lambda: "running", pid=lambda: 11
        ),
        Process=lambda _pid: SimpleNamespace(create_time=lambda: 99.0),
    )
    calls = []
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *_a, **_k: calls.append(True))

    with pytest.raises(RuntimeError, match="identity changed"):
        update_cmd._stop_windows_gateway_service(
            "HermesGateway", expected_service_identity=(11, 11.0)
        )

    assert calls == []
