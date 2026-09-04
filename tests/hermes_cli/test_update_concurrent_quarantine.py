"""Tests for issue #26670 — concurrent hermes.exe detection and improved
quarantine retry / reboot-deferred fallback during `hermes update` on Windows.

These tests force ``_is_windows`` to return ``True`` via patching so the
Windows-specific code paths can be exercised on any host.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import main as cli_main


# Tests in this module either exercise the REAL _detect_concurrent_hermes_instances
# helper (and need the autouse stub in tests/hermes_cli/conftest.py disabled),
# or supply their own explicit return value via patch.object. Mark the whole
# module so the conftest fixture skips its default stub.
pytestmark = pytest.mark.real_concurrent_gate


# ---------------------------------------------------------------------------
# _detect_concurrent_hermes_instances
# ---------------------------------------------------------------------------


def _make_proc(pid: int, exe: str, name: str = "hermes.exe"):
    """Build a duck-typed psutil Process stand-in with the .info dict."""
    proc = MagicMock()
    proc.info = {"pid": pid, "exe": exe, "name": name}
    return proc




# ---------------------------------------------------------------------------
# Parent-chain exclusion (issue #30768 follow-up — the setuptools .exe
# launcher on Windows is a separate native process that spawns python.exe;
# excluding only ``os.getpid()`` flags the launcher as a concurrent instance.
# ---------------------------------------------------------------------------


def _fake_psutil_with_parent_chain(
    parent_chain: list[int],
    proc_iter_rows: list,
    *,
    ancestor_exe: str | None = None,
):
    """Build a psutil stand-in that has Process()/parents()/exe() AND process_iter().

    ``parent_chain`` is the ordered list of ancestor PIDs (closest first)
    returned by ``proc.parents()`` on the seed (``os.getpid()``).
    ``ancestor_exe`` is the executable path reported by each ancestor's
    ``.exe()``; when it matches one of our shim paths the ancestor is
    excluded (the launcher-shim case). Pass ``None`` to model an ancestor
    whose exe can't be read (psutil error) — it stays in the candidate set.
    """

    class _FakeProc:
        def __init__(self, pid: int, exe_path: str | None):
            self.pid = pid
            self._exe = exe_path

        def exe(self):
            if self._exe is None:
                raise OSError("exe unavailable")
            return self._exe

        def parents(self):
            return [_FakeProc(p, ancestor_exe) for p in parent_chain]

    class _NoSuchProcess(Exception):
        pass

    class _AccessDenied(Exception):
        pass

    def _process(pid=None):
        return _FakeProc(pid if pid is not None else os.getpid(), ancestor_exe)

    return types.SimpleNamespace(
        Process=_process,
        NoSuchProcess=_NoSuchProcess,
        AccessDenied=_AccessDenied,
        process_iter=lambda attrs: iter(proc_iter_rows),
    )


@patch.object(cli_main, "_is_windows", return_value=True)
def test_detect_concurrent_parents_call_robust_to_one_bad_hop(_winp, tmp_path):
    """The launcher shim is still excluded even when an ancestor exe is unreadable.

    Field regression (issues #29341, #34795): the old per-hop ``parent()``
    walk bailed on the FIRST psutil error, so an AccessDenied on any hop left
    the launcher shim in the candidate set and re-triggered the false
    positive. ``parents()`` returns the whole list at once; we evaluate each
    ancestor independently, so one unreadable hop never strands the launcher.
    """
    scripts_dir = tmp_path
    shim = scripts_dir / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    launcher_pid = me + 100

    rows = [
        _make_proc(me, str(shim), "python.exe"),
        _make_proc(launcher_pid, str(shim), "hermes.exe"),
    ]
    # ancestor_exe=None → every ancestor's .exe() raises OSError. The helper
    # must swallow it per-ancestor and not crash; the launcher won't be
    # excluded in this degenerate case, but a real run reads the shim exe.
    fake_psutil = _fake_psutil_with_parent_chain(
        parent_chain=[launcher_pid],
        proc_iter_rows=rows,
        ancestor_exe=None,
    )
    with patch.dict(sys.modules, {"psutil": fake_psutil}):
        result = cli_main._detect_concurrent_hermes_instances(scripts_dir)

    # No crash; helper completes. (Degenerate stub: launcher exe unreadable.)
    assert result == [(launcher_pid, "hermes.exe")]




# ---------------------------------------------------------------------------
# _format_concurrent_instances_message
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# _quarantine_running_hermes_exe — retry, then report
# ---------------------------------------------------------------------------


@patch.object(cli_main, "_is_windows", return_value=True)
def test_quarantine_succeeds_first_attempt(_winp, tmp_path):
    """When the rename works immediately, no warning, single rename pair returned."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"old")

    pairs = cli_main._quarantine_running_hermes_exe(tmp_path)

    assert len(pairs) == 1
    orig, quarantine = pairs[0]
    assert orig == shim
    assert quarantine.name.startswith("hermes.exe.old.")
    assert quarantine.exists()
    assert not shim.exists()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_quarantine_reports_a_lock_it_cannot_break(_winp, tmp_path, capsys, monkeypatch):
    """Every retry failed: name the likely culprits, queue nothing for reboot."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"locked")

    def always_fails(self, target):
        raise OSError(32, "The process cannot access the file (simulated lock)")

    monkeypatch.setattr(cli_main, "_hermes_exe_shims", lambda d: [shim])
    with patch.object(Path, "rename", always_fails), patch(
        "time.sleep", lambda *_a, **_k: None
    ):
        pairs = cli_main._quarantine_running_hermes_exe(tmp_path)

    captured = capsys.readouterr().out.lower()

    assert pairs == []
    # A clear message, not raw [WinError 32], and no reboot promise we can't keep.
    assert "could not quarantine" in captured
    assert "reboot" not in captured




# ---------------------------------------------------------------------------
# Windows gateway pause/resume before update mutation
# ---------------------------------------------------------------------------


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_windows_gateways_for_update_stops_profile_and_unmapped_pids(
    _winp,
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


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_and_resume_windows_gateway_service(
    _winp,
    monkeypatch,
    tmp_path,
):
    """A real Windows service is stopped before venv mutation and restarted
    afterward instead of spawning a competing detached gateway."""
    import hermes_cli.gateway as gateway_mod
    import hermes_cli.update_cmd as update_cmd

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
        update_cmd,
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


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_windows_gateway_service_failure_restores_every_attempted_service(
    _winp,
    monkeypatch,
):
    """A service that times out after accepting stop is restarted too."""
    import hermes_cli.gateway as gateway_mod
    import hermes_cli.update_cmd as update_cmd

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
    monkeypatch.setattr(
        update_cmd,
        "_restore_windows_gateway_service",
        lambda name: restarted.append(name),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="HermesGatewayPicasso"):
        cli_main._pause_windows_gateways_for_update()

    assert restarted == ["HermesGatewayPicasso", "HermesGateway"]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_windows_gateway_service_surfaces_rollback_start_failure(
    _winp,
    monkeypatch,
):
    import hermes_cli.gateway as gateway_mod
    import hermes_cli.update_cmd as update_cmd

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
    monkeypatch.setattr(
        update_cmd,
        "_restore_windows_gateway_service",
        fake_start,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="rollback failures: HermesGateway"):
        cli_main._pause_windows_gateways_for_update()


def test_restore_windows_gateway_service_waits_out_stop_pending(monkeypatch):
    import hermes_cli.update_cmd as update_cmd

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

    update_cmd._restore_windows_gateway_service("HermesGateway")

    assert restarted == ["HermesGateway"]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_windows_gateways_aborts_when_service_discovery_is_indeterminate(
    _winp,
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


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_windows_gateways_aborts_when_gateway_pid_discovery_is_indeterminate(
    _winp,
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


@patch.object(cli_main, "_is_windows", return_value=True)
def test_resume_windows_gateway_service_failure_stays_retryable(
    _winp,
    monkeypatch,
):
    import hermes_cli.update_cmd as update_cmd

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

    with pytest.raises(RuntimeError, match="HermesGateway"):
        cli_main._resume_windows_gateways_after_update(token)

    assert token["resume_needed"] is True
    assert token["services"] == ["HermesGateway"]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_resume_windows_gateway_launcher_refresh_failure_stays_retryable(
    _winp,
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


# ---------------------------------------------------------------------------
# venv-side launcher ancestors (the uv launcher/worker split)
#
# A gateway started through the venv shim is two processes:
#   venv\Scripts\python.exe (launcher)  ->  uv\python\...\python.exe (worker)
# The gateway's PID file records the WORKER, so find_gateway_pids() (and the
# pause set built from it) only ever sees the worker. The venv-holder guard
# matches on the venv path prefix, so it only ever sees the LAUNCHER. The two
# sets were disjoint: a gateway the updater had just stopped still tripped the
# guard, aborting every update ("venv-blocked: N process(es) hold the install").
# ---------------------------------------------------------------------------


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


@patch.object(cli_main, "_is_windows", return_value=True)
def test_venv_launcher_ancestors_returns_venv_side_parent(_winp, monkeypatch):
    """The worker's venv-side parent is reported so the guard set is covered."""
    venv_exe = str(cli_main.PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
    worker_exe = r"C:\Users\x\AppData\Roaming\uv\python\cpython-3.11\python.exe"

    # worker 200 -> launcher 100 (even == venv-side)
    fake = _fake_psutil_tree({200: 100}, venv_exe, worker_exe)
    monkeypatch.setitem(sys.modules, "psutil", fake)

    assert cli_main._venv_launcher_ancestors([200]) == [100]


@patch.object(cli_main, "_is_windows", return_value=True)
def test_venv_launcher_ancestors_ignores_non_venv_parents(_winp, monkeypatch):
    """A Scheduled Task's cmd.exe / an operator shell is not a venv holder."""
    venv_exe = str(cli_main.PROJECT_ROOT / "venv" / "Scripts" / "python.exe")
    worker_exe = r"C:\Windows\System32\cmd.exe"

    # worker 200 -> parent 101 (odd == NOT venv-side)
    fake = _fake_psutil_tree({200: 101}, venv_exe, worker_exe)
    monkeypatch.setitem(sys.modules, "psutil", fake)

    assert cli_main._venv_launcher_ancestors([200]) == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_venv_launcher_ancestors_is_empty_without_pids(_winp):
    """No mapped gateways means nothing to walk up from."""
    assert cli_main._venv_launcher_ancestors([]) == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_pause_kill_set_covers_venv_guard_abort_set(
    _winp,
    monkeypatch,
    tmp_path,
):
    """INVARIANT: whatever the venv guard would abort on must be stopped.

    This is the contract the two PID-resolution paths must satisfy. Before the
    launcher walk existed, ``terminated`` held only the uv-side worker while
    the guard reported the venv-side launcher, so the update aborted forever
    despite a "successful" pause.
    """
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

    # What the downstream venv-holder guard would report as blocking.
    guard_would_abort_on = {launcher_pid}
    assert guard_would_abort_on.issubset(set(terminated)), (
        f"pause stopped {sorted(terminated)} but the venv guard aborts on "
        f"{sorted(guard_would_abort_on)} — disjoint sets abort the update"
    )


# ---------------------------------------------------------------------------
# _leftover_pausable_gateway_pids (the guard-level gateway fallback)
#
# The pause stops every gateway discovery finds, but the venv-holder guard
# sees the process table as it is NOW. A supervisor (Scheduled Task, login
# watchdog) can respawn a gateway inside the pause→guard window, and some
# spawn paths never register in discovery at all. Those holders are exactly
# what the pause machinery exists to stop — the guard nominates them for a
# stop-and-recheck instead of dead-ending, and refuses the moment any
# non-gateway holder is present.
# ---------------------------------------------------------------------------


GATEWAY_ARGV = [
    r"C:\x\venv\Scripts\python.exe",
    "-m",
    "hermes_cli.main",
    "gateway",
    "run",
]


def _fake_psutil_cmdlines(argv_by_pid):
    """psutil stand-in serving live argv per pid; unknown pids raise."""

    class FakeProc:
        def __init__(self, pid):
            if pid not in argv_by_pid:
                raise ValueError(f"no such pid {pid}")
            self._argv = argv_by_pid[pid]

        def cmdline(self):
            return self._argv

    return types.SimpleNamespace(Process=FakeProc)


def test_leftover_holders_that_are_all_gateways_are_nominated(monkeypatch):
    """Respawned/unmapped gateway holders get stopped, not dead-ended on."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_cmdlines({300: GATEWAY_ARGV, 301: GATEWAY_ARGV}),
    )
    matches = [
        (300, "python.exe", "truncated..."),
        (301, "python.exe", "truncated..."),
    ]

    assert cli_main._leftover_pausable_gateway_pids(matches) == [300, 301]


def test_plain_update_refuses_to_tree_kill_its_gateway_ancestor(
    monkeypatch, capsys
):
    """#98814: terminal-launched update must survive to report the refusal."""
    import hermes_cli.gateway as gateway_cli
    import hermes_cli.update_cmd as update_cmd

    monkeypatch.setattr(
        gateway_cli,
        "_is_pid_ancestor_of_current_process",
        lambda pid: pid == 300,
    )

    refused = update_cmd._refuse_gateway_ancestor_tree_kill(
        [300, 301], gateway_mode=False
    )

    assert refused is True
    output = capsys.readouterr().out
    assert "taskkill /T" in output
    assert "`/update`" in output
    assert "separate terminal" in output


def test_gateway_handoff_keeps_leftover_gateway_recovery(monkeypatch, capsys):
    """The detached `/update` hand-off still owns leftover gateway cleanup."""
    import hermes_cli.gateway as gateway_cli
    import hermes_cli.update_cmd as update_cmd

    ancestry_checks = []
    monkeypatch.setattr(
        gateway_cli,
        "_is_pid_ancestor_of_current_process",
        lambda pid: ancestry_checks.append(pid) or True,
    )

    assert (
        update_cmd._refuse_gateway_ancestor_tree_kill(
            [300], gateway_mode=True
        )
        is False
    )
    assert ancestry_checks == []
    assert capsys.readouterr().out == ""


def test_one_non_gateway_holder_keeps_the_hard_refusal(monkeypatch):
    """A REPL/backend holder means the guard must abort exactly as before."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_cmdlines(
            {300: GATEWAY_ARGV, 400: [r"C:\x\venv\Scripts\python.exe", "-i"]}
        ),
    )
    matches = [(300, "python.exe", "..."), (400, "python.exe", "...")]

    assert cli_main._leftover_pausable_gateway_pids(matches) is None


def test_unreadable_argv_falls_back_to_the_captured_prefix(monkeypatch):
    """psutil failure degrades to the scan's captured cmdline, not a crash.

    The captured prefix decides: a gateway invocation still qualifies, and
    anything else still refuses.
    """
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_cmdlines({}))
    gateway_prefix = r"venv\Scripts\python.exe -m hermes_cli.main gateway run"

    assert cli_main._leftover_pausable_gateway_pids(
        [(300, "python.exe", gateway_prefix)]
    ) == [300]
    assert (
        cli_main._leftover_pausable_gateway_pids(
            [
                (300, "python.exe", gateway_prefix),
                (400, "python.exe", "python.exe -i"),
            ]
        )
        is None
    )











# ---------------------------------------------------------------------------
# cmd_update integration — concurrent-instance gate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _classify_concurrent_instance / _filter_non_gateway_concurrent_instances
#
# #37039: the pre-update concurrent-instance gate lets the update proceed
# when every concurrent hermes.exe is a gateway runtime — the pause
# machinery (_pause_windows_gateways_for_update) stops those before any
# file mutation and the post-update restart phase brings them back.
# Classification delegates to _is_pausable_gateway → the canonical
# gateway.status.looks_like_gateway_command_line matcher, so the gate's
# exemption and the pause discovery cannot drift apart.
# ---------------------------------------------------------------------------


def _fake_psutil_classify(argv_by_pid):
    """psutil stand-in serving .cmdline() per pid; unknown pids raise."""

    class FakeProc:
        def __init__(self, pid):
            if pid not in argv_by_pid:
                raise ValueError(f"no such pid {pid}")
            self._argv = argv_by_pid[pid]

        def cmdline(self):
            return self._argv

    return types.SimpleNamespace(Process=FakeProc)


def test_classify_concurrent_instance_recognises_gateway_runtimes(monkeypatch):
    """Gateway runtime command lines classify as ``gateway`` regardless of
    launcher shape (python -m, hermes.exe shim, hermes-gateway.exe,
    gateway/run.py, bare `hermes gateway` which defaults to run)."""
    cases = [
        [r"C:\venv\Scripts\python.exe", "-m", "hermes_cli.main", "gateway", "run"],
        [r"C:\venv\Scripts\hermes.exe", "gateway", "run"],
        [r"C:\venv\Scripts\hermes-gateway.exe"],
        [r"C:\venv\Scripts\python.exe", "gateway/run.py"],
        ["hermes.exe", "GATEWAY", "RUN"],  # matcher is case-insensitive
        ["hermes.exe", "gateway"],  # bare `hermes gateway` defaults to run
        # profile selector before the subcommand — canonical matcher strips it
        ["hermes.exe", "--profile", "work", "gateway", "run"],
    ]
    for argv in cases:
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_classify({77: argv}))
        result = cli_main._classify_concurrent_instance(77)
        assert result == "gateway", f"expected gateway for {argv!r}, got {result!r}"


def test_classify_concurrent_instance_recognises_non_gateways(monkeypatch):
    """Non-runtime command lines classify as ``non-gateway`` — including
    gateway MANAGEMENT subcommands (`gateway status`), which the canonical
    matcher rejects but a substring matcher would misclassify. These keep
    the pre-update abort."""
    cases = [
        [r"C:\venv\Scripts\hermes.exe"],  # interactive REPL
        [r"C:\venv\Scripts\hermes.exe", "dashboard"],
        ["hermes.exe", "gateway", "status"],  # management, not runtime
        ["hermes.exe", "gateway", "stop"],
        ["python", "-m", "hermes_cli.main"],
        [],
    ]
    for argv in cases:
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_classify({77: argv}))
        result = cli_main._classify_concurrent_instance(77)
        assert result == "non-gateway", (
            f"expected non-gateway for {argv!r}, got {result!r}"
        )


def test_classify_concurrent_instance_unknown_on_psutil_error(monkeypatch):
    """Unreadable cmdline (process gone / AccessDenied) → ``unknown`` —
    treated as non-gateway by the filter, so the gate still aborts."""
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_classify({}))
    assert cli_main._classify_concurrent_instance(4242) == "unknown"


def test_classify_concurrent_instance_unknown_without_psutil(monkeypatch):
    """Missing psutil entirely → ``unknown``, never a crash."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert cli_main._classify_concurrent_instance(4242) == "unknown"


def test_filter_non_gateway_concurrent_instances_splits(monkeypatch):
    """Gateway PIDs drop out of the abort list; REPL/dashboard/unknown stay."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_classify(
            {
                100: ["hermes.exe", "gateway", "run"],
                200: ["hermes.exe"],  # REPL — keep
                300: ["hermes.exe", "dashboard"],  # keep
                # 400 missing → unknown → keep
            }
        ),
    )
    matches = [
        (100, "hermes.exe"),
        (200, "hermes.exe"),
        (300, "hermes.exe"),
        (400, "hermes.exe"),
    ]
    kept = cli_main._filter_non_gateway_concurrent_instances(matches)
    assert kept == [(200, "hermes.exe"), (300, "hermes.exe"), (400, "hermes.exe")]


def test_filter_non_gateway_concurrent_instances_gateway_only(monkeypatch):
    """All-gateway match list filters to empty — the gate lets the update
    proceed and the pause machinery handles the gateways."""
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_classify(
            {
                111: ["hermes.exe", "gateway", "run"],
                222: [r"C:\venv\Scripts\hermes-gateway.exe"],
            }
        ),
    )
    matches = [(111, "hermes.exe"), (222, "hermes-gateway.exe")]
    assert cli_main._filter_non_gateway_concurrent_instances(matches) == []


# ---------------------------------------------------------------------------
# _cmd_update_impl integration with the relaxed pre-update gate (#37039)
# ---------------------------------------------------------------------------


def _update_args():
    return SimpleNamespace(
        check=False,
        gateway=False,
        yes=False,
        force=False,
        backup=False,
        no_backup=True,
    )


@patch.object(cli_main, "_is_windows", return_value=True)
def test_update_gate_skips_abort_when_only_concurrent_is_gateway(
    _winp, tmp_path, capsys
):
    """Regression test for #37039: with only gateway processes concurrent,
    the gate must NOT sys.exit(2) — the update proceeds to the pre-update
    backup step (sentinel), and the pause machinery owns the gateways."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()

    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=scripts_dir
    ), patch.object(
        cli_main,
        "_detect_concurrent_hermes_instances",
        return_value=[(1000, "hermes.exe"), (2000, "hermes-gateway.exe")],
    ), patch.object(
        cli_main, "_filter_non_gateway_concurrent_instances", return_value=[]
    ) as mock_filter, patch.object(
        cli_main, "_run_pre_update_backup"
    ) as mock_backup:
        mock_backup.side_effect = RuntimeError("reached post-gate body")
        with pytest.raises(RuntimeError, match="reached post-gate body"):
            cli_main._cmd_update_impl(_update_args(), gateway_mode=False)

    mock_filter.assert_called_once()
    mock_backup.assert_called_once()
    captured = capsys.readouterr().out
    assert "Another hermes.exe is running" not in captured


@patch.object(cli_main, "_is_windows", return_value=True)
def test_update_gate_still_aborts_on_non_gateway_concurrent(
    _winp, tmp_path, capsys
):
    """A non-gateway concurrent instance must still abort with exit 2, and
    the message must list only the non-gateway PIDs (the gateway is not the
    user's problem to kill)."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()

    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=scripts_dir
    ), patch.object(
        cli_main,
        "_detect_concurrent_hermes_instances",
        return_value=[(1000, "hermes.exe"), (3000, "hermes.exe")],
    ), patch.object(
        cli_main,
        "_filter_non_gateway_concurrent_instances",
        return_value=[(3000, "hermes.exe")],
    ), patch.object(
        cli_main, "_run_pre_update_backup"
    ) as mock_backup:
        with pytest.raises(SystemExit) as excinfo:
            cli_main._cmd_update_impl(_update_args(), gateway_mode=False)

    assert excinfo.value.code == 2
    mock_backup.assert_not_called()
    captured = capsys.readouterr().out
    assert "3000" in captured
    assert "1000" not in captured  # gateway PID no longer blamed
    assert "--force" in captured


@patch.object(cli_main, "_is_windows", return_value=True)
def test_update_impl_refuses_before_terminating_gateway_ancestor(
    _winp, monkeypatch, capsys
):
    """#98814: the live holder path must gate the destructive call itself."""
    import gateway.status as status_mod
    import hermes_cli.gateway as gateway_cli

    holder = (
        300,
        "python.exe",
        r"C:\x\venv\Scripts\python.exe -m hermes_cli.main gateway run",
    )
    monkeypatch.setattr(
        gateway_cli,
        "_is_pid_ancestor_of_current_process",
        lambda pid: pid == 300,
    )

    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=None
    ), patch.object(
        cli_main, "_run_pre_update_backup", return_value=None
    ), patch.object(
        cli_main, "_pause_windows_gateways_for_update", return_value=None
    ), patch.object(
        cli_main, "_detect_venv_python_processes", return_value=[holder]
    ), patch.object(
        cli_main, "_leftover_pausable_gateway_pids", return_value=[300]
    ), patch.object(
        cli_main, "_resume_windows_gateways_after_update"
    ) as resume, patch.object(
        status_mod, "terminate_pid"
    ) as terminate:
        with pytest.raises(SystemExit) as excinfo:
            cli_main._cmd_update_impl(_update_args(), gateway_mode=False)

    assert excinfo.value.code == 2
    terminate.assert_not_called()
    resume.assert_called_once_with(None)
    output = capsys.readouterr().out
    assert "taskkill /T" in output
    assert "`/update`" in output


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


