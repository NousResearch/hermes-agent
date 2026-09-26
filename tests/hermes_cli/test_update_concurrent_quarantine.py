"""`hermes update` concurrent-instance gate and Windows gateway service helpers
(#26670, #37039, #98814): concurrent shim detection, shim quarantine,
gateway/non-gateway classification, gateway pause/resume (SCM services,
profile and unmapped gateways, venv launcher ancestors), leftover venv holder
nomination, gateway-ancestor tree-kill refusal, and SCM stop/restore.

Windows-only code paths carry ``@pytest.mark.windows_only`` and run on the
native Windows lane; nothing here fakes the host OS.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import dashboard_procs
from hermes_cli import main as cli_main
from hermes_cli import main_install_repair
from hermes_cli import update_cmd
from hermes_constants import project_venv_dir


# Tests in this module either exercise the REAL _detect_concurrent_hermes_instances
# helper (and need the autouse stub in tests/hermes_cli/conftest.py disabled),
# or supply their own explicit return value via patch.object. Mark the whole
# module so the conftest fixture skips its default stub.
pytestmark = pytest.mark.real_concurrent_gate


def _project_venv_root() -> Path:
    """Mirror the guards' lookup: the checkout's real venv/.venv, else venv/."""
    venv_dir = project_venv_dir(cli_main.PROJECT_ROOT) or cli_main.PROJECT_ROOT / "venv"
    return venv_dir.resolve()  # the guards prefix-match on the resolved path


# ---------------------------------------------------------------------------
# _detect_concurrent_hermes_instances — parent-chain exclusion
#
# The setuptools .exe launcher is a separate native process that spawns
# python.exe, so excluding only os.getpid() flags our own launcher as a
# concurrent instance (#30768 follow-up, #29341, #34795).
# ---------------------------------------------------------------------------


def _fake_psutil_with_ancestors(ancestors, rows):
    """psutil stand-in: ``Process(seed).parents()`` yields *ancestors*.

    *ancestors* is ``[(pid, exe_or_None), ...]`` closest first; ``None`` models
    an ancestor whose ``.exe()`` raises (AccessDenied / exited). *rows* are
    ``(pid, exe, name)`` tuples served by ``process_iter``.
    """

    class _Ancestor:
        def __init__(self, pid, exe):
            self.pid = pid
            self._exe = exe

        def exe(self):
            if self._exe is None:
                raise OSError("exe unavailable (simulated AccessDenied)")
            return self._exe

    class _Seed:
        def __init__(self, pid=None):
            self.pid = pid

        def parents(self):
            return [_Ancestor(pid, exe) for pid, exe in ancestors]

    return types.SimpleNamespace(
        Process=_Seed,
        process_iter=lambda _attrs: iter(
            SimpleNamespace(info={"pid": pid, "exe": exe, "name": name})
            for pid, exe, name in rows
        ),
    )


@pytest.mark.windows_only
def test_detect_concurrent_parents_call_robust_to_one_bad_hop(tmp_path, monkeypatch):
    """One unreadable ancestor never strands the launcher shim (#29341, #34795).

    The old per-hop ``parent()`` walk bailed on the FIRST psutil error, so an
    AccessDenied on any hop left the launcher shim in the candidate set and the
    update refused on our own launcher. Each ancestor is evaluated on its own:
    the launcher behind the bad hop is excluded, we are excluded, and a real
    concurrent shim process is still reported.
    """
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"")
    me = os.getpid()
    bad_hop_pid, launcher_pid, other_pid = me + 50, me + 100, me + 200
    unrelated = tmp_path / "elsewhere" / "python.exe"

    rows = [
        (me, str(shim), "python.exe"),
        (bad_hop_pid, None, "cmd.exe"),
        (launcher_pid, str(shim), "hermes.exe"),
        (other_pid, str(shim), "hermes.exe"),
        (me + 300, str(unrelated), "python.exe"),
    ]
    fake_psutil = _fake_psutil_with_ancestors(
        [(bad_hop_pid, None), (launcher_pid, str(shim))], rows
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    result = dashboard_procs._detect_concurrent_hermes_instances(tmp_path)

    assert result == [(other_pid, "hermes.exe")]


# ---------------------------------------------------------------------------
# _quarantine_running_hermes_exe — retry, then report
# ---------------------------------------------------------------------------


@pytest.mark.windows_only
def test_quarantine_succeeds_first_attempt(tmp_path):
    """A free shim is renamed aside once; nothing is reported as failed."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"old")
    failed: list[str] = []

    pairs = main_install_repair._quarantine_running_hermes_exe(
        tmp_path, failed_out=failed
    )

    assert len(pairs) == 1
    orig, quarantine = pairs[0]
    assert orig == shim
    assert quarantine.name.startswith("hermes.exe.old.")
    assert quarantine.exists()
    assert not shim.exists()
    assert failed == []


@pytest.mark.windows_only
def test_quarantine_reports_a_lock_it_cannot_break(tmp_path, capsys, monkeypatch):
    """Every retry failed: name the culprits, queue nothing for reboot, and
    report the shim so the dependency sync refuses instead of half-applying."""
    shim = tmp_path / "hermes.exe"
    shim.write_bytes(b"locked")
    failed: list[str] = []

    def always_fails(self, target):
        raise OSError(32, "The process cannot access the file (simulated lock)")

    monkeypatch.setattr(main_install_repair, "_hermes_exe_shims", lambda d: [shim])
    with patch.object(Path, "rename", always_fails), patch(
        "time.sleep", lambda *_a, **_k: None
    ):
        pairs = main_install_repair._quarantine_running_hermes_exe(
            tmp_path, failed_out=failed
        )

    captured = capsys.readouterr().out.lower()

    assert pairs == []
    assert failed == ["hermes.exe"]
    assert shim.exists()
    # A clear message, not raw [WinError 32], and no reboot promise we can't keep.
    assert "could not quarantine" in captured
    assert "reboot" not in captured


# ---------------------------------------------------------------------------
# Windows gateway pause/resume before update mutation
# ---------------------------------------------------------------------------


@pytest.mark.windows_only
def test_pause_windows_gateways_for_update_stops_profile_and_unmapped_pids(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Profile gateways get a planned-stop marker + graceful drain; unmapped
    gateways are force-stopped with their argv captured for respawn."""
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
    monkeypatch.setattr("gateway.restart.get_restart_drain_timeout", lambda: 0.1)
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


@pytest.mark.windows_only
def test_pause_and_resume_windows_gateway_service(
    monkeypatch,
    tmp_path,
):
    """A real Windows service is stopped before venv mutation and restarted
    afterward instead of spawning a competing detached gateway."""
    import hermes_cli.gateway as gateway_mod
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
        gateway_mod, "find_windows_gateway_services", lambda **_k: [service]
    )
    monkeypatch.setattr("gateway.restart.get_restart_drain_timeout", lambda: 0.1)

    stopped = []
    started = []
    for mod in (update_cmd, update_cmd_windows):
        monkeypatch.setattr(
            mod,
            "_stop_windows_gateway_service",
            lambda name, **_kwargs: stopped.append(name),
            raising=False,
        )
        monkeypatch.setattr(
            mod,
            "_start_windows_gateway_service",
            lambda name: started.append(name),
            raising=False,
        )
    monkeypatch.setattr(cli_main, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        cli_main,
        "_cold_start_windows_gateway_after_update",
        lambda *_a: (_ for _ in ()).throw(
            AssertionError("service resume must not cold-start")
        ),
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
    assert token["resume_needed"] is False


def _two_services():
    return [
        SimpleNamespace(name="HermesGateway", profile="default", service_pid=11, service_create_time=11.0, gateway_pid=101, gateway_create_time=101.0, descendant_identities=()),
        SimpleNamespace(name="HermesGatewayPicasso", profile="picasso", service_pid=22, service_create_time=22.0, gateway_pid=202, gateway_create_time=202.0, descendant_identities=()),
    ]


def _patch_service_discovery(monkeypatch, services):
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda **_k: [])
    monkeypatch.setattr(gateway_mod, "find_profile_gateway_processes", lambda **_k: [])
    monkeypatch.setattr(
        gateway_mod, "find_windows_gateway_services", lambda **_k: services
    )


def _patch_service_stop_restore(monkeypatch, fake_stop, fake_restore):
    import hermes_cli.update_cmd_windows as update_cmd_windows

    for mod in (update_cmd, update_cmd_windows):
        monkeypatch.setattr(mod, "_stop_windows_gateway_service", fake_stop)
        monkeypatch.setattr(
            mod, "_restore_windows_gateway_service", fake_restore, raising=False
        )


def _stop_times_out_on_picasso(name, **_kwargs):
    if name == "HermesGatewayPicasso":
        raise RuntimeError("simulated stop timeout")


@pytest.mark.windows_only
def test_pause_windows_gateway_service_failure_restores_every_attempted_service(
    monkeypatch,
):
    """A service that times out after accepting stop is restarted too."""
    _patch_service_discovery(monkeypatch, _two_services())
    restarted = []
    _patch_service_stop_restore(
        monkeypatch, _stop_times_out_on_picasso, restarted.append
    )

    with pytest.raises(RuntimeError, match="HermesGatewayPicasso"):
        cli_main._pause_windows_gateways_for_update()

    assert restarted == ["HermesGatewayPicasso", "HermesGateway"]


@pytest.mark.windows_only
def test_pause_windows_gateway_service_surfaces_rollback_start_failure(
    monkeypatch,
):
    """A rollback restart that fails is named, never swallowed."""
    _patch_service_discovery(monkeypatch, _two_services())

    def fake_restore(name):
        if name == "HermesGateway":
            raise RuntimeError("simulated rollback start failure")

    _patch_service_stop_restore(monkeypatch, _stop_times_out_on_picasso, fake_restore)

    with pytest.raises(RuntimeError, match="rollback failures: HermesGateway"):
        cli_main._pause_windows_gateways_for_update()


@pytest.mark.windows_only
def test_pause_windows_gateways_aborts_when_service_discovery_is_indeterminate(
    monkeypatch,
):
    """An indeterminate SCM scan aborts before any gateway is torn down."""
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "find_profile_gateway_processes", lambda **_k: [])
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


@pytest.mark.windows_only
def test_pause_windows_gateways_aborts_when_gateway_pid_discovery_is_indeterminate(
    monkeypatch,
):
    """Failed PID discovery aborts instead of reading as "no gateways running"."""
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(gateway_mod, "find_profile_gateway_processes", lambda **_k: [])
    monkeypatch.setattr(gateway_mod, "find_windows_gateway_services", lambda **_k: [])
    monkeypatch.setattr(
        gateway_mod,
        "find_gateway_pids",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("PID discovery failed")),
    )

    with pytest.raises(RuntimeError, match="PID discovery failed"):
        cli_main._pause_windows_gateways_for_update()


@pytest.mark.windows_only
def test_resume_windows_gateway_service_failure_stays_retryable(
    monkeypatch,
):
    """A service that fails to restart stays on the token for the retry."""
    import hermes_cli.update_cmd_windows as update_cmd_windows

    token = {
        "resume_needed": True,
        "profiles": {},
        "unmapped": [],
        "services": ["HermesGateway"],
    }
    monkeypatch.setattr(cli_main, "_refresh_windows_gateway_launchers", lambda: None)
    for mod in (update_cmd, update_cmd_windows):
        monkeypatch.setattr(
            mod,
            "_start_windows_gateway_service",
            lambda _name: (_ for _ in ()).throw(RuntimeError("simulated start failure")),
        )

    with pytest.raises(RuntimeError, match="HermesGateway"):
        cli_main._resume_windows_gateways_after_update(token)

    assert token["resume_needed"] is True
    assert token["services"] == ["HermesGateway"]


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
    """psutil stand-in where ``tree`` maps worker pid -> parent pid.

    Even pids are venv-side (``venv_exe``); odd pids are unrelated ancestors
    (``worker_exe``) that must NOT be returned. Pids in ``dead`` (a live
    reference — later additions count) are uninspectable: construction raises,
    exactly like psutil.NoSuchProcess for an exited process.
    """

    dead_set = dead if dead is not None else set()

    class FakeProc:
        def __init__(self, pid=None):
            self.pid = pid
            if pid in dead_set:
                raise ValueError(f"process {pid} has exited")
            if pid is not None and pid not in tree and pid not in tree.values():
                raise ValueError(f"no such pid {pid}")

        def parent(self):
            ppid = tree.get(self.pid)
            return FakeProc(ppid) if ppid else None

        def parents(self):
            return []

        def exe(self):
            return venv_exe if self.pid % 2 == 0 else worker_exe

    return types.SimpleNamespace(Process=FakeProc)


@pytest.mark.windows_only
def test_venv_launcher_ancestors_returns_venv_side_parent(monkeypatch):
    """The worker's venv-side parent is reported so the guard set is covered."""
    venv_exe = str(_project_venv_root() / "Scripts" / "python.exe")
    worker_exe = r"C:\Users\x\AppData\Roaming\uv\python\cpython-3.11\python.exe"

    # worker 200 -> launcher 100 (even == venv-side)
    monkeypatch.setitem(
        sys.modules, "psutil", _fake_psutil_tree({200: 100}, venv_exe, worker_exe)
    )

    assert cli_main._venv_launcher_ancestors([200]) == [100]


@pytest.mark.windows_only
def test_venv_launcher_ancestors_ignores_non_venv_parents(monkeypatch):
    """A Scheduled Task's cmd.exe / an operator shell is not a venv holder."""
    venv_exe = str(_project_venv_root() / "Scripts" / "python.exe")
    worker_exe = r"C:\Windows\System32\cmd.exe"

    # worker 200 -> parent 101 (odd == NOT venv-side)
    monkeypatch.setitem(
        sys.modules, "psutil", _fake_psutil_tree({200: 101}, venv_exe, worker_exe)
    )

    assert cli_main._venv_launcher_ancestors([200]) == []


@pytest.mark.windows_only
def test_pause_kill_set_covers_venv_guard_abort_set(
    monkeypatch,
    tmp_path,
):
    """INVARIANT: whatever the venv guard would abort on must be stopped.

    Before the launcher walk existed, ``terminated`` held only the uv-side
    worker while the guard reported the venv-side launcher, so the update
    aborted forever despite a "successful" pause. The worker exits during the
    graceful drain and is uninspectable afterwards, so the launcher must be
    resolved BEFORE draining.
    """
    import gateway.status as status_mod
    import hermes_cli.gateway as gateway_mod

    venv_exe = str(_project_venv_root() / "Scripts" / "python.exe")
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
    monkeypatch.setattr("gateway.restart.get_restart_drain_timeout", lambda: 0.1)
    drained_dead: set[int] = set()

    def _drain_marks_workers_dead(pids, *, timeout):
        drained_dead.update(int(p) for p in pids)
        return set()

    monkeypatch.setattr(
        cli_main,
        "_wait_for_windows_update_gateway_exit",
        _drain_marks_workers_dead,
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_tree(
            {worker_pid: launcher_pid}, venv_exe, worker_exe, dead=drained_dead
        ),
    )

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
        result = update_cmd._classify_concurrent_instance(77)
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
        result = update_cmd._classify_concurrent_instance(77)
        assert result == "non-gateway", (
            f"expected non-gateway for {argv!r}, got {result!r}"
        )


def test_classify_concurrent_instance_unknown_on_psutil_error(monkeypatch):
    """Unreadable cmdline (process gone / AccessDenied) → ``unknown`` —
    treated as non-gateway by the filter, so the gate still aborts."""
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_classify({}))
    assert update_cmd._classify_concurrent_instance(4242) == "unknown"


def test_classify_concurrent_instance_unknown_without_psutil(monkeypatch):
    """Missing psutil entirely → ``unknown``, never a crash."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert update_cmd._classify_concurrent_instance(4242) == "unknown"


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


@pytest.mark.windows_only
def test_update_gate_skips_abort_when_only_concurrent_is_gateway(
    tmp_path, capsys, monkeypatch
):
    """#37039: with only gateway runtimes concurrent, the gate must NOT
    sys.exit(2) — the update proceeds to the pre-update backup step, and the
    pause machinery owns the gateways. The real classifier decides."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_classify(
            {
                1000: ["hermes.exe", "gateway", "run"],
                2000: [r"C:\venv\Scripts\hermes-gateway.exe"],
            }
        ),
    )

    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=scripts_dir
    ), patch.object(
        cli_main,
        "_detect_concurrent_hermes_instances",
        return_value=[(1000, "hermes.exe"), (2000, "hermes-gateway.exe")],
    ), patch.object(
        cli_main, "_run_pre_update_backup"
    ) as mock_backup:
        mock_backup.side_effect = RuntimeError("reached post-gate body")
        with pytest.raises(RuntimeError, match="reached post-gate body"):
            update_cmd._cmd_update_impl(_update_args(), gateway_mode=False)

    mock_backup.assert_called_once()
    assert "Another hermes.exe is running" not in capsys.readouterr().out


@pytest.mark.windows_only
def test_update_gate_still_aborts_on_non_gateway_concurrent(
    tmp_path, capsys, monkeypatch
):
    """A non-gateway concurrent instance still aborts with exit 2 before any
    mutation, and the message lists only the non-gateway PID (the gateway is
    not the user's problem to kill)."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        _fake_psutil_classify(
            {1000: ["hermes.exe", "gateway", "run"], 3000: ["hermes.exe"]}
        ),
    )

    with patch.object(
        cli_main, "_venv_scripts_dir", return_value=scripts_dir
    ), patch.object(
        cli_main,
        "_detect_concurrent_hermes_instances",
        return_value=[(1000, "hermes.exe"), (3000, "hermes.exe")],
    ), patch.object(
        cli_main, "_run_pre_update_backup"
    ) as mock_backup:
        with pytest.raises(SystemExit) as excinfo:
            update_cmd._cmd_update_impl(_update_args(), gateway_mode=False)

    assert excinfo.value.code == 2
    mock_backup.assert_not_called()
    captured = capsys.readouterr().out
    assert "3000" in captured
    assert "1000" not in captured  # gateway PID no longer blamed
    assert "--force" in captured


@pytest.mark.windows_only
def test_update_impl_refuses_before_terminating_gateway_ancestor(
    monkeypatch, capsys
):
    """#98814: the live venv-holder path gates the destructive call itself —
    a gateway holder that is our own ancestor is never terminated."""
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
            update_cmd._cmd_update_impl(_update_args(), gateway_mode=False)

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
