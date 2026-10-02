"""Windows gateway service invariants and launcher identity after PM cutover."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main, update_cmd, update_cmd_windows

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


@pytest.mark.platforms("windows")
def test_pause_stops_launcher_after_worker_drain(
    monkeypatch,
    tmp_path,
):
    """Capture the launcher identity while its worker is still inspectable."""
    import hermes_cli.gateway as gateway_mod
    import gateway.status as status_mod

    # The install venv is whatever hermes_constants.project_venv_dir resolves for the checkout (a
    # CI checkout has no venv/ and the test interpreter lives elsewhere); pin it to the fixture layout.
    monkeypatch.setattr("hermes_constants.project_venv_dir", lambda root: cli_main.PROJECT_ROOT / "venv")
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

    assert update_cmd_windows._leftover_pausable_gateway_pids(matches) == [300, 301]


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

    assert update_cmd_windows._leftover_pausable_gateway_pids(matches) is None


def test_unreadable_argv_falls_back_to_the_captured_prefix(monkeypatch):
    """psutil failure degrades to the scan's captured cmdline, not a crash.

    The captured prefix decides: a gateway invocation still qualifies, and
    anything else still refuses.
    """
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil_cmdlines({}))
    gateway_prefix = r"venv\Scripts\python.exe -m hermes_cli.main gateway run"

    assert update_cmd_windows._leftover_pausable_gateway_pids(
        [(300, "python.exe", gateway_prefix)]
    ) == [300]
    assert (
        update_cmd_windows._leftover_pausable_gateway_pids(
            [
                (300, "python.exe", gateway_prefix),
                (400, "python.exe", "python.exe -i"),
            ]
        )
        is None
    )


# Current PM updates select immutable dependency generations. The retired
# concurrent shim filter below must hand control to takeover without inspecting
# or quarantining live launchers. Native recovery continues to use the canonical
# gateway matcher; actual updater concurrency is enforced by UpdateLock.

@pytest.mark.parametrize("argv", [
    [r"C:\venv\Scripts\python.exe", "-m", "hermes_cli.main", "gateway", "run"],
    [r"C:\venv\Scripts\hermes.exe", "gateway", "run"],
    [r"C:\venv\Scripts\hermes-gateway.exe"],
    [r"C:\venv\Scripts\python.exe", "gateway/run.py"],
    ["hermes.exe", "GATEWAY", "RUN"],
    ["hermes.exe", "gateway"],
    ["hermes.exe", "--profile", "work", "gateway", "run"],
])
def test_canonical_holder_classifier_recognises_gateway_runtimes(argv):
    """Current native recovery still shares the canonical gateway matcher."""
    from hermes_cli._scan_venv_blockers import _is_pausable_gateway

    assert _is_pausable_gateway(" ".join(argv)) is True


@pytest.mark.parametrize("argv", [
    [r"C:\venv\Scripts\hermes.exe"],
    [r"C:\venv\Scripts\hermes.exe", "dashboard"],
    ["hermes.exe", "gateway", "status"],
    ["hermes.exe", "gateway", "stop"],
    ["python", "-m", "hermes_cli.main"],
    [],
])
def test_canonical_holder_classifier_rejects_non_gateways(argv):
    from hermes_cli._scan_venv_blockers import _is_pausable_gateway

    assert _is_pausable_gateway(" ".join(argv)) is False


@pytest.mark.parametrize("psutil_module", [None, _fake_psutil_cmdlines({})], ids=["missing", "unreadable"])
def test_unknown_holder_cannot_gain_gateway_recovery_exemption(monkeypatch, psutil_module):
    """Unavailable live argv and an unknown captured prefix keep refusal."""
    monkeypatch.setitem(sys.modules, "psutil", psutil_module)
    assert update_cmd_windows._leftover_pausable_gateway_pids([(4242, "python", "")]) is None


@pytest.mark.parametrize("matches", [
    [(100, "hermes.exe"), (200, "hermes.exe"), (300, "hermes.exe"), (400, "hermes.exe")],
    [(111, "hermes.exe"), (222, "hermes-gateway.exe")],
], ids=["mixed", "gateway-only"])
def test_retired_concurrent_filter_hands_off_without_classifying_or_mutating(monkeypatch, matches):
    """PM uses immutable generations; historical gates must escape, not quarantine."""
    before = list(matches)
    calls = []

    def stop():
        calls.append("takeover")
        raise SystemExit(17)

    # Stub before invoking this retired hook: never spawn an actual updater.
    monkeypatch.setattr(update_cmd, "stop_for_relaunch", stop)
    monkeypatch.setattr(update_cmd_windows, "_leftover_pausable_gateway_pids", lambda *_: pytest.fail("retired filter must not rediscover processes"))
    with pytest.raises(SystemExit) as stopped:
        update_cmd._filter_non_gateway_concurrent_instances(matches)
    assert stopped.value.code == 17
    assert matches == before
    assert calls == ["takeover"]


@pytest.mark.platforms('windows')
def test_current_update_lock_admits_new_owner_without_rewriting_shims(tmp_path, monkeypatch):
    """A new PM update owns its marker while existing executable bytes survive."""
    from hermes_cli import update_lock

    marker = tmp_path / "home" / update_lock.MARKER_NAME
    executable = tmp_path / "Scripts" / "hermes.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"synthetic running shim")
    with update_lock.UpdateLock(path=marker) as lock:
        assert lock.acquired is True
        assert marker.is_file()
        assert executable.read_bytes() == b"synthetic running shim"
    assert not marker.exists()
    assert executable.read_bytes() == b"synthetic running shim"
    assert list(executable.parent.iterdir()) == [executable]


@pytest.mark.platforms('windows')
def test_current_update_lock_refuses_unrelated_live_updater_without_mutation(tmp_path, monkeypatch):
    """Update concurrency now means an update owner, not any REPL process."""
    from hermes_cli import update_lock

    marker = tmp_path / update_lock.MARKER_NAME
    original = "987654321\n" + str(int(update_lock.time.time())) + "\n"
    marker.write_text(original, encoding="utf-8")
    monkeypatch.delenv(update_lock.HANDOFF_PID_ENV, raising=False)
    monkeypatch.setattr(update_lock, "_pid_alive", lambda pid: pid == 987654321)
    monkeypatch.setattr(update_lock, "_is_ancestor_pid", lambda pid: False)
    with update_lock.UpdateLock(path=marker) as lock:
        assert lock.acquired is False
        assert lock.holder.pid == 987654321
        assert marker.read_text(encoding="utf-8") == original
    assert marker.read_text(encoding="utf-8") == original


@pytest.mark.platforms('windows')
def test_update_guard_refuses_before_terminating_gateway_ancestor(monkeypatch, capsys):
    """Current native holder guard checks ancestry before any destructive call."""
    import gateway.status as status_mod
    import hermes_cli.gateway as gateway_cli

    holder = (300, "python.exe", r"C:\x\venv\Scripts\python.exe -m hermes_cli.main gateway run")
    monkeypatch.setattr(gateway_cli, "_is_pid_ancestor_of_current_process", lambda pid: pid == 300)
    monkeypatch.setattr(cli_main, "_detect_venv_python_processes", lambda: [holder])
    monkeypatch.setattr(cli_main, "_leftover_pausable_gateway_pids", lambda matches: [300])
    with patch.object(cli_main, "_resume_windows_gateways_after_update") as resume, patch.object(status_mod, "terminate_pid") as terminate:
        with pytest.raises(SystemExit) as stopped:
            update_cmd_windows._clear_windows_venv_holders_or_exit(SimpleNamespace(), False, {"resume_needed": False})
    assert stopped.value.code == 2
    terminate.assert_not_called()
    resume.assert_called_once_with({"resume_needed": False})
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
