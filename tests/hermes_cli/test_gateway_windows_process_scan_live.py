"""Native Windows coverage for the gateway process-table fallback.

The fast path must inspect real Windows process command lines without starting
PowerShell/WMIC.  The sleepers below only carry Hermes-shaped argv; they never
import or execute the gateway, and every owned child is reaped on scope exit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from hermes_cli import gateway, gateway_windows


pytestmark = pytest.mark.windows_only


def _spawn_gateway_shaped_sleeper(profile: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-c",
            "import time; time.sleep(120)",
            "-m",
            "hermes_cli.main",
            "--profile",
            profile,
            "gateway",
            "run",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _stop_owned(child: subprocess.Popen) -> None:
    if child.poll() is None:
        child.kill()
    child.wait(timeout=10)


def _wait_for_gateway_pid(pid: int, timeout: float = 10.0) -> None:
    """Wait for a newly spawned lookalike to become visible to the native process scan."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid in gateway.find_gateway_pids():
            return
        time.sleep(0.05)
    raise AssertionError(f"replacement gateway process {pid} did not become visible")


@pytest.mark.spawns_gateway_lookalike
def test_native_scan_avoids_wmi_and_preserves_profile_scope_and_exclusions(
    tmp_path, monkeypatch
):
    profile = f"scan-{os.getpid()}"
    sibling_profile = f"{profile}-2"
    profile_home = tmp_path / "hermes-home" / "profiles" / profile
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    with ExitStack() as owned_children:
        own = _spawn_gateway_shaped_sleeper(profile)
        owned_children.callback(_stop_owned, own)
        sibling = _spawn_gateway_shaped_sleeper(sibling_profile)
        owned_children.callback(_stop_owned, sibling)

        def _wmi_must_not_run():
            raise AssertionError("native psutil snapshot unexpectedly fell back to WMIC/PowerShell")

        monkeypatch.setattr(gateway, "_windows_process_listing", _wmi_must_not_run)

        scoped = gateway._scan_gateway_pids(set())
        assert own.pid in scoped
        assert sibling.pid not in scoped

        excluded = gateway._scan_gateway_pids({own.pid})
        assert own.pid not in excluded
        assert sibling.pid not in excluded

        all_profiles = gateway._scan_gateway_pids(set(), all_profiles=True)
        assert own.pid in all_profiles
        assert sibling.pid in all_profiles


def test_scan_retains_bounded_listing_fallback_when_native_snapshot_is_unavailable(
    tmp_path, monkeypatch
):
    profile = f"fallback-{os.getpid()}"
    profile_home = tmp_path / "hermes-home" / "profiles" / profile
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr(gateway_windows, "_snapshot_process_command_lines", lambda: None)

    calls = []

    def _fallback_listing():
        calls.append(True)
        return "\n".join(
            [
                f"CommandLine=python.exe -m hermes_cli.main --profile {profile} gateway run",
                "ProcessId=11001",
                f"CommandLine=python.exe -m hermes_cli.main --profile {profile}-2 gateway run",
                "ProcessId=11002",
                f"CommandLine=python.exe -m hermes_cli.main --profile {profile} gateway run",
                "ProcessId=11003",
            ]
        )

    monkeypatch.setattr(gateway, "_windows_process_listing", _fallback_listing)

    assert gateway._scan_gateway_pids({11003}) == [11001]
    assert calls == [True]


def test_native_snapshot_preserves_argument_boundaries_and_fails_over_cleanly(monkeypatch):
    psutil = pytest.importorskip("psutil")
    argv = [
        r"C:\Program Files\Hermes\python.exe",
        "-m",
        "hermes_cli.main",
        "--profile",
        "work",
        "gateway",
        "run",
    ]
    entries = [
        SimpleNamespace(info={"pid": 12001, "cmdline": argv}),
        SimpleNamespace(info={"pid": 12002, "cmdline": None}),
        SimpleNamespace(info={"pid": 12003, "cmdline": []}),
    ]
    monkeypatch.setattr(psutil, "process_iter", lambda *args, **kwargs: iter(entries))

    assert gateway_windows._snapshot_process_command_lines() == [
        (12001, subprocess.list2cmdline(argv))
    ]

    def _failed_snapshot(*_args, **_kwargs):
        raise psutil.Error("synthetic process-table failure")

    monkeypatch.setattr(psutil, "process_iter", _failed_snapshot)
    assert gateway_windows._snapshot_process_command_lines() is None


def test_empty_orphan_scan_does_not_wait_for_task_scheduler(tmp_path, monkeypatch):
    """A fresh profile with no gateway has no process for the supervisor to protect."""
    profile_home = tmp_path / "hermes-home" / "profiles" / f"empty-{os.getpid()}"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    def _unexpected_probe(_name):
        raise AssertionError("empty orphan sweep must not wait for Task Scheduler")

    def _unexpected_kill(_pid, _signal):
        raise AssertionError("an empty profile must not target any process")

    monkeypatch.setattr(gateway, "_windows_scheduled_task_supervises", _unexpected_probe)
    monkeypatch.setattr(gateway.os, "kill", _unexpected_kill)
    assert gateway._reap_unsupervised_gateway_orphans() is False


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.parametrize("state", ["Running", "Ready", "Queued", "Disabled", "MISSING", None])
def test_nonempty_orphan_sweep_keeps_supervision_and_rescans_after_probe(
    tmp_path, monkeypatch, state
):
    """A task probe cannot authorize killing a stale scan or a supervised child."""
    profile = f"reap-{os.getpid()}"
    profile_home = tmp_path / "hermes-home" / "profiles" / profile
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    supervised = state in {"Running", "Ready", "Queued"}
    real_kill = os.kill
    killed = []
    probes = []
    # Hosted Windows runners may themselves descend from services.exe. Keep this
    # test focused on Task Scheduler state; the parent-chain backstop is covered
    # independently and must not classify the test sleeper as service-owned.
    monkeypatch.setattr(gateway, "_reaper_candidate_is_supervisor_owned", lambda _pid: False)

    with ExitStack() as owned_children:
        original = _spawn_gateway_shaped_sleeper(profile)
        owned_children.callback(_stop_owned, original)
        replacement = None

        def _task_state(name):
            nonlocal replacement
            probes.append(name)
            if not supervised:
                # The process table can change during a slow scheduler query.
                _stop_owned(original)
                replacement = _spawn_gateway_shaped_sleeper(profile)
                owned_children.callback(_stop_owned, replacement)
                _wait_for_gateway_pid(replacement.pid)
            return state

        def _kill_owned_only(pid, sig):
            assert replacement is not None and pid == replacement.pid
            killed.append(pid)
            return real_kill(pid, sig)

        monkeypatch.setattr(gateway, "_windows_scheduled_task_state", _task_state)
        monkeypatch.setattr(gateway.os, "kill", _kill_owned_only)

        assert gateway._reap_unsupervised_gateway_orphans() is (not supervised)
        assert probes == [gateway_windows.get_task_name()]
        if supervised:
            assert original.poll() is None
            assert killed == []
        else:
            assert replacement is not None
            assert killed == [replacement.pid]
            assert replacement.wait(timeout=10) is not None
