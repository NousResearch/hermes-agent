"""Tests for the hosted-runner process-tree watchdog."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from scripts.ci import timebox_process
from scripts.ci.timebox_process import KILL_SIGNAL, TIMEOUT_EXIT_CODE, run_with_timebox


class FakeProcess:
    def __init__(self, outcomes: list[int | BaseException]) -> None:
        self.outcomes = outcomes
        self.pid = 4242

    def wait(self, timeout: int | None = None) -> int:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return int(outcome)

    def poll(self) -> None:
        return None


class FakeTracker:
    def start(self) -> None:
        pass

    def stop(self) -> list[psutil.Process]:
        return []


def _fake_tracker(_pid: int) -> FakeTracker:
    return FakeTracker()


def test_returns_command_exit_code() -> None:
    process = FakeProcess([0])

    result = run_with_timebox(
        ["tests-command"],
        timeout_seconds=10,
        grace_seconds=2,
        popen=lambda *args, **kwargs: process,
        tracker_factory=_fake_tracker,
    )

    assert result == 0


def test_timeout_terminates_the_process_group() -> None:
    process = FakeProcess([subprocess.TimeoutExpired(["tests-command"], 10), -15])
    signals: list[tuple[int, int]] = []

    result = run_with_timebox(
        ["tests-command"],
        timeout_seconds=10,
        grace_seconds=2,
        popen=lambda *args, **kwargs: process,
        terminate_group=lambda pid, signum: signals.append((pid, signum)),
        tracker_factory=_fake_tracker,
    )

    assert result == TIMEOUT_EXIT_CODE
    assert signals == [(4242, signal.SIGTERM)]


def test_timeout_escalates_to_kill_after_grace_period() -> None:
    process = FakeProcess([
        subprocess.TimeoutExpired(["tests-command"], 10),
        subprocess.TimeoutExpired(["tests-command"], 2),
        -9,
    ])
    signals: list[tuple[int, int]] = []

    result = run_with_timebox(
        ["tests-command"],
        timeout_seconds=10,
        grace_seconds=2,
        popen=lambda *args, **kwargs: process,
        terminate_group=lambda pid, signum: signals.append((pid, signum)),
        tracker_factory=_fake_tracker,
    )

    assert result == TIMEOUT_EXIT_CODE
    assert signals == [(4242, signal.SIGTERM), (4242, KILL_SIGNAL)]


def test_keyboard_interrupt_terminates_the_process_group() -> None:
    process = FakeProcess([KeyboardInterrupt(), -signal.SIGINT])
    signals: list[tuple[int, int]] = []

    with pytest.raises(KeyboardInterrupt):
        run_with_timebox(
            ["tests-command"],
            timeout_seconds=10,
            grace_seconds=2,
            popen=lambda *args, **kwargs: process,
            terminate_group=lambda pid, signum: signals.append((pid, signum)),
            tracker_factory=_fake_tracker,
        )

    assert signals == [(4242, signal.SIGINT)]


def test_repeated_cancellation_does_not_interrupt_cleanup() -> None:
    handler = timebox_process._CancellationForwarder()

    class RepeatedSignalProcess(FakeProcess):
        wait_calls = 0

        def wait(self, timeout: int | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                handler(signal.SIGTERM, None)
            if self.wait_calls == 2:
                handler(signal.SIGTERM, None)
            return super().wait(timeout)

    process = RepeatedSignalProcess([
        subprocess.TimeoutExpired(["tests-command"], 2),
        -9,
    ])
    signals: list[tuple[int, int]] = []

    with pytest.raises(timebox_process._CancellationSignal):
        run_with_timebox(
            ["tests-command"],
            timeout_seconds=10,
            grace_seconds=2,
            popen=lambda *args, **kwargs: process,
            terminate_group=lambda pid, signum: signals.append((pid, signum)),
            tracker_factory=_fake_tracker,
        )

    assert signals == [(4242, signal.SIGTERM), (4242, KILL_SIGNAL)]


def test_descendant_tracker_retains_identity_after_leader_exit() -> None:
    class TrackedProcess:
        pid = 4343

        def create_time(self) -> float:
            return 123.0

    descendant = TrackedProcess()
    snapshots = iter([[descendant], []])
    tracker = timebox_process._DescendantTracker(
        4242,
        refresh_seconds=60,
        snapshot_descendants=lambda pid: next(snapshots),
    )

    tracker.start()
    retained = tracker.stop()

    assert retained == [descendant]


_GRANDCHILD_SOURCE = """
import os
from pathlib import Path
import signal
import sys
import time

if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
while True:
    time.sleep(1)
"""

_PARENT_SOURCE = """
import os
from pathlib import Path
import subprocess
import sys
import time

creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
subprocess.Popen(
    [sys.executable, "-c", sys.argv[1], sys.argv[2]],
    start_new_session=os.name != "nt",
    creationflags=creationflags,
)
Path(sys.argv[3]).write_text(str(os.getpid()), encoding="utf-8")
while True:
    time.sleep(1)
"""


def _nested_process_command(parent_pid_file: Path, child_pid_file: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        _PARENT_SOURCE,
        _GRANDCHILD_SOURCE,
        str(child_pid_file),
        str(parent_pid_file),
    ]


def _read_pid(path: Path, *, timeout_seconds: float = 5) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            return int(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"process did not publish its pid to {path}")


def _is_running(pid: int, *, marker: Path) -> bool:
    try:
        process = psutil.Process(pid)
        if str(marker) not in " ".join(process.cmdline()):
            return False
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return False


def _assert_processes_stop(
    pids: list[int], *, marker: Path, timeout_seconds: float = 5
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and any(
        _is_running(pid, marker=marker) for pid in pids
    ):
        time.sleep(0.05)
    assert not [pid for pid in pids if _is_running(pid, marker=marker)]


def _force_cleanup(pids: list[int], *, marker: Path) -> None:
    for pid in reversed(pids):
        try:
            process = psutil.Process(pid)
            if str(marker) not in " ".join(process.cmdline()):
                continue
            process.kill()
            process.wait(timeout=5)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass


def test_force_cleanup_ignores_a_reused_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnrelatedProcess:
        killed = False

        def cmdline(self) -> list[str]:
            return ["unrelated-command"]

        def kill(self) -> None:
            self.killed = True

    unrelated = UnrelatedProcess()
    monkeypatch.setattr(psutil, "Process", lambda pid: unrelated)

    _force_cleanup([99999], marker=tmp_path)

    assert unrelated.killed is False


@pytest.mark.live_system_guard_bypass
def test_timeout_kills_a_descendant_in_its_own_session(tmp_path: Path) -> None:
    """A worker-created session must not outlive a timed-out test wrapper."""
    parent_pid_file = tmp_path / "parent.pid"
    child_pid_file = tmp_path / "child.pid"
    pids: list[int] = []

    try:
        result = run_with_timebox(
            _nested_process_command(parent_pid_file, child_pid_file),
            timeout_seconds=5,
            grace_seconds=1,
        )
        pids = [_read_pid(parent_pid_file), _read_pid(child_pid_file)]

        assert result == TIMEOUT_EXIT_CODE
        _assert_processes_stop(pids, marker=tmp_path)
    finally:
        _force_cleanup(pids, marker=tmp_path)


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_sigterm_cleans_up_the_complete_process_tree(tmp_path: Path) -> None:
    """A hosted-runner cancellation must not leave test workers running."""
    parent_pid_file = tmp_path / "parent.pid"
    child_pid_file = tmp_path / "child.pid"
    pids: list[int] = []
    watchdog = subprocess.Popen(
        [
            sys.executable,
            str(Path("scripts/ci/timebox_process.py")),
            "--timeout-seconds",
            "30",
            "--grace-seconds",
            "1",
            "--",
            *_nested_process_command(parent_pid_file, child_pid_file),
        ],
        start_new_session=True,
    )
    watchdog_process = psutil.Process(watchdog.pid)
    watchdog_process.create_time()
    try:
        pids = [_read_pid(parent_pid_file), _read_pid(child_pid_file)]
        watchdog_process.send_signal(signal.SIGTERM)

        assert watchdog.wait(timeout=5) == 128 + signal.SIGTERM
        _assert_processes_stop(pids, marker=tmp_path)
    finally:
        if watchdog.poll() is None:
            watchdog.kill()
            watchdog.wait(timeout=5)
        for path in (parent_pid_file, child_pid_file):
            if path.exists():
                pid = int(path.read_text(encoding="utf-8"))
                if pid not in pids:
                    pids.append(pid)
        _force_cleanup(pids, marker=tmp_path)
