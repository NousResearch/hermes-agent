"""Regression tests for cron child-process termination semantics."""

import concurrent.futures
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _process_adapter(proc):
    class Adapter:
        pid = proc.pid

        @staticmethod
        def is_alive():
            return proc.poll() is None

        @staticmethod
        def terminate():
            proc.terminate()

        @staticmethod
        def kill():
            proc.kill()

    return Adapter()


def _wait_dead(proc, timeout=3.0):
    deadline = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    return proc.poll() is not None


def _wait_for_pid_file(path: Path, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return int(path.read_text())
        time.sleep(0.02)
    raise AssertionError("child did not publish its descendant PID")


def _pid_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_is_running(pid):
    """Return false for exited or zombie processes."""
    if not _pid_is_alive(pid):
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, OSError, IndexError):
        return False
    return state != "Z"


def _handshake_result_child(job, child_conn, ready_conn, start_conn, *args):
    ready_conn.send("ready")
    ready_conn.close()
    assert start_conn.recv() == "start"
    child_conn.send(("result", (True, "output", "response", None)))
    child_conn.close()
    os._exit(0)


def _slow_start_child(job, child_conn, ready_conn, start_conn, *args):
    time.sleep(1.0)
    ready_conn.send("ready")
    ready_conn.close()
    os._exit(0)


def _handshake_start_marker_child(job, child_conn, ready_conn, start_conn, *args):
    ready_conn.send("ready")
    ready_conn.close()
    assert start_conn.recv() == "start"
    Path(job["start_file"]).write_text("released")
    child_conn.send(("result", (True, "output", "response", None)))
    child_conn.close()
    os._exit(0)


def _stubborn_child(job, child_conn, ready_conn, start_conn, *args):
    ready_conn.send("ready")
    ready_conn.close()
    start_conn.recv()
    if os.name == "posix":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)


def _pipe_failure_child(job, child_conn, ready_conn, start_conn, *args):
    ready_conn.send("ready")
    ready_conn.close()
    start_conn.recv()
    os._exit(7)


def _orphaning_child(job, child_conn, ready_conn, start_conn, *args):
    """Exit the group leader after starting a descendant in its own group."""
    if os.name != "posix":
        os._exit(0)
    os.setsid()
    ready_conn.send("ready")
    ready_conn.close()
    assert start_conn.recv() == "start"
    descendant_code = (
        f"import os,time; open({job['orphan_pid_file']!r}, 'w').write(str(os.getpid())); "
        f"time.sleep({job.get('orphan_sleep_seconds', 30)})"
    )
    subprocess.Popen(
        [sys.executable, "-c", descendant_code],
        start_new_session=job.get("orphan_start_new_session", True),
    )
    deadline = time.monotonic() + 1.0
    while not os.path.exists(job["orphan_pid_file"]) and time.monotonic() < deadline:
        time.sleep(0.01)
    os._exit(124)


def _inactivity_timeout_child(job, child_conn, ready_conn, start_conn, *args):
    """Exercise the child-side non-cooperative inactivity termination path."""
    import cron.scheduler as scheduler

    ready_conn.send("ready")
    ready_conn.close()
    assert start_conn.recv() == "start"

    class Agent:
        def interrupt(self, reason):
            self.reason = reason

    agent = Agent()
    future = concurrent.futures.Future()
    future.set_running_or_notify_cancel()
    if scheduler._interrupt_and_wait_for_cron_future(
        agent, future, "inactivity", timeout=0.05
    ):
        os._exit(2)
    # This is the authoritative child boundary: the parent must observe the
    # exit and reap the process instead of waiting on the non-cooperative
    # worker future forever.
    os._exit(124)


def test_force_termination_kills_the_complete_child_process_group(tmp_path):
    """A stubborn child and its descendant must not survive escalation."""
    if os.name != "posix":
        pytest.skip("process-group signal semantics are POSIX-specific")
    import cron.scheduler as scheduler

    pid_file = tmp_path / "descendant.pid"
    descendant_code = (
        f"import os,time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    code = (
        "import os, signal, subprocess, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"subprocess.Popen([os.sys.executable, '-c', {descendant_code!r}]); "
        "time.sleep(30)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        descendant_pid = _wait_for_pid_file(pid_file)
        assert _pid_is_alive(descendant_pid)
        scheduler._terminate_cron_process_tree(_process_adapter(proc), force=False)
        assert not _wait_dead(proc, timeout=0.3), "SIGTERM should be ignored by the fixture"
        scheduler._terminate_cron_process_tree(_process_adapter(proc), force=True)
        assert _wait_dead(proc)
        assert proc.returncode is not None
        deadline = time.monotonic() + 3
        while _pid_is_alive(descendant_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _pid_is_alive(descendant_pid)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)


def test_interrupt_wait_is_bounded_for_non_cooperative_agent():
    """A stuck worker must not make timeout cleanup wait forever."""
    import cron.scheduler as scheduler

    class Agent:
        def __init__(self):
            self.interrupted = None

        def interrupt(self, reason):
            self.interrupted = reason

    agent = Agent()
    future = concurrent.futures.Future()
    future.set_running_or_notify_cancel()
    started = time.monotonic()
    stopped = scheduler._interrupt_and_wait_for_cron_future(
        agent, future, "inactivity", timeout=0.05
    )
    elapsed = time.monotonic() - started
    assert stopped is False
    assert agent.interrupted == "inactivity"
    assert elapsed < 1.0


def test_isolated_supervisor_handshake_result_and_registry_cleanup(monkeypatch):
    """The production supervisor owns handshake, result, reaping, and removal."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_cron_child_entry", _handshake_result_child)
    job = {"id": "isolated-success", "name": "isolated success"}
    assert scheduler._run_isolated_cron_job(job) == (True, "output", "response", None)
    with scheduler._running_lock:
        assert job["id"] not in scheduler._cron_processes


def test_isolated_supervisor_applies_wall_bound_during_startup(monkeypatch):
    """A sub-10-second cap must bound readiness, not just agent execution."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_cron_child_entry", _slow_start_child)
    monkeypatch.setattr(scheduler, "_cron_max_runtime_seconds", lambda: 0.1)
    job = {"id": "isolated-startup-timeout", "name": "startup timeout"}
    started = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError)):
        scheduler._run_isolated_cron_job(job)
    assert time.monotonic() - started < 3.0
    with scheduler._running_lock:
        assert job["id"] not in scheduler._cron_processes


def test_isolated_supervisor_reports_boundary_fallback_status(monkeypatch):
    """Unavailable hard containment is explicit while the child remains safe."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_cron_child_entry", _handshake_result_child)
    monkeypatch.setattr(
        scheduler,
        "allocate_boundary",
        lambda job_id: (_ for _ in ()).throw(
            scheduler.BoundaryUnavailable("delegation unavailable")
        ),
    )
    job = {"id": "isolated-fallback", "name": "isolated fallback"}
    assert scheduler._run_isolated_cron_job(job)[0] is True
    assert scheduler.get_cron_boundary_status(job["id"]) == "unavailable"
    with scheduler._running_lock:
        assert job["id"] not in scheduler._cron_processes


def test_isolated_supervisor_reports_unsupported_boundary_status(monkeypatch):
    """Non-Linux platforms never report hard containment as available."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_cron_child_entry", _handshake_result_child)
    monkeypatch.setattr(scheduler.sys, "platform", "darwin")
    monkeypatch.setattr(
        scheduler,
        "allocate_boundary",
        lambda job_id: (_ for _ in ()).throw(
            scheduler.BoundaryUnavailable("hard boundary unsupported")
        ),
    )
    job = {"id": "isolated-unsupported", "name": "isolated unsupported"}
    assert scheduler._run_isolated_cron_job(job)[0] is True
    assert scheduler.get_cron_boundary_status(job["id"]) == "unsupported"


class _FakeBoundary:
    def __init__(self, *, assignment_error=None, terminate_result=True):
        self.assignment_error = assignment_error
        self.terminate_result = terminate_result
        self.assign_called = False
        self.terminate_called = 0

    def assign_and_verify(self, pid):
        self.assign_called = True
        if self.assignment_error:
            raise self.assignment_error

    def terminate(self, *, force, timeout):
        self.terminate_called += 1
        return self.terminate_result


def test_isolated_supervisor_keeps_child_parked_on_assignment_failure(monkeypatch, tmp_path):
    import cron.scheduler as scheduler

    boundary = _FakeBoundary(
        assignment_error=RuntimeError("identity mismatch"), terminate_result=False
    )
    monkeypatch.setattr(scheduler, "_cron_child_entry", _handshake_start_marker_child)
    monkeypatch.setattr(scheduler, "allocate_boundary", lambda job_id: boundary)
    job = {
        "id": "isolated-assignment-failure",
        "name": "isolated assignment failure",
        "start_file": str(tmp_path / "released"),
    }
    with pytest.raises(RuntimeError, match="identity mismatch"):
        scheduler._run_isolated_cron_job(job)
    assert not Path(job["start_file"]).exists()
    assert boundary.assign_called
    assert scheduler.get_cron_boundary_status(job["id"]) == "cleanup_failed"
    with scheduler._running_lock:
        assert scheduler._cron_boundaries[job["id"]] is boundary
        process = scheduler._cron_processes[job["id"]]
    process.kill()
    process.join(timeout=3)
    with scheduler._running_lock:
        scheduler._cron_processes.pop(job["id"], None)
        scheduler._cron_boundaries.pop(job["id"], None)


def test_isolated_supervisor_reaps_parked_child_after_successful_empty_teardown(monkeypatch):
    """Assignment failure must not strand a child after empty-boundary cleanup."""
    import cron.scheduler as scheduler

    boundary = _FakeBoundary(
        assignment_error=RuntimeError("identity mismatch"), terminate_result=True
    )
    monkeypatch.setattr(scheduler, "_cron_child_entry", _handshake_start_marker_child)
    monkeypatch.setattr(scheduler, "allocate_boundary", lambda job_id: boundary)
    job = {
        "id": "isolated-empty-teardown",
        "name": "isolated empty teardown",
        "start_file": "/tmp/never-released",
    }

    with pytest.raises(RuntimeError, match="identity mismatch"):
        scheduler._run_isolated_cron_job(job)

    assert boundary.assign_called
    assert boundary.terminate_called == 1
    assert scheduler.get_cron_boundary_status(job["id"]) == "terminated"
    with scheduler._running_lock:
        assert job["id"] not in scheduler._cron_processes
        assert job["id"] not in scheduler._cron_boundaries
        assert job["id"] not in scheduler._cron_start_parents


def test_isolated_supervisor_retains_boundary_when_allocation_cleanup_fails(monkeypatch):
    import cron.scheduler as scheduler

    boundary = _FakeBoundary(terminate_result=False)
    monkeypatch.setattr(scheduler, "_cron_child_entry", _handshake_result_child)
    monkeypatch.setattr(
        scheduler,
        "allocate_boundary",
        lambda job_id: (_ for _ in ()).throw(
            scheduler.BoundaryUnavailable(
                "allocation probe failed", boundary=boundary, cleanup_failed=True
            )
        ),
    )
    job = {"id": "isolated-allocation-cleanup-failure", "name": "allocation cleanup failure"}
    with pytest.raises(RuntimeError, match="allocation cleanup failed"):
        scheduler._run_isolated_cron_job(job)
    assert scheduler.get_cron_boundary_status(job["id"]) == "cleanup_failed"
    with scheduler._running_lock:
        assert scheduler._cron_boundaries[job["id"]] is boundary
        process = scheduler._cron_processes[job["id"]]
    process.kill()
    process.join(timeout=3)
    with scheduler._running_lock:
        scheduler._cron_processes.pop(job["id"], None)
        scheduler._cron_boundaries.pop(job["id"], None)


def test_isolated_supervisor_retains_boundary_when_termination_fails(monkeypatch):
    import cron.scheduler as scheduler

    boundary = _FakeBoundary(terminate_result=False)
    monkeypatch.setattr(scheduler, "_cron_child_entry", _stubborn_child)
    monkeypatch.setattr(scheduler, "allocate_boundary", lambda job_id: boundary)
    monkeypatch.setattr(scheduler, "_cron_max_runtime_seconds", lambda: 0.3)
    job = {"id": "isolated-cleanup-failure", "name": "isolated cleanup failure"}
    result = scheduler._run_isolated_cron_job(job)
    assert result[0] is False
    assert scheduler.get_cron_boundary_status(job["id"]) == "cleanup_failed"
    with scheduler._running_lock:
        assert scheduler._cron_boundaries[job["id"]] is boundary
        process = scheduler._cron_processes[job["id"]]
    process.kill()
    process.join(timeout=3)
    with scheduler._running_lock:
        scheduler._cron_processes.pop(job["id"], None)
        scheduler._cron_boundaries.pop(job["id"], None)


def test_isolated_supervisor_escalates_and_reaps_on_wall_timeout(monkeypatch):
    """A child that ignores normal completion is terminated and reaped."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_cron_child_entry", _stubborn_child)
    monkeypatch.setattr(scheduler, "_cron_max_runtime_seconds", lambda: 0.5)
    terminate_calls = []
    real_terminate = scheduler._terminate_cron_process_tree

    def record_terminate(process, *, force, boundary=None):
        terminate_calls.append(force)
        return real_terminate(process, force=force, boundary=boundary)

    monkeypatch.setattr(scheduler, "_terminate_cron_process_tree", record_terminate)
    job = {"id": "isolated-timeout", "name": "isolated timeout"}
    started = time.monotonic()
    result = scheduler._run_isolated_cron_job(job)
    assert result[0] is False
    assert result[3] and "exceeded wall-clock limit" in result[3]
    assert terminate_calls and terminate_calls[0] is False
    assert time.monotonic() - started < 8.0
    with scheduler._running_lock:
        assert job["id"] not in scheduler._cron_processes


def test_isolated_supervisor_reports_pipe_failure_and_reaps(monkeypatch):
    """Child exit without a result is surfaced as a bounded failure."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_cron_child_entry", _pipe_failure_child)
    job = {"id": "isolated-pipe-failure", "name": "isolated pipe failure"}
    try:
        scheduler._run_isolated_cron_job(job)
    except RuntimeError as exc:
        assert "without a result" in str(exc)
    else:
        raise AssertionError("expected child pipe failure")
    with scheduler._running_lock:
        assert job["id"] not in scheduler._cron_processes


def test_isolated_supervisor_best_effort_group_cleanup_after_leader_exit(
    monkeypatch, tmp_path
):
    """Fallback cleanup kills descendants that remain in the process group."""
    if os.name != "posix":
        pytest.skip("process-group signal semantics are POSIX-specific")
    import cron.scheduler as scheduler

    pid_file = tmp_path / "orphan.pid"

    monkeypatch.setattr(
        scheduler,
        "allocate_boundary",
        lambda job_id: (_ for _ in ()).throw(
            scheduler.BoundaryUnavailable("delegation unavailable")
        ),
    )
    monkeypatch.setattr(scheduler, "_cron_child_entry", _orphaning_child)
    job = {
        "id": "isolated-group-fallback",
        "name": "isolated group fallback",
        "orphan_pid_file": str(pid_file),
        "orphan_start_new_session": False,
        "orphan_sleep_seconds": 2,
    }
    with pytest.raises(RuntimeError, match="without a result"):
        scheduler._run_isolated_cron_job(job)
    orphan_pid = _wait_for_pid_file(pid_file)
    assert scheduler.get_cron_boundary_status(job["id"]) == "unavailable"
    deadline = time.monotonic() + 3.0
    while _pid_is_alive(orphan_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _pid_is_alive(orphan_pid)
    with scheduler._running_lock:
        assert job["id"] not in scheduler._cron_processes


def test_isolated_supervisor_reaps_child_after_non_cooperative_inactivity(monkeypatch):
    """Inactivity cleanup must terminate the child, not wait on its Future."""
    import cron.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_cron_child_entry", _inactivity_timeout_child)
    monkeypatch.setattr(scheduler, "_cron_max_runtime_seconds", lambda: None)
    job = {"id": "isolated-inactivity", "name": "isolated inactivity"}
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="without a result"):
        scheduler._run_isolated_cron_job(job)
    assert time.monotonic() - started < 3.0
    with scheduler._running_lock:
        assert job["id"] not in scheduler._cron_processes


def test_real_cgroup_supervisor_path_is_capability_gated(monkeypatch, tmp_path):
    """The real supervisor path is exercised only when cgroup-v2 is usable."""
    import cron.process_boundary as boundary_module
    import cron.scheduler as scheduler

    try:
        boundary_module._current_cgroup_parent()
        probe_boundary = boundary_module.allocate_boundary("capability-gate-probe")
        assert probe_boundary.terminate(force=True, timeout=3.0)
    except boundary_module.BoundaryUnavailable as exc:
        pytest.skip(f"cgroup hard-containment capability unavailable: {exc}")
    monkeypatch.setattr(scheduler, "_cron_child_entry", _orphaning_child)
    descendant_file = tmp_path / "supervisor-descendant.pid"
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    job = {
        "id": "real-supervisor-path",
        "name": "real supervisor path",
        "orphan_pid_file": str(descendant_file),
    }
    try:
        with pytest.raises(RuntimeError, match="without a result"):
            scheduler._run_isolated_cron_job(job)
        descendant_pid = _wait_for_pid_file(descendant_file)
        deadline = time.monotonic() + 3.0
        while _pid_is_running(descendant_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _pid_is_running(descendant_pid)
        assert _pid_is_alive(unrelated.pid)
        assert scheduler.get_cron_boundary_status(job["id"]) == "contained"
    finally:
        if unrelated.poll() is None:
            unrelated.kill()
            unrelated.wait(timeout=3)


def test_real_cgroup_boundary_contains_detached_descendant_and_preserves_unrelated(tmp_path):
    """Exercise the authoritative boundary when this host delegates one."""
    import cron.process_boundary as boundary_module

    try:
        boundary = boundary_module.allocate_boundary("real-boundary-test")
    except boundary_module.BoundaryUnavailable as exc:
        pytest.skip(f"cgroup hard-containment capability unavailable: {exc}")

    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    descendant_file = str(tmp_path / "real-descendant.pid")
    descendant_code = (
        "import time; time.sleep(30)"
    )
    child_code = (
        "import os,subprocess,sys; "
        "sys.stdin.buffer.read(1); "
        f"f=open({descendant_file!r}, 'w'); "
        f"p=subprocess.Popen([sys.executable, '-c', {descendant_code!r}], "
        "start_new_session=True); f.write(str(p.pid)); f.close(); os._exit(0)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdin=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        boundary.assign_and_verify(child.pid)
        child.stdin.write(b"x")
        child.stdin.close()
        assert child.wait(timeout=3) == 0
        descendant_pid = _wait_for_pid_file(tmp_path / "real-descendant.pid")
        assert _pid_is_running(descendant_pid)
        assert boundary.terminate(force=True, timeout=3.0)
        deadline = time.monotonic() + 3.0
        while _pid_is_running(descendant_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _pid_is_running(descendant_pid)
        assert _pid_is_alive(unrelated.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)
        if unrelated.poll() is None:
            unrelated.kill()
            unrelated.wait(timeout=3)
        if boundary.path.exists():
            boundary.terminate(force=True, timeout=3.0)
