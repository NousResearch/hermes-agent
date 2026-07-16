"""Lifecycle tests for the detached Kanban worker supervisor."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_worker_supervisor as supervisor


def _wait_until_gone(pid: int, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def _start_frozen_term_ignoring_supervisor(tmp_path, monkeypatch, *, max_runtime=None):
    """Create the adversarial boundary: stopped supervisor + live stubborn child."""
    from hermes_cli import kanban_db as kb

    child_pid_path = tmp_path / "child.pid"
    spec_path = tmp_path / "worker-spec.json"
    handshake_path = tmp_path / "supervisor.pid"
    log_path = tmp_path / "worker.log"
    child_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    spec_path.write_text(
        json.dumps(
            {
                "command": [sys.executable, "-c", child_code, str(child_pid_path)],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "placeholder",
                "run_id": 1,
                "claim_lock": "placeholder",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="stubborn tree", assignee="worker",
            max_runtime_seconds=max_runtime,
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert claimed.claim_lock is not None
        assert claimed.current_run_id is not None

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec.update(
        task_id=task_id,
        run_id=claimed.current_run_id,
        claim_lock=claimed.claim_lock,
    )
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    repo_root = Path(__file__).parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.kanban_worker_supervisor", str(spec_path)],
        cwd=repo_root,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not handshake_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert int(handshake_path.read_text(encoding="utf-8")) == proc.pid
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        # Freeze after the supervisor has published and spawned.  A direct
        # supervisor SIGKILL at this interleaving leaves the stubborn child
        # orphaned; group SIGTERM/KILL must remove both.
        os.kill(proc.pid, signal.SIGSTOP)
        monkeypatch.setattr(kb, "_WORKER_TREE_TERMINATE_GRACE_SECONDS", 0.12)
        monkeypatch.setattr(kb, "_WORKER_TREE_TERMINATE_POLL_SECONDS", 0.01)
        return kb, task_id, claimed, proc, child_pid
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise


def _cleanup_process_group(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_manual_reclaim_kills_frozen_supervisor_tree_before_release(tmp_path, monkeypatch):
    kb, task_id, _claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            assert kb.reclaim_task(conn, task_id, reason="real tree regression")
            assert _wait_until_gone(child_pid), "manual reclaim orphaned child"
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.status == "ready"
            assert task.claim_lock is None
            assert task.worker_pid is None
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_stale_reclaim_kills_frozen_supervisor_tree_before_release(tmp_path, monkeypatch):
    kb, task_id, _claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch,
    )
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            conn.execute(
                "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? WHERE id = ?",
                (int(time.time()) - 1, int(time.time()) - 7200, task_id),
            )
            assert kb.release_stale_claims(conn) == 1
            assert _wait_until_gone(child_pid), "stale reclaim orphaned child"
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.status == "ready"
            assert task.claim_lock is None
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_max_runtime_kills_frozen_supervisor_tree_before_timeout_release(tmp_path, monkeypatch):
    kb, task_id, claimed, proc, child_pid = _start_frozen_term_ignoring_supervisor(
        tmp_path, monkeypatch, max_runtime=1,
    )
    try:
        with kb.connect() as conn:
            assert kb._set_worker_pid(conn, task_id, proc.pid)
            old = int(time.time()) - 10
            conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (old, task_id))
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (old, claimed.current_run_id),
            )
            assert kb.enforce_max_runtime(conn) == [task_id]
            assert _wait_until_gone(child_pid), "max runtime orphaned child"
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.status == "ready"
            assert task.claim_lock is None
            assert task.consecutive_failures == 1
    finally:
        _cleanup_process_group(proc)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group semantics only")
def test_reclaim_identity_mismatch_defers_without_signalling_unrelated_group(tmp_path):
    """A reused/non-leader PID is never converted into killpg(pid, ...)."""
    from hermes_cli import kanban_db as kb

    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="pid reuse", assignee="worker")
            assert kb.claim_task(conn, task_id) is not None
            assert kb._set_worker_pid(conn, task_id, unrelated.pid)
            assert not kb.reclaim_task(conn, task_id, reason="identity check")
            assert not kb.reassign_task(
                conn, task_id, "other-worker", reclaim_first=True,
            )
            task = kb.get_task(conn, task_id)
            events = kb.list_events(conn, task_id)
        assert unrelated.poll() is None
        assert task is not None
        assert task.status == "running"
        assert task.assignee == "worker"
        deferred = [event for event in events if event.kind == "reclaim_deferred"]
        assert len(deferred) == 2
        for event in deferred:
            assert event.payload is not None
            assert event.payload["identity"] == "process_group_mismatch"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_windows_reclaim_tree_cleanup_uses_taskkill(monkeypatch):
    """Windows reclaim uses taskkill /T /F instead of direct PID SIGKILL."""
    from hermes_cli import kanban_db as kb

    calls = []
    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        kb.subprocess,
        "run",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )
    info = kb._terminate_reclaimed_worker(91015, f"{kb._claimer_id().split(':', 1)[0]}:x")
    assert info["terminated"] is True
    assert calls[0][0] == ["taskkill", "/PID", "91015", "/T", "/F"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups only")
@pytest.mark.parametrize("supervisor_mode", ["timeout", "early_exit"])
def test_parent_handshake_failure_kills_real_supervisor_tree_and_cleans_artifacts(
    tmp_path, monkeypatch, supervisor_mode,
):
    """A no-handshake supervisor must not leave its real child behind."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(kb, "_WORKER_PID_HANDSHAKE_TIMEOUT_SECONDS", 0.08)
    db_path = kb.kanban_db_path()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()

    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    supervisor_code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); "
        + ("sys.exit(23)" if supervisor_mode == "early_exit" else "time.sleep(60)")
    )
    real_popen = subprocess.Popen

    def spawn_no_handshake(_cmd, **_kwargs):
        proc = real_popen(
            [
                sys.executable,
                "-c",
                supervisor_code,
                str(child_pid_path),
                child_code,
            ],
            start_new_session=True,
        )
        deadline = time.monotonic() + 2
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.exists()
        return proc

    monkeypatch.setattr(kb.subprocess, "Popen", spawn_no_handshake)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="tree cleanup", assignee="worker")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        with pytest.raises(RuntimeError, match=(
            "timed out" if supervisor_mode == "timeout" else "exited with code 23"
        )):
            kb._default_spawn(task, str(tmp_path))

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert _wait_until_gone(child_pid), "handshake cleanup orphaned the worker"
    log_dir = home / "kanban" / "logs"
    assert not list(log_dir.glob("*.spawn.json"))
    assert not list(log_dir.glob("*.worker.pid"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics only")
def test_real_pid_publication_failure_kills_and_reaps_worker_tree(tmp_path):
    """Supervisor-side publication failure kills a real TERM-ignoring worker."""
    child_pid_path = tmp_path / "child.pid"
    spec_path = tmp_path / "worker-spec.json"
    log_path = tmp_path / "worker.log"
    handshake_path = tmp_path / "pid-directory"
    handshake_path.mkdir()
    child_code = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    spec_path.write_text(
        json.dumps(
            {
                "command": [sys.executable, "-c", child_code, str(child_pid_path)],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "t_worker",
                "run_id": 1,
                "claim_lock": "host:worker",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )

    real_popen = subprocess.Popen
    captured = {}

    def capture_worker(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        captured["pid"] = proc.pid
        return proc

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(supervisor.subprocess, "Popen", capture_worker)
    try:
        with pytest.raises(OSError):
            supervisor.run(spec_path)
    finally:
        monkeypatch.undo()

    child_pid = captured["pid"]
    assert _wait_until_gone(child_pid), "publication failure orphaned the worker"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics only")
def test_direct_supervisor_sigterm_forwards_to_and_reaps_real_worker(tmp_path):
    """Stop/reclaim signaling the registered supervisor PID reaches its child."""
    child_pid_path = tmp_path / "child.pid"
    spec_path = tmp_path / "worker-spec.json"
    handshake_path = tmp_path / "supervisor.pid"
    log_path = tmp_path / "worker.log"
    child_code = (
        "import os, sys, time; "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    spec_path.write_text(
        json.dumps(
            {
                "command": [sys.executable, "-c", child_code, str(child_pid_path)],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "t_worker",
                "run_id": 1,
                "claim_lock": "host:worker",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.kanban_worker_supervisor", str(spec_path)],
        cwd=repo_root,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not handshake_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert int(handshake_path.read_text(encoding="utf-8")) == proc.pid
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        proc.terminate()
        proc.wait(timeout=5)
        assert _wait_until_gone(child_pid), "SIGTERM left the real worker alive"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_windows_tree_cleanup_uses_taskkill_and_reaps_supervisor(monkeypatch):
    """Windows takes the explicit taskkill /T path rather than POSIX signals."""
    from hermes_cli import kanban_db as kb

    calls = []

    class FakeSupervisor:
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            return 0

    monkeypatch.setattr(kb, "_IS_WINDOWS", True)
    monkeypatch.setattr(
        kb.subprocess,
        "run",
        lambda args, **kwargs: calls.append(("taskkill", args, kwargs)),
    )

    kb._terminate_supervisor_tree(91013, supervisor=FakeSupervisor())

    assert calls[0][0] == "taskkill"
    assert calls[0][1] == ["taskkill", "/PID", "91013", "/T", "/F"]
    assert calls[1] == ("wait", 5)


def test_handshake_timeout_preserves_original_error_when_cleanup_raises(
    tmp_path, monkeypatch,
):
    """A cleanup fault cannot replace the handshake timeout reported to dispatch."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    db_path = kb.kanban_db_path()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    monkeypatch.setattr(kb, "_WORKER_PID_HANDSHAKE_TIMEOUT_SECONDS", 0.01)

    class FakeSupervisor:
        pid = 91014
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(kb.subprocess, "Popen", lambda *_args, **_kwargs: FakeSupervisor())
    monkeypatch.setattr(
        kb,
        "_terminate_supervisor_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cleanup failure")),
    )
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="preserve timeout", assignee="worker")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        with pytest.raises(RuntimeError, match="timed out before reporting the supervisor pid"):
            kb._default_spawn(task, str(tmp_path))


def test_pid_publication_failure_terminates_kills_and_reaps_worker(tmp_path, monkeypatch):
    """Never leave a worker alive when its PID cannot be published to dispatch."""
    spec_path = tmp_path / "worker-spec.json"
    log_path = tmp_path / "worker.log"
    handshake_path = tmp_path / "worker.pid"
    spec_path.write_text(
        json.dumps(
            {
                "command": ["worker"],
                "cwd": None,
                "log_path": str(log_path),
                "handshake_path": str(handshake_path),
                "task_id": "t_worker",
                "run_id": 1,
                "claim_lock": "host:worker",
                "board": "default",
            },
        ),
        encoding="utf-8",
    )

    class Worker:
        pid = 91004

        def __init__(self):
            self.terminated = 0
            self.killed = 0
            self.wait_timeouts: list[float | None] = []

        def terminate(self):
            self.terminated += 1

        def kill(self):
            self.killed += 1

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if self.killed == 0:
                raise subprocess.TimeoutExpired(["worker"], timeout or 0.0)
            return -9

    worker = Worker()
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_args, **_kwargs: worker)

    def fail_pid_publication(_path, _pid):
        raise OSError("handshake disk failure")

    monkeypatch.setattr(supervisor, "_write_supervisor_pid", fail_pid_publication)

    with pytest.raises(OSError, match="handshake disk failure"):
        supervisor.run(spec_path)

    assert worker.terminated == 1
    assert worker.killed == 1
    assert len(worker.wait_timeouts) == 2
    assert worker.wait_timeouts[0] is not None
    assert worker.wait_timeouts[1] is None
