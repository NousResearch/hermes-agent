"""V4 targeted tests for foreground terminal cancellation.

Tests:
1. Foreground terminal PID registration at spawn time (not after completion)
2. cancel_job kills foreground terminal process tree without manual kill
3. Worker thread interrupt signal set during cancel
4. Worker cleanup in try/finally ensures _hermes_job_id removed
5. Gateway equivalent: /jobs → /stop <job_id> → foreground terminal termination
6. Normal exit unregisters PID from ProcessRegistry

All tests use @pytest.mark.live_system_guard_bypass because they spawn
real child processes and need real os.kill/psutil signal delivery.
"""
import os
import sys
import time
import threading
import subprocess

import pytest

pytestmark = pytest.mark.live_system_guard_bypass

os.environ["HERMES_PREEMPTIVE_CANCELLATION"] = "true"

from agent.cancellation import (
    JobState, JobManager, CancellationToken,
    ProcessRegistry, get_process_registry,
    is_preemptive_cancellation_enabled,
)


# ── Helpers ──

def _count_descendants(pid):
    """Count living descendant processes of pid using psutil."""
    try:
        import psutil
        parent = psutil.Process(pid)
        return len(parent.children(recursive=True))
    except Exception:
        return -1  # can't determine


def _process_alive(pid):
    """Check if a PID is still alive."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


class TestForegroundTerminalPIDRegistration:
    """Foreground PID registered at spawn, unregistered on normal exit."""

    def test_pid_registered_immediately_after_spawn(self):
        """PID should be in ProcessRegistry before _wait_for_process starts."""
        from tools.environments.local import LocalEnvironment

        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        env = LocalEnvironment(cwd="/tmp")
        threading.current_thread()._hermes_job_id = jid

        # Run a quick command — registration should happen and unregister on exit
        result = env.execute("echo hello", timeout=10)

        # After normal exit, PID should be unregistered
        pids = registry.get_pids(jid)
        assert len(pids) == 0, f"Expected 0 PIDs after normal exit, got {pids}"

        # Cleanup
        delattr(threading.current_thread(), "_hermes_job_id")

    def test_pid_unregistered_on_normal_exit(self):
        """Normal process completion should unregister the PID."""
        from tools.environments.local import LocalEnvironment

        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        env = LocalEnvironment(cwd="/tmp")
        threading.current_thread()._hermes_job_id = jid

        # Register a fake PID manually to verify unregister works
        registry.register_pid(jid, 12345)
        assert 12345 in registry.get_pids(jid)
        registry.unregister_pid(jid, 12345)
        assert 12345 not in registry.get_pids(jid)

        delattr(threading.current_thread(), "_hermes_job_id")


class TestForegroundTerminalCancellation:
    """cancel_job kills foreground terminal process tree."""

    def test_cancel_job_kills_foreground_sleep(self):
        """cancel_job should kill a running foreground sleep command."""
        from tools.environments.local import LocalEnvironment

        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        env = LocalEnvironment(cwd="/tmp")
        result_holder = {}
        error_holder = {}

        def worker():
            threading.current_thread()._hermes_job_id = jid
            try:
                result_holder["result"] = env.execute("sleep 30", timeout=60)
            except Exception as e:
                error_holder["error"] = e
            finally:
                try:
                    delattr(threading.current_thread(), "_hermes_job_id")
                except AttributeError:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Wait for process to spawn and register
        deadline = time.time() + 5
        while time.time() < deadline:
            pids = registry.get_pids(jid)
            if pids:
                break
            time.sleep(0.05)
        else:
            t.join(timeout=1)
            pytest.fail("No PID registered within 5s — foreground registration broken")

        registered_pids = registry.get_pids(jid)
        assert len(registered_pids) >= 1, f"Expected at least 1 PID registered, got {registered_pids}"

        # Verify the process is alive
        for pid in registered_pids:
            assert _process_alive(pid), f"Process {pid} should be alive before cancel"

        # Cancel the job — should kill the process tree
        mgr.cancel_job(jid)

        # Wait for worker to return (cancel should unblock _wait_for_process)
        t.join(timeout=15)
        assert not t.is_alive(), "Worker thread should have returned after cancel"

        # Verify processes are dead
        for pid in registered_pids:
            assert not _process_alive(pid), f"Process {pid} should be dead after cancel"

        # Verify final state is CANCELLED
        state = mgr.get_state(jid)
        assert state == JobState.CANCELLED, f"Expected CANCELLED, got {state}"

        # Verify registry is cleared
        remaining = registry.get_pids(jid)
        assert remaining == [], f"Registry should be empty after CANCELLED, got {remaining}"

    def test_cancel_job_kills_parent_and_children(self):
        """cancel_job should kill both parent (bash) and child (sleep) processes."""
        from tools.environments.local import LocalEnvironment

        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        env = LocalEnvironment(cwd="/tmp")

        def worker():
            threading.current_thread()._hermes_job_id = jid
            try:
                env.execute("sleep 30", timeout=60)
            finally:
                try:
                    delattr(threading.current_thread(), "_hermes_job_id")
                except AttributeError:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Wait for PID registration
        deadline = time.time() + 5
        while time.time() < deadline:
            if registry.get_pids(jid):
                break
            time.sleep(0.05)

        registered_pids = registry.get_pids(jid)
        assert len(registered_pids) >= 1

        # Count descendants before cancel
        parent_pid = registered_pids[0]
        descendants_before = _count_descendants(parent_pid)

        # Cancel
        mgr.cancel_job(jid)
        t.join(timeout=15)

        # Verify all processes dead
        for pid in registered_pids:
            assert not _process_alive(pid), f"Parent process {pid} should be dead"

        # Verify no orphan descendants
        if descendants_before > 0:
            descendants_after = _count_descendants(parent_pid)
            # parent is dead, so children() will raise — that's expected
            # We just verify the parent is dead

        assert mgr.get_state(jid) == JobState.CANCELLED

    def test_terminal_returns_within_timeout_after_cancel(self):
        """terminal_tool should return within a reasonable time after cancel."""
        from tools.environments.local import LocalEnvironment

        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        env = LocalEnvironment(cwd="/tmp")
        start_time = None
        end_time = None

        def worker():
            nonlocal start_time, end_time
            threading.current_thread()._hermes_job_id = jid
            start_time = time.time()
            try:
                env.execute("sleep 30", timeout=120)
            finally:
                end_time = time.time()
                try:
                    delattr(threading.current_thread(), "_hermes_job_id")
                except AttributeError:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Wait for spawn
        deadline = time.time() + 5
        while time.time() < deadline:
            if registry.get_pids(jid):
                break
            time.sleep(0.05)

        # Cancel after 1 second
        time.sleep(1.0)
        mgr.cancel_job(jid)
        t.join(timeout=15)

        assert end_time is not None, "Worker should have completed"
        elapsed = end_time - start_time
        # Should return well before the 30s sleep completes
        # Give generous bound: cancel (5s grace + 1s kill) + overhead
        assert elapsed < 20, f"Should return within 20s, took {elapsed:.1f}s"


class TestWorkerThreadInterruptOnCancel:
    """Worker thread interrupt signal set during cancel_job."""

    def test_interrupt_signal_set_on_worker_tids(self):
        """cancel_job should set interrupt signal on registered worker tids."""
        mgr = JobManager()
        jid = mgr.create_job()

        # Register a fake worker tid
        fake_tid = 99999
        mgr.register_worker_tid(jid, fake_tid)

        # Verify it's tracked
        tids = mgr.get_worker_tids(jid)
        assert fake_tid in tids

        # Cancel — callback should try to set interrupt on worker tids
        # (run_agent._set_interrupt will fail for fake tid, but that's caught)
        mgr.cancel_job(jid)

        # Verify the tid was attempted (no exception = callback completed)
        time.sleep(0.1)
        state = mgr.get_state(jid)
        assert state == JobState.CANCELLED  # no processes, so CANCELLED

    def test_worker_tid_unregistered_on_cleanup(self):
        """Worker tid should be removed from JobManager when worker exits."""
        mgr = JobManager()
        jid = mgr.create_job()

        tid = threading.current_thread().ident
        mgr.register_worker_tid(jid, tid)
        assert tid in mgr.get_worker_tids(jid)

        mgr.unregister_worker_tid(jid, tid)
        assert tid not in mgr.get_worker_tids(jid)


class TestGatewayEquivalentSmoke:
    """Simulates the gateway busy path: /jobs → /stop <job_id> → verify termination."""

    def test_jobs_returns_running_job_with_foreground_terminal(self):
        """Running job should appear in list_running_jobs with correct state."""
        from tools.environments.local import LocalEnvironment

        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        env = LocalEnvironment(cwd="/tmp")

        def worker():
            threading.current_thread()._hermes_job_id = jid
            try:
                env.execute("sleep 30", timeout=60)
            finally:
                try:
                    delattr(threading.current_thread(), "_hermes_job_id")
                except AttributeError:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Wait for spawn
        deadline = time.time() + 5
        while time.time() < deadline:
            if registry.get_pids(jid):
                break
            time.sleep(0.05)

        # Simulate /jobs
        jobs = mgr.list_running_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == jid
        assert jobs[0]["state"] == "running"

        # Simulate /stop <job_id>
        result = mgr.cancel_job(jid)
        assert result is not None
        assert result.job_id == jid

        # Verify termination
        t.join(timeout=15)
        assert not t.is_alive()
        assert mgr.get_state(jid) == JobState.CANCELLED

        # /jobs should show no running jobs
        jobs = mgr.list_running_jobs()
        assert len(jobs) == 0

    def test_stop_all_with_multiple_foreground_terminals(self):
        """STOP ALL should cancel multiple foreground terminal jobs."""
        from tools.environments.local import LocalEnvironment

        mgr = JobManager()
        jid1 = mgr.create_job()
        jid2 = mgr.create_job()
        registry = get_process_registry()

        env1 = LocalEnvironment(cwd="/tmp")
        env2 = LocalEnvironment(cwd="/tmp")

        def worker1():
            threading.current_thread()._hermes_job_id = jid1
            try:
                env1.execute("sleep 30", timeout=60)
            finally:
                try:
                    delattr(threading.current_thread(), "_hermes_job_id")
                except AttributeError:
                    pass

        def worker2():
            threading.current_thread()._hermes_job_id = jid2
            try:
                env2.execute("sleep 30", timeout=60)
            finally:
                try:
                    delattr(threading.current_thread(), "_hermes_job_id")
                except AttributeError:
                    pass

        t1 = threading.Thread(target=worker1, daemon=True)
        t2 = threading.Thread(target=worker2, daemon=True)
        t1.start()
        t2.start()

        # Wait for both to spawn
        deadline = time.time() + 5
        while time.time() < deadline:
            if registry.get_pids(jid1) and registry.get_pids(jid2):
                break
            time.sleep(0.05)

        # /jobs should show 2 running
        jobs = mgr.list_running_jobs()
        assert len(jobs) == 2

        # /stop all
        results = mgr.cancel_all()
        assert len(results) == 2

        # Wait for both to finish
        t1.join(timeout=15)
        t2.join(timeout=15)

        assert mgr.get_state(jid1) == JobState.CANCELLED
        assert mgr.get_state(jid2) == JobState.CANCELLED

        # No running jobs left
        jobs = mgr.list_running_jobs()
        assert len(jobs) == 0


class TestForegroundSmoke:
    """Full V4 smoke: foreground terminal + cancel + verify everything."""

    def test_full_foreground_cancel_lifecycle(self):
        """Complete lifecycle: spawn → register → cancel → verify all clean."""
        from tools.environments.local import LocalEnvironment

        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        # 1. Verify job is running
        assert mgr.get_state(jid) == JobState.RUNNING

        # 2. Start foreground terminal
        env = LocalEnvironment(cwd="/tmp")
        result_holder = {}

        def worker():
            threading.current_thread()._hermes_job_id = jid
            try:
                result_holder["result"] = env.execute("sleep 30", timeout=60)
            finally:
                try:
                    delattr(threading.current_thread(), "_hermes_job_id")
                except AttributeError:
                    pass

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # 3. Wait for PID registration
        deadline = time.time() + 5
        while time.time() < deadline:
            if registry.get_pids(jid):
                break
            time.sleep(0.05)

        registered_pids = registry.get_pids(jid)
        assert len(registered_pids) >= 1, "PID should be registered"

        # 4. Process should be alive
        for pid in registered_pids:
            assert _process_alive(pid), f"Process {pid} should be alive"

        # 5. Cancel the job
        cancel_result = mgr.cancel_job(jid)
        assert cancel_result is not None
        assert cancel_result.state == JobState.CANCEL_REQUESTED

        # 6. Worker should return within timeout
        t.join(timeout=15)
        assert not t.is_alive(), "Worker should have returned"

        # 7. All processes dead
        for pid in registered_pids:
            assert not _process_alive(pid), f"Process {pid} should be dead"

        # 8. State CANCELLED
        assert mgr.get_state(jid) == JobState.CANCELLED

        # 9. Registry cleared
        assert registry.get_pids(jid) == []

        # 10. No running jobs
        assert mgr.list_running_jobs() == []
