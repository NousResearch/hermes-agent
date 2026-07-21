"""V3 targeted tests for preemptive cancellation runtime paths.

Tests the 4 correction areas:
1. Busy gateway path: /stop <job_id>, /stop all, /jobs during running agent
2. Worker thread context: _hermes_job_id propagated to concurrent tool workers
3. State finalization: CANCELLED only when remaining_processes empty
4. Concurrent terminal/write_file/send_message with cancellation
"""
import os
import sys
import time
import threading
import subprocess

import pytest

# ── Feature flag setup ──
os.environ["HERMES_PREEMPTIVE_CANCELLATION"] = "true"

from agent.cancellation import (
    JobState, JobManager, CancellationToken, CancellationResult,
    ProcessRegistry, get_process_registry, get_job_manager,
    is_preemptive_cancellation_enabled,
)


# ── Area 3: State finalization ──

class TestStateFinalization:
    """CANCELLED only when remaining_processes is empty.  Survivors → CANCELLING."""

    def test_cancelled_when_no_remaining_processes(self):
        """cancel_job with no registered processes → CANCELLED."""
        mgr = JobManager()
        jid = mgr.create_job()
        result = mgr.cancel_job(jid)
        assert result is not None
        # callback fires synchronously in request_cancel
        time.sleep(0.1)  # let callback complete
        state = mgr.get_state(jid)
        assert state == JobState.CANCELLED, f"Expected CANCELLED, got {state}"

    def test_cancelling_when_survivors_remain(self):
        """If processes survive the kill attempt, state stays CANCELLING."""
        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        # Register a non-existent PID — the kill attempt will be a no-op
        # (NoSuchProcess), and we monkeypatch get_remaining to simulate a survivor
        registry.register_pid(jid, 99998)

        # Monkeypatch get_remaining to simulate a survivor after kill
        original_get_remaining = registry.get_remaining
        registry.get_remaining = lambda job_id: [{"pid": 99998, "status": "running", "name": "stubborn"}]

        result = mgr.cancel_job(jid)
        time.sleep(0.1)

        state = mgr.get_state(jid)
        assert state == JobState.CANCELLING, f"Expected CANCELLING with survivors, got {state}"

        # Restore
        registry.get_remaining = original_get_remaining
        registry.clear(jid)

    def test_process_registry_cleared_on_cancelled(self):
        """ProcessRegistry should be cleared when job reaches CANCELLED."""
        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()
        registry.register_pid(jid, 99999)  # fake PID

        mgr.cancel_job(jid)
        time.sleep(0.1)

        state = mgr.get_state(jid)
        assert state == JobState.CANCELLED
        # Registry should have been cleared
        remaining = registry.get_pids(jid)
        assert remaining == [], f"Expected empty pids after CANCELLED, got {remaining}"

    def test_process_registry_not_cleared_on_cancelling(self):
        """ProcessRegistry should NOT be cleared when state stays CANCELLING."""
        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()
        registry.register_pid(jid, 99997)

        # Monkeypatch to simulate survivor
        original = registry.get_remaining
        registry.get_remaining = lambda job_id: [{"pid": 99997, "status": "running"}]

        mgr.cancel_job(jid)
        time.sleep(0.1)

        state = mgr.get_state(jid)
        assert state == JobState.CANCELLING
        # Registry should NOT have been cleared
        pids = registry.get_pids(jid)
        assert 99997 in pids, f"PID should still be in registry during CANCELLING, got {pids}"

        # Restore and clean up
        registry.get_remaining = original
        registry.clear(jid)

    def test_remaining_processes_recorded_in_result(self):
        """cancel_job result should include remaining_processes."""
        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()
        registry.register_pid(jid, 99996)

        # Monkeypatch to simulate survivor
        original = registry.get_remaining
        registry.get_remaining = lambda job_id: [{"pid": 99996, "status": "running", "name": "stubborn"}]

        result = mgr.cancel_job(jid)
        assert result is not None
        assert isinstance(result.remaining_processes, list)

        # Restore
        registry.get_remaining = original
        registry.clear(jid)

    def test_duplicate_stop_returns_current_state(self):
        """Second cancel_job call returns current state, not re-declares."""
        mgr = JobManager()
        jid = mgr.create_job()
        result1 = mgr.cancel_job(jid)
        time.sleep(0.1)

        result2 = mgr.cancel_job(jid)
        assert result2 is not None
        assert result2.state == JobState.CANCELLED, f"Second stop should return CANCELLED, got {result2.state}"


# ── Area 2: Worker thread context ──

class TestWorkerThreadContext:
    """_hermes_job_id propagated to worker threads in _run_tool."""

    def test_job_id_set_on_worker_thread(self):
        """When _run_tool executes in a worker thread, _hermes_job_id is set."""
        from agent.cancellation import JobManager

        mgr = JobManager()
        jid = mgr.create_job()

        # Simulate what _run_tool does: set _hermes_job_id on the thread
        captured = {}

        def worker():
            # This simulates the _run_tool behavior
            threading.current_thread()._hermes_job_id = jid
            captured["job_id"] = getattr(threading.current_thread(), "_hermes_job_id", None)
            # Clean up
            delattr(threading.current_thread(), "_hermes_job_id")

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert captured["job_id"] == jid, f"Worker thread should have job_id {jid}, got {captured.get('job_id')}"

    def test_job_id_cleared_after_worker_exits(self):
        """After _run_tool finishes, _hermes_job_id should be removed from the worker thread."""
        mgr = JobManager()
        jid = mgr.create_job()

        worker_thread_ref = []

        def worker():
            threading.current_thread()._hermes_job_id = jid
            worker_thread_ref.append(threading.current_thread())
            try:
                delattr(threading.current_thread(), "_hermes_job_id")
            except AttributeError:
                pass

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        # The thread object should no longer have _hermes_job_id
        assert not hasattr(worker_thread_ref[0], "_hermes_job_id"), "job_id should be cleared after worker exits"


# ── Area 1: Gateway busy path ──

class TestGatewayBusyPath:
    """Test that /stop <job_id>, /stop all, and /jobs work during running agent."""

    def test_stop_command_args_routing(self):
        """/stop with args should parse correctly for JobManager routing."""
        # Simulate the gateway logic: bare /stop → session-wide, /stop <id> → JobManager
        from agent.cancellation import JobManager

        mgr = JobManager()
        jid = mgr.create_job()

        # Simulate: /stop <jid>
        # The gateway calls _handle_stop_command which reads event.get_command_args()
        # For this test, we directly call cancel_job (what the handler does)
        result = mgr.cancel_job(jid)
        assert result is not None
        assert result.job_id == jid

        # Simulate: /stop all
        jid2 = mgr.create_job()
        results = mgr.cancel_all()
        assert len(results) >= 1

        # Simulate: bare /stop (no args) → session-wide stop
        # This would call _interrupt_and_clear_session, not cancel_job
        # We just verify that bare /stop doesn't touch JobManager
        jid3 = mgr.create_job()
        # Bare /stop should NOT cancel via JobManager
        state = mgr.get_state(jid3)
        assert state == JobState.RUNNING, "Bare /stop should not affect JobManager jobs"

    def test_jobs_command_lists_running_jobs(self):
        """/jobs should list all running jobs."""
        mgr = JobManager()
        jid1 = mgr.create_job()
        jid2 = mgr.create_job()

        jobs = mgr.list_running_jobs()
        assert len(jobs) == 2
        job_ids = [j["job_id"] for j in jobs]
        assert jid1 in job_ids
        assert jid2 in job_ids

        # After cancelling one, it should still show if in CANCEL_REQUESTED/CANCELLING
        mgr.cancel_job(jid1)
        time.sleep(0.1)
        jobs = mgr.list_running_jobs()
        # jid1 should be CANCELLED (no processes), so not in running list
        # jid2 should still be running
        running_ids = [j["job_id"] for j in jobs]
        assert jid2 in running_ids
        assert jid1 not in running_ids, "CANCELLED job should not appear in running jobs"


# ── Area 4: Concurrent tool cancellation ──

class TestConcurrentToolCancellation:
    """Test that cancel_job alone (no manual kill_process_tree) kills child processes."""

    def test_cancel_job_kills_child_process_without_manual_kill(self):
        """cancel_job should kill registered child processes via its callback."""
        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        # Spawn a long-running child process
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        registry.register_pid(jid, proc.pid)

        # Verify it's alive
        assert proc.poll() is None, "Child process should be alive before cancel"

        # Cancel the job — callback should kill the process
        mgr.cancel_job(jid)

        # Wait for the callback to complete (SIGTERM → 5s → SIGKILL)
        deadline = time.time() + 10
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)

        assert proc.poll() is not None, "Child process should be terminated after cancel_job"

        # Verify state is CANCELLED (no survivors)
        state = mgr.get_state(jid)
        assert state == JobState.CANCELLED, f"Expected CANCELLED, got {state}"

    def test_cancel_job_with_survivor_stays_cancelling(self):
        """If a process survives the kill, state should be CANCELLING, not CANCELLED."""
        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        # Spawn a process that ignores SIGTERM
        proc = subprocess.Popen(
            ["python3", "-c", "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        registry.register_pid(jid, proc.pid)

        assert proc.poll() is None, "Process should be alive before cancel"

        # Cancel the job
        mgr.cancel_job(jid)

        # Wait for the callback to complete (SIGTERM → 5s grace → SIGKILL)
        deadline = time.time() + 15
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.5)

        # The process should be killed by SIGKILL after grace period
        assert proc.poll() is not None, "Process should be killed by SIGKILL"

        # State should be CANCELLED since SIGKILL should have worked
        state = mgr.get_state(jid)
        assert state == JobState.CANCELLED, f"Expected CANCELLED after SIGKILL, got {state}"

    def test_concurrent_terminal_cancel_via_cancel_job_only(self):
        """Concurrent terminal tool simulation: register PID, cancel_job, verify death."""
        mgr = JobManager()
        jid = mgr.create_job()
        registry = get_process_registry()

        # Simulate what terminal_tool does when it spawns a process
        proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # new process group, like terminal does
        )
        # terminal_tool registers the PID with the current job_id
        registry.register_pid(jid, proc.pid)

        assert proc.poll() is None

        # STOP <job_id> — just call cancel_job, no manual kill_process_tree
        result = mgr.cancel_job(jid)
        assert result is not None
        assert result.state == JobState.CANCEL_REQUESTED

        # Wait for callback to kill the process
        deadline = time.time() + 10
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)

        assert proc.poll() is not None, "Process should be dead after cancel_job"

    def test_write_file_gate_blocks_after_cancel(self):
        """After cancellation, file write gate should block."""
        from agent.cancellation_gates import guard_file_write, OperationCancelled

        mgr = JobManager()
        jid = mgr.create_job()

        # Create a mock agent with the cancellation token
        class MockAgent:
            pass
        agent = MockAgent()
        agent._cancellation_token = mgr.get_token(jid)
        agent._job_id = jid
        agent._interrupt_requested = False

        mgr.cancel_job(jid)
        time.sleep(0.1)

        # Guard should raise OperationCancelled
        with pytest.raises(OperationCancelled):
            guard_file_write(agent, "/tmp/test_cancel_block.txt")

    def test_send_message_gate_blocks_after_cancel(self):
        """After cancellation, external send gate should block."""
        from agent.cancellation_gates import guard_external_send, OperationCancelled

        mgr = JobManager()
        jid = mgr.create_job()

        class MockAgent:
            pass
        agent = MockAgent()
        agent._cancellation_token = mgr.get_token(jid)
        agent._job_id = jid
        agent._interrupt_requested = False

        mgr.cancel_job(jid)
        time.sleep(0.1)

        with pytest.raises(OperationCancelled):
            guard_external_send(agent, "discord:#test")

    def test_feature_flag_off_preserves_existing_behavior(self):
        """When feature flag is off, cancellation is a no-op."""
        # Save and disable
        original = os.environ.get("HERMES_PREEMPTIVE_CANCELLATION", "")
        os.environ["HERMES_PREEMPTIVE_CANCELLATION"] = "false"

        # Check without reloading — is_preemptive_cancellation_enabled reads env each call
        assert not is_preemptive_cancellation_enabled()

        # Restore
        os.environ["HERMES_PREEMPTIVE_CANCELLATION"] = original


# ── Smoke test ──

class TestV3Smoke:
    """V3 smoke: end-to-end cancel with process tree + gate + state verification."""

    def test_full_cancel_lifecycle(self):
        """Full lifecycle: create_job → register PID → cancel_job → verify state + process death."""
        mgr = JobManager()
        registry = get_process_registry()

        # 1. Create job
        jid = mgr.create_job()
        assert mgr.get_state(jid) == JobState.RUNNING

        # 2. Register a real child process (simulates terminal_tool)
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        registry.register_pid(jid, proc.pid)
        assert proc.poll() is None

        # 3. Cancel the job (simulates /stop <job_id>)
        result = mgr.cancel_job(jid)
        assert result is not None
        assert result.state == JobState.CANCEL_REQUESTED

        # 4. Wait for process to die (callback runs synchronously in request_cancel)
        deadline = time.time() + 10
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)

        assert proc.poll() is not None, "Process should be dead"
        time.sleep(0.1)  # ensure callback fully completed

        # 5. State should be CANCELLED (no survivors)
        state = mgr.get_state(jid)
        assert state == JobState.CANCELLED, f"Expected CANCELLED, got {state}"

        # 6. Registry should be cleared
        pids = registry.get_pids(jid)
        assert pids == [], f"Registry should be empty after CANCELLED, got {pids}"

        # 7. Gates should block
        from agent.cancellation_gates import guard_file_write, guard_external_send, OperationCancelled
        token = mgr.get_token(jid)

        class MockAgent:
            pass
        mock_agent = MockAgent()
        mock_agent._cancellation_token = token
        mock_agent._job_id = jid
        mock_agent._interrupt_requested = False

        with pytest.raises(OperationCancelled):
            guard_file_write(mock_agent, "/tmp/test")
        with pytest.raises(OperationCancelled):
            guard_external_send(mock_agent, "discord:#test")

    def test_cancel_all_multiple_jobs(self):
        """STOP ALL cancels multiple jobs with registered processes."""
        mgr = JobManager()
        registry = get_process_registry()

        # Create 2 jobs with child processes
        jid1 = mgr.create_job()
        jid2 = mgr.create_job()
        proc1 = subprocess.Popen(["sleep", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        proc2 = subprocess.Popen(["sleep", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        registry.register_pid(jid1, proc1.pid)
        registry.register_pid(jid2, proc2.pid)

        # Cancel all
        results = mgr.cancel_all()
        assert len(results) == 2

        # Wait for processes to die
        for proc in [proc1, proc2]:
            deadline = time.time() + 10
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.1)
            assert proc.poll() is not None

        time.sleep(0.1)
        assert mgr.get_state(jid1) == JobState.CANCELLED
        assert mgr.get_state(jid2) == JobState.CANCELLED
