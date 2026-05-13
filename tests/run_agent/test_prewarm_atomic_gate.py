"""Regression tests for the atomic prewarm gate in AIAgent.__init__.

Issue #24651: the former Event.is_set() + Event.set() pattern was not atomic.
Two concurrent AIAgent instantiations could both see is_set()==False and each
spawn a fetch_model_metadata thread, defeating the "spawn exactly once" guard.

The fix replaces the Event with a Lock used via acquire(blocking=False).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import run_agent


class TestPrewarmAtomicGate:
    """Verify _openrouter_prewarm_lock spawns the prewarm thread exactly once."""

    def setup_method(self):
        """Reset the module-level lock before each test."""
        self._orig_lock = run_agent._openrouter_prewarm_lock
        run_agent._openrouter_prewarm_lock = threading.Lock()

    def teardown_method(self):
        run_agent._openrouter_prewarm_lock = self._orig_lock

    def test_prewarm_thread_spawned_exactly_once_under_concurrency(self):
        """50 concurrent callers racing the gate must win the lock exactly once."""
        spawn_count = []
        spawn_lock = threading.Lock()
        barrier = threading.Barrier(50)

        def simulate_init():
            barrier.wait()  # maximise contention at the gate
            if run_agent._openrouter_prewarm_lock.acquire(blocking=False):
                with spawn_lock:
                    spawn_count.append(1)

        with ThreadPoolExecutor(max_workers=50) as pool:
            futures = [pool.submit(simulate_init) for _ in range(50)]
            for f in futures:
                f.result()

        assert len(spawn_count) == 1, (
            f"Expected exactly 1 prewarm thread spawn, got {len(spawn_count)}"
        )

    def test_lock_stays_held_after_first_acquisition(self):
        """Lock must remain held after the winning acquire so late callers fail."""
        assert run_agent._openrouter_prewarm_lock.acquire(blocking=False) is True
        # Lock is never released — a second acquire must fail immediately
        assert run_agent._openrouter_prewarm_lock.acquire(blocking=False) is False
