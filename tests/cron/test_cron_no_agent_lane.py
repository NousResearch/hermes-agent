"""A job whose mode needs no model must never queue behind one that may hold a worker for hours.

Card t_ee19b72d. Measured 2026-09-27 on the farmer-bob profile (``cron.max_parallel_jobs: 1``):
the profile's weekly local-model Vault lint ran 9 h 40 m, and because EVERY due job of the
profile was submitted to ONE per-home FIFO pool, the profile's own ``no_agent`` script jobs
(``vault-commit.sh`` 0.3 s, ``daily-brief.sh``) sat ``claimed`` with ``started_at = NULL`` for
the same 9 h 40 m -- indistinguishable from a hang to the operator reading the board.

The fix is one pool per lane, keyed by home as before: the agent lane, and a no-agent lane whose
jobs are exactly those ``run_job`` short-circuits on (``job.get("no_agent")``) before any model is
resolved. Each lane keeps the profile's ``max_parallel_jobs``, so the config key keeps its
meaning; in-flight de-duplication is untouched because ``_inflight_key`` is per (home, job id).

Every test here fails on the pre-change bytes: the lane argument, the second executor, and the
script job's thread name do not exist there, and the script job waits for the agent job.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

import cron.scheduler as sched
from cron import scheduler_tick
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

AGENT_JOB = {
    "id": "ee19-agent",
    "name": "agent job",
    "no_agent": False,
    "schedule": {"kind": "recurring"},
}
SCRIPT_JOB = {
    "id": "ee19-script",
    "name": "script job",
    "no_agent": True,
    "script": "echo hi",
    "schedule": {"kind": "recurring"},
}

AGENT_PREFIX = "cron-parallel"
NO_AGENT_PREFIX = "cron-parallel-no-agent"


class _HomeScope:
    """Bind the cron home the way a per-profile tick does."""

    def __init__(self, home: Path):
        self._home = str(home)

    def __enter__(self):
        self._token = set_hermes_home_override(self._home)
        return self

    def __exit__(self, *_exc):
        reset_hermes_home_override(self._token)
        return False


class _Patched:
    """Temporarily replace attributes on cron.scheduler (the module the tick dispatches through)."""

    def __init__(self, **attrs):
        self._attrs = attrs
        self._saved = {}

    def __enter__(self):
        for name, value in self._attrs.items():
            self._saved[name] = getattr(sched, name)
            setattr(sched, name, value)
        return self

    def __exit__(self, *_exc):
        for name, value in self._saved.items():
            setattr(sched, name, value)
        return False


class _PoolTestCase(unittest.TestCase):
    def setUp(self):
        sched._shutdown_parallel_pool()
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        self.scope = _HomeScope(self.home)
        self.scope.__enter__()

    def tearDown(self):
        try:
            sched._shutdown_parallel_pool()
        finally:
            self.scope.__exit__(None, None, None)
            self._tmp.cleanup()


class LanePoolTests(_PoolTestCase):
    def test_lanes_are_separate_pools_sized_by_the_same_key(self):
        agent = sched._get_parallel_pool(1)
        script = sched._get_parallel_pool(1, lane=sched.PARALLEL_LANE_NO_AGENT)
        self.assertIsNot(agent, script, "one pool for both job modes")
        self.assertEqual(agent._max_workers, 1)
        self.assertEqual(script._max_workers, 1, "the config key must size each lane")
        self.assertIs(sched._get_parallel_pool(1), agent, "lane identity is not stable")
        self.assertIs(
            sched._get_parallel_pool(1, lane=sched.PARALLEL_LANE_NO_AGENT),
            script,
            "lane identity is not stable",
        )

    def test_script_lane_starts_while_the_agent_lane_is_busy(self):
        """The reported defect: a 0.3 s script job waited 9 h 40 m behind an agent job."""
        agent = sched._get_parallel_pool(1)
        script = sched._get_parallel_pool(1, lane=sched.PARALLEL_LANE_NO_AGENT)
        started: list[float] = []

        t0 = time.monotonic()
        long_future = agent.submit(time.sleep, 1.5)
        time.sleep(0.2)
        short_future = script.submit(started.append, time.monotonic())
        short_future.result(timeout=5)
        waited = started[0] - t0
        self.assertLess(
            waited,
            0.5,
            f"a no_agent job waited {waited:.2f}s behind an agent job -- lanes are not separate",
        )
        long_future.result(timeout=10)

    def test_fifo_within_the_agent_lane_is_unchanged(self):
        agent = sched._get_parallel_pool(1)
        order: list[str] = []

        def _hold():
            time.sleep(0.4)
            order.append("first")

        long_future = agent.submit(_hold)
        time.sleep(0.05)
        second_future = agent.submit(order.append, "second")
        second_future.result(timeout=5)
        long_future.result(timeout=5)
        self.assertEqual(order, ["first", "second"], "one lane must stay a FIFO of its own")

    def test_discard_drops_every_lane_of_a_home(self):
        sched._get_parallel_pool(1)
        sched._get_parallel_pool(1, lane=sched.PARALLEL_LANE_NO_AGENT)
        key = sched.hermes_home_key(self.home)
        self.assertIn(key, sched._parallel_pools)
        self.assertEqual(len(sched._parallel_pools[key]), 2)
        sched.discard_parallel_pools({key})
        self.assertNotIn(key, sched._parallel_pools)
        self.assertNotIn(key, sched._parallel_pool_max_workers)

    def test_shutdown_closes_every_lane(self):
        agent = sched._get_parallel_pool(1)
        script = sched._get_parallel_pool(1, lane=sched.PARALLEL_LANE_NO_AGENT)
        sched._shutdown_parallel_pool()
        for pool in (agent, script):
            self.assertTrue(getattr(pool, "_shutdown", False), "a lane executor was left alive")
        self.assertEqual(sched._parallel_pools, {})


class TickLaneRoutingTests(_PoolTestCase):
    """The tick itself must route a no_agent job to the no-agent lane (real ``_submit_with_guard``)."""

    def _run_tick(self, jobs):
        seen: dict = {}
        state: dict = {}
        lock_dir = self.home / "cron"

        def _process_due_job(job, adapters, loop, verbose):
            seen[job["id"]] = threading.current_thread().name
            if job["id"] == AGENT_JOB["id"]:
                state["agent_started"] = time.monotonic()
                time.sleep(1.5)
            else:
                state["script_delay"] = time.monotonic() - state["t0"]
            return True

        with _Patched(
            _should_yield_tick_to_fresh_gateway=lambda: None,
            _get_lock_paths=lambda: (lock_dir, lock_dir / "tick.lock"),
            _maybe_reap_dead_owners=lambda: None,
            _maybe_run_worktree_maintenance=lambda: None,
            get_due_jobs=lambda: jobs,
            _sweep_stale_inflight_for_tick=lambda due: None,
            advance_next_runs=lambda ids: None,
            _resolve_max_parallel_workers=lambda: 1,
            _process_due_job=_process_due_job,
            _sweep_mcp_orphans=lambda: None,
            _sweep_mcp_orphans_when_all_done=lambda futures: None,
        ):
            state["t0"] = time.monotonic()
            ran = scheduler_tick.tick(verbose=False, sync=True)
        self.assertGreaterEqual(ran, 1, "the tick dispatched nothing")
        return seen, state

    def test_no_agent_job_is_dispatched_on_the_no_agent_lane(self):
        seen, state = self._run_tick([AGENT_JOB, SCRIPT_JOB])
        self.assertTrue(
            seen[AGENT_JOB["id"]].startswith(AGENT_PREFIX),
            f"agent job ran on {seen[AGENT_JOB['id']]}",
        )
        self.assertTrue(
            seen[SCRIPT_JOB["id"]].startswith(NO_AGENT_PREFIX),
            f"no_agent job ran on {seen[SCRIPT_JOB['id']]} -- it shared the agent lane",
        )
        self.assertLess(
            state["script_delay"],
            0.5,
            f"the no_agent job queued {state['script_delay']:.2f}s behind the agent job",
        )

    def test_a_tick_with_no_script_jobs_starts_no_second_executor(self):
        seen, _state = self._run_tick([AGENT_JOB])
        self.assertTrue(seen[AGENT_JOB["id"]].startswith(AGENT_PREFIX))
        lanes = sched._parallel_pools[sched.hermes_home_key(self.home)]
        self.assertEqual(sorted(lanes), [sched.PARALLEL_LANE_AGENT])


if __name__ == "__main__":
    unittest.main(verbosity=2)
