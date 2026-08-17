"""RED-first regression test for the cron in-flight claim leak (t_27b59583).

The leak
--------
``cron/scheduler.py`` tracks in-flight cron jobs in the module-level
``_running_job_ids`` set. ``_submit_with_guard`` adds a job id BEFORE the
future that owns its release exists: everything between the add and
``pool.submit`` — ``create_execution``, ``contextvars.copy_context()``, and
once running, the whole pre-future body of ``run_one_job`` (SessionDB
construction around L3150-3161, agent import/build, config load) — has no
``finally`` that discards the id. If any of it throws or hangs, the release
path in ``_run_and_release``'s ``finally`` never runs. Every later tick then
short-circuits with ``cron.scheduler: Job '<x>' already running — skipping``
with no ``last_error``, no failure counter, and no alert, until the whole
gateway process restarts (incident: jarvis ``board-pm-triage-*`` jobs,
2026-08-02).

This file is committed BEFORE the fix (red-first). Against the unfixed
scheduler these tests MUST FAIL: the stale id is never released, so the
primary assertion (``job_id not in get_running_job_ids()`` after a tick)
fails. The implementation task (t_3778a491) makes them pass by adding the
bounded stale-entry guard: on each tick, a claim older than
``max(2 * interval, floor)`` with no live future is force-released, logged
with a countable ``cron.inflight.forced_release`` signal, and surfaced via
``mark_job_run(..., success=False, error=...)`` as ``last_error``.

Design notes
------------
- The job store persists ``schedule`` as an already-parsed DICT
  (``{"kind": "interval", "minutes": N}``), not the string form
  ``parse_schedule`` consumes — the fixtures use the persisted dict shape.
- The guard's age bookkeeping (``_running_since`` / ``_running_futures`` /
  ``get_inflight_guard_stats``) does not exist yet on the unfixed scheduler.
  The helpers reference it defensively (``getattr``/``hasattr``) so the SAME
  file runs cleanly against both the red (unfixed) and the green (fixed)
  implementation; the leak simulation is identical either way — an id in
  ``_running_job_ids`` with no future ever installed.
"""

import time
from unittest.mock import patch

import pytest

import cron.scheduler as sched


@pytest.fixture(autouse=True)
def _clean_inflight():
    """Reset the in-memory running set so tests are isolated.

    Clears the guard bookkeeping defensively: on the unfixed scheduler only
    ``_running_job_ids`` exists; the age/future dicts and counters appear
    with the fix, and clearing them keeps the same file hermetic on both.
    """
    sched._running_job_ids.clear()
    for attr in ("_running_since", "_running_futures", "_forced_releases"):
        obj = getattr(sched, attr, None)
        if obj is not None:
            obj.clear()
    if hasattr(sched, "_forced_release_count"):
        sched._forced_release_count = 0
    yield
    sched._running_job_ids.clear()
    for attr in ("_running_since", "_running_futures", "_forced_releases"):
        obj = getattr(sched, attr, None)
        if obj is not None:
            obj.clear()


def _job(job_id="wedged", minutes=60, kind="interval", cron_expr=None,
         repeat=None):
    """Build a job row using the PERSISTED schedule dict shape."""
    if kind == "interval":
        schedule = {
            "kind": "interval",
            "minutes": minutes,
            "display": f"every {minutes}m",
        }
    elif kind == "cron":
        expr = cron_expr or "0 9 * * 1"
        schedule = {"kind": "cron", "expr": expr, "display": expr}
    else:
        schedule = {
            "kind": "once",
            "run_at": "2030-01-01T00:00:00",
            "display": "once at 2030-01-01 00:00",
        }
    job = {
        "id": job_id,
        "name": f"board-pm-triage-{job_id}",
        "schedule": schedule,
    }
    if repeat is not None:
        job["repeat"] = repeat
    return job


def _inject_stale_claim(job_id: str) -> None:
    """Simulate the leak exactly as the incident left it: the job id is in
    the running set but no future was ever installed, so the release path in
    the worker's ``finally`` can never run.

    On the unfixed scheduler ``_running_job_ids`` is the only bookkeeping, so
    this is precisely the shape of the real wedge. Once the bounded guard
    lands it also records an old start time — 6h ago, far past
    ``max(2 * 60m interval, 30m floor)`` — so the claim is past its
    allowance on the first sweep.
    """
    sched._running_job_ids.add(job_id)
    running_since = getattr(sched, "_running_since", None)
    if running_since is not None:
        running_since[job_id] = time.time() - 6 * 60 * 60  # 6h old


class TestStaleInflightLeak:
    def test_stale_claim_is_force_released_and_reported_after_tick(
        self, tmp_path, caplog
    ):
        """The regression: a leaked in-flight claim must be force-released by
        the next tick, surface as ``last_error``, and emit a countable
        signal — instead of silently skipping every fire until the gateway
        process restarts."""
        job = _job(job_id="board-pm-triage-wedged", minutes=60)
        job_id = job["id"]
        _inject_stale_claim(job_id)

        with caplog.at_level("WARNING"), \
             patch.object(sched, "_get_hermes_home", return_value=tmp_path), \
             patch("cron.jobs.load_jobs", return_value=[job]), \
             patch.object(sched, "get_due_jobs", return_value=[]), \
             patch.object(sched, "mark_job_run") as mark:
            sched.tick(verbose=False)

        # RED: on the unfixed scheduler the tick never releases the id — it
        # short-circuits with "already running — skipping" and the id stays
        # in the set forever, so this assertion FAILS and proves the leak.
        assert job_id not in sched.get_running_job_ids()

        # GREEN (after the bounded guard): the release surfaces as a failure
        # on the job row instead of silence…
        assert mark.call_count == 1
        args = mark.call_args.args
        assert args[0] == job_id
        assert args[1] is False
        assert "in-flight" in args[2]

        # …and emits the countable forced-release signal (log + probe stats).
        assert any(
            "cron.inflight.forced_release" in r.message for r in caplog.records
        )
        stats = sched.get_inflight_guard_stats()
        assert stats["forced_releases"] == 1

    def test_young_inflight_claim_is_not_force_released(self, tmp_path):
        """Bound the guard: a claim younger than its allowance (and with no
        future) is left alone — the sweep must not double-dispatch healthy
        long-running jobs. Passes on both the red and the fixed code."""
        job = _job(job_id="young", minutes=60)
        job_id = job["id"]
        sched._running_job_ids.add(job_id)
        running_since = getattr(sched, "_running_since", None)
        if running_since is not None:
            running_since[job_id] = time.time() - 60  # 1 minute old

        with patch.object(sched, "_get_hermes_home", return_value=tmp_path), \
             patch("cron.jobs.load_jobs", return_value=[job]), \
             patch.object(sched, "get_due_jobs", return_value=[]), \
             patch.object(sched, "mark_job_run") as mark:
            sched.tick(verbose=False)

        assert job_id in sched.get_running_job_ids()
        mark.assert_not_called()


class TestJobIntervalMinutes:
    """Allowance inputs come from the PERSISTED dict schedule shape, not the
    string form parse_schedule consumes (review Blocker 1)."""

    def test_reads_persisted_interval_dict(self):
        job = _job(minutes=4320)
        assert sched._job_interval_minutes(job) == 4320.0

    def test_reads_persisted_cron_dict(self):
        # */15 every 15 minutes → cadence 15m.
        job = _job(kind="cron", cron_expr="*/15 * * * *")
        assert sched._job_interval_minutes(job) == 15.0

    def test_reads_persisted_weekly_cron_dict(self):
        # 0 9 * * 1 fires weekly → cadence 7*24*60 = 10080m.
        job = _job(kind="cron", cron_expr="0 9 * * 1")
        assert sched._job_interval_minutes(job) == 7 * 24 * 60

    def test_string_fallback_still_works(self):
        # Defensive fallback for programmatic callers.
        job = {"id": "x", "schedule": "every 60m"}
        assert sched._job_interval_minutes(job) == 60.0

    def test_oneshot_has_no_interval(self):
        job = _job(kind="once")
        assert sched._job_interval_minutes(job) is None

    def test_garbage_returns_none(self):
        job = {"id": "x", "schedule": {"kind": "bogus"}}
        assert sched._job_interval_minutes(job) is None


class TestStaleInflightSweep:
    """Unit-level bound checks on sweep_stale_inflight itself."""

    def test_allowance_is_at_least_two_intervals(self, tmp_path):
        """A slow-but-healthy 6h job is not clipped by the 30m floor."""
        job = _job(minutes=360)
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 4 * 60 * 60  # 4h < 12h

        with patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == []
        assert job["id"] in sched.get_running_job_ids()

    def test_allowance_honors_persisted_4320m_row(self, tmp_path):
        """The real guide-curator row (4320m) gets a 144h allowance, not the
        30m floor — the Blocker-1 regression against the live store shape."""
        job = _job(job_id="guide-curator", minutes=4320)
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 4 * 60 * 60  # 4h ≪ 144h

        with patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == []
        assert job["id"] in sched.get_running_job_ids()

    def test_cron_allowance_not_clipped_to_floor(self, tmp_path):
        """A weekly cron job (cadence 10080m) is not clipped at 30m: a 24h
        claim is still healthy (allowance 20160m)."""
        job = _job(job_id="weekly", kind="cron", cron_expr="0 9 * * 1")
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 24 * 60 * 60

        with patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == []
        assert job["id"] in sched.get_running_job_ids()

    def test_live_future_is_never_released(self, tmp_path):
        """A claim with a genuinely executing future is left alone even when
        old — the sweep must not double-dispatch healthy long-running jobs."""
        import concurrent.futures

        job = _job()
        fut: concurrent.futures.Future = concurrent.futures.Future()
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 10 * 60 * 60
        sched._running_futures[job["id"]] = fut

        with patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == []
        assert job["id"] in sched.get_running_job_ids()
        fut.set_result(True)

        # Once the future is done but the id somehow survived, it IS stale.
        with patch.object(sched, "mark_job_run"), \
             patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == [job["id"]]

    def test_pending_sentinel_released_when_submit_hung(self, tmp_path):
        """A claim whose submit path hung stays _FUTURE_PENDING past its
        allowance (the SessionDB-init wedge class) and must be released."""
        job = _job()
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 5 * 60 * 60
        sched._running_futures[job["id"]] = sched._FUTURE_PENDING

        with patch.object(sched, "mark_job_run") as mark, \
             patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == [job["id"]]
        assert mark.call_count == 1

    def test_pending_sentinel_young_claim_is_not_released(self, tmp_path):
        """A young pending claim (submit still in flight) is safe."""
        job = _job()
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 60  # 1 minute
        sched._running_futures[job["id"]] = sched._FUTURE_PENDING

        with patch.object(sched, "mark_job_run") as mark, \
             patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == []
        assert job["id"] in sched.get_running_job_ids()
        mark.assert_not_called()

    def test_finite_repeat_job_released_without_mark_job_run(self, tmp_path):
        """A forced release must not consume a finite repeat budget or
        auto-delete the row; the claim is released, the row untouched."""
        job = _job(repeat={"times": 1, "completed": 0})
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 5 * 60 * 60

        with patch.object(sched, "mark_job_run") as mark, \
             patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == [job["id"]]
        assert job["id"] not in sched.get_running_job_ids()
        mark.assert_not_called()
        assert sched.get_inflight_guard_stats()["forced_releases"] == 1

    def test_claim_without_timestamp_is_adopted_then_swept(self, tmp_path):
        """An id injected with no recorded start (pre-guard claim) must not be
        released immediately, but must become sweepable."""
        job = _job()
        sched._running_job_ids.add(job["id"])

        with patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            assert sched.sweep_stale_inflight([job]) == []
            assert job["id"] in sched._running_since
            sched._running_since[job["id"]] -= 5 * 60 * 60
            with patch.object(sched, "mark_job_run"):
                assert sched.sweep_stale_inflight([job]) == [job["id"]]

    def test_forced_release_logs_a_warning(self, tmp_path, caplog):
        job = _job()
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 5 * 60 * 60

        with caplog.at_level("WARNING"), \
             patch.object(sched, "mark_job_run"), \
             patch.object(sched, "_get_hermes_home", return_value=tmp_path):
            sched.sweep_stale_inflight([job])

        assert any("cron.inflight.forced_release" in r.message for r in caplog.records)


class TestWedgedJobRefiresWithoutRestart:
    def test_tick_sweeps_then_dispatches_the_previously_wedged_job(self, tmp_path):
        """End-to-end symptom: before the fix, tick() returned 0 forever."""
        job = dict(_job(), enabled=True, next_run_at="2020-01-01T00:00:00",
                   deliver="local")
        sched._running_job_ids.add(job["id"])
        sched._running_since[job["id"]] = time.time() - 6 * 60 * 60

        with patch.object(sched, "_get_hermes_home", return_value=tmp_path), \
             patch.object(sched, "get_due_jobs", return_value=[job]), \
             patch("cron.jobs.load_jobs", return_value=[job]), \
             patch.object(sched, "advance_next_runs"), \
             patch.object(sched, "mark_job_run"), \
             patch.object(sched, "create_execution", return_value={"id": "exec-1"}), \
             patch.object(sched, "finish_execution"), \
             patch.object(sched, "run_one_job", return_value=True):
            n = sched.tick(verbose=False)

        assert n == 1, "wedged job must fire again without a gateway restart"
        assert job["id"] not in sched.get_running_job_ids()
        assert sched.get_inflight_guard_stats()["forced_releases"] == 1
