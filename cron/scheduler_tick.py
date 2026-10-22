"""In-process tick admission and dispatch; job execution stays in scheduler.py."""

import concurrent.futures
import contextlib
import logging
import sqlite3
import time

from cron import store_health
from cron.constants import is_recurring

logger = logging.getLogger("cron.scheduler")

# Log the yield at most once per episode (reset when the skew changes) to avoid per-interval spam.
_YIELD_LOG_INTERVAL_SECONDS = 3600.0
_last_yield_log: dict[str, object] = {}


def _log_tick_yield_once(reason: str) -> None:
    """Log the yield at error level once per episode (skew signature)."""
    global _last_yield_log
    now = time.monotonic()
    last_reason = _last_yield_log.get("reason")
    last_at = _last_yield_log.get("at", 0.0)
    if last_reason != reason or (now - float(last_at)) >= _YIELD_LOG_INTERVAL_SECONDS:
        logger.error(
            "Cron tick yielded: this process is running stale code (%s) and a "
            "fresher gateway owns the runtime lock — jobs will fire from that "
            "process. Restart this one to reclaim its ticks.",
            reason)
    _last_yield_log = {"reason": reason, "at": now}


def tick(verbose=True, adapters=None, loop=None, sync=True, *, can_dispatch=None,
         headless: bool = False):
    from hermes_cli.backend_retirement import retirement

    # Hold admission through the entire scan/advance/submit handoff. A predicate alone races
    # prepare after the check but before a due job enters the running-job ledger.
    with retirement.work() as admitted:
        if not admitted:
            return 0
        return _tick_admitted(verbose, adapters, loop, sync, can_dispatch=can_dispatch,
                              headless=headless)


def _acquire_tick_lock_or_degrade(_sched):
    """The tick lock fd, or None when this tick must not run: another ticker holds the lock, or
    the store cannot be written (read-only/full/denied: the lock file cannot even be created), in
    which case the store is marked degraded and the MCP orphan sweep still runs."""
    lock_dir, lock_file = _sched._get_lock_paths()
    _sched._ensure_cron_dir(lock_dir)
    try:
        return _sched._acquire_tick_lock(lock_file)
    except OSError as exc:
        if exc.errno not in store_health.UNWRITABLE_ERRNOS:
            raise  # EMFILE/ENFILE etc. stay a FAILED tick (#87644)
        store_health.note_unwritable(exc, "tick lock unavailable", "lock", cron_dir=lock_dir)
        _sched._sweep_mcp_orphans()
        return None


def _advance_or_drop_recurring(_sched, due_jobs: list) -> list:
    """Persist the recurring jobs' advance before any runs; when the store refuses it, drop them
    (no durable advance -> a crash mid-run would re-fire them, so skipping is the at-most-once
    side). One-shots still go through their own fire claim."""
    try:
        _sched.advance_next_runs([job["id"] for job in due_jobs])
        return due_jobs
    except OSError as exc:
        skipped = [j for j in due_jobs if is_recurring(j)]
        store_health.note_unwritable(exc, f"skipped {len(skipped)} recurring job(s)", "advance", skipped)
        return [j for j in due_jobs if not is_recurring(j)]


def _pre_dispatch_housekeeping(_sched, headless: bool) -> None:
    """Admission reconcile, legacy upgrade drain and dead-owner reap, run under the tick lock
    before due jobs are read."""
    from cron.scheduler_authority import reconcile_pending
    if headless:
        reconcile_pending(allow_connect=False)
    else:
        reconcile_pending()
    # Upgrade drain for the retired CLI lane's records: never let it stop due jobs below.
    try:
        from cron.bot_chat_legacy import drain_legacy_pending
        drain_legacy_pending()
    except Exception as _legacy_exc:
        _sched.logger.warning("Legacy Bot Chat pending drain failed: %s", _legacy_exc, exc_info=True)
    _sched._maybe_reap_dead_owners()


def _tick_admitted(
    verbose: bool = True, adapters=None, loop=None, sync: bool = True, *, can_dispatch=None,
    headless: bool = False):
    """Check and run all due jobs. File-locked so only one tick scans and dispatches at a time
    (gateway ticker vs standalone daemon / manual tick); the lock is released once the due jobs
    are in the pool, also when ``sync`` then waits for them. ``can_dispatch``: optional gate; false
    leaves due jobs for the next allowed tick. ``headless``: the tick runs outside any gateway (system crontab, ``hermes
    cron tick``) and must refuse agent jobs rather than spawn one. Returns the number of jobs
    executed (0 if another tick holds the lock)."""
    from cron import scheduler as _sched

    # Stale-code yield gate — BEFORE the lock race. A process whose checkout was updated under it
    # serves mixed sys.modules (jobs die on ImportErrors); if a fresher gateway holds the runtime
    # lock, ITS ticker dispatches. With no fresh holder (desktop-standalone) the tick proceeds.
    _skew = _sched._should_yield_tick_to_fresh_gateway()
    if _skew is not None:
        _log_tick_yield_once(f"boot={_skew[0]} disk={_skew[1]}")
        raise _sched.CronTickYielded(_skew[0], _skew[1])

    lock_fd = _acquire_tick_lock_or_degrade(_sched)
    if lock_fd is None:
        return 0

    _results: list = []
    _pending_futures: list = []
    try:
        # `hermes pause` ESTOP: skip dispatch, never touch in-flight runs; check_paused logs once.
        with contextlib.suppress(ImportError):
            from agent.estop import check_paused as _estop_check_paused
            if _estop_check_paused("cron", _sched.logger):
                return 0

        if can_dispatch is not None and not can_dispatch():
            _sched.logger.debug("Cron dispatch paused while gateway drains existing work")
            return 0

        _pre_dispatch_housekeeping(_sched, headless)
        # Periodic worktree GC (6h, threaded) — the only sweep gateway-only boxes get.
        try:
            _sched._maybe_run_worktree_maintenance()
        except Exception as _wt_exc:
            _sched.logger.debug("Worktree maintenance dispatch failed: %s", _wt_exc)

        due_jobs = _sched.get_due_jobs()
        _sched._sweep_stale_inflight_for_tick(due_jobs)
        if headless:
            from cron.scheduler_gateway_gate import refuse_agent_jobs_without_gateway
            due_jobs = refuse_agent_jobs_without_gateway(due_jobs)

        if not due_jobs:
            # Idle tick: skip config load + pool setup, but still reap crashed jobs' MCP orphans.
            if verbose:
                # Idle tick: skip config load + pool partitioning entirely (#33612 — the gateway ticker
                # calls tick(verbose=False) every 60s, so idle ticks previously fell through to
                # load_config()). Still run the post-tick MCP orphan sweep: main intentionally sweeps on
                # idle ticks so orphaned stdio children from crashed jobs are reaped even when nothing is
                # due.
                _sched.logger.info("%s - No jobs due", _sched._hermes_now().strftime('%H:%M:%S'))
            store_health.recheck_idle()
            _sched._sweep_mcp_orphans()
            return 0

        if store_health.dispatch_blocked(due_jobs):
            # Known-unwritable store, re-probed at most once a minute: every advance/claim would
            # fail, so skip them (the skip is recorded) and keep the tick's housekeeping alive.
            _sched._sweep_mcp_orphans()
            return 0

        if verbose:
            _sched.logger.info("%s - %s job(s) due", _sched._hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        # Advance next_run_at for recurring jobs FIRST, under the lock, before any execution
        # (at-most-once). Re-advancing running jobs keeps the grace window alive; mark_job_run
        # overwrites it on completion. Composes with the claim-time advance in claim_job_for_fire.
        due_jobs = _advance_or_drop_recurring(_sched, due_jobs)
        if not due_jobs:
            _sched._sweep_mcp_orphans()
            return 0

        _max_workers = _sched._resolve_max_parallel_workers()
        _split_scripts, _script_workers = _sched._resolve_max_parallel_script_workers()
        if verbose:
            _sched.logger.info(
                "Running %d job(s) in parallel (max_workers=%s%s)",
                len(due_jobs),
                _max_workers if _max_workers else "unbounded",
                f", scripts={_script_workers or 'unbounded'}" if _split_scripts else "")

        def _process_job(job: dict) -> bool:
            return _sched._process_due_job(job, adapters, loop, verbose)

        # Persistent pool, non-blocking dispatch. Already-running jobs are skipped; mark_job_run
        # re-arms next_run_at on completion, so no catch-up queue is needed.
        _all_futures: list = []
        pool = _sched._get_parallel_pool(_max_workers)
        script_pool = (_sched._get_parallel_pool(_script_workers, lane=_sched.SCRIPT_POOL_LANE)
                       if _split_scripts else pool)
        for job in due_jobs:
            fut = _sched._submit_with_guard(
                job, script_pool if job.get("no_agent") else pool, _process_job)
            if fut is None:
                continue
            _all_futures.append(fut)
            if not sync:
                _results.append(True)  # optimistically counted

        if not sync:
            _sched._sweep_mcp_orphans_when_all_done(_all_futures)
            return sum(_results)
        # sync (``hermes cron tick`` from a system crontab): wait below, AFTER the lock is
        # released. Holding it for the runs starved every other ticker (the gateway's) for the
        # whole length of the longest job; due-ness was already advanced above and each run
        # holds its own fire claim, so a concurrent tick cannot re-fire these jobs.
        _pending_futures = _all_futures
    finally:
        _sched._release_tick_lock(lock_fd)

    _wait_refreshing_queued_leases(_sched, _pending_futures, _results)
    _sched._sweep_mcp_orphans()
    return sum(_results)


# Well inside FIRE_CLAIM_TTL_SECONDS (the pending-slot lease) and the live-owner reap bound.
_QUEUED_LEASE_REFRESH_SECONDS = 60.0


def _wait_refreshing_queued_leases(_sched, futures: list, results: list) -> None:
    """Collect *futures* into *results*. Outside the tick lock another ticker may scan while
    these runs wait in the pool, so every run not yet started keeps its ``pending_slot`` stamp
    and execution row fresh: a queued run must look owned, not abandoned, or the other ticker
    restores the slot and the occurrence runs twice."""
    pending = set(futures)
    while pending:
        done, pending = concurrent.futures.wait(
            pending, timeout=_QUEUED_LEASE_REFRESH_SECONDS,
            return_when=concurrent.futures.FIRST_COMPLETED)
        for f in done:
            try:
                results.append(f.result())
            except (Exception, concurrent.futures.CancelledError) as exc:  # health: allow BLE001 -- moved from the sync wait: one job's failure is logged and counted, never aborts the tick
                _sched.logger.error("Cron job future failed: %s", exc)
                results.append(False)
        queued = [f for f in pending if not f.running() and not f.done()]
        if queued:
            _refresh_queued_leases(queued)


def _refresh_queued_leases(queued: list) -> None:
    from cron.executions import touch_execution_progress
    from cron.occurrences import refresh_pending_slots

    for f in queued:
        execution_id = getattr(f, "cron_execution_id", None)
        if execution_id:
            with contextlib.suppress(OSError, sqlite3.Error):
                touch_execution_progress(execution_id)
    job_ids = [j for j in (getattr(f, "cron_job_id", None) for f in queued) if j]
    try:
        refresh_pending_slots(job_ids)
    except OSError as exc:
        logger.warning("Could not refresh queued cron runs' pending slots: %s", exc)
