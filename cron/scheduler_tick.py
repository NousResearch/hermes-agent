"""In-process tick admission and dispatch; job execution stays in scheduler.py."""

import concurrent.futures
import contextlib
import logging
import time

logger = logging.getLogger("cron.scheduler")

# One WARNING per errno per window: an unwritable store fails every 60s tick until fixed.
_STORE_WARN_INTERVAL_SECONDS = 900.0
_last_store_warning: dict = {}


def warn_store_unwritable(exc: OSError, skipped: str) -> None:
    """Rate-limited WARNING for a cron store write that failed (ENOSPC/EROFS/EACCES). The caller
    skips the dispatch that needed the write: no job runs without a durable advance/fire claim."""
    now = time.monotonic()
    last = _last_store_warning.get(exc.errno)
    if last is not None and now - last < _STORE_WARN_INTERVAL_SECONDS:
        return
    _last_store_warning[exc.errno] = now
    logger.warning(
        "Cron store is unwritable (%s); skipped %s. Jobs stay due and fire once the store "
        "accepts writes again.", exc, skipped)


def tick(verbose=True, adapters=None, loop=None, sync=True, *, can_dispatch=None):
    from hermes_cli.backend_retirement import retirement

    # Hold admission through the entire scan/advance/submit handoff. A predicate alone races
    # prepare after the check but before a due job enters the running-job ledger.
    with retirement.work() as admitted:
        if not admitted:
            return 0
        return _tick_admitted(verbose, adapters, loop, sync, can_dispatch=can_dispatch)


def _tick_admitted(
    verbose: bool = True, adapters=None, loop=None, sync: bool = True, *, can_dispatch=None):
    """Check and run all due jobs. File-locked so only one tick runs at a time (gateway ticker vs
    standalone daemon / manual tick). ``can_dispatch``: optional gate; false leaves due jobs for the
    next allowed tick. Returns the number of jobs executed (0 if another tick holds the lock)."""
    from cron import scheduler as _sched

    # Stale-code yield gate — BEFORE the lock race. A process whose checkout was updated under it
    # serves mixed sys.modules (jobs die on ImportErrors); if a fresher gateway holds the runtime
    # lock, ITS ticker dispatches. With no fresh holder (desktop-standalone) the tick proceeds.
    _skew = _sched._should_yield_tick_to_fresh_gateway()
    if _skew is not None:
        _sched._log_tick_yield_once(f"boot={_skew[0]} disk={_skew[1]}")
        raise _sched.CronTickYielded(_skew[0], _skew[1])

    lock_dir, lock_file = _sched._get_lock_paths()
    _sched._ensure_cron_dir(lock_dir)
    lock_fd = _sched._acquire_tick_lock(lock_file)
    if lock_fd is None:
        return 0

    try:
        # `hermes pause` ESTOP: skip dispatch, never touch in-flight runs; check_paused logs once.
        with contextlib.suppress(ImportError):
            from agent.estop import check_paused as _estop_check_paused
            if _estop_check_paused("cron", _sched.logger):
                return 0

        if can_dispatch is not None and not can_dispatch():
            _sched.logger.debug("Cron dispatch paused while gateway drains existing work")
            return 0

        from cron.bot_chat_delivery import drain, drain_in_background
        if sync:
            drain()
        else:
            drain_in_background()
        _sched._maybe_reap_dead_owners()
        # Periodic worktree GC (6h, threaded) — the only sweep gateway-only boxes get.
        try:
            _sched._maybe_run_worktree_maintenance()
        except Exception as _wt_exc:
            _sched.logger.debug("Worktree maintenance dispatch failed: %s", _wt_exc)

        due_jobs = _sched.get_due_jobs()
        _sched._sweep_stale_inflight_for_tick(due_jobs)

        if not due_jobs:
            # Idle tick: skip config load + pool setup, but still reap crashed jobs' MCP orphans.
            if verbose:
                # Idle tick: skip config load + pool partitioning entirely (#33612 — the gateway ticker
                # calls tick(verbose=False) every 60s, so idle ticks previously fell through to
                # load_config()). Still run the post-tick MCP orphan sweep: main intentionally sweeps on
                # idle ticks so orphaned stdio children from crashed jobs are reaped even when nothing is
                # due.
                _sched.logger.info("%s - No jobs due", _sched._hermes_now().strftime('%H:%M:%S'))
            _sched._sweep_mcp_orphans()
            return 0

        if verbose:
            _sched.logger.info("%s - %s job(s) due", _sched._hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        # Advance next_run_at for recurring jobs FIRST, under the lock, before any execution
        # (at-most-once). Re-advancing running jobs keeps the grace window alive; mark_job_run
        # overwrites it on completion. Composes with the claim-time advance in claim_job_for_fire.
        try:
            _sched.advance_next_runs([job["id"] for job in due_jobs])
        except OSError as exc:
            # No durable advance -> a crash mid-run would re-fire recurring jobs; skipping is the
            # at-most-once side. One-shots still go through their own fire claim.
            recurring = [j for j in due_jobs if j.get("schedule", {}).get("kind") in {"cron", "interval"}]
            warn_store_unwritable(exc, f"{len(recurring)} recurring job(s)")
            due_jobs = [j for j in due_jobs if j not in recurring]
            if not due_jobs:
                return 0

        _max_workers = _sched._resolve_max_parallel_workers()
        if verbose:
            _sched.logger.info(
                "Running %d job(s) in parallel (max_workers=%s)",
                len(due_jobs),
                _max_workers if _max_workers else "unbounded")

        def _process_job(job: dict) -> bool:
            return _sched._process_due_job(job, adapters, loop, verbose)

        # Persistent pool, non-blocking dispatch. Already-running jobs are skipped; mark_job_run
        # re-arms next_run_at on completion, so no catch-up queue is needed.
        _results: list = []
        _all_futures: list = []
        pool = _sched._get_parallel_pool(_max_workers)
        for job in due_jobs:
            fut = _sched._submit_with_guard(job, pool, _process_job)
            if fut is None:
                continue
            _all_futures.append(fut)
            if not sync:
                _results.append(True)  # optimistically counted

        if sync:
            for f in concurrent.futures.as_completed(_all_futures):
                try:
                    _results.append(f.result())
                except Exception as exc:
                    _sched.logger.error("Cron job future failed: %s", exc)
                    _results.append(False)
            _sched._sweep_mcp_orphans()
            return sum(_results)

        _sched._sweep_mcp_orphans_when_all_done(_all_futures)
        return sum(_results)
    finally:
        _sched._release_tick_lock(lock_fd)
