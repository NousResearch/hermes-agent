"""Cron: teardown of a worker that outlived its ``run_job``.

``ThreadPoolExecutor.shutdown(wait=False)`` after an inactivity timeout does not stop a
worker already inside ``run_conversation``. Finalizing its SessionDB from ``run_job``'s
``finally`` would close a handle the worker is still writing to — the checkpoint/WAL-unlink
overlap behind #102827. The worker's Future owns the teardown instead.

Also: release of a restart-safe worker's transient scope unit once its ledger row is terminal.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

SCOPE_RELEASE_GRACE_SECONDS = 60.0  # let cleanly-exiting stragglers drain first
SCOPE_RELEASE_POLL_SECONDS = 5.0
# BEST-EFFORT bound on how long a terminalized execution's unit stays loaded-active: the
# release thread is a daemon and dies with a gateway restart while its worker still drains
# (the restart-safe case: no waiter survives). Such residual units are handled operationally
# by ledger-checked exact-name cleanup (safe_worker_cleanup) and the zombie-scope sentinel.
SCOPE_RELEASE_DEADLINE_SECONDS = 1800.0


def _close_late_session_db_result(future: concurrent.futures.Future) -> None:
    """Done-callback: close a SessionDB whose constructor finished after run_job's init timeout
    (worker abandoned via ``shutdown(wait=False)``), else its .db/WAL/SHM handles leak to EMFILE.

    If the constructor later completes inside that abandoned worker, the Future's result — an open
    SessionDB holding .db / WAL / SHM file handles — would be orphaned and never closed, leaking
    descriptors until EMFILE (#72782). This callback retrieves and closes that eventual late result.
    """
    with contextlib.suppress(Exception):
        db = future.result()
        if db is not None:
            from hermes_state_registry import release_or_close
            release_or_close(db)


def defer_teardown_to_running_worker(
    future: Optional[concurrent.futures.Future], session_db, agent, job_id: str, job_name: str,
    cron_session_id: str, workdir: Optional[str] = None,
) -> bool:
    """Return True when the worker is still running and its Future will finalize the session
    and tear the agent down on completion; False when the caller must do it now."""
    if future is None or future.done():
        return False
    from cron.scheduler import _finalize_cron_session, _teardown_cron_agent

    def _finish(_future) -> None:
        try:
            if session_db:
                _finalize_cron_session(session_db, agent, job_id, job_name, cron_session_id,
                                       workdir=workdir)
        finally:
            _teardown_cron_agent(agent, job_id)

    # Runs inline if the worker finished between done() and here — still exactly once.
    future.add_done_callback(_finish)
    return True


def reap_terminal_worker_in_background(process: subprocess.Popen) -> None:
    """Keep the reap contract when the waiter returns before the worker exits.

    The ledger turning terminal lets ``_wait_for_external_cron_worker_body``
    return while the worker is still in final teardown. The gateway remains the
    worker's parent, so if nobody calls ``wait()`` afterwards the worker lingers
    as a zombie (STAT=Z) under the gateway until it is restarted (#114509). A
    short-lived daemon thread holds that single responsibility and ends with
    the process exit it waits for.
    """
    threading.Thread(
        target=process.wait,
        name=f"cron-worker-reap-{getattr(process, 'pid', '?')}",
        daemon=True,
    ).start()


def _scope_unit_is_active(unit_name: str) -> bool:
    """True only when ``systemctl --user is-active`` reports active/activating.

    Any failure (no systemctl, no user bus, timeout) reads as NOT active: never stop
    what we cannot see.
    """
    import shutil

    from tools.process_registry import systemd_user_bus_env

    binary = shutil.which("systemctl")
    if binary is None:
        return False
    try:
        result = subprocess.run(
            [binary, "--user", "is-active", unit_name],
            capture_output=True, timeout=10, stdin=subprocess.DEVNULL,
            env=systemd_user_bus_env(), check=False,
        )
    except Exception as exc:
        logger.warning("systemctl --user is-active %s failed: %s", unit_name, exc, exc_info=True)
        return False
    return (result.stdout or b"").decode(errors="replace").strip() in ("active", "activating")


def schedule_scope_unit_release(
    process: subprocess.Popen,
    scope_unit: str,
    *,
    grace_seconds: float = SCOPE_RELEASE_GRACE_SECONDS,
    poll_seconds: float = SCOPE_RELEASE_POLL_SECONDS,
    deadline_seconds: float = SCOPE_RELEASE_DEADLINE_SECONDS,
) -> None:
    """Reap the worker, then release its transient scope unit if something still holds it.

    A restart-safe worker runs in a ``systemd-run --scope --collect`` unit, which systemd
    only deactivates once its cgroup is empty. Any process still in it after the worker
    committed its terminal row and exited — a ``start_new_session`` background job, a
    grandchild detached by the bot session — keeps the unit ``loaded active running``
    for days. Nothing else ever stops that unit.

    Only-then semantics: this is scheduled exclusively from waiter paths that have
    already confirmed the ledger row terminal or the owner provably gone, so nothing in
    the unit still owns the execution. A worker stuck in teardown past
    ``deadline_seconds`` is killed, as kanban's ``reap_terminal_workers`` does for a
    worker that outlived its terminal ledger row (#111791). It subsumes
    :func:`reap_terminal_worker_in_background` (#114509): the same thread calls
    ``wait()``.

    Stragglers get ``grace_seconds`` to drain on their own; whatever remains is
    stopped by the EXACT unit name. Operator lesson: never glob ``hermes-worker-*``,
    which would also stop live workers and background jobs in their own units.
    """
    threading.Thread(
        target=_release_scope_unit,
        args=(process, scope_unit, grace_seconds, poll_seconds, deadline_seconds),
        name=f"cron-scope-release-{getattr(process, 'pid', '?')}",
        daemon=True,
    ).start()


def _release_scope_unit(
    process: subprocess.Popen, scope_unit: str, grace_seconds: float, poll_seconds: float,
    deadline_seconds: float,
) -> None:
    """Thread body of :func:`schedule_scope_unit_release`."""
    try:
        try:
            process.wait(timeout=deadline_seconds)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Cron worker pid %s still running %s s after its ledger row went terminal; "
                "killing it to release scope unit %s",
                process.pid, deadline_seconds, scope_unit,
            )
            with contextlib.suppress(Exception):
                process.kill()
            with contextlib.suppress(Exception):
                process.wait(timeout=30)
        grace_ends = time.monotonic() + grace_seconds
        while time.monotonic() < grace_ends:
            if not _scope_unit_is_active(scope_unit):
                return
            time.sleep(poll_seconds)
        from tools.process_registry import _stop_systemd_unit

        if _stop_systemd_unit(scope_unit):
            logger.info(
                "Released restart-safe cron worker scope unit %s (worker reaped, ledger terminal)",
                scope_unit,
            )
        else:
            logger.warning("Could not release restart-safe cron worker scope unit %s", scope_unit)
    except Exception as exc:
        # Best-effort reclamation: must never surface to the gateway tick.
        logger.debug("Releasing cron worker scope unit %s failed: %s", scope_unit, exc, exc_info=True)
