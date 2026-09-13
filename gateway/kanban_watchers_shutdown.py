"""Bounded scoped-worker shutdown before releasing dispatcher ownership."""

from __future__ import annotations

import hermes_cli.kanban_db_boards as _owner_kanban_boards

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as _kbc
from hermes_cli import kanban_worker_recovery
from hermes_cli import kanban_worker_stop

import threading
import time

from gateway.kanban_watchers_common import logger

_SHUTDOWN_STOP_BASE_SECONDS = 15.0


def stop_scoped_workers_on_shutdown(_load_config) -> None:
    """Honour ``kanban.worker_isolation_stop_on_shutdown`` (default
    false).

    Default behaviour: scoped workers live in their own transient
    user systemd scopes, so they SURVIVE a gateway restart and the
    next gateway re-adopts them (claim rewritten, run continues).
    When the knob is true, a graceful shutdown instead stops every
    scoped worker this host still claims — verified teardown of
    the whole cgroup, not a pid kill — before the gateway exits.

    Runs on a daemon thread with a join bounded by ONE deadline
    covering the whole stop (pre-scan + worker join + service
    drain; each verified stop is capped per unit in
    tools.process_registry), so a wedged ``systemctl`` cannot hang
    gateway shutdown AND the dispatcher lock is not released while
    cleanup is still inside its budget (review findings g + I).
    When the budget does expire, exactly what was left — pending
    AND in-flight units — is logged before the lock goes; the next
    gateway's adoption sweep re-adopts or reclaims it.
    """
    try:
        cfg = _load_config()
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("worker_isolation_stop_on_shutdown", False):
            return
    except Exception:
        logger.exception("kanban shutdown: cannot load config; leaving scoped workers for re-adoption")
        return

    def _collect_expected_units() -> list:
        """Cheap pre-scan: which units this host will try to stop.

        Checks the abort flag between boards so a cancelled
        shutdown stops scanning too (finding Q). Progress lands
        in ``state`` live (pass 8, Y): the units list, boards
        scanned, and boards total are readable while the scan
        runs, so a scan that outlives the base budget yields a
        truthful "incomplete" summary instead of a fabricated
        zero."""
        units: list = state["expected"]
        try:
            boards = _owner_kanban_boards.list_boards(include_archived=False)
        except Exception:
            boards = [_owner_kanban_boards.read_board_metadata(_owner_kanban_boards.DEFAULT_BOARD)]
        state["boards_total"] = len(boards)
        host_prefix = f"{kanban_db._claimer_id().split(':', 1)[0]}:"
        for b in boards:
            if _should_abort_cleanup():
                break
            slug = b.get("slug") or _owner_kanban_boards.DEFAULT_BOARD
            conn = None
            try:
                conn = _kbc.connect(board=slug)
                for row in conn.execute(
                    "SELECT claim_lock, worker_scope FROM tasks "
                    "WHERE status = 'running' "
                    "  AND worker_scope IS NOT NULL"
                ).fetchall():
                    if (row["claim_lock"] or "").startswith(host_prefix):
                        units.append(row["worker_scope"])
            except Exception:
                pass
            finally:
                state["boards_scanned"] = state.get("boards_scanned", 0) + 1
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
        return units

    state: dict = {"expected": []}
    collected = threading.Event()
    # Finding Q: the cleanup thread must not keep scanning or
    # stopping scopes after the caller's budget expires and the
    # dispatcher lock is released — the next gateway may be
    # re-adopting those very units. The caller sets this event
    # the moment its deadline passes; every board scan and every
    # unit stop checks it between units and stands down.
    cleanup_cancelled = threading.Event()

    def _should_abort_cleanup() -> bool:
        if cleanup_cancelled.is_set():
            return True
        deadline = state.get("deadline")
        return deadline is not None and time.monotonic() >= deadline

    def _run() -> None:
        _collect_expected_units()
        collected.set()
        try:
            boards = _owner_kanban_boards.list_boards(include_archived=False)
        except Exception:
            boards = [_owner_kanban_boards.read_board_metadata(_owner_kanban_boards.DEFAULT_BOARD)]
        for b in boards:
            if _should_abort_cleanup():
                break
            slug = b.get("slug") or _owner_kanban_boards.DEFAULT_BOARD
            conn = None
            try:
                conn = _kbc.connect(board=slug)
                stopped = kanban_worker_recovery.stop_all_scoped_workers(
                    conn,
                    should_abort=_should_abort_cleanup,
                    cancel_event=cleanup_cancelled,
                    deadline=state.get("deadline"),
                )
            except Exception:
                logger.exception("kanban shutdown [%s]: scoped-worker stop failed", slug)
                continue
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
            if stopped:
                state.setdefault("stopped", []).extend(stopped)
                logger.info(
                    "kanban shutdown [%s]: stopped %d scoped worker(s): %s",
                    slug, len(stopped), ", ".join(stopped),
                )

    worker = threading.Thread(
        target=_run, name="kanban-scope-shutdown", daemon=True,
    )
    worker.start()
    # Wait for the pre-scan so the budget scales with the real
    # work instead of a flat 15 s: base covers the board scans,
    # each expected unit adds its own verified-stop deadline.
    # The pre-scan wait is INSIDE that budget: one deadline from
    # the moment the thread started bounds BOTH joins below, so
    # the stated budget is a ceiling on the whole shutdown stop,
    # not a per-join allowance stacked on top of each other
    # (review finding I — the old code joined the worker for the
    # full budget and THEN added a whole per-unit timeout for
    # the service drain, up to double the stated budget).
    shutdown_started = time.monotonic()
    scan_done = collected.wait(timeout=_SHUTDOWN_STOP_BASE_SECONDS)
    if not scan_done:
        # The pre-scan outlived the base budget (slow boards or
        # a wedged board listing). Its live progress still feeds
        # the budget below, but the final summary must never
        # present the unscanned boards' units as "zero
        # unstopped" (pass 8, Y) — the count is unknowable by
        # construction, so the incompleteness is what gets
        # reported.
        logger.warning(
            "kanban shutdown: scope pre-scan incomplete after "
            "%.0fs — %d of %s board(s) scanned, %d unit(s) "
            "enumerated so far; the stop budget covers only "
            "those",
            _SHUTDOWN_STOP_BASE_SECONDS,
            state.get("boards_scanned", 0),
            str(state.get("boards_total", "?")),
            len(state.get("expected") or []),
        )
    try:
        from tools.process_registry_scope import SCOPE_STOP_VERIFY_BOUND_SECONDS
    except Exception:  # pragma: no cover — import is process-local
        SCOPE_STOP_VERIFY_BOUND_SECONDS = 31.0
    expected = state.get("expected") or []
    budget = _SHUTDOWN_STOP_BASE_SECONDS + len(expected) * (
        SCOPE_STOP_VERIFY_BOUND_SECONDS + 2.0
    )
    deadline = shutdown_started + budget
    state["deadline"] = deadline  # the cleanup thread honors it too
    worker.join(timeout=max(0.0, deadline - time.monotonic()))
    # The budget is gone: stand the cleanup thread down NOW — it
    # must not keep scanning boards or stopping scopes while the
    # next gateway re-adopts them (finding Q). The event also
    # reaches any IN-FLIGHT verified stop through
    # stop_all_scoped_workers' cancel plumbing (pass 8, Y).
    cleanup_cancelled.set()
    # Tick-queued verified stops must not outlive the lock either:
    # drain the background service within the SAME deadline
    # before the caller releases the dispatcher lock.
    leftover = kanban_worker_stop.join_scope_stop_service(
        timeout=max(0.0, deadline - time.monotonic()),
        # The same cancel signal the direct cleanup propagates
        # (pass 9, AH): on expiry the join cancels the service's
        # in-flight stop too, so no systemctl signalling outlives
        # the budget or the lock release.
        cancel_event=cleanup_cancelled,
    )
    if worker.is_alive() or leftover:
        stopped_units = set(state.get("stopped") or [])
        still_running = sorted(
            (set(expected) - stopped_units)
            | {str(u) for u in leftover}
        )
        if scan_done or collected.is_set():
            scan_note = ""
            count_txt = "%d unit(s) still stopping (%s)" % (
                len(still_running),
                ", ".join(still_running) or "<unknown>",
            )
        elif state.get("boards_total") is None:
            # Not even the board listing returned — nothing about
            # the expected set is knowable (pass 8, Y): say so
            # instead of printing a fabricated zero.
            scan_note = (
                " — scan incomplete: boards not enumerated "
                "(board listing stalled)"
            )
            count_txt = "an unknown number of unit(s) still stopping"
        else:
            unscanned = max(
                0,
                state["boards_total"]
                - state.get("boards_scanned", 0),
            )
            scan_note = (
                " — scan incomplete: units on %d unscanned "
                "board(s) not enumerated" % unscanned
            )
            count_txt = "an unknown number of unit(s) still stopping"
        logger.warning(
            "kanban shutdown: scoped-worker stop did not finish "
            "within its budget (%.0fs)%s — releasing the dispatcher "
            "lock with %s; the next gateway's adoption sweep will "
            "re-adopt or reclaim whatever remains",
            budget, scan_note, count_txt,
        )
