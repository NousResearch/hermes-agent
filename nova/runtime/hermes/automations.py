"""Read and govern the Hermes cron store, one agent at a time.

A NOVA agent **is** a Hermes profile, and a Hermes cron store lives under the profile's
home (``cron/jobs.json``). So an automation belongs to exactly one agent by construction,
and an agent belongs to exactly one tenant — the isolation is the directory, not a check
somebody has to remember to write. Proven in this module's tests: reading agent A's store
cannot see, pause or delete a job in agent B's, because it is a different file.

**Nothing here reimplements the scheduler.** Reads and writes go through
``cron.jobs``' own API under :func:`cron.jobs.use_cron_store`, so schedule parsing,
``next_run_at`` computation, the on-disk lock and the job-state machine all stay in
Hermes. NOVA decides *who may act*; Hermes decides *what the act means*.

Two findings from the audit shape this file:

* ``use_cron_store`` is a **contextvar**, so it is safe to use inside a threaded
  server — unlike the module-global ``cron.executions.EXECUTIONS_FILE``.
* ``use_cron_store`` retargets ``jobs.json`` and the ticker markers but **not** the
  execution ledger: ``cron.executions`` resolves its path from ``get_hermes_home()``,
  which a contextvar does not move. ``cron.jobs.list_jobs`` therefore attaches a
  ``latest_execution`` read from the wrong store. This module ignores that field and
  reads ``<profile>/cron/executions.db`` itself, read-only.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from nova.runtime.base import AutomationRunView, AutomationView, SchedulerHealth

#: Cron store, relative to an agent's profile home.
CRON_DIRNAME = "cron"

#: Execution ledger inside that store.
EXECUTIONS_DB = "executions.db"

#: How stale the ticker heartbeat may be before the scheduler is reported as not running.
#: The ticker loops once a minute, so three missed iterations is a real stall rather than
#: a slow tick.
HEARTBEAT_STALE_SECONDS = 180.0


@contextmanager
def _store(profile_dir: Path) -> Iterator[Any]:
    """Point ``cron.jobs`` at one agent's store for the duration of the block.

    Imported lazily and inside the function: ``tests/platform/test_boundaries.py``
    requires that nothing under ``nova/`` imports the runtime at module scope, so a
    NOVA install without Hermes still loads.
    """
    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(profile_dir):
        yield cron_jobs


def _open_ledger(profile_dir: Path) -> Optional[sqlite3.Connection]:
    """Read-only connection to this agent's execution ledger, or None.

    Read-only by URI, exactly as :mod:`nova.runtime.hermes.work` opens the task store:
    the control plane observes the runtime, it never writes to it. Absence is normal —
    the ledger is created the first time a job actually fires.
    """
    path = profile_dir / CRON_DIRNAME / EXECUTIONS_DB
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error:
        return None


def _recent_runs(
    profile_dir: Path, job_ids: tuple[str, ...], *, per_job: int
) -> dict[str, tuple[AutomationRunView, ...]]:
    """Recent execution attempts per job, newest first.

    One query for every job rather than one per job: an agent with fifty automations
    would otherwise cost fifty connections' worth of work to render one screen.
    """
    if not job_ids or per_job < 1:
        return {}
    connection = _open_ledger(profile_dir)
    if connection is None:
        return {}
    try:
        placeholders = ",".join("?" for _ in job_ids)
        try:
            rows = connection.execute(
                f"SELECT job_id, id, status, claimed_at, started_at, finished_at, error "  # noqa: S608
                f"FROM executions WHERE job_id IN ({placeholders}) "
                "ORDER BY claimed_at DESC, id DESC",
                job_ids,
            ).fetchall()
        except sqlite3.Error:
            return {}
    finally:
        connection.close()

    grouped: dict[str, list[AutomationRunView]] = {}
    for row in rows:
        bucket = grouped.setdefault(str(row["job_id"]), [])
        if len(bucket) >= per_job:
            continue
        bucket.append(
            AutomationRunView(
                run_id=str(row["id"] or ""),
                status=str(row["status"] or ""),
                claimed_at=str(row["claimed_at"] or ""),
                started_at=str(row["started_at"] or "") or None,
                finished_at=str(row["finished_at"] or "") or None,
                error=str(row["error"] or ""),
            )
        )
    return {job_id: tuple(runs) for job_id, runs in grouped.items()}


def scheduler_health(profile_dir: Path) -> SchedulerHealth:
    """Whether anything is actually going to run these schedules.

    This is the most important thing the Automations screen says, and the reason it
    exists. Hermes' own CLI calls it their number-one support report: *"the cron ticker
    only runs inside the gateway … without a running gateway, next_run_at passes but jobs
    never fire and last_run_at stays null"*. A dashboard that lists schedules without
    saying whether a scheduler is attached is showing intentions as if they were
    commitments.

    Two markers, deliberately distinct, both written by the ticker into this profile's
    own store: ``ticker_heartbeat`` (the loop iterated) and ``ticker_last_success`` (the
    loop iterated *without raising*). A ticker stuck failing every tick keeps the first
    fresh and the second stale — "running, but failing" rather than "healthy".
    """
    with _store(profile_dir) as cron_jobs:
        heartbeat = cron_jobs.get_ticker_heartbeat_age()
        success = cron_jobs.get_ticker_success_age()
        last_error = cron_jobs.get_ticker_last_error() or ""

    if heartbeat is None:
        # Missing is "cannot determine", not "dead" — but for a customer deciding whether
        # to trust a schedule, an unobservable scheduler and a stopped one mean the same
        # thing, so the caveat is spelled out rather than resolved optimistically.
        return SchedulerHealth(
            running=False,
            detail=(
                "No scheduler heartbeat found for this agent. Automations are recorded "
                "but nothing is running them — in Hermes the ticker lives inside the "
                "gateway, and there is no standalone cron daemon."
            ),
        )
    if heartbeat > HEARTBEAT_STALE_SECONDS:
        return SchedulerHealth(
            running=False,
            heartbeat_age_seconds=heartbeat,
            success_age_seconds=success,
            last_error=last_error,
            detail=(
                f"The scheduler last checked in {int(heartbeat)}s ago, past the "
                f"{int(HEARTBEAT_STALE_SECONDS)}s staleness bound. Schedules will not fire."
            ),
        )
    if success is None or success > HEARTBEAT_STALE_SECONDS:
        return SchedulerHealth(
            running=True,
            healthy=False,
            heartbeat_age_seconds=heartbeat,
            success_age_seconds=success,
            last_error=last_error,
            detail=(
                "The scheduler is running but has not completed a tick without error "
                "recently. Automations may be silently failing."
            ),
        )
    return SchedulerHealth(
        running=True,
        healthy=True,
        heartbeat_age_seconds=heartbeat,
        success_age_seconds=success,
        last_error=last_error,
        detail="",
    )


def _to_view(record: dict, agent_id: str, runs: tuple[AutomationRunView, ...]) -> AutomationView:
    """One stored cron record as the control plane presents it.

    ``schedule_display`` is Hermes' own rendering of the schedule ("every day at 07:00").
    Re-deriving it from the cron expression would be a second implementation of something
    the runtime already does, and the two would drift.
    """
    schedule = record.get("schedule") or {}
    return AutomationView(
        automation_id=str(record.get("id") or ""),
        name=str(record.get("name") or "") or str(record.get("id") or ""),
        agent_id=agent_id,
        schedule_display=str(
            record.get("schedule_display") or schedule.get("display") or ""
        ),
        schedule_kind=str(schedule.get("kind") or ""),
        schedule_expression=str(schedule.get("expr") or ""),
        enabled=bool(record.get("enabled", True)),
        state=str(record.get("state") or ""),
        next_run_at=str(record.get("next_run_at") or "") or None,
        last_run_at=str(record.get("last_run_at") or "") or None,
        last_status=str(record.get("last_status") or ""),
        last_error=str(record.get("last_error") or ""),
        failure_streak=int(record.get("failure_streak") or 0),
        paused_reason=str(record.get("paused_reason") or ""),
        created_at=str(record.get("created_at") or "") or None,
        runs=runs,
    )


def list_automations(
    profile_dir: Path, agent_id: str, *, history: int = 5
) -> list[AutomationView]:
    """Every automation this agent owns, disabled ones included.

    Disabled ones included on purpose: a paused automation is exactly what an
    administrator opens this screen to find.
    """
    with _store(profile_dir) as cron_jobs:
        try:
            # include_disabled: a paused automation is still an automation.
            records = cron_jobs.list_jobs(include_disabled=True)
        except Exception:  # noqa: BLE001 — an unreadable store is "none", not a 500
            return []

    ids = tuple(str(r.get("id") or "") for r in records if r.get("id"))
    runs = _recent_runs(profile_dir, ids, per_job=history)
    return [
        _to_view(record, agent_id, runs.get(str(record.get("id") or ""), ()))
        for record in records
    ]


def get_automation(
    profile_dir: Path, agent_id: str, automation_id: str, *, history: int = 20
) -> Optional[AutomationView]:
    """One automation, or None when this agent does not own it.

    None rather than a refusal: from outside the agent's store the job does not exist,
    which is both true and the answer that leaks nothing.
    """
    with _store(profile_dir) as cron_jobs:
        record = cron_jobs.get_job(automation_id)
    if not record:
        return None
    runs = _recent_runs(profile_dir, (automation_id,), per_job=history)
    return _to_view(record, agent_id, runs.get(automation_id, ()))


def set_enabled(
    profile_dir: Path, agent_id: str, automation_id: str, *, enabled: bool, reason: str = ""
) -> Optional[AutomationView]:
    """Pause or resume one automation. None when this agent does not own it.

    Delegates to ``cron.jobs.pause_job`` / ``resume_job`` rather than writing
    ``enabled`` into the record: resuming recomputes ``next_run_at`` and clears the pause
    marker, and a hand-written flag would leave a job that looks active and never fires.

    Only the enabled/disabled transition is offered. Creating an automation means
    handing an agent a new instruction that NOVA never compiled and no policy reviewed,
    which is a governance hole rather than a feature — see PHASE_11_AUTOMATIONS.md.
    """
    with _store(profile_dir) as cron_jobs:
        if cron_jobs.get_job(automation_id) is None:
            return None
        if enabled:
            record = cron_jobs.resume_job(automation_id)
        else:
            record = cron_jobs.pause_job(automation_id, reason or None)
    if not record:
        return None
    runs = _recent_runs(profile_dir, (automation_id,), per_job=5)
    return _to_view(record, agent_id, runs.get(automation_id, ()))


def create(profile_dir: Path, agent_id: str, compiled) -> AutomationView:
    """Create the runtime job for a compiled automation.

    ``job_kwargs`` comes from the compiler and is passed through unchanged — nothing is
    added here, so what was reviewed is what runs. Schedule parsing, id minting and
    ``next_run_at`` remain the runtime's.
    """
    with _store(profile_dir) as cron_jobs:
        cron_jobs.ensure_dirs()
        record = cron_jobs.create_job(**compiled.job_kwargs)
    return _to_view(record, agent_id, ())


def delete(profile_dir: Path, agent_id: str, automation_id: str) -> bool:
    """Remove one automation. False when this agent does not own it.

    Ownership is re-checked against this agent's own store rather than trusted from a
    listing the caller may have obtained earlier.
    """
    with _store(profile_dir) as cron_jobs:
        if cron_jobs.get_job(automation_id) is None:
            return False
        return bool(cron_jobs.remove_job(automation_id))


def validate_schedule(schedule: str):
    """Parse a schedule phrase with the runtime's own parser, or raise ``SpecError``.

    Lives in the adapter because this is the one package allowed to name the runtime.
    The compiler takes this as an injected callable, so NOVA never grows a second
    schedule grammar that could drift from the one that decides when jobs actually run.
    """
    from nova.errors import SpecError

    from cron.jobs import parse_schedule

    try:
        return parse_schedule(schedule)
    except Exception as exc:  # noqa: BLE001 — the runtime's own message is the useful one
        raise SpecError(f"schedule {schedule!r} is not valid: {exc}") from None
