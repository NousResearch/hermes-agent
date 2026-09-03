"""Timezone-migration silent misfire on the cron fire path.

Production incident: after upgrading from a build that scheduled in UTC to
one that honours the profile timezone (Europe/Brussels), daily cron jobs
stopped running. Their ``jobs.json`` rows still held pre-migration instants
like ``2026-09-02T04:00:00+00:00`` for expr ``0 4 * * *``. ``_ensure_aware``
normalizes that to ``06:00+02``, which ``0 4 * * *`` excludes, so the
stale-expression guard (#93049) classified it as a direct ``jobs.json`` edit,
logged exactly that, and re-anchored to tomorrow WITHOUT firing — the due
occurrence vanished with no failure anywhere.

The fix classifies the mismatch instead of assuming an edit: an instant whose
own wall clock is a legal occurrence, and which only left the lattice because
normalization changed its offset, is a representation migration and fires.

These exercise the real store against a temp ``HERMES_HOME`` (no mocks) per
the E2E-over-mocks discipline for file-touching code.
"""

from __future__ import annotations

from datetime import datetime

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def _reset_migration_counters(monkeypatch):
    """Module-level telemetry counters must not leak between tests."""
    from cron import jobs as J

    monkeypatch.setattr(J, "_timezone_migration_catchups", 0)
    monkeypatch.setattr(J, "_timezone_migration_catchups_recent", [])
    yield


# Europe/Brussels is +02:00 on this date; the legacy row was written by a
# build that scheduled everything at the UTC offset.
_BRUSSELS_NOW = datetime.fromisoformat("2026-09-02T06:05:00+02:00")
_LEGACY_UTC_NEXT_RUN = "2026-09-02T04:00:00+00:00"
_DAILY_0400 = "0 4 * * *"


def _write_cron_job(expr: str, next_run_at: str, name: str = "t") -> str:
    """Persist a cron job with a pinned next_run_at (the legacy-row shape)."""
    from cron.jobs import create_job, load_jobs, save_jobs

    job = create_job(prompt="x", schedule="every 5m", name=name)
    jobs = load_jobs()
    for j in jobs:
        if j["id"] == job["id"]:
            j["schedule"] = {"kind": "cron", "expr": expr}
            j["next_run_at"] = next_run_at
    save_jobs(jobs)
    return job["id"]


def test_legacy_utc_offset_next_run_still_fires(temp_home, monkeypatch):
    """The incident case: a pre-migration +00:00 instant for a Brussels
    ``0 4 * * *`` job must fire its due occurrence, not be re-anchored away."""
    from cron.jobs import get_due_jobs, get_timezone_migration_catchup_stats

    monkeypatch.setattr("cron.jobs._hermes_now", lambda: _BRUSSELS_NOW)
    jid = _write_cron_job(_DAILY_0400, _LEGACY_UTC_NEXT_RUN)

    due = get_due_jobs()

    assert jid in [j["id"] for j in due]
    stats = get_timezone_migration_catchup_stats()
    assert stats["timezone_migration_catchups"] == 1
    record = stats["recent"][0]
    assert record["job_id"] == jid
    assert record["expr"] == _DAILY_0400
    assert record["stored_next_run_at"] == _LEGACY_UTC_NEXT_RUN
    assert record["normalized_next_run_at"] == "2026-09-02T06:00:00+02:00"


def test_legacy_offset_catchup_fires_at_most_once(temp_home, monkeypatch):
    """The catch-up run is a single fire: once the scheduler advances the
    job, the legacy instant is gone and a second scan finds nothing due."""
    from cron.jobs import advance_next_run, get_due_jobs, get_job

    monkeypatch.setattr("cron.jobs._hermes_now", lambda: _BRUSSELS_NOW)
    jid = _write_cron_job(_DAILY_0400, _LEGACY_UTC_NEXT_RUN)

    assert jid in [j["id"] for j in get_due_jobs()]
    assert advance_next_run(jid) is True

    # Re-anchored to tomorrow's occurrence, expressed in the configured zone.
    assert get_job(jid)["next_run_at"] == "2026-09-03T04:00:00+02:00"
    assert [j["id"] for j in get_due_jobs() if j["id"] == jid] == []


def test_genuine_expr_edit_still_reanchors_without_firing(temp_home, monkeypatch):
    """#93049 protection intact: a stale instant in the CURRENT offset (no
    representation change) is still treated as an edit and does not fire."""
    from cron.jobs import get_due_jobs, get_job, get_timezone_migration_catchup_stats

    monkeypatch.setattr("cron.jobs._hermes_now", lambda: _BRUSSELS_NOW)
    # Stored at the configured offset, but the expr was edited to 09:00.
    jid = _write_cron_job("0 9 * * *", "2026-09-02T04:00:00+02:00")

    due = get_due_jobs()

    assert [j["id"] for j in due if j["id"] == jid] == []
    assert get_job(jid)["next_run_at"] == "2026-09-02T09:00:00+02:00"
    assert (
        get_timezone_migration_catchup_stats()["timezone_migration_catchups"] == 0
    )


def test_expr_edit_on_a_legacy_offset_row_still_does_not_fire(temp_home, monkeypatch):
    """A legacy +00:00 row whose expr was ALSO edited must not fire: the
    stored wall clock is not an occurrence of the new expression either, so
    the migration escape hatch does not open."""
    from cron.jobs import get_due_jobs, get_job, get_timezone_migration_catchup_stats

    monkeypatch.setattr("cron.jobs._hermes_now", lambda: _BRUSSELS_NOW)
    jid = _write_cron_job("0 9 * * *", _LEGACY_UTC_NEXT_RUN)

    due = get_due_jobs()

    assert [j["id"] for j in due if j["id"] == jid] == []
    assert get_job(jid)["next_run_at"] == "2026-09-02T09:00:00+02:00"
    assert (
        get_timezone_migration_catchup_stats()["timezone_migration_catchups"] == 0
    )


def test_future_local_wall_clock_is_left_scheduled(temp_home, monkeypatch):
    """A legacy row whose normalized instant has not arrived yet is simply
    not due — no catch-up, no re-anchor, no telemetry."""
    from cron.jobs import get_due_jobs, get_job, get_timezone_migration_catchup_stats

    before_due = datetime.fromisoformat("2026-09-02T05:00:00+02:00")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: before_due)
    jid = _write_cron_job(_DAILY_0400, _LEGACY_UTC_NEXT_RUN)

    due = get_due_jobs()

    assert [j["id"] for j in due if j["id"] == jid] == []
    assert get_job(jid)["next_run_at"] == _LEGACY_UTC_NEXT_RUN
    assert (
        get_timezone_migration_catchup_stats()["timezone_migration_catchups"] == 0
    )


def test_future_stored_wall_clock_still_takes_the_offset_repair_path(
    temp_home, monkeypatch
):
    """#28934 regression: a westward TZ move (+10 -> +02) that makes a still-
    future wall clock look due recomputes rather than firing early, and is
    NOT reclassified as a migration catch-up."""
    from cron.jobs import get_due_jobs, get_job, get_timezone_migration_catchup_stats

    scan_time = datetime.fromisoformat("2026-09-02T14:00:00+02:00")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: scan_time)
    jid = _write_cron_job("0 21 * * *", "2026-09-02T21:00:00+10:00")

    due = get_due_jobs()

    assert [j["id"] for j in due if j["id"] == jid] == []
    assert get_job(jid)["next_run_at"] == "2026-09-02T21:00:00+02:00"
    assert (
        get_timezone_migration_catchup_stats()["timezone_migration_catchups"] == 0
    )


def test_classifier_separates_migration_from_edit(temp_home):
    """Unit-level: the three classifications the fire path branches on."""
    from cron.jobs import (
        STALE_CRON_EXPR_EDIT,
        STALE_CRON_MATCH,
        STALE_CRON_TIMEZONE_MIGRATION,
        _classify_stale_cron_next_run,
    )

    daily = {"kind": "cron", "expr": _DAILY_0400}
    raw_legacy = datetime.fromisoformat(_LEGACY_UTC_NEXT_RUN)
    normalized = datetime.fromisoformat("2026-09-02T06:00:00+02:00")
    on_lattice = datetime.fromisoformat("2026-09-02T04:00:00+02:00")

    # Stored instant already occurs under the current expression.
    assert (
        _classify_stale_cron_next_run(daily, on_lattice, on_lattice)
        == STALE_CRON_MATCH
    )
    # Only the offset representation changed.
    assert (
        _classify_stale_cron_next_run(daily, raw_legacy, normalized)
        == STALE_CRON_TIMEZONE_MIGRATION
    )
    # Wall clock never moved, so a mismatch can only be a schedule edit.
    assert (
        _classify_stale_cron_next_run(
            {"kind": "cron", "expr": "0 9 * * *"}, on_lattice, on_lattice
        )
        == STALE_CRON_EXPR_EDIT
    )
    # Wall clock moved, but the stored wall clock is not an occurrence either.
    assert (
        _classify_stale_cron_next_run(
            {"kind": "cron", "expr": "0 9 * * *"}, raw_legacy, normalized
        )
        == STALE_CRON_EXPR_EDIT
    )
