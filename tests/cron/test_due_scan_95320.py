"""Regression tests for #95320 — the cron due scan must not deep-copy or
double-validate the jobs store on every tick.

Three behaviors are pinned:

1. An idle tick performs NO ``copy.deepcopy`` of the store (the pre-fix
   code deep-copied every record each tick even though ``load_jobs()``
   had just parsed a fresh, unshared structure).
2. Malformed records (id-less / bad schedule / bad timestamps) are still
   repaired exactly once and the repairs reach disk — normalize-once must
   be behaviorally identical to the old validate-both-views loops.
3. The retention sweep's survivor filter keeps every non-swept job
   (set-based membership replaces the quadratic any() scan), and config
   reads for the retention windows see edits to config.yaml — the cached
   validated config is invalidated when the file changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cron.jobs import (
    create_job,
    get_due_jobs,
    get_job,
    load_jobs,
    save_jobs,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _future_iso(minutes: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


def _write_minimal_config():
    """Drop a minimal config.yaml into the active HERMES_HOME.

    With no config file on disk, hermes_cli.config cannot build a cache
    signature and re-merges DEFAULT_CONFIG (with defensive deepcopies) on
    every read. Production installs always have one, so tests that pin
    steady-state tick costs provide it. Resolved via get_config_path()
    because the global conftest redirects HERMES_HOME to a per-test
    sandbox that is not this module's tmp_path.
    """
    from hermes_cli.config import get_config_path

    cfg = Path(get_config_path())
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("model:\n  default: test-model\n", encoding="utf-8")


# =========================================================================
# 1. No per-tick deepcopy of the store
# =========================================================================


def test_idle_tick_performs_no_deepcopy(tmp_cron_dir, monkeypatch):
    """An idle tick must not deepcopy the store (#95320).

    load_jobs() returns a fresh json.loads() product with no shared
    references; the pre-fix `copy.deepcopy(raw_jobs)` per tick was pure
    allocation churn scaling with store size while holding the jobs lock.
    """
    create_job(prompt="Recurring", schedule="every 1h")
    _write_minimal_config()

    # Warm up: populate the validated-config cache so the retention
    # lookup below is a signature check, and absorb one-off import-time
    # work. Only steady-state tick work is measured.
    get_due_jobs()

    calls = {"n": 0}

    import copy as _copy

    real_deepcopy = _copy.deepcopy

    def _counting_deepcopy(x, memo=None, *a, **kw):
        calls["n"] += 1
        return real_deepcopy(x, memo=memo)

    monkeypatch.setattr(_copy, "deepcopy", _counting_deepcopy)
    due = get_due_jobs()
    assert due == []
    assert calls["n"] == 0, (
        "idle due scan called copy.deepcopy %d time(s); the scan must "
        "normalize the fresh parse in place instead of copying it" % calls["n"]
    )


def test_due_scan_still_fires_without_deepcopy(tmp_cron_dir):
    """Control for the spy test above: a due job still fires normally."""
    job = create_job(prompt="Due now", schedule="every 1h")
    jobs = load_jobs()
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    for j in jobs:
        if j["id"] == job["id"]:
            j["next_run_at"] = past
    save_jobs(jobs)

    due = get_due_jobs()
    assert [j["id"] for j in due] == [job["id"]]


# =========================================================================
# 2. Malformed records: normalized once, persisted, healthy siblings safe
# =========================================================================


def _write_raw_store(jobs_file: Path, records):
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(json.dumps({"jobs": records}), encoding="utf-8")


def test_malformed_records_repaired_once_and_persisted(tmp_cron_dir, monkeypatch):
    """Id-less / malformed records are repaired and the repairs land on disk.

    The pre-fix code ran each repair pass twice (once over a deep-copied
    view, once over raw). Normalizing only the canonical records first,
    then deriving the scheduler view, must leave the same end state.
    """
    from cron import jobs as cj

    healthy_id = "healthy01"
    legacy = {  # id-less record from an older writer
        "job_id": "legacy012",
        "name": "legacy",
        "prompt": "x",
        "enabled": False,
        "schedule": {"kind": "once", "run_at": _future_iso(60)},
    }
    broken_schedule = {
        "id": "brokensch",
        "name": "broken-schedule",
        "prompt": "x",
        "enabled": False,
        "schedule": None,  # null schedule used to abort the whole scan
    }
    broken_next_run = {
        "id": "brokennext",
        "name": "broken-next-run",
        "prompt": "x",
        "enabled": False,
        "schedule": {"kind": "interval", "minutes": 10},
        "next_run_at": 12345,  # non-string ISO value
    }
    broken_last_run = {
        "id": "brokenlast",
        "name": "broken-last-run",
        "prompt": "x",
        "enabled": False,
        "schedule": {"kind": "interval", "minutes": 10},
        "next_run_at": _future_iso(30),
        "last_run_at": "not-a-date",  # unparseable string
    }
    healthy = {
        "id": healthy_id,
        "name": "healthy",
        "prompt": "x",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 60},
        # overdue but inside the catch-up grace window → fires this tick
        "next_run_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
    }

    _write_raw_store(
        cj.JOBS_FILE,
        [legacy, broken_schedule, broken_next_run, broken_last_run, healthy],
    )

    due = get_due_jobs()
    assert [j["id"] for j in due] == [healthy_id]

    repaired = {r["name"]: r for r in load_jobs()}
    # id recovered from the drifted "job_id" key
    assert repaired["legacy"]["id"] == "legacy012"
    # null schedule normalized to {}
    assert repaired["broken-schedule"]["schedule"] == {}
    # invalid next_run_at stripped (absent → recovery path recomputes)
    assert "next_run_at" not in repaired["broken-next-run"]
    # unparseable last_run_at stripped
    assert "last_run_at" not in repaired["broken-last-run"]
    # healthy sibling untouched
    stored_healthy = repaired["healthy"]
    assert stored_healthy["next_run_at"] == healthy["next_run_at"]
    assert stored_healthy["schedule"]["kind"] == "interval"


# =========================================================================
# 3. Retention sweep survivor filter (set-based)
# =========================================================================


class TestSweepSurvivorFilter:
    def _completed_oneshot(self, age_days: float, name: str) -> str:
        # NOTE: "30m" is a RECURRING interval in the current schedule grammar; the one-shot
        # spelling is "in 30m"/a timestamp, and the retention sweep only prunes schedule.kind
        # == "once" records. Building these with "30m" would silently stop testing the sweep.
        job = create_job(prompt=name, schedule="in 30m", repeat=1, name=name)
        from cron.jobs import mark_job_run

        mark_job_run(job["id"], success=True)
        stamp = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == job["id"]:
                j["last_run_at"] = stamp
        save_jobs(jobs)
        return job["id"]

    def test_sweep_keeps_every_survivor(self, tmp_cron_dir):
        """Expired one-shots go; recurring + recent + future records stay.

        Guards the set-based survivor filter that replaced the quadratic
        any()-scan after _sweep_completed_oneshots removed records.
        """
        old_a = self._completed_oneshot(age_days=30, name="old-a")
        old_b = self._completed_oneshot(age_days=45, name="old-b")
        recent = self._completed_oneshot(age_days=1, name="recent")
        recurring = create_job(prompt="Rec", schedule="every 1h", name="rec")
        future = create_job(
            prompt="Later",
            schedule="every 1h",
            name="later",
        )

        get_due_jobs()

        ids = {j["id"] for j in load_jobs()}
        assert old_a not in ids
        assert old_b not in ids
        assert recent in ids
        assert recurring["id"] in ids
        assert future["id"] in ids

        # Second tick is stable: nothing else disappears.
        get_due_jobs()
        ids2 = {j["id"] for j in load_jobs()}
        assert ids2 == ids

    def test_due_view_excludes_swept_records(self, tmp_cron_dir):
        """A swept record must not surface as due in the same tick."""
        stale = create_job(prompt="Old", schedule="in 30m", repeat=1, name="old")
        from cron.jobs import mark_job_run

        mark_job_run(stale["id"], success=True)
        jobs = load_jobs()
        for j in jobs:
            if j["id"] == stale["id"]:
                j["last_run_at"] = (
                    datetime.now(timezone.utc) - timedelta(days=30)
                ).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert stale["id"] not in [j["id"] for j in due]


# =========================================================================
# 4. Config reads: cached validated config invalidated on file change
# =========================================================================


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _config_path(home: Path) -> Path:
    from hermes_cli.config import get_config_path

    return Path(get_config_path())


def _set_cron_config(home: Path, key: str, value):
    """Write config.yaml directly (external-edit shape: no save_config)."""
    cfg = _config_path(home)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    text = f"model:\n  default: test-model\ncron:\n  {key}: {value}\n"
    cfg.write_text(text, encoding="utf-8")


def test_retention_window_follows_config_edit(hermes_home):
    """Editing cron.completed_retention_days takes effect without restart.

    The tick reads the retention window through the process-wide validated
    config cache; this pins that the cache's file-signature check
    invalidates it when config.yaml changes on disk (the pre-fix code paid
    a full defensive deepcopy of the entire merged config for the same
    two scalar keys on every 60s tick).
    """
    from cron.jobs import _completed_oneshot_retention_days

    _set_cron_config(hermes_home, "completed_retention_days", 7)
    assert _completed_oneshot_retention_days() == 7.0

    # External edit (different length so size+mtime both move).
    _set_cron_config(hermes_home, "completed_retention_days", 30)
    assert _completed_oneshot_retention_days() == 30.0


def test_output_retention_follows_config_edit(hermes_home):
    """Same cache-invalidation contract for cron.output_retention."""
    from cron.jobs import _cron_output_keep

    _set_cron_config(hermes_home, "output_retention", 5)
    assert _cron_output_keep() == 5

    _set_cron_config(hermes_home, "output_retention", 11)
    assert _cron_output_keep() == 11


def test_retention_defaults_when_config_absent(hermes_home):
    """No config.yaml → documented defaults, no crash."""
    from cron.jobs import (
        COMPLETED_ONESHOT_RETENTION_DAYS,
        _completed_oneshot_retention_days,
        _cron_output_keep,
    )

    assert _completed_oneshot_retention_days() == float(
        COMPLETED_ONESHOT_RETENTION_DAYS
    )
    assert _cron_output_keep() >= 0
