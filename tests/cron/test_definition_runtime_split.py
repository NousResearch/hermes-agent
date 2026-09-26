"""#75607 — jobs.json holds declarations; scheduler state lives in cron/runtime.db.

Covers the contracts the split must keep: a fire never rewrites jobs.json, a pre-split (or
downgrade-written) combined store migrates without losing a value, a save interrupted between the
two artifacts is finished (or safely discarded) by the next load, and the degraded-lock writer
cannot roll back rows it never touched.
"""

from __future__ import annotations

import json
import sqlite3

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    import cron.jobs

    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    return home


def _jobs_file(home):
    return home / "cron" / "jobs.json"


def _disk_records(home):
    return json.loads(_jobs_file(home).read_text(encoding="utf-8"))["jobs"]


def _runtime_rows(home):
    from cron.runtime_state import load_runtime_states
    return load_runtime_states(home / "cron")


def _write_combined(home, records):
    _jobs_file(home).write_text(json.dumps({"jobs": records}), encoding="utf-8")


def _combined_record(job_id="legacy1", **overrides):
    record = {
        "id": job_id, "name": "legacy", "prompt": "summarize", "schedule": {
            "kind": "interval", "minutes": 120, "display": "every 120m"},
        "schedule_display": "every 120m", "repeat": {"times": 5, "completed": 2},
        "enabled": True, "state": "scheduled", "deliver": "local",
        "created_at": "2026-08-01T00:00:00+00:00",
        "next_run_at": "2099-01-01T00:00:00+00:00", "last_run_at": "2026-09-01T10:00:00+00:00",
        "last_status": "ok", "last_error": None, "failure_streak": 1,
        "fire_claim": {"by": "host:1", "at": "2026-09-01T10:00:00+00:00"},
        "pending_slot": {"slot": "2026-09-01T12:00:00+00:00"},
    }
    record.update(overrides)
    return record


def test_run_bookkeeping_does_not_rewrite_jobs_json(hermes_env):
    from cron.jobs import create_job, load_jobs, mark_job_run

    job = create_job(prompt="ping", schedule="every 2h", name="pinger")
    before = _jobs_file(hermes_env).read_bytes()

    mark_job_run(job["id"], True)

    assert _jobs_file(hermes_env).read_bytes() == before
    [record] = _disk_records(hermes_env)
    assert "last_status" not in record and "next_run_at" not in record
    assert "completed" not in record["repeat"]
    [merged] = load_jobs()
    assert merged["last_status"] == "ok" and merged["last_run_at"]
    assert _runtime_rows(hermes_env)[job["id"]]["last_status"] == "ok"


def test_combined_store_migrates_without_losing_a_value(hermes_env):
    from cron.jobs import load_jobs

    record = _combined_record()
    _write_combined(hermes_env, [record])

    assert load_jobs() == [record]
    [definition] = _disk_records(hermes_env)
    for key in ("next_run_at", "last_run_at", "last_status", "failure_streak", "fire_claim",
                "pending_slot", "state"):
        assert key not in definition, key
    assert definition["repeat"] == {"times": 5}
    assert definition["prompt"] == "summarize" and definition["created_at"] == record["created_at"]
    row = _runtime_rows(hermes_env)["legacy1"]
    assert row["repeat_completed"] == 2 and row["fire_claim"] == record["fire_claim"]
    assert load_jobs() == [record]


def test_state_written_by_an_older_hermes_wins_over_runtime_db(hermes_env):
    """Downgrade then upgrade: the combined record the older build wrote is the newest state."""
    from cron.jobs import create_job, load_jobs, mark_job_run

    job = create_job(prompt="ping", schedule="every 2h", name="pinger")
    mark_job_run(job["id"], True)
    newer = dict(load_jobs()[0], last_status="error", last_error="boom",
                 repeat={"times": None, "completed": 7})
    _write_combined(hermes_env, [newer])

    [merged] = load_jobs()
    assert merged["last_status"] == "error" and merged["last_error"] == "boom"
    assert merged["repeat"]["completed"] == 7
    assert "last_status" not in _disk_records(hermes_env)[0]


def test_one_hand_added_runtime_key_does_not_discard_the_rest_of_the_row(hermes_env):
    """A stray scheduler key in an otherwise-split record overrides only that field: losing
    repeat.completed would let a capped job fire past its limit."""
    from cron.jobs import create_job, load_jobs, mark_job_run

    job = create_job(prompt="ping", schedule="every 2h", name="capped", repeat=10)
    for _ in range(3):
        mark_job_run(job["id"], True)
    before = load_jobs()[0]
    assert before["repeat"]["completed"] == 3
    records = _disk_records(hermes_env)
    records[0]["paused_reason"] = "hand edit"
    _write_combined(hermes_env, records)

    [merged] = load_jobs()
    assert merged["paused_reason"] == "hand edit"
    assert merged["repeat"]["completed"] == 3
    assert merged["last_run_at"] == before["last_run_at"]
    assert merged["next_run_at"] == before["next_run_at"]
    assert _runtime_rows(hermes_env)[job["id"]]["repeat_completed"] == 3


def test_transient_retry_bookkeeping_stays_out_of_jobs_json(hermes_env):
    from cron import unreachable_retry
    from cron.jobs import create_job, load_jobs, save_jobs

    create_job(prompt="ping", schedule="every 2h")
    before = _jobs_file(hermes_env).read_bytes()
    jobs = load_jobs()
    jobs[0][unreachable_retry.STATE_KEY] = {"attempt": 1}
    save_jobs(jobs)

    assert _jobs_file(hermes_env).read_bytes() == before
    assert load_jobs()[0][unreachable_retry.STATE_KEY] == {"attempt": 1}


def _crash_on_jobs_json_replace(monkeypatch):
    import cron.jobs

    real = cron.jobs.atomic_replace

    def _crash(src, dst, *a, **kw):
        if str(dst).endswith("jobs.json"):
            raise OSError("simulated crash before rename")
        return real(src, dst, *a, **kw)

    monkeypatch.setattr(cron.jobs, "atomic_replace", _crash)


def test_interrupted_save_is_finished_by_the_next_load(hermes_env, monkeypatch):
    from cron.jobs import create_job, load_jobs, update_job
    from cron.runtime_state import load_pending_definitions

    job = create_job(prompt="ping", schedule="every 2h", name="before")
    with monkeypatch.context() as m:
        _crash_on_jobs_json_replace(m)
        with pytest.raises(OSError):
            update_job(job["id"], {"name": "after"})

    assert _disk_records(hermes_env)[0]["name"] == "before"
    assert load_pending_definitions(hermes_env / "cron")[0] is not None

    assert load_jobs()[0]["name"] == "after"
    assert _disk_records(hermes_env)[0]["name"] == "after"
    assert load_pending_definitions(hermes_env / "cron") == (None, None, None)


def test_interrupted_save_yields_to_a_later_edit_of_jobs_json(hermes_env, monkeypatch):
    from cron.jobs import create_job, load_jobs, update_job

    job = create_job(prompt="ping", schedule="every 2h", name="before")
    with monkeypatch.context() as m:
        _crash_on_jobs_json_replace(m)
        with pytest.raises(OSError):
            update_job(job["id"], {"name": "journaled"})
    records = _disk_records(hermes_env)
    records[0]["name"] = "operator edit"
    _write_combined(hermes_env, records)

    assert load_jobs()[0]["name"] == "operator edit"


def test_schedule_edited_outside_the_scheduler_drops_derived_state(hermes_env):
    from cron.jobs import create_job, load_jobs

    job = create_job(prompt="ping", schedule="every 2h", name="pinger")
    assert load_jobs()[0]["next_run_at"]
    records = _disk_records(hermes_env)
    records[0]["schedule"] = {"kind": "interval", "minutes": 5, "display": "every 5m"}
    _write_combined(hermes_env, records)

    [merged] = load_jobs()
    assert merged["id"] == job["id"] and "next_run_at" not in merged


def test_remove_job_deletes_its_runtime_row(hermes_env):
    from cron.jobs import create_job, remove_job

    keep = create_job(prompt="keep", schedule="every 2h")
    gone = create_job(prompt="gone", schedule="every 2h")
    assert remove_job(gone["id"])

    assert set(_runtime_rows(hermes_env)) == {keep["id"]}


def test_stale_save_leaves_rows_it_did_not_change(hermes_env):
    """A writer holding an old snapshot (degraded lock) must not roll back a sibling's newer state
    for a job it never touched."""
    import cron.jobs as jobs
    from cron.runtime_state import write_runtime_states

    a = jobs.create_job(prompt="a", schedule="every 2h")
    b = jobs.create_job(prompt="b", schedule="every 2h")
    with jobs._jobs_lock():
        snapshot = jobs.load_jobs()
        sibling = dict(_runtime_rows(hermes_env)[b["id"]], last_status="ok-from-sibling")
        write_runtime_states(hermes_env / "cron", {b["id"]: sibling})
        for job in snapshot:
            if job["id"] == a["id"]:
                job["last_status"] = "ok-from-writer"
        jobs._save_jobs_unlocked(snapshot)

    rows = _runtime_rows(hermes_env)
    assert rows[a["id"]]["last_status"] == "ok-from-writer"
    assert rows[b["id"]]["last_status"] == "ok-from-sibling"


def test_runtime_db_from_the_earlier_store_revision_upgrades_in_place(hermes_env):
    """#75833's first revision wrote the same tables with a narrower journal and extra row keys."""
    from cron.jobs import load_jobs

    record = {k: v for k, v in _combined_record().items()
              if k in {"id", "name", "prompt", "schedule", "schedule_display", "enabled",
                       "deliver", "created_at"}}
    record["repeat"] = {"times": 1}
    _write_combined(hermes_env, [record])
    with sqlite3.connect(hermes_env / "cron" / "runtime.db") as conn:
        conn.execute("CREATE TABLE job_runtime (job_id TEXT PRIMARY KEY, state_json TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE pending_definitions (singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
            "definitions_json TEXT NOT NULL)")
        conn.execute("INSERT INTO job_runtime VALUES (?, ?)", ("legacy1", json.dumps({
            "last_status": "ok", "repeat_completed": 1, "runtime_tombstone": True,
            "_definition_digest": "d", "_schedule_digest": "s"})))

    [merged] = load_jobs()
    assert merged["state"] == "completed" and merged["enabled"] is False
    assert merged["repeat"] == {"times": 1, "completed": 1} and merged["last_status"] == "ok"
    assert not any(key.startswith("_") or key == "runtime_tombstone" for key in merged)


def test_each_profile_keeps_its_own_runtime_db(hermes_env, tmp_path):
    from cron.jobs import create_job, load_jobs, use_cron_store

    other = tmp_path / "other-profile"
    (other / "cron").mkdir(parents=True)
    with use_cron_store(other):
        job = create_job(prompt="elsewhere", schedule="every 2h")
        assert [j["id"] for j in load_jobs()] == [job["id"]]

    assert (other / "cron" / "runtime.db").exists()
    assert job["id"] not in _runtime_rows(hermes_env)
    assert load_jobs() == []


def test_quick_snapshot_restores_run_state_with_definitions(hermes_env):
    from cron.jobs import create_job, load_jobs, mark_job_run, remove_job
    from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

    job = create_job(prompt="ping", schedule="every 2h")
    mark_job_run(job["id"], True)
    snapshot_id = create_quick_snapshot(hermes_home=hermes_env)
    assert snapshot_id
    remove_job(job["id"])

    assert restore_quick_snapshot(snapshot_id, hermes_home=hermes_env)
    [restored] = load_jobs()
    assert restored["id"] == job["id"] and restored["last_status"] == "ok"
