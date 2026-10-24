"""#75607 — jobs.json holds declarations; scheduler state lives in cron/runtime.db.

Covers the contracts the split must keep: a fire never rewrites jobs.json, a pre-split (or
downgrade-written) combined store migrates without losing a value, a save interrupted between the
two artifacts is finished (or safely discarded) by the next load, and the degraded-lock writer
cannot roll back rows it never touched.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

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
    return json.loads(_jobs_file(home).read_text(encoding="utf-8-sig"))["jobs"]


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


def test_quota_parked_failure_does_not_rewrite_jobs_json(hermes_env):
    """A fire parked through a closed provider window (cron/quota_hold.py) is per-fire scheduler
    bookkeeping like any other: the hold marker lives in runtime.db, jobs.json stays untouched."""
    import hashlib

    from cron import quota_hold
    from cron.jobs import create_job, load_jobs, mark_job_run

    job = create_job(prompt="ping", schedule="every 2h", name="held")
    before = hashlib.sha256(_jobs_file(hermes_env).read_bytes()).hexdigest()

    assert mark_job_run(job["id"], False, "quota exhausted", quota_hold_seconds=20 * 60 * 60)

    assert hashlib.sha256(_jobs_file(hermes_env).read_bytes()).hexdigest() == before
    [held] = load_jobs()
    assert held[quota_hold.STATE_KEY] and held["next_run_at"] == held[quota_hold.STATE_KEY]
    assert held["enabled"] is True


def _fully_authored_job(home):
    """A job built by create_job with every optional argument set, then fired once."""
    from cron.jobs import create_job, mark_job_run

    (home / "scripts").mkdir(exist_ok=True)
    (home / "scripts" / "collect.py").write_text("print(1)\n", encoding="utf-8")
    (home / "workdir").mkdir(exist_ok=True)
    upstream = create_job(prompt="upstream", schedule="every 1h")
    job = create_job(
        prompt="summarize", schedule="every 2h", name="full", repeat=5, deliver="local",
        origin={"platform": "telegram", "chat_id": "1"}, skills=["a", "b"], model="m",
        provider="openrouter", base_url="https://llm.example/v1", script="collect.py",
        context_from=upstream["id"], enabled_toolsets=["web"], workdir=str(home / "workdir"),
        attach_to_session=True, monitor_url="https://example.com/feed", reasoning_effort="high",
        failure_deliver="local")
    mark_job_run(job["id"], False, "boom")
    return job["id"]


def test_split_keeps_the_definition_schema_in_jobs_json_and_round_trips(hermes_env):
    from cron.constants import DECLARATIVE_JOB_EXTRAS, JOB_DEFINITION_FIELDS
    from cron.jobs import _merge_job, _split_job, get_job

    job = get_job(_fully_authored_job(hermes_env))
    definition, runtime = _split_job(job)

    declarative = JOB_DEFINITION_FIELDS | DECLARATIVE_JOB_EXTRAS
    assert not declarative & set(runtime)
    assert declarative & set(job) <= set(definition)
    assert "completed" not in definition["repeat"]
    assert _merge_job(*_split_job(job)) == job


def test_profile_import_keeps_runtime_state_of_a_split_store(hermes_env):
    """Profile distributions refresh authored fields through load_jobs/save_jobs; the counter and
    the quota hold that live only in runtime.db must survive the import."""
    from cron import quota_hold
    from cron.job_definition import import_job_definitions
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="old prompt", schedule="every 2h", repeat=10)
    assert mark_job_run(job["id"], True)
    assert mark_job_run(job["id"], False, "quota", quota_hold_seconds=20 * 60 * 60)
    held = get_job(job["id"])
    assert "completed" not in _disk_records(hermes_env)[0]["repeat"]

    import_job_definitions(
        {job["id"]: {"prompt": "new prompt", "schedule": held["schedule"], "repeat": {"times": 10}}},
        paused_reason="test")

    after = get_job(job["id"])
    assert after["prompt"] == "new prompt"
    assert after["repeat"] == {"times": 10, "completed": 2}
    assert after[quota_hold.STATE_KEY] == held[quota_hold.STATE_KEY]


def test_orphaned_runtime_row_never_resurrects_a_job(hermes_env):
    """A runtime row whose job left jobs.json (hand delete, hand-edited id) is ignored."""
    from cron.jobs import create_job, load_jobs, mark_job_run, update_job

    keep = create_job(prompt="keep", schedule="every 2h")
    gone = create_job(prompt="gone", schedule="every 2h")
    mark_job_run(gone["id"], True)
    records = [r for r in _disk_records(hermes_env) if r["id"] != gone["id"]]
    records[0]["id"] = "renamed"
    _write_combined(hermes_env, records)

    assert [j["id"] for j in load_jobs()] == ["renamed"]
    update_job("renamed", {"name": "still here"})
    [renamed] = load_jobs()
    assert renamed["id"] == "renamed" and "last_run_at" not in renamed
    assert {r["id"] for r in _disk_records(hermes_env)} == {"renamed"}
    assert keep["id"] in _runtime_rows(hermes_env)  # the orphan is kept, just never merged


def test_schedule_edit_clears_a_quota_hold_kept_in_runtime_db(hermes_env):
    from cron import quota_hold
    from cron.jobs import create_job, get_job, mark_job_run, update_job

    job = create_job(prompt="ping", schedule="every 2h")
    assert mark_job_run(job["id"], False, "quota", quota_hold_seconds=20 * 60 * 60)
    assert quota_hold.STATE_KEY in _runtime_rows(hermes_env)[job["id"]]

    update_job(job["id"], {"schedule": "every 15m"})

    assert quota_hold.STATE_KEY not in _runtime_rows(hermes_env)[job["id"]]
    assert quota_hold.STATE_KEY not in get_job(job["id"])


def test_hand_added_key_moves_to_runtime_db_and_is_never_lost(hermes_env):
    """Keys outside the definition schema are scheduler-side by rule: a key an operator adds to
    jobs.json by hand is moved into runtime.db, still merged back on every load."""
    from cron.jobs import create_job, load_jobs, mark_job_run

    job = create_job(prompt="ping", schedule="every 2h")
    records = _disk_records(hermes_env)
    records[0]["operator_note"] = {"owner": "ops"}
    _write_combined(hermes_env, records)

    assert load_jobs()[0]["operator_note"] == {"owner": "ops"}
    assert "operator_note" not in _disk_records(hermes_env)[0]
    mark_job_run(job["id"], True)
    assert load_jobs()[0]["operator_note"] == {"owner": "ops"}


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


def test_interrupted_removal_is_finished_with_its_runtime_row(hermes_env, monkeypatch):
    from cron.jobs import create_job, load_jobs, mark_job_run, remove_job

    keep = create_job(prompt="keep", schedule="every 2h")
    gone = create_job(prompt="gone", schedule="every 2h")
    mark_job_run(gone["id"], True)
    with monkeypatch.context() as m:
        _crash_on_jobs_json_replace(m)
        with pytest.raises(OSError):
            remove_job(gone["id"])

    assert [j["id"] for j in load_jobs()] == [keep["id"]]
    assert set(_runtime_rows(hermes_env)) == {keep["id"]}


def test_discarded_removal_keeps_the_jobs_runtime_state(hermes_env, monkeypatch):
    """Remove A crashes before the jobs.json rename, then the operator edits B: the newer file
    wins, A stays declared — and must keep its repeat progress and run history with it."""
    from cron.jobs import create_job, load_jobs, mark_job_run, remove_job

    capped = create_job(prompt="capped", schedule="every 2h", repeat=5)
    other = create_job(prompt="other", schedule="every 2h")
    for _ in range(2):
        mark_job_run(capped["id"], True)
    with monkeypatch.context() as m:
        _crash_on_jobs_json_replace(m)
        with pytest.raises(OSError):
            remove_job(capped["id"])
    records = _disk_records(hermes_env)
    for record in records:
        if record["id"] == other["id"]:
            record["name"] = "operator edit"
    _write_combined(hermes_env, records)

    kept = {j["id"]: j for j in load_jobs()}[capped["id"]]
    assert kept["repeat"] == {"times": 5, "completed": 2}
    assert kept["last_status"] == "ok"


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


def _definition_record(job_id="split1"):
    return {k: v for k, v in _combined_record(job_id).items()
            if k in {"id", "name", "prompt", "schedule", "schedule_display", "enabled",
                     "deliver", "created_at"}}


@pytest.mark.parametrize("raw", [
    pytest.param(lambda: json.dumps([]), id="empty-bare-list"),
    pytest.param(lambda: json.dumps([_definition_record()]), id="bare-list"),
    pytest.param(lambda: json.dumps({"jobs": [_definition_record()]}).replace(
        "summarize", "sum\tmarize"), id="raw-control-character"),
])
def test_structural_repair_is_written_once(hermes_env, caplog, raw):
    """A repaired outer shape (bare list, raw control character) must reach jobs.json, or the
    repair is logged again on every load."""
    from cron.jobs import load_jobs

    _jobs_file(hermes_env).write_text(raw(), encoding="utf-8")
    first = load_jobs()

    assert isinstance(json.loads(_jobs_file(hermes_env).read_text(encoding="utf-8-sig")), dict)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cron.jobs"):
        assert load_jobs() == first
    assert not [r for r in caplog.records if "Auto-repaired" in r.getMessage()]


def test_legacy_job_id_record_keeps_identity_and_state(hermes_env):
    """Older writers stored the identity as ``job_id``; migrating such a record must keep that id
    and the scheduler state keyed by it."""
    from cron.jobs import load_jobs

    record = _combined_record()
    record["job_id"] = record.pop("id")
    _write_combined(hermes_env, [record])

    load_jobs()
    [job] = load_jobs()
    assert job["id"] == "legacy1"
    assert job["repeat"] == {"times": 5, "completed": 2}
    assert job["last_status"] == "ok" and job["fire_claim"] == record["fire_claim"]


def test_quick_snapshot_never_splits_a_job_from_its_runtime_row(hermes_env, monkeypatch):
    """A removal landing while the snapshot is being taken must not leave the snapshot holding a
    definition whose runtime row is already gone."""
    import hermes_cli.backup as backup
    from cron.jobs import create_job, load_jobs, mark_job_run, remove_job

    job = create_job(prompt="ping", schedule="every 2h")
    mark_job_run(job["id"], True)
    real_copy = backup._safe_copy_db
    racers = []

    def copy_racing_a_removal(src, dst, *args, **kwargs):
        if Path(src).name == "runtime.db" and not racers:
            racer = threading.Thread(target=remove_job, args=(job["id"],))
            racers.append(racer)
            racer.start()
            racer.join(timeout=2)
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(backup, "_safe_copy_db", copy_racing_a_removal)
    snapshot_id = backup.create_quick_snapshot(hermes_home=hermes_env)
    racers[0].join()
    assert snapshot_id and backup.restore_quick_snapshot(snapshot_id, hermes_home=hermes_env)

    restored = [j for j in load_jobs() if j["id"] == job["id"]]
    assert all(j.get("last_status") == "ok" for j in restored)


# Another process removing a job. Its lock wait is cut from 30 s to 1 s so that "the backup held
# the store lock across a copy longer than the wait" (a multi-GB executions.db) happens in a test:
# the remover then proceeds degraded, exactly as it would after the real timeout.
_REMOVER = ("import sys, cron.jobs as j; j._JOBS_LOCK_TIMEOUT_SECONDS = 1.0; "
            "sys.exit(0 if j.remove_job(sys.argv[1]) else 3)")


def _remove_in_another_process(job_id):
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, "-c", _REMOVER, job_id], cwd=repo_root,
                          env=dict(os.environ), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def _cron_lock_is_free():
    """Whether another thread can take the cron store lock right now (it gives up after 3 s)."""
    import cron.jobs

    def take_and_release():
        with cron.jobs._jobs_lock():
            pass

    probe = threading.Thread(target=take_and_release, daemon=True)
    probe.start()
    probe.join(timeout=3)
    return not probe.is_alive()


def _race_a_removal_with_the_executions_db_copy(home, monkeypatch, job_id):
    """While the backup copies cron/executions.db (a non-pair file), probe the store lock and let
    another process remove ``job_id``. Returns the probe results."""
    import hermes_cli.backup as backup

    sqlite3.connect(home / "cron" / "executions.db").execute(
        "CREATE TABLE IF NOT EXISTS runs (id INTEGER)").connection.close()
    real_copy = backup._safe_copy_db
    lock_free = []

    def slow_executions_copy(src, dst, *args, **kwargs):
        if Path(src).name == "executions.db" and not lock_free:
            lock_free.append(_cron_lock_is_free())
            _remove_in_another_process(job_id)
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(backup, "_safe_copy_db", slow_executions_copy)
    return lock_free


def _assert_pair_consistent(cron_dir):
    """Every job the copied jobs.json declares has its runtime row in the copied runtime.db."""
    from cron.runtime_state import load_runtime_states

    declared = {r["id"] for r in json.loads(
        (cron_dir / "jobs.json").read_text(encoding="utf-8-sig"))["jobs"]}
    rows = load_runtime_states(cron_dir)
    assert declared <= set(rows), f"definitions without runtime rows: {declared - set(rows)}"


def test_quick_snapshot_pair_survives_a_removal_during_a_slow_cron_copy(hermes_env, monkeypatch):
    """A removal by another process while a large non-pair cron file (executions.db) is being
    copied must not leave the snapshot's jobs.json declaring a job whose runtime row is gone, and
    the store lock must not be held across that copy."""
    import hermes_cli.backup as backup
    from cron.jobs import create_job, load_jobs, mark_job_run

    job = create_job(prompt="ping", schedule="every 2h")
    mark_job_run(job["id"], True)
    lock_free = _race_a_removal_with_the_executions_db_copy(hermes_env, monkeypatch, job["id"])

    snapshot_id = backup.create_quick_snapshot(hermes_home=hermes_env)

    assert snapshot_id and [j["id"] for j in load_jobs()] == []
    _assert_pair_consistent(hermes_env / "state-snapshots" / snapshot_id / "cron")
    assert lock_free == [True], "cron store lock held while copying cron/executions.db"


def test_full_backup_pair_survives_a_removal_during_a_slow_cron_copy(
        hermes_env, monkeypatch, tmp_path):
    """Same contract for the full zip backup (pre-update, pre-migration, `hermes backup`): its
    cron/jobs.json and cron/runtime.db must be a pair that existed together."""
    import zipfile

    import hermes_cli.backup as backup
    from cron.jobs import create_job, mark_job_run

    job = create_job(prompt="ping", schedule="every 2h")
    mark_job_run(job["id"], True)
    lock_free = _race_a_removal_with_the_executions_db_copy(hermes_env, monkeypatch, job["id"])
    # A directory listing may return the cron files in any order; take the one that puts the
    # large non-pair file between the two halves of the store.
    real_iter = backup._iter_backup_files
    rank = {"jobs.json": 0, "executions.db": 1, "runtime.db": 2}
    monkeypatch.setattr(backup, "_iter_backup_files", lambda *a, **kw: sorted(
        real_iter(*a, **kw), key=lambda entry: rank.get(entry[1].name, 3)))

    archive = backup.create_pre_update_backup(hermes_home=hermes_env)

    assert archive is not None
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted, members=["cron/jobs.json", "cron/runtime.db"])
    _assert_pair_consistent(extracted / "cron")
    assert lock_free == [True], "cron store lock held while archiving cron/executions.db"


def test_update_recovery_brings_back_run_state_of_restored_jobs(hermes_env):
    """The post-update safety net restores lost job definitions from the pre-update snapshot; with
    the split store it must bring back their scheduler state too (repeat progress, run history),
    while a job that survived keeps its newer live state."""
    from cron.jobs import create_job, get_job, mark_job_run, remove_job
    from hermes_cli.backup import create_quick_snapshot, restore_cron_jobs_if_emptied

    capped = create_job(prompt="capped", schedule="every 2h", repeat=5)
    survivor = create_job(prompt="survivor", schedule="every 2h")
    for _ in range(2):
        mark_job_run(capped["id"], True)
    mark_job_run(survivor["id"], True)
    before = get_job(capped["id"])
    snapshot_id = create_quick_snapshot(label="pre-update", hermes_home=hermes_env, keep=5)
    remove_job(capped["id"])  # lost during the update: definition and runtime row both gone
    mark_job_run(survivor["id"], False, "boom")  # newer than the snapshot

    assert restore_cron_jobs_if_emptied(snapshot_id, hermes_home=hermes_env)

    restored = get_job(capped["id"])
    assert restored["repeat"] == {"times": 5, "completed": 2}
    assert restored["last_run_at"] == before["last_run_at"]
    assert restored["last_status"] == "ok"
    live = get_job(survivor["id"])
    assert live["last_status"] == "error" and live["last_error"] == "boom"



def test_full_backup_takes_runtime_db_created_by_a_migration_after_the_scan(
        hermes_env, monkeypatch, tmp_path):
    """A pre-split store scanned before a concurrent load migrates it: by copy time its jobs.json
    is stripped and the state lives in a runtime.db the scan never listed. The archive must still
    carry that state, not the stripped definitions alone."""
    import zipfile

    import hermes_cli.backup as backup
    from cron.jobs import load_jobs
    from cron.runtime_state import load_runtime_states

    _write_combined(hermes_env, [_combined_record()])
    real_iter = backup._iter_backup_files

    def scan_then_migrate(*args, **kwargs):
        scanned = list(real_iter(*args, **kwargs))
        load_jobs()  # another process's first load of the new code migrates the store
        return scanned

    monkeypatch.setattr(backup, "_iter_backup_files", scan_then_migrate)

    archive = backup.create_pre_update_backup(hermes_home=hermes_env)

    assert archive is not None
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as zf:
        assert "cron/runtime.db" in zf.namelist()
        zf.extractall(extracted, members=["cron/jobs.json", "cron/runtime.db"])
    _assert_pair_consistent(extracted / "cron")
    assert load_runtime_states(extracted / "cron")["legacy1"]["repeat_completed"] == 2


def test_update_recovery_with_nothing_to_fold_writes_under_the_store_lock(
        hermes_env, monkeypatch):
    """When every restored job still has its live runtime row there is nothing to fold, but the
    restored jobs.json must still be written under the store lock: a removal slipping in between
    would otherwise have its definition brought back without its runtime row."""
    import hermes_cli.backup as backup
    import cron.jobs as jobs
    from cron.jobs import create_job, get_job, mark_job_run

    lost = create_job(prompt="lost", schedule="every 2h")
    other = create_job(prompt="other", schedule="every 2h")
    mark_job_run(lost["id"], True)
    mark_job_run(other["id"], True)
    snapshot_id = backup.create_quick_snapshot(label="pre-update", hermes_home=hermes_env, keep=5)
    # The update drops one definition; both runtime rows survive, so nothing needs folding.
    _write_combined(hermes_env, [r for r in _disk_records(hermes_env) if r["id"] != lost["id"]])

    live_jobs_json = _jobs_file(hermes_env).resolve()
    removers = []

    def remove_other_while_restoring(dst):
        if Path(dst).resolve() == live_jobs_json and not removers:
            remover = threading.Thread(target=jobs.remove_job, args=(other["id"],), daemon=True)
            removers.append(remover)
            remover.start()
            remover.join(timeout=2)  # finishes at once unless the restore holds the store lock

    real_copy2 = backup.shutil.copy2
    real_write = jobs._write_definitions_file

    def copy2(src, dst, *args, **kwargs):
        remove_other_while_restoring(dst)
        return real_copy2(src, dst, *args, **kwargs)

    def write_definitions(path, definitions):
        remove_other_while_restoring(path)
        return real_write(path, definitions)

    monkeypatch.setattr(backup.shutil, "copy2", copy2)
    monkeypatch.setattr(jobs, "_write_definitions_file", write_definitions)

    assert backup.restore_cron_jobs_if_emptied(snapshot_id, hermes_home=hermes_env)
    assert removers
    removers[0].join(timeout=60)
    assert not removers[0].is_alive()

    _assert_pair_consistent(hermes_env / "cron")
    assert get_job(lost["id"])["last_status"] == "ok"

@pytest.mark.platforms("posix")
def test_root_writer_hands_runtime_db_to_the_store_owner(hermes_env, monkeypatch):
    """A root CLI command migrating another user's store must not leave runtime.db root-owned 0600:
    the unprivileged gateway could then no longer load any job (#68483 for jobs.json)."""
    import cron.jobs as jobs

    _write_combined(hermes_env, [_combined_record()])
    cron_dir = (hermes_env / "cron").resolve()
    owned = {cron_dir, (cron_dir / "jobs.json")}
    real_stat = os.stat

    class _OwnedByGateway:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_uid = 1000
            self.st_gid = 1000

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if isinstance(path, (str, os.PathLike)) and Path(path).resolve() in owned:
            return _OwnedByGateway(result)
        return result

    chowned = []
    monkeypatch.setattr(jobs.os, "stat", fake_stat)
    monkeypatch.setattr(jobs.os, "geteuid", lambda: 0)
    monkeypatch.setattr(jobs.os, "getegid", lambda: 0)
    monkeypatch.setattr(jobs.os, "chown", lambda p, uid, gid: chowned.append((Path(p).resolve(), uid, gid)))

    jobs.load_jobs()

    assert (cron_dir / "runtime.db", 1000, 1000) in chowned


def _runtime_only_save(home):
    """A save that changes only scheduler state (the per-fire advance), after one normal save."""
    import cron.jobs as jobs

    record = _combined_record()
    jobs.save_jobs([record])
    before = _jobs_file(home).read_bytes()
    record["last_error"] = "x" * 200_000  # needs new pages in runtime.db
    return jobs, record, before


def test_full_runtime_db_raises_enospc_so_the_store_degrades(hermes_env, monkeypatch):
    """A full disk fails the runtime.db commit with SQLITE_FULL. The scheduler degrades an unwritable
    store (cron/store_health.py) only on OSError, so the commit must surface as ENOSPC like the
    jobs.json write beside it, not as a sqlite3 error that fails the whole tick."""
    import hermes_cli.sqlite_util as sqlite_util
    from cron import store_health

    jobs, record, before = _runtime_only_save(hermes_env)
    real_open_db = sqlite_util.open_db

    def full_disk_open_db(*args, **kwargs):
        conn = real_open_db(*args, **kwargs)
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        conn.execute(f"PRAGMA max_page_count = {pages}")
        return conn

    monkeypatch.setattr(sqlite_util, "open_db", full_disk_open_db)
    with pytest.raises(OSError) as raised:
        jobs.save_jobs([record])
    assert raised.value.errno in store_health.UNWRITABLE_ERRNOS
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    assert _jobs_file(hermes_env).read_bytes() == before


@pytest.mark.platforms("posix")
def test_read_only_runtime_db_raises_an_unwritable_errno(hermes_env):
    from cron import store_health

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root writes through file modes")
    jobs, record, _ = _runtime_only_save(hermes_env)
    db = hermes_env / "cron" / "runtime.db"
    for path in (db, Path(f"{db}-wal"), Path(f"{db}-shm")):
        if path.exists():
            path.chmod(0o444)
    with pytest.raises(OSError) as raised:
        jobs.save_jobs([record])
    assert raised.value.errno in store_health.UNWRITABLE_ERRNOS
    assert isinstance(raised.value.__cause__, sqlite3.Error)
