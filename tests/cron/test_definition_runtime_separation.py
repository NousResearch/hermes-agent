"""Contracts for declarative cron definitions and volatile runtime state."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

import pytest

import cron.jobs as jobs


_RUNTIME_FIELDS = {
    "created_at",
    "fire_claim",
    "last_delivery_error",
    "last_error",
    "last_run_at",
    "last_status",
    "model_snapshot",
    "next_run_at",
    "paused_at",
    "provider_snapshot",
    "run_claim",
    "state",
}


@pytest.fixture()
def cron_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Route cron definition and runtime storage to a temporary profile."""
    home = tmp_path / "profile"
    cron_dir = home / "cron"
    monkeypatch.setattr(jobs, "HERMES_DIR", home)
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(
        jobs,
        "_compute_provider_model_snapshots",
        lambda **_kwargs: ("provider-at-create", "model-at-create"),
    )
    yield home


def _raw_definitions(home: Path) -> list[dict]:
    """Read only the persisted definitions from a temporary profile."""
    data = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))
    return data["jobs"]


def test_recurring_run_does_not_rewrite_definition_artifact(cron_store: Path) -> None:
    """Execution metadata and counters must change outside jobs.json."""
    created = jobs.create_job(
        prompt="check health",
        schedule="every 1h",
        name="health",
        repeat=3,
        deliver="local",
    )
    definitions_path = cron_store / "cron" / "jobs.json"
    before = definitions_path.read_bytes()

    jobs.mark_job_run(created["id"], True)

    assert definitions_path.read_bytes() == before
    loaded = jobs.get_job(created["id"])
    assert loaded is not None
    assert loaded["last_status"] == "ok"
    assert loaded["last_run_at"]
    assert loaded["next_run_at"]
    assert loaded["repeat"] == {"times": 3, "completed": 1}
    assert (cron_store / "cron" / "runtime.db").exists()


def test_legacy_combined_store_migrates_without_losing_runtime(
    cron_store: Path,
) -> None:
    """Migration must preserve cadence, counters, and an active fire lease."""
    cron_dir = cron_store / "cron"
    cron_dir.mkdir(parents=True)
    now = jobs._hermes_now().isoformat()
    legacy = {
        "id": "legacy-job",
        "name": "legacy",
        "prompt": "run",
        "schedule": {"kind": "interval", "minutes": 30, "display": "every 30m"},
        "schedule_display": "every 30m",
        "repeat": {"times": 5, "completed": 2},
        "enabled": True,
        "state": "scheduled",
        "created_at": now,
        "next_run_at": now,
        "last_run_at": now,
        "last_status": "error",
        "last_error": "prior failure",
        "last_delivery_error": "prior delivery failure",
        "fire_claim": {"at": now, "by": "live-owner"},
        "run_claim": None,
        "paused_at": None,
        "provider_snapshot": "provider-at-create",
        "model_snapshot": "model-at-create",
        "deliver": "local",
    }
    (cron_dir / "jobs.json").write_text(
        json.dumps({"jobs": [legacy], "updated_at": now}),
        encoding="utf-8",
    )

    migrated = jobs.load_jobs()

    assert len(migrated) == 1
    assert migrated[0]["next_run_at"] == now
    assert migrated[0]["last_status"] == "error"
    assert migrated[0]["fire_claim"] == {"at": now, "by": "live-owner"}
    assert migrated[0]["repeat"] == {"times": 5, "completed": 2}
    definition = _raw_definitions(cron_store)[0]
    assert not (_RUNTIME_FIELDS & definition.keys())
    assert definition["repeat"] == {"times": 5}
    assert (cron_dir / "runtime.db").exists()


def test_migration_retries_after_definition_write_failure(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after the runtime commit must leave a lossless retry path."""
    cron_dir = cron_store / "cron"
    cron_dir.mkdir(parents=True)
    now = jobs._hermes_now().isoformat()
    legacy = {
        "id": "retry-job",
        "prompt": "retry",
        "schedule": {"kind": "interval", "minutes": 15},
        "repeat": {"times": 3, "completed": 1},
        "enabled": True,
        "next_run_at": now,
        "fire_claim": {"at": now, "by": "owner-before-crash"},
    }
    jobs_path = cron_dir / "jobs.json"
    jobs_path.write_text(json.dumps({"jobs": [legacy]}), encoding="utf-8")
    original_write = jobs._write_job_definitions_unlocked

    def fail_definition_write(_definitions: list[dict]) -> None:
        """Simulate termination after runtime.db commits."""
        raise OSError("simulated definition write failure")

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", fail_definition_write)
    with pytest.raises(OSError, match="simulated definition write failure"):
        jobs.load_jobs()

    still_legacy = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"][0]
    assert still_legacy["next_run_at"] == now
    assert still_legacy["fire_claim"]["by"] == "owner-before-crash"
    assert (cron_dir / "runtime.db").exists()

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", original_write)
    recovered = jobs.load_jobs()[0]
    assert recovered["next_run_at"] == now
    assert recovered["fire_claim"]["by"] == "owner-before-crash"
    assert recovered["repeat"] == {"times": 3, "completed": 1}
    assert "next_run_at" not in _raw_definitions(cron_store)[0]


def test_definition_export_excludes_runtime_and_timestamps(cron_store: Path) -> None:
    """Stable export must contain operator intent only."""
    created = jobs.create_job(
        prompt="report",
        schedule="every 2h",
        name="report",
        repeat=4,
        deliver="local",
    )
    jobs.mark_job_run(created["id"], False, "transient failure")

    exported = jobs.export_job_definitions()

    assert len(exported) == 1
    definition = exported[0]
    assert not (_RUNTIME_FIELDS & definition.keys())
    assert definition["repeat"] == {"times": 4}
    assert definition["id"] == created["id"]
    assert definition["prompt"] == "report"


def test_definition_only_reconcile_preserves_existing_runtime(cron_store: Path) -> None:
    """Reapplying stable definitions must not erase cadence or run history."""
    created = jobs.create_job(
        prompt="reconcile",
        schedule="every 3h",
        name="reconcile",
        repeat=4,
        deliver="local",
    )
    jobs.mark_job_run(created["id"], True)
    definitions = jobs.export_job_definitions()
    before = jobs.get_job(created["id"])
    assert before is not None

    jobs.save_jobs(definitions)

    after = jobs.get_job(created["id"])
    assert after is not None
    assert after["next_run_at"] == before["next_run_at"]
    assert after["last_run_at"] == before["last_run_at"]
    assert after["last_status"] == "ok"
    assert after["repeat"] == {"times": 4, "completed": 1}


def test_journal_recovers_interrupted_ordinary_definition_update(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed JSON materialization must roll forward on the next read."""
    created = jobs.create_job(
        prompt="before",
        schedule="every 1h",
        name="journal",
        deliver="local",
    )
    original_write = jobs._write_job_definitions_unlocked
    definitions_path = cron_store / "cron" / "jobs.json"
    before = definitions_path.read_bytes()
    merged = jobs.load_jobs()
    merged[0]["prompt"] = "after"

    def fail_definition_write(_definitions: list[dict]) -> None:
        """Simulate interruption after the SQLite journal commits."""
        raise OSError("simulated materialization failure")

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", fail_definition_write)
    with pytest.raises(OSError, match="simulated materialization failure"):
        jobs.save_jobs(merged)
    assert definitions_path.read_bytes() == before

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", original_write)
    recovered = jobs.get_job(created["id"])
    assert recovered is not None
    assert recovered["prompt"] == "after"
    assert _raw_definitions(cron_store)[0]["prompt"] == "after"


def test_journal_recovers_interrupted_definition_delete(
    cron_store: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupted explicit deletion must roll forward, never resurrect runtime."""
    created = jobs.create_job(
        prompt="delete",
        schedule="every 1h",
        name="delete",
        deliver="local",
    )
    original_write = jobs._write_job_definitions_unlocked

    def fail_definition_write(_definitions: list[dict]) -> None:
        """Simulate interruption after deletion is journaled."""
        raise OSError("simulated delete materialization failure")

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", fail_definition_write)
    with pytest.raises(OSError, match="simulated delete materialization failure"):
        jobs.remove_job(created["id"])

    monkeypatch.setattr(jobs, "_write_job_definitions_unlocked", original_write)
    assert jobs.get_job(created["id"]) is None
    assert _raw_definitions(cron_store) == []


def test_runtime_database_is_owner_only(cron_store: Path) -> None:
    """Volatile state must retain the jobs store's owner-only mode."""
    jobs.create_job(
        prompt="secure",
        schedule="every 1h",
        name="secure",
        deliver="local",
    )
    mode = (cron_store / "cron" / "runtime.db").stat().st_mode & 0o777
    assert mode == 0o600


def test_schedule_reconcile_resets_stale_cadence_and_claims(cron_store: Path) -> None:
    """Source-controlled cadence changes must not inherit old leases/counters."""
    created = jobs.create_job(
        prompt="cadence",
        schedule="every 1h",
        name="cadence",
        repeat=5,
        deliver="local",
    )
    jobs.mark_job_run(created["id"], True)
    definitions = jobs.export_job_definitions()
    definitions[0]["schedule"] = {
        "kind": "interval",
        "minutes": 10,
        "display": "every 10m",
    }
    definitions[0]["schedule_display"] = "every 10m"
    definitions[0]["repeat"] = {"times": 2}

    jobs.save_jobs(definitions)

    reconciled = jobs.get_job(created["id"])
    assert reconciled is not None
    assert reconciled["repeat"] == {"times": 2, "completed": 0}
    assert reconciled.get("next_run_at") is None
    assert reconciled.get("fire_claim") is None
    assert reconciled.get("run_claim") is None
    assert reconciled["state"] == "scheduled"
    assert reconciled["last_status"] == "ok"


def test_terminal_one_shot_tombstones_without_definition_write(
    cron_store: Path,
) -> None:
    """Terminal lifecycle must hide the job but retain reproducible intent."""
    created = jobs.create_job(
        prompt="once",
        schedule="1m",
        name="once",
        deliver="local",
    )
    definitions_path = cron_store / "cron" / "jobs.json"
    before = definitions_path.read_bytes()

    jobs.mark_job_run(created["id"], True)

    assert definitions_path.read_bytes() == before
    assert jobs.get_job(created["id"]) is None
    assert jobs.list_jobs(include_disabled=True) == []
    internal = jobs.load_jobs()
    assert internal[0]["runtime_tombstone"]["reason"] == "repeat_limit"
    assert jobs.export_job_definitions()[0]["id"] == created["id"]


def test_missing_definitions_fail_closed_without_erasing_runtime(
    cron_store: Path,
) -> None:
    """Transient JSON absence cannot destroy cadence or terminal tombstones."""
    from cron.runtime_state import load_runtime_states

    created = jobs.create_job(
        prompt="once",
        schedule="1m",
        name="preserve-on-missing",
        deliver="local",
    )
    jobs_path = cron_store / "cron" / "jobs.json"
    definitions_bytes = jobs_path.read_bytes()
    jobs.mark_job_run(created["id"], True)
    runtime_before = load_runtime_states(cron_store / "cron")
    assert runtime_before[created["id"]]["runtime_tombstone"]

    jobs_path.unlink()
    with pytest.raises(RuntimeError, match="jobs.json is missing"):
        jobs.load_jobs()
    assert load_runtime_states(cron_store / "cron") == runtime_before

    jobs_path.write_bytes(definitions_bytes)
    recovered = jobs.load_jobs()
    assert recovered[0]["runtime_tombstone"]["reason"] == "repeat_limit"


def test_missing_definitions_are_valid_for_true_first_run(cron_store: Path) -> None:
    """No JSON and no runtime rows still represents an empty first-run store."""
    assert jobs.load_jobs() == []


def test_large_runtime_snapshot_avoids_sql_variable_limit(cron_store: Path) -> None:
    """Complete snapshots must work above common 999-variable SQLite limits."""
    records = [
        {
            "id": f"job-{index}",
            "prompt": "bulk",
            "schedule": {"kind": "interval", "minutes": 60},
            "repeat": {"times": None, "completed": index},
            "enabled": True,
            "state": "scheduled",
            "next_run_at": jobs._hermes_now().isoformat(),
        }
        for index in range(1_100)
    ]

    jobs.save_jobs(records)

    assert len(jobs.load_jobs()) == 1_100
    assert len(_raw_definitions(cron_store)) == 1_100


def test_runtime_state_is_profile_local(tmp_path: Path) -> None:
    """Identical job IDs in two profiles must never share runtime state."""
    job_id = "shared-id"
    definition = {
        "id": job_id,
        "name": "shared",
        "prompt": "run",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 1h"},
        "schedule_display": "every 1h",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "deliver": "local",
    }
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"

    with jobs.use_cron_store(home_a):
        jobs.save_jobs([definition])
        jobs.mark_job_run(job_id, True)
    with jobs.use_cron_store(home_b):
        jobs.save_jobs([definition])
        jobs.mark_job_run(job_id, False, "profile-b-only")

    with jobs.use_cron_store(home_a):
        loaded_a = jobs.get_job(job_id)
        assert loaded_a is not None
        assert loaded_a["last_status"] == "ok"
    with jobs.use_cron_store(home_b):
        loaded_b = jobs.get_job(job_id)
        assert loaded_b is not None
        assert loaded_b["last_status"] == "error"
        assert loaded_b["last_error"] == "profile-b-only"


def test_concurrent_fire_claims_remain_atomic_without_definition_write(
    cron_store: Path,
) -> None:
    """Exactly one contender wins while jobs.json remains byte-stable."""
    created = jobs.create_job(
        prompt="claim",
        schedule="every 1h",
        name="claim",
        deliver="local",
    )
    definitions_path = cron_store / "cron" / "jobs.json"
    before = definitions_path.read_bytes()
    barrier = threading.Barrier(3)
    results: list[bool] = []

    def contend() -> None:
        """Attempt one fire claim after both contenders are ready."""
        barrier.wait()
        results.append(jobs.claim_job_for_fire(created["id"]))

    threads = [threading.Thread(target=contend) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(results) == [False, True]
    assert definitions_path.read_bytes() == before
