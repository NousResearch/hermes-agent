"""The local missed-run policy preserves grace and manual triggers."""
import threading
from datetime import timedelta

import pytest

from cron import jobs


def _claim_and_record(due_job):
    claimed = jobs.claim_job_for_fire(due_job["id"], return_job=True)
    assert isinstance(claimed, dict)
    for field in ("_misfire_event", "_count_catch_up_occurrence"):
        if field in due_job:
            claimed[field] = due_job[field]
    assert jobs.record_claimed_misfire(claimed)


@pytest.mark.parametrize("catch_up", [True, False])
def test_missed_policy_preserves_grace_and_manual(tmp_path, monkeypatch, catch_up):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"cron:\n  catch_up_missed: {str(catch_up).lower()}\n", encoding="utf-8")
    with jobs.use_cron_store(tmp_path / "cron"):
        now = jobs._hermes_now()
        for name, lag in [("stale", 14400), ("grace", 30), ("manual", 14400)]:
            job = jobs.create_job(prompt=name, schedule="every 1h", model="fixture", deliver="local")
            stored = jobs.load_jobs()
            row = next(row for row in stored if row["id"] == job["id"])
            row["next_run_at"] = (now - timedelta(seconds=lag)).isoformat()
            if name == "manual":
                row["manual_run_at"] = row["next_run_at"]
            jobs.save_jobs(stored)
        due = {job["prompt"] for job in jobs.get_due_jobs()}
        assert due == ({"stale", "grace", "manual"} if catch_up else {"grace", "manual"})
        stale = next(row for row in jobs.load_jobs() if row["prompt"] == "stale")
        assert jobs._ensure_aware(jobs.datetime.fromisoformat(stale["next_run_at"])) > now


@pytest.mark.parametrize("uncomputable", [False, True])
def test_default_and_uncomputable_still_catch_up(tmp_path, monkeypatch, uncomputable):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if uncomputable:
        (tmp_path / "config.yaml").write_text("cron:\n  catch_up_missed: false\n", encoding="utf-8")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="default", schedule="every 1h", model="fixture", deliver="local")
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(hours=4)).isoformat()
        jobs.save_jobs(stored)
        if uncomputable:
            monkeypatch.setattr(jobs, "compute_next_run", lambda *args: None)
        assert [row["id"] for row in jobs.get_due_jobs()] == [job["id"]]


@pytest.mark.parametrize("catch_up", [True, False])
def test_per_job_policy_overrides_global_and_audits_decision(tmp_path, monkeypatch, catch_up):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"cron:\n  catch_up_missed: {str(not catch_up).lower()}\n", encoding="utf-8")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(
            prompt="override", schedule="every 1h", model="fixture", deliver="local",
            catch_up=catch_up)
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(hours=4)).isoformat()
        jobs.save_jobs(stored)

        due = jobs.get_due_jobs()

        assert bool(due) is catch_up
        if catch_up:
            assert "last_misfire" not in jobs.load_jobs()[0]
            _claim_and_record(due[0])
        event = jobs.load_jobs()[0]["last_misfire"]
        assert event["action"] == ("ran" if catch_up else "skipped")
        assert event["policy_source"] == "job"
        audit = (jobs._current_cron_store().cron_dir / "misfires.jsonl").read_text(encoding="utf-8")
        assert f'"job_id": "{job["id"]}"' in audit


def test_per_job_grace_controls_lateness_and_is_audited(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(
            prompt="grace", schedule="every 1h", model="fixture", deliver="local",
            catch_up=False, misfire_grace_seconds=5 * 60 * 60)
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(hours=4)).isoformat()
        jobs.save_jobs(stored)

        due = jobs.get_due_jobs()

        assert [row["id"] for row in due] == [job["id"]]
        _claim_and_record(due[0])
        event = jobs.load_jobs()[0]["last_misfire"]
        assert event["action"] == "ran"
        assert event["grace_seconds"] == 5 * 60 * 60


def test_small_grace_catch_up_is_audited_after_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(
            prompt="small grace", schedule="every 1h", model="fixture", deliver="local",
            catch_up=True, misfire_grace_seconds=0)
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(seconds=30)).isoformat()
        jobs.save_jobs(stored)

        due = jobs.get_due_jobs()
        assert due[0]["last_dispatch"]["kind"] == "catch_up"
        _claim_and_record(due[0])

        event = jobs.load_jobs()[0]["last_misfire"]
        assert event["grace_seconds"] == 0
        assert event["lateness_seconds"] < jobs._LATE_DISPATCH_TOLERANCE_SECONDS


def test_failed_claim_is_not_audited_and_restored_slot_records_once(tmp_path, monkeypatch):
    from cron import scheduler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with jobs.use_cron_store(tmp_path / "cron"):
        audit_path = jobs._current_cron_store().cron_dir / "misfires.jsonl"
        job = jobs.create_job(
            prompt="restore", schedule="every 1h", model="fixture", deliver="local",
            misfire_grace_seconds=0)
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = (jobs._hermes_now() - timedelta(minutes=10)).isoformat()
        jobs.save_jobs(stored)

        first = jobs.get_due_jobs()[0]
        assert jobs.get_catch_up_occurrence_count() == 0
        with monkeypatch.context() as failed_claim:
            failed_claim.setattr(scheduler, "claim_job_for_fire", lambda *args, **kwargs: False)
            failed_claim.setattr(scheduler, "finish_execution", lambda *args, **kwargs: None)
            assert scheduler._process_due_job(
                dict(first, execution_id="failed"), None, None, False)

        assert "last_misfire" not in jobs.get_job(job["id"])
        assert jobs.get_catch_up_occurrence_count() == 0
        assert not audit_path.exists()

        restored = jobs.get_due_jobs()[0]
        _claim_and_record(restored)
        assert jobs.get_catch_up_occurrence_count() == 1
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1


def test_concurrent_claimed_catchups_increment_counter_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with jobs.use_cron_store(tmp_path / "cron"):
        first_write_entered = threading.Event()
        release_first_write = threading.Event()
        real_atomic_write = jobs.atomic_write_text
        calls = 0

        def delayed_atomic_write(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_write_entered.set()
                assert release_first_write.wait(timeout=2)
            return real_atomic_write(*args, **kwargs)

        def record_claimed_catch_up():
            with jobs.use_cron_store(tmp_path / "cron"):
                assert jobs.record_claimed_misfire({"_count_catch_up_occurrence": True})

        monkeypatch.setattr(jobs, "atomic_write_text", delayed_atomic_write)
        first = threading.Thread(target=record_claimed_catch_up)
        second = threading.Thread(target=record_claimed_catch_up)

        first.start()
        assert first_write_entered.wait(timeout=2)
        second.start()
        release_first_write.set()
        first.join(timeout=2)
        second.join(timeout=2)

        assert not first.is_alive()
        assert not second.is_alive()
        assert jobs.get_catch_up_occurrence_count() == 2


def test_oneshot_rejects_misfire_overrides_and_reports_fixed_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    run_at = (jobs._hermes_now() + timedelta(hours=1)).isoformat()
    with jobs.use_cron_store(tmp_path / "cron"):
        with pytest.raises(ValueError, match="only supported for recurring"):
            jobs.create_job(prompt="once", schedule=run_at, catch_up=True)
        with pytest.raises(ValueError, match="only supported for recurring"):
            jobs.create_job(prompt="once", schedule=run_at, misfire_grace_seconds=3600)

        legacy = jobs.create_job(prompt="legacy once", schedule=run_at)
        legacy["catch_up"] = True
        legacy["misfire_grace_seconds"] = 3600

        assert jobs.resolve_job_misfire_policy(legacy) == {
            "catch_up": False,
            "catch_up_source": "one-shot",
            "misfire_grace_seconds": jobs.ONESHOT_GRACE_SECONDS,
            "misfire_grace_source": "one-shot",
        }


def test_schedule_edits_enforce_recurring_only_misfire_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    run_at = (jobs._hermes_now() + timedelta(hours=1)).isoformat()
    with jobs.use_cron_store(tmp_path / "cron"):
        recurring = jobs.create_job(
            prompt="switch", schedule="every 1h", catch_up=False,
            misfire_grace_seconds=60)

        with pytest.raises(ValueError, match="only supported for recurring"):
            jobs.update_job(recurring["id"], {"schedule": run_at})

        switched = jobs.update_job(recurring["id"], {
            "schedule": run_at,
            "catch_up": None,
            "misfire_grace_seconds": None,
        })
        assert switched["schedule"]["kind"] == "once"
        assert "catch_up" not in switched
        assert "misfire_grace_seconds" not in switched

        with pytest.raises(ValueError, match="only supported for recurring"):
            jobs.update_job(switched["id"], {"catch_up": True})

        recurring_again = jobs.update_job(switched["id"], {
            "schedule": "every 1h",
            "catch_up": True,
            "misfire_grace_seconds": 60,
        })
        assert recurring_again["catch_up"] is True
        assert recurring_again["misfire_grace_seconds"] == 60
