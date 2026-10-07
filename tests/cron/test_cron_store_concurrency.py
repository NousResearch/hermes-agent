"""Concurrency + guard tests for the cron store (jobs vanishing / run state lost).

Covers the 2026-09-20 incident class: two writers race against one
``jobs.json`` under the degraded flock path (#60703), and a whole-file
read-modify-write from a stale in-memory copy silently reverts the sibling's
work. Two distinct losses:

* a concurrently-created job disappears entirely;
* a concurrently-written *field* (``last_run_at`` and friends) reverts,
  leaving a job that demonstrably ran looking like it never did.

Plus the vanished-job guard's three outcomes (ok / vanished / unavailable).
"""

import json
import os

import pytest

import cron.jobs as jobs_mod
from cron.jobs import create_job, load_jobs, remove_job, save_jobs
from cron.lifecycle_journal import (
    STATUS_OK,
    STATUS_UNAVAILABLE,
    STATUS_VANISHED,
    check_vanished_jobs,
    read_entries,
    record_created,
    record_removed,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


@pytest.fixture()
def degraded_lock(monkeypatch):
    """Force ``_jobs_lock()`` into its in-process-only mode.

    This is the state the live gateway logged repeatedly on the incident day
    ("Timed out after 30s waiting for the cron jobs lock ... proceeding with
    in-process locking only"), and the only state in which two writers can
    both believe they own the store.
    """
    monkeypatch.setattr(jobs_mod, "fcntl", None)
    monkeypatch.setattr(jobs_mod, "msvcrt", None)


def _write_store_directly(tmp_path, jobs):
    """Simulate a SIBLING process writing jobs.json out from under us."""
    path = tmp_path / "cron" / "jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": jobs, "updated_at": "sibling"}),
                    encoding="utf-8")


def _read_store(tmp_path):
    return json.loads((tmp_path / "cron" / "jobs.json").read_text())["jobs"]


def _by_id(jobs, jid):
    return next((j for j in jobs if j.get("id") == jid), None)


class TestConcurrentWriters:
    """A stale writer must not revert a sibling's create or field update."""

    def test_concurrent_create_is_not_clobbered(self, tmp_cron_dir, degraded_lock):
        """Instance-2 shape: a job created between our load and our save."""
        with jobs_mod._jobs_lock():
            stale = load_jobs()  # empty snapshot

            # A sibling process creates a job while we hold our stale copy.
            _write_store_directly(tmp_cron_dir, [
                {"id": "SIBLING", "name": "sibling", "schedule": {"kind": "once"}},
            ])

            stale.append({"id": "OURS", "name": "ours",
                          "schedule": {"kind": "once"}})
            save_jobs(stale)

        ids = {j["id"] for j in _read_store(tmp_cron_dir)}
        assert ids == {"OURS", "SIBLING"}, "a concurrent create was clobbered"

    def test_concurrent_run_state_update_is_not_reverted(
        self, tmp_cron_dir, degraded_lock
    ):
        """Instance-3 shape: the job survives but its run state reverts.

        This is the loss the by-id merge exists for: nobody deleted anything,
        and the pre-existing id-presence merge sees both sides holding the id,
        so without field-level reconciliation the stale copy wins outright.
        """
        _write_store_directly(tmp_cron_dir, [{
            "id": "RAN", "name": "oneshot", "schedule": {"kind": "once"},
            "last_run_at": None, "last_status": None, "state": "scheduled",
        }])

        with jobs_mod._jobs_lock():
            stale = load_jobs()  # last_run_at is None in this snapshot
            assert _by_id(stale, "RAN")["last_run_at"] is None

            # A sibling scheduler completes the run and records it.
            _write_store_directly(tmp_cron_dir, [{
                "id": "RAN", "name": "oneshot", "schedule": {"kind": "once"},
                "last_run_at": "2026-09-20T16:07:28+00:00",
                "last_status": "ok", "state": "completed",
            }])

            # We edit an UNRELATED field from our stale snapshot.
            _by_id(stale, "RAN")["paused_reason"] = "edited concurrently"
            save_jobs(stale)

        persisted = _by_id(_read_store(tmp_cron_dir), "RAN")
        assert persisted["last_run_at"] == "2026-09-20T16:07:28+00:00", \
            "the sibling's run state was reverted by a stale writer"
        assert persisted["last_status"] == "ok"
        assert persisted["state"] == "completed"
        # ...and our own edit still landed.
        assert persisted["paused_reason"] == "edited concurrently"

    def test_our_own_field_edit_wins_over_stale_disk_value(
        self, tmp_cron_dir, degraded_lock
    ):
        """Reconciliation must not invert into "disk always wins"."""
        _write_store_directly(tmp_cron_dir, [
            {"id": "J", "name": "old-name", "schedule": {"kind": "once"}},
        ])
        with jobs_mod._jobs_lock():
            jobs = load_jobs()
            _write_store_directly(tmp_cron_dir, [
                {"id": "J", "name": "old-name", "schedule": {"kind": "once"},
                 "unrelated": 1},
            ])
            _by_id(jobs, "J")["name"] = "new-name"
            save_jobs(jobs)

        persisted = _by_id(_read_store(tmp_cron_dir), "J")
        assert persisted["name"] == "new-name", "our deliberate edit was lost"
        assert persisted["unrelated"] == 1, "the sibling's new field was dropped"

    def test_intentional_field_deletion_is_not_resurrected(
        self, tmp_cron_dir, degraded_lock
    ):
        """Popping a field is a real edit, not an absence to be refilled."""
        _write_store_directly(tmp_cron_dir, [
            {"id": "J", "schedule": {"kind": "once"}, "drift_alerted": True},
        ])
        with jobs_mod._jobs_lock():
            jobs = load_jobs()
            _write_store_directly(tmp_cron_dir, [
                {"id": "J", "schedule": {"kind": "once"}, "drift_alerted": True,
                 "last_run_at": "2026-09-20T16:07:28+00:00"},
            ])
            _by_id(jobs, "J").pop("drift_alerted")
            save_jobs(jobs)

        persisted = _by_id(_read_store(tmp_cron_dir), "J")
        assert "drift_alerted" not in persisted, "a deleted field came back"
        assert persisted["last_run_at"] == "2026-09-20T16:07:28+00:00"

    def test_intentional_removal_still_removes(self, tmp_cron_dir, degraded_lock):
        """The merge must never resurrect a declared deletion."""
        job = create_job(prompt="x", schedule="in 10 hours", name="doomed")
        assert remove_job(job["id"]) is True
        assert _by_id(_read_store(tmp_cron_dir), job["id"]) is None


class TestVanishedJobGuard:
    """expected-vs-present reconciliation, all three outcomes."""

    def test_ok_when_every_removal_is_accounted_for(self, tmp_cron_dir):
        keeper = create_job(prompt="a", schedule="in 10 hours", name="keeper")
        doomed = create_job(prompt="b", schedule="in 10 hours", name="doomed")
        assert remove_job(doomed["id"]) is True

        report = check_vanished_jobs()
        assert report.status == STATUS_OK
        assert report.vanished == []
        assert report.should_alert is False
        assert keeper["id"] in {j["id"] for j in load_jobs()}

    def test_vanished_fires_on_a_synthetic_deletion(self, tmp_cron_dir):
        """Rip a job out behind the store's back — the guard must catch it."""
        job = create_job(prompt="a", schedule="in 10 hours", name="ghost")
        remaining = [j for j in _read_store(tmp_cron_dir) if j["id"] != job["id"]]
        _write_store_directly(tmp_cron_dir, remaining)

        report = check_vanished_jobs()
        assert report.status == STATUS_VANISHED
        assert [v["job_id"] for v in report.vanished] == [job["id"]]
        assert report.vanished[0]["name"] == "ghost"
        assert report.should_alert is True
        assert job["id"] in report.summary()

    def test_unavailable_when_the_journal_cannot_be_read(self, tmp_cron_dir):
        """A guard that cannot see must never report green."""
        create_job(prompt="a", schedule="in 10 hours", name="x")
        journal = tmp_cron_dir / "cron" / "lifecycle.jsonl"
        os.chmod(journal, 0o000)
        try:
            report = check_vanished_jobs()
        finally:
            os.chmod(journal, 0o600)

        assert report.status == STATUS_UNAVAILABLE
        assert report.should_alert is True, \
            "an unreadable journal must not be reported as ok"

    def test_recreated_id_is_not_excused_by_an_older_removal(self, tmp_cron_dir):
        """A stale removal record must not cover a LATER create's loss."""
        _write_store_directly(tmp_cron_dir, [])
        record_removed("REUSED", reason="first life")
        record_created("REUSED", name="second life")

        report = check_vanished_jobs()
        assert report.status == STATUS_VANISHED
        assert [v["job_id"] for v in report.vanished] == ["REUSED"]

    def test_journal_records_creates_and_removals(self, tmp_cron_dir):
        job = create_job(prompt="a", schedule="in 10 hours", name="tracked")
        remove_job(job["id"])

        events = [(e["event"], e["job_id"]) for e in read_entries()]
        assert ("created", job["id"]) in events
        assert ("removed", job["id"]) in events

    def test_journal_survives_a_torn_line(self, tmp_cron_dir):
        """A crash-torn final line must not blind the guard to earlier entries."""
        create_job(prompt="a", schedule="in 10 hours", name="x")
        journal = tmp_cron_dir / "cron" / "lifecycle.jsonl"
        with open(journal, "a", encoding="utf-8") as f:
            f.write('{"event": "created", "job_id": "trunc')

        report = check_vanished_jobs()
        assert report.status == STATUS_OK
        assert report.created_count == 1

    def test_prune_drops_entries_past_retention_and_keeps_recent(self, tmp_cron_dir):
        """Retention must bound the journal without eating live evidence."""
        from cron.lifecycle_journal import prune

        create_job(prompt="a", schedule="in 10 hours", name="recent")
        journal = tmp_cron_dir / "cron" / "lifecycle.jsonl"
        with open(journal, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "created", "job_id": "ANCIENT",
                "at": "2020-01-01T00:00:00+00:00",
            }) + "\n")

        assert prune() == 1
        remaining = journal.read_text()
        assert "ANCIENT" not in remaining
        assert "recent" in remaining


class TestCronStatusSurface:
    """The guard must reach an operator, not just return a dataclass."""

    def test_status_summary_reports_a_vanished_job(self, tmp_cron_dir, capsys):
        from hermes_cli.cron import _print_active_jobs_summary

        job = create_job(prompt="a", schedule="in 10 hours", name="ghost")
        remaining = [j for j in _read_store(tmp_cron_dir) if j["id"] != job["id"]]
        _write_store_directly(tmp_cron_dir, remaining)

        _print_active_jobs_summary([])
        out = capsys.readouterr().out
        assert "VANISHED" in out
        assert job["id"] in out
        assert "ghost" in out

    def test_status_summary_is_quiet_on_a_healthy_store(self, tmp_cron_dir, capsys):
        create_job(prompt="a", schedule="in 10 hours", name="fine")
        from hermes_cli.cron import _print_active_jobs_summary

        _print_active_jobs_summary(load_jobs())
        out = capsys.readouterr().out
        assert "VANISHED" not in out
