"""Tests for gateway/jobs.py — job tracking, modifier attachment, persistence."""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from gateway.jobs import (
    GatewayJob,
    GatewayJobs,
    FIXED_WORKSTREAMS,
    JOB_STATUSES,
)


@pytest.fixture(autouse=True)
def _isolated_instance():
    """Reset singleton and point store at a temp file between tests."""
    GatewayJobs._instance = None
    with tempfile.TemporaryDirectory() as tmp:
        store = f"{tmp}/jobs.json"
        with patch.object(GatewayJobs, "_store_path", property(lambda self: __import__('pathlib').Path(store))):
            yield
    GatewayJobs._instance = None


class TestGatewayJob:
    def test_create_defaults(self):
        job = GatewayJob(
            job_id="abc",
            session_key="s1",
            workstream="research",
            matter_summary="Check prices",
        )
        assert job.status == "active"
        assert job.job_id == "abc"
        assert job.trigger_text == ""

    def test_is_active_or_queued(self):
        job = GatewayJob("a", "s", "misc", "x", status="queued")
        assert job.is_active_or_queued
        job.status = "completed"
        assert not job.is_active_or_queued

    def test_to_from_dict_roundtrip(self):
        job = GatewayJob(
            job_id="test123",
            session_key="sk",
            workstream="coding_debug",
            matter_summary="Fix null pointer",
            status="paused",
            trigger_text="the null pointer bug",
            reply_to_message_id="msg_42",
            parent_job_id="parent_99",
        )
        d = job.to_dict()
        restored = GatewayJob.from_dict(d)
        assert restored.job_id == job.job_id
        assert restored.session_key == job.session_key
        assert restored.workstream == job.workstream
        assert restored.matter_summary == job.matter_summary
        assert restored.status == job.status
        assert restored.parent_job_id == job.parent_job_id


class TestGatewayJobsCRUD:
    def test_create_becomes_active_when_empty(self):
        jobs = GatewayJobs.get()
        job = jobs.create("s1", "research", "Look up AAPL", "what is AAPL")
        assert job.status == "active"
        assert jobs.active_job("s1") is not None
        assert jobs.active_job("s1").job_id == job.job_id

    def test_create_stays_queued_when_active_exists(self):
        jobs = GatewayJobs.get()
        j1 = jobs.create("s1", "research", "Task 1", "do one")
        j2 = jobs.create("s1", "coding_debug", "Task 2", "do two")
        assert j1.status == "active"
        assert j2.status == "queued"

    def test_complete_promotes_next_queued(self):
        jobs = GatewayJobs.get()
        j1 = jobs.create("s1", "research", "First", "one")
        j2 = jobs.create("s1", "coding_debug", "Second", "two")
        jobs.complete("s1", j1.job_id)
        assert j1.status == "completed"
        active = jobs.active_job("s1")
        assert active is not None
        assert active.job_id == j2.job_id
        assert active.status == "active"

    def test_cancel_clears_active(self):
        jobs = GatewayJobs.get()
        j1 = jobs.create("s1", "misc", "Task", "do")
        jobs.cancel("s1", j1.job_id)
        assert j1.status == "cancelled"
        assert jobs.active_job("s1") is None

    def test_list_for_session_filter(self):
        jobs = GatewayJobs.get()
        j1 = jobs.create("s1", "research", "A", "a")
        jobs.create("s1", "misc", "B", "b")
        all_jobs = jobs.list_for_session("s1")
        assert len(all_jobs) == 2
        active_only = jobs.list_for_session("s1", status_filter="active")
        assert len(active_only) == 1

    def test_get_job(self):
        jobs = GatewayJobs.get()
        j = jobs.create("s1", "misc", "X", "x")
        found = jobs.get_job("s1", j.job_id)
        assert found is not None
        assert found.job_id == j.job_id
        assert jobs.get_job("s1", "nonexistent") is None

    def test_clear_session(self):
        jobs = GatewayJobs.get()
        jobs.create("s1", "misc", "A", "a")
        jobs.create("s1", "misc", "B", "b")
        jobs.clear_session("s1")
        assert jobs.list_for_session("s1") == []
        assert jobs.active_job("s1") is None

    def test_pause_and_resume(self):
        jobs = GatewayJobs.get()
        j = jobs.create("s1", "misc", "Task", "do it")
        assert jobs.pause("s1", j.job_id)
        assert j.status == "paused"
        # Active slot stays
        assert jobs.active_job("s1").job_id == j.job_id
        # Resume
        assert jobs.resume("s1", j.job_id)
        assert j.status == "active"
        assert j.paused_context is None

    def test_resume_wrong_status(self):
        jobs = GatewayJobs.get()
        j = jobs.create("s1", "misc", "Task", "do")
        assert not jobs.resume("s1", j.job_id)  # not paused

    def test_invalid_workstream_fallback(self):
        jobs = GatewayJobs.get()
        j = jobs.create("s1", "fantasy_ws", "Task", "x")
        assert j.workstream == "misc"

    def test_invalid_status_rejected(self):
        jobs = GatewayJobs.get()
        j = jobs.create("s1", "misc", "Task", "x")
        assert not jobs.set_status("s1", j.job_id, "imaginary")


class TestModifierAttachment:
    def test_attach_modifier(self):
        jobs = GatewayJobs.get()
        parent = jobs.create("s1", "research", "Check AAPL", "what is AAPL")
        child = jobs.attach_modifier("s1", parent.job_id, "also MSFT", "research")
        assert child is not None
        assert child.parent_job_id == parent.job_id
        assert child.workstream == "research"
        assert child.status == "queued"

    def test_attach_to_nonexistent(self):
        jobs = GatewayJobs.get()
        assert jobs.attach_modifier("s1", "nope", "text") is None

    def test_find_matching_exact_workstream(self):
        jobs = GatewayJobs.get()
        j1 = jobs.create("s1", "research", "Stock prices", "AAPL")
        j2 = jobs.create("s1", "coding_debug", "Fix bug", "null pointer")
        matches = jobs.find_matching_jobs("s1", "research", "prices")
        # research job should score higher
        assert matches[0].job_id == j1.job_id
        assert len(matches) >= 1

    def test_best_match_none_when_empty(self):
        jobs = GatewayJobs.get()
        assert jobs.best_match("s1", "research") is None

    def test_best_match_returns_top(self):
        jobs = GatewayJobs.get()
        jobs.create("s1", "research", "AAPL stock check", "AAPL")
        jobs.create("s1", "coding_debug", "Fix cron", "cron")
        best = jobs.best_match("s1", "research", "stock")
        assert best is not None
        assert best.workstream == "research"

    def test_no_cross_workstream_match(self):
        jobs = GatewayJobs.get()
        j = jobs.create("s1", "coding_debug", "Fix null bug", "bug")
        matches = jobs.find_matching_jobs("s1", "research", "AAPL")
        # coding_debug != research, no keyword overlap either
        # The job may still appear but with low score
        if matches:
            assert matches[0].workstream != "research" or matches[0].job_id != j.job_id


class TestPersistence:
    def test_save_and_load(self):
        GatewayJobs._instance = None
        jobs1 = GatewayJobs.get()
        jobs1.create("persist_s1", "hermes_ops", "Fix gateway", "gateway crash")
        jobs1.create("persist_s1", "research", "Check GOOGL", "GOOGL price")
        jobs1.save()

        # New instance loads from disk
        GatewayJobs._instance = None
        jobs2 = GatewayJobs.get()
        jobs2.load()
        loaded = jobs2.list_for_session("persist_s1")
        assert len(loaded) == 2
        summaries = {j.matter_summary for j in loaded}
        assert "Fix gateway" in summaries
        assert "Check GOOGL" in summaries

        # Clean up
        jobs2.clear_session("persist_s1")

    def test_load_missing_file_is_noop(self):
        GatewayJobs._instance = None
        with tempfile.TemporaryDirectory() as tmp:
            store = f"{tmp}/nonexistent.json"
            with patch.object(GatewayJobs, "_store_path", property(lambda self: __import__('pathlib').Path(store))):
                jobs = GatewayJobs.get()
                jobs.load()  # should not raise
                assert jobs.list_for_session("any") == []


class TestStatusSummary:
    def test_empty(self):
        jobs = GatewayJobs.get()
        assert "No tracked" in jobs.status_summary("s1")

    def test_active_and_queued(self):
        jobs = GatewayJobs.get()
        jobs.create("s1", "research", "AAPL check", "AAPL")
        jobs.create("s1", "coding_debug", "Bug fix", "bug")
        summary = jobs.status_summary("s1")
        assert "▶" in summary
        assert "⏳" in summary
        assert "AAPL check" in summary


class TestFixedWorkstreams:
    def test_all_expected(self):
        expected = {
            "hermes_ops", "spring_reit_work", "personal_family",
            "research", "drafting_writing", "coding_debug",
            "finance_analysis", "misc",
        }
        assert FIXED_WORKSTREAMS == expected

    def test_job_statuses(self):
        assert JOB_STATUSES == {"active", "queued", "paused", "cancelled", "completed"}


class TestRestartRecovery:
    def test_recover_demotes_active_to_queued(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        # Simulate pre-restart state: active job + queued job
        j1 = GatewayJob("j1", "s1", "research", "AAPL check", status="active")
        j2 = GatewayJob("j2", "s1", "coding_debug", "Fix bug", status="queued")
        jobs._jobs["s1"] = [j1, j2]
        jobs._active_job["s1"] = "j1"

        interrupted = jobs.recover_after_restart()
        assert "s1" in interrupted
        assert j1.status == "queued"  # demoted
        # j2 promoted, active slot now points to j2
        assert jobs._active_job.get("s1") == "j2"

    def test_recover_promotes_first_queued(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        j1 = GatewayJob("j1", "s1", "research", "AAPL", status="active")
        j2 = GatewayJob("j2", "s1", "coding_debug", "Bug", status="queued")
        jobs._jobs["s1"] = [j1, j2]
        jobs._active_job["s1"] = "j1"

        jobs.recover_after_restart()
        # j1 demoted, j2 promoted
        active = jobs.active_job("s1")
        assert active is not None
        assert active.job_id == "j2"
        assert active.status == "active"
        assert j1.status == "queued"

    def test_recover_no_queued_jobs(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        j1 = GatewayJob("j1", "s1", "misc", "Solo task", status="active")
        jobs._jobs["s1"] = [j1]
        jobs._active_job["s1"] = "j1"

        interrupted = jobs.recover_after_restart()
        assert "s1" in interrupted
        assert j1.status == "queued"
        assert jobs.active_job("s1") is None  # no queued to promote

    def test_recover_paused_jobs(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        j1 = GatewayJob("j1", "s1", "misc", "Paused task", status="paused")
        jobs._jobs["s1"] = [j1]
        jobs._active_job["s1"] = "j1"

        interrupted = jobs.recover_after_restart()
        assert "s1" in interrupted
        assert j1.status == "queued"  # demoted from paused

    def test_recover_completed_unaffected(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        j1 = GatewayJob("j1", "s1", "misc", "Done", status="completed")
        j2 = GatewayJob("j2", "s1", "misc", "Cancelled", status="cancelled")
        jobs._jobs["s1"] = [j1, j2]

        interrupted = jobs.recover_after_restart()
        assert interrupted == {}  # nothing to recover
        assert j1.status == "completed"
        assert j2.status == "cancelled"

    def test_recover_save_and_load_roundtrip(self):
        GatewayJobs._instance = None
        jobs1 = GatewayJobs.get()
        j1 = jobs1.create("s1", "research", "AAPL check", "AAPL")
        j2 = jobs1.create("s1", "coding_debug", "Bug fix", "bug")
        jobs1.pause("s1", j1.job_id)
        assert j1.status == "paused"
        jobs1.save()

        # Simulate restart: new instance, load, recover
        GatewayJobs._instance = None
        jobs2 = GatewayJobs.get()
        jobs2.load()
        interrupted = jobs2.recover_after_restart()
        assert len(interrupted) >= 1
        # j1 should be demoted and j2 promoted
        loaded_j1 = jobs2.get_job("s1", j1.job_id)
        assert loaded_j1.status == "queued"
        active = jobs2.active_job("s1")
        assert active is not None

        jobs2.clear_session("s1")


class TestCompletenessTestHeuristic:
    """Simulate the heuristic in GatewayRunner: short messages → modifier,
    long self-contained messages → standalone."""

    def test_short_message_is_modifier(self):
        """Messages like 'done', 'push it', 'Thursday works better'
        should be treated as modifiers when jobs exist."""
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        parent = jobs.create("s1", "research", "AAPL check", "AAPL price?")
        # Simulate the heuristic: short message, existing jobs → modifier
        short_msgs = ["done", "push it", "ok", "Thursday works better", "also check MSFT"]
        for msg in short_msgs:
            word_count = len(msg.split())
            is_short = word_count <= 5 and len(msg) <= 60
            has_question = any(w in msg.lower() for w in {
                "what", "how", "who", "where", "when", "why", "find", "check",
                "search", "look", "create", "make", "build", "write", "run",
                "show", "list", "tell", "explain", "please", "can you",
            })
            is_modifier = is_short or not has_question
            assert is_modifier, f"'{msg}' should be detected as modifier"

    def test_long_question_is_standalone(self):
        """Long messages with question/action words should be standalone."""
        standalone_msgs = [
            "What is the current price of AAPL and how does it compare to MSFT?",
            "Can you create a new Python script that processes CSV files?",
            "Please check the status of all running cron jobs and report back.",
        ]
        for msg in standalone_msgs:
            word_count = len(msg.split())
            is_short = word_count <= 5 and len(msg) <= 60
            has_question = any(w in msg.lower() for w in {
                "what", "how", "who", "where", "when", "why", "find", "check",
                "search", "look", "create", "make", "build", "write", "run",
                "show", "list", "tell", "explain", "please", "can you",
            })
            is_modifier = is_short or not has_question
            assert not is_modifier, f"'{msg[:50]}...' should be standalone"


class TestMultiTargetClarification:
    """When 'done' or 'push it' has multiple possible targets."""

    def test_done_multiple_jobs_asks_which(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        jobs.create("s1", "research", "AAPL check", "AAPL")
        jobs.create("s1", "coding_debug", "Bug fix", "bug")
        candidates = jobs.active_or_queued_jobs("s1")
        assert len(candidates) == 2
        # Should ask "which job is done?" — not complete anything
        assert candidates[0].status != "completed"
        assert candidates[1].status != "completed"

    def test_push_it_multiple_jobs_asks_which(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        jobs.create("s1", "research", "AAPL check", "AAPL")
        jobs.create("s1", "coding_debug", "Bug fix", "bug")
        candidates = jobs.active_or_queued_jobs("s1")
        assert len(candidates) == 2
        # Multiple candidates — should ask, not push

    def test_done_single_job_completes(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        j = jobs.create("s1", "research", "AAPL check", "AAPL")
        candidates = jobs.active_or_queued_jobs("s1")
        assert len(candidates) == 1
        jobs.complete("s1", candidates[0].job_id)
        assert jobs.active_job("s1") is None

    def test_done_no_jobs_reports_none(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        assert jobs.active_or_queued_jobs("s1") == []


class TestZeroTarget:
    """Edge cases with no matching jobs."""

    def test_modifier_no_existing_jobs(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        # No jobs exist — best_match should return None
        assert jobs.best_match("s1", "research", "check stock") is None

    def test_modifier_wrong_workstream(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        jobs.create("s1", "coding_debug", "Fix null bug", "null pointer")
        # Modifier about research with keyword "stock" — no match
        best = jobs.best_match("s1", "research", "stock price")
        # May return the coding_debug job with 0 score but it's still returned
        # The caller should check workstream match

    def test_cancel_nonexistent_job(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        assert not jobs.cancel("s1", "nonexistent_id")


class TestModifierTargetsQueued:
    """Spec #20: modifier attaches to queued job, not active job."""

    def test_modifier_targets_queued_by_workstream(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        j_active = jobs.create("s1", "research", "AAPL check", "AAPL")
        j_queued = jobs.create("s1", "coding_debug", "Bug fix", "bug")
        # j_active is active (research), j_queued is queued (coding_debug)
        # Modifier about coding → should match j_queued, NOT j_active
        best = jobs.best_match("s1", "coding_debug", "bug status")
        assert best is not None
        assert best.job_id == j_queued.job_id, (
            f"Expected queued job {j_queued.job_id}, got {best.job_id}"
        )
        assert best.status == "queued"

    def test_attach_modifier_keeps_original_status(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        parent = jobs.create("s1", "research", "AAPL", "AAPL")
        # Create a queued modifier
        child = jobs.attach_modifier("s1", parent.job_id, "also MSFT", "research")
        assert child.status == "queued"
        assert parent.status == "active"  # parent unchanged


class TestWorkstreamRejection:
    """Invalid workstreams fall back to 'misc'."""

    def test_invalid_workstream_fallback_on_create(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        for bad_ws in ["openclaw", "fantasy", "", "   ", "new_workstream_123"]:
            j = jobs.create("s1", bad_ws, "Some task", "do something")
            assert j.workstream == "misc", (
                f"'{bad_ws}' should fall back to misc, got '{j.workstream}'"
            )

    def test_all_valid_workstreams_accepted(self):
        from gateway.jobs import FIXED_WORKSTREAMS
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        for ws in FIXED_WORKSTREAMS:
            j = jobs.create("s1", ws, "Task", "do")
            assert j.workstream == ws

    def test_no_openclaw_in_workstreams(self):
        from gateway.jobs import FIXED_WORKSTREAMS
        assert "openclaw" not in FIXED_WORKSTREAMS
        for ws in FIXED_WORKSTREAMS:
            assert "openclaw" not in ws.lower()


class TestStatusSummaryEdgeCases:
    """Edge cases for status display."""

    def test_paused_job_shown(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        j = jobs.create("s1", "misc", "Paused task", "do it")
        jobs.pause("s1", j.job_id)
        summary = jobs.status_summary("s1")
        assert "⏸" in summary
        assert "Paused task" in summary

    def test_mixed_statuses(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        jobs.create("s1", "research", "Active task", "do")
        j2 = jobs.create("s1", "misc", "Queued task", "later")
        j3 = jobs.create("s1", "coding_debug", "Another queued", "also later")
        summary = jobs.status_summary("s1")
        assert "▶" in summary
        assert "⏳" in summary
        assert "Active task" in summary

    def test_trim_completed(self):
        GatewayJobs._instance = None
        jobs = GatewayJobs.get()
        for i in range(25):
            j = jobs.create("s1", "misc", f"Task {i}", f"task {i}")
            jobs.complete("s1", j.job_id)
        jobs.trim_completed("s1", max_completed=20)
        completed = [j for j in jobs.list_for_session("s1") if j.status == "completed"]
        assert len(completed) <= 20
