"""Tests for /loop slash command — parsing, creation, loop evaluation, dynamic mode."""

import hashlib
import json
import pytest
from unittest.mock import patch, MagicMock
from subprocess import CompletedProcess, TimeoutExpired

from hermes_cli.loop_command import handle_loop_command, _parse_create_args, _classify_schedule
from cron.scheduler import _evaluate_loop_tick, _run_loop_verify, _build_job_prompt


# =========================================================================
# _classify_schedule — interval/prompt/mode splitting
# =========================================================================

class TestClassifySchedule:
    def test_bare_minutes_normalized_to_recurring(self):
        # Loops are recurring: a bare "5m" must become "every 5m", not a one-shot.
        assert _classify_schedule("5m check") == ("every 5m", "check", False, None)
        assert _classify_schedule("30m do it") == ("every 30m", "do it", False, None)

    def test_bare_hours_and_days(self):
        assert _classify_schedule("2h monitor") == ("every 2h", "monitor", False, None)
        assert _classify_schedule("1d sweep") == ("every 1d", "sweep", False, None)

    def test_explicit_every_form(self):
        assert _classify_schedule("every 2h monitor") == ("every 2h", "monitor", False, None)

    def test_no_interval_is_dynamic(self):
        assert _classify_schedule("check deploy") == ("", "check deploy", True, None)

    def test_single_word_is_dynamic(self):
        assert _classify_schedule("status") == ("", "status", True, None)

    def test_every_without_interval_is_dynamic(self):
        assert _classify_schedule("every day check") == ("", "every day check", True, None)

    def test_bare_subminute_rejected(self):
        sched, prompt, dynamic, error = _classify_schedule("30s check")
        assert error is not None and "Sub-minute" in error

    def test_every_subminute_rejected(self):
        sched, prompt, dynamic, error = _classify_schedule("every 30s check")
        assert error is not None and "Sub-minute" in error


# =========================================================================
# _parse_create_args — timed mode
# =========================================================================

class TestParseCreateArgsTimed:
    def test_basic_interval_and_prompt(self):
        r = _parse_create_args("5m check the deployment")
        assert r["schedule"] == "every 5m"
        assert r["prompt"] == "check the deployment"
        assert r["dynamic"] is False
        assert r["skills"] is None
        assert r["verify"] is None
        assert r["error"] is None

    def test_every_prefix(self):
        r = _parse_create_args("every 2h monitor disk usage")
        assert r["schedule"] == "every 2h"
        assert r["prompt"] == "monitor disk usage"
        assert r["dynamic"] is False

    def test_skills_flag(self):
        r = _parse_create_args("30m check logs --skills devops,networking")
        assert r["schedule"] == "every 30m"
        assert r["skills"] == ["devops", "networking"]
        assert r["prompt"] == "check logs"

    def test_verify_flag_bare(self):
        r = _parse_create_args("5m fix tests --verify pytest")
        assert r["schedule"] == "every 5m"
        assert r["verify"] == "pytest"
        assert r["prompt"] == "fix tests"

    def test_verify_flag_quoted(self):
        r = _parse_create_args('5m fix tests --verify "npm test -- -u"')
        assert r["verify"] == "npm test -- -u"

    def test_verify_flag_single_quotes(self):
        r = _parse_create_args("5m fix tests --verify 'pytest -x -v'")
        assert r["verify"] == "pytest -x -v"

    def test_skills_and_verify_combined(self):
        r = _parse_create_args("1h review PRs --skills github-code-review --verify 'gh pr checks'")
        assert r["schedule"] == "every 1h"
        assert r["prompt"] == "review PRs"
        assert r["skills"] == ["github-code-review"]
        assert r["verify"] == "gh pr checks"

    def test_verify_before_skills_ordering(self):
        """--verify without quotes should NOT eat --skills when --skills is parsed first."""
        r = _parse_create_args("30m check status --verify pytest --skills devops")
        assert r["schedule"] == "every 30m"
        assert r["prompt"] == "check status"
        assert r["skills"] == ["devops"]
        assert r["verify"] == "pytest"

    def test_name_flag(self):
        r = _parse_create_args("30m check logs --name log-watcher")
        assert r["schedule"] == "every 30m"
        assert r["prompt"] == "check logs"
        assert r["name"] == "log-watcher"

    def test_name_with_skills_and_verify(self):
        r = _parse_create_args("1h review PRs --name pr-check --skills github-code-review --verify 'gh pr checks'")
        assert r["schedule"] == "every 1h"
        assert r["prompt"] == "review PRs"
        assert r["name"] == "pr-check"
        assert r["skills"] == ["github-code-review"]
        assert r["verify"] == "gh pr checks"

    def test_missing_prompt(self):
        r = _parse_create_args("5m")
        assert r["error"] == "Missing prompt text"

    def test_missing_every_prompt(self):
        r = _parse_create_args("every 30m")
        assert r["error"] == "Missing prompt text"

    def test_subminute_interval_rejected(self):
        r = _parse_create_args("30s check deploy")
        assert r["error"] is not None
        assert "Sub-minute" in r["error"]


# =========================================================================
# _parse_create_args — dynamic mode (no interval)
# =========================================================================

class TestParseCreateArgsDynamic:
    def test_prompt_only(self):
        r = _parse_create_args("check if tests are passing")
        assert r["dynamic"] is True
        assert r["schedule"] == ""
        assert r["prompt"] == "check if tests are passing"
        assert r["error"] is None

    def test_prompt_with_verify(self):
        r = _parse_create_args("check deployment --verify curl localhost/health")
        assert r["dynamic"] is True
        assert r["prompt"] == "check deployment"
        assert r["verify"] == "curl localhost/health"

    def test_prompt_with_skills(self):
        r = _parse_create_args("review open PRs --skills github-code-review")
        assert r["dynamic"] is True
        assert r["prompt"] == "review open PRs"
        assert r["skills"] == ["github-code-review"]

    def test_prompt_with_all_flags(self):
        r = _parse_create_args("monitor disk --name disk-check --verify 'df -h' --skills devops")
        assert r["dynamic"] is True
        assert r["prompt"] == "monitor disk"
        assert r["name"] == "disk-check"
        assert r["verify"] == "df -h"
        assert r["skills"] == ["devops"]

    def test_single_word_prompt(self):
        r = _parse_create_args("status")
        assert r["dynamic"] is True
        assert r["prompt"] == "status"

    def test_every_without_valid_interval(self):
        """'every' followed by non-interval token should be dynamic."""
        r = _parse_create_args("every day check something")
        assert r["dynamic"] is True
        assert r["prompt"] == "every day check something"


# =========================================================================
# handle_loop_command
# =========================================================================

class TestHandleLoopCommand:
    def test_empty_returns_usage(self):
        result = json.loads(handle_loop_command(""))
        assert result["success"] is True
        assert "Usage" in result["message"]

    def test_status_returns_success(self):
        result = json.loads(handle_loop_command("status"))
        assert result["success"] is True

    def test_pause_missing_id(self):
        result = json.loads(handle_loop_command("pause"))
        assert result["success"] is False
        assert "Usage" in result["error"]

    def test_resume_missing_id(self):
        result = json.loads(handle_loop_command("resume"))
        assert result["success"] is False
        assert "Usage" in result["error"]

    def test_stop_missing_id(self):
        result = json.loads(handle_loop_command("stop"))
        assert result["success"] is False
        assert "Usage" in result["error"]

    def test_remove_missing_id(self):
        result = json.loads(handle_loop_command("remove"))
        assert result["success"] is False
        assert "Usage" in result["error"]

    def test_list_alias(self):
        result = json.loads(handle_loop_command("list"))
        assert result["success"] is True

    def test_help_alias(self):
        result = json.loads(handle_loop_command("help"))
        assert result["success"] is True
        assert "Usage" in result["message"]


# =========================================================================
# _run_loop_verify (mocked subprocess)
# =========================================================================

class TestRunLoopVerify:
    def test_no_verify_command(self):
        job = {"loop_verify": None}
        assert _run_loop_verify(job) is None

    def test_no_verify_key(self):
        job = {}
        assert _run_loop_verify(job) is None

    @patch("cron.scheduler.subprocess.run")
    def test_verify_success(self, mock_run):
        mock_run.return_value = CompletedProcess([], 0, stdout="", stderr="")
        job = {"loop_verify": "echo ok"}
        assert _run_loop_verify(job) is None
        mock_run.assert_called_once_with(
            "echo ok", shell=True, capture_output=True, text=True, timeout=60,
        )

    @patch("cron.scheduler.subprocess.run")
    def test_verify_failure_with_stderr(self, mock_run):
        mock_run.return_value = CompletedProcess(
            [], 1, stdout="output here", stderr="something broke",
        )
        job = {"loop_verify": "pytest"}
        result = _run_loop_verify(job)
        assert result is not None
        assert "exit 1" in result
        assert "something broke" in result
        assert "output here" in result

    @patch("cron.scheduler.subprocess.run")
    def test_verify_failure_stdout_only(self, mock_run):
        mock_run.return_value = CompletedProcess([], 2, stdout="FAIL: 3 tests", stderr="")
        job = {"loop_verify": "pytest"}
        result = _run_loop_verify(job)
        assert result is not None
        assert "exit 2" in result
        assert "FAIL: 3 tests" in result
        assert "stderr" not in result

    @patch("cron.scheduler.subprocess.run")
    def test_verify_timeout(self, mock_run):
        mock_run.side_effect = TimeoutExpired("pytest", 60)
        job = {"loop_verify": "pytest --slow"}
        result = _run_loop_verify(job)
        assert result is not None
        assert "timed out" in result
        assert "60s" in result

    @patch("cron.scheduler.subprocess.run")
    def test_verify_generic_exception(self, mock_run):
        mock_run.side_effect = OSError("no such file")
        job = {"loop_verify": "/nonexistent/cmd"}
        result = _run_loop_verify(job)
        assert result is not None
        assert "no such file" in result


# =========================================================================
# _evaluate_loop_tick — integration tests
# =========================================================================

def _make_loop_job(**overrides):
    """Create a minimal loop job dict for testing."""
    job = {
        "id": "test123",
        "name": "test-loop",
        "prompt": "check deployment status",
        "loop": True,
        "loop_dynamic": False,
        "loop_verify": None,
        "loop_no_progress_threshold": 3,
        "loop_no_progress_count": 0,
        "loop_last_output_hash": None,
        "loop_last_response": None,
        "loop_last_delivered_hash": None,
        "loop_last_verify_error": None,
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
    }
    job.update(overrides)
    return job


class TestEvaluateLoopTickIntegration:
    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_first_run_delivers(self, mock_pause, mock_update):
        job = _make_loop_job()
        alert, skip = _evaluate_loop_tick(job, "some output")
        assert alert is None
        assert skip is False
        mock_update.assert_called_once()
        mock_pause.assert_not_called()
        updates = mock_update.call_args[0][1]
        assert updates["loop_last_output_hash"] is not None
        assert updates["loop_no_progress_count"] == 0

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_same_output_increments_count(self, mock_pause, mock_update):
        response_hash = hashlib.sha256(b"same output").hexdigest()[:16]
        job = _make_loop_job(loop_last_output_hash=response_hash)
        alert, skip = _evaluate_loop_tick(job, "same output")
        assert alert is None
        assert skip is False
        updates = mock_update.call_args[0][1]
        assert updates["loop_no_progress_count"] == 1
        mock_pause.assert_not_called()

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_same_output_threshold_triggers_pause(self, mock_pause, mock_update):
        response_hash = hashlib.sha256(b"stuck output").hexdigest()[:16]
        job = _make_loop_job(
            loop_last_output_hash=response_hash,
            loop_no_progress_count=2,
        )
        alert, skip = _evaluate_loop_tick(job, "stuck output")
        assert alert is not None
        assert "auto-paused" in alert
        assert skip is True
        mock_pause.assert_called_once_with("test123", reason="no progress detected")

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_different_output_resets_count(self, mock_pause, mock_update):
        old_hash = hashlib.sha256(b"old output").hexdigest()[:16]
        job = _make_loop_job(
            loop_last_output_hash=old_hash,
            loop_no_progress_count=2,
        )
        alert, skip = _evaluate_loop_tick(job, "new output")
        assert alert is None
        updates = mock_update.call_args[0][1]
        assert updates["loop_no_progress_count"] == 0
        mock_pause.assert_not_called()

    @patch("hermes_cli.goals.judge_goal", return_value=("done", "goal achieved", False))
    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_judge_done_increments_count(self, mock_pause, mock_update, mock_judge):
        job = _make_loop_job()
        alert, skip = _evaluate_loop_tick(job, "all tests passing")
        assert alert is None
        updates = mock_update.call_args[0][1]
        assert updates["loop_no_progress_count"] == 1
        mock_pause.assert_not_called()

    @patch("hermes_cli.goals.judge_goal", return_value=("done", "goal achieved", False))
    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_judge_done_threshold_triggers_pause(self, mock_pause, mock_update, mock_judge):
        response_hash = hashlib.sha256(b"everything done").hexdigest()[:16]
        job = _make_loop_job(
            loop_last_output_hash=response_hash,
            loop_no_progress_count=1,
        )
        alert, skip = _evaluate_loop_tick(job, "everything done")
        assert alert is not None
        assert "auto-paused" in alert
        mock_pause.assert_called_once()

    @patch("hermes_cli.goals.judge_goal", side_effect=Exception("model unavailable"))
    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_judge_failure_fail_open(self, mock_pause, mock_update, mock_judge):
        job = _make_loop_job()
        alert, skip = _evaluate_loop_tick(job, "some output")
        assert alert is None
        updates = mock_update.call_args[0][1]
        assert updates["loop_no_progress_count"] == 0
        mock_pause.assert_not_called()

    @patch("cron.scheduler._run_loop_verify", return_value="verify failed: exit 1")
    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_verify_error_stored(self, mock_pause, mock_update, mock_verify):
        job = _make_loop_job(loop_verify="pytest")
        alert, skip = _evaluate_loop_tick(job, "some output")
        assert alert is None
        updates = mock_update.call_args[0][1]
        assert updates["loop_last_verify_error"] == "verify failed: exit 1"

    @patch("cron.scheduler._run_loop_verify", return_value=None)
    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_verify_success_clears_error(self, mock_pause, mock_update, mock_verify):
        job = _make_loop_job(loop_verify="pytest", loop_last_verify_error="old error")
        alert, skip = _evaluate_loop_tick(job, "some output")
        assert alert is None
        updates = mock_update.call_args[0][1]
        assert updates["loop_last_verify_error"] is None

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_delivery_gating_same_as_last_delivered(self, mock_pause, mock_update):
        response_hash = hashlib.sha256(b"same output").hexdigest()[:16]
        job = _make_loop_job(
            loop_last_output_hash="different_hash",
            loop_last_delivered_hash=response_hash,
        )
        alert, skip = _evaluate_loop_tick(job, "same output")
        assert alert is None
        assert skip is True

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_delivery_gating_different_from_last(self, mock_pause, mock_update):
        job = _make_loop_job(
            loop_last_output_hash="old_hash",
            loop_last_delivered_hash="old_hash",
        )
        alert, skip = _evaluate_loop_tick(job, "new output")
        assert alert is None
        assert skip is False

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_empty_response_hashes_consistently(self, mock_pause, mock_update):
        job = _make_loop_job()
        alert, skip = _evaluate_loop_tick(job, "")
        assert alert is None
        updates = mock_update.call_args[0][1]
        assert updates["loop_last_output_hash"] is not None
        assert "loop_last_delivered_hash" not in updates


# =========================================================================
# _evaluate_loop_tick — adaptive interval tests
# =========================================================================

class TestAdaptiveIntervals:
    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_dynamic_output_changed_halves_interval(self, mock_pause, mock_update):
        """Dynamic job: output changes → interval halves."""
        job = _make_loop_job(
            loop_dynamic=True,
            schedule={"kind": "interval", "minutes": 10, "display": "every 10m"},
            loop_last_output_hash="old_hash",
        )
        alert, skip = _evaluate_loop_tick(job, "new output")
        assert alert is None
        updates = mock_update.call_args[0][1]
        assert updates["schedule"]["minutes"] == 5
        assert updates["schedule_display"] == "every 5m"

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_dynamic_output_stable_doubles_interval(self, mock_pause, mock_update):
        """Dynamic job: same output → interval doubles."""
        response_hash = hashlib.sha256(b"same output").hexdigest()[:16]
        job = _make_loop_job(
            loop_dynamic=True,
            schedule={"kind": "interval", "minutes": 5, "display": "every 5m"},
            loop_last_output_hash=response_hash,
        )
        alert, skip = _evaluate_loop_tick(job, "same output")
        assert alert is None
        updates = mock_update.call_args[0][1]
        assert updates["schedule"]["minutes"] == 10
        assert updates["schedule_display"] == "every 10m"

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_dynamic_interval_clamped_min(self, mock_pause, mock_update):
        """Dynamic job: interval won't go below 1 minute."""
        job = _make_loop_job(
            loop_dynamic=True,
            schedule={"kind": "interval", "minutes": 1, "display": "every 1m"},
            loop_last_output_hash="old_hash",
        )
        alert, skip = _evaluate_loop_tick(job, "new output")
        updates = mock_update.call_args[0][1]
        # Already at 1, halving would give 0, clamped to 1 — no change, no schedule update
        assert "schedule" not in updates

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_dynamic_interval_clamped_max(self, mock_pause, mock_update):
        """Dynamic job: interval won't go above 1440 minutes (24h)."""
        response_hash = hashlib.sha256(b"stable").hexdigest()[:16]
        job = _make_loop_job(
            loop_dynamic=True,
            schedule={"kind": "interval", "minutes": 1440, "display": "every 1440m"},
            loop_last_output_hash=response_hash,
        )
        alert, skip = _evaluate_loop_tick(job, "stable")
        updates = mock_update.call_args[0][1]
        # Already at 1440, doubling would give 2880, clamped to 1440 — no change
        assert "schedule" not in updates

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_non_dynamic_job_no_schedule_change(self, mock_pause, mock_update):
        """Non-dynamic job: schedule never changes."""
        job = _make_loop_job(
            loop_dynamic=False,
            schedule={"kind": "interval", "minutes": 5, "display": "every 5m"},
            loop_last_output_hash="old_hash",
        )
        alert, skip = _evaluate_loop_tick(job, "new output")
        updates = mock_update.call_args[0][1]
        assert "schedule" not in updates
        assert "schedule_display" not in updates

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_dynamic_first_run_no_schedule_change(self, mock_pause, mock_update):
        """Dynamic job first run (no previous hash) → output is "changed" → halves."""
        job = _make_loop_job(
            loop_dynamic=True,
            schedule={"kind": "interval", "minutes": 5, "display": "every 5m"},
        )
        alert, skip = _evaluate_loop_tick(job, "first output")
        updates = mock_update.call_args[0][1]
        # First run: no last_hash, so output_changed=True, halves from 5 → 2
        assert updates["schedule"]["minutes"] == 2

    @patch("cron.jobs.update_job")
    @patch("cron.jobs.pause_job")
    def test_dynamic_cron_schedule_ignored(self, mock_pause, mock_update):
        """Dynamic job with cron schedule (not interval) → no adaptive change."""
        job = _make_loop_job(
            loop_dynamic=True,
            schedule={"kind": "cron", "expr": "*/5 * * * *", "display": "*/5 * * * *"},
            loop_last_output_hash="old_hash",
        )
        alert, skip = _evaluate_loop_tick(job, "new output")
        updates = mock_update.call_args[0][1]
        assert "schedule" not in updates


# =========================================================================
# Create path — end-to-end against a real job store (no create_job mock)
#
# These guard the contract that mocked tests can't: that the job actually
# created by /loop is RECURRING. Bug history: a bare "30m" was passed straight
# to create_job, which parsed it as a one-shot (kind="once") — so the loop
# fired once and died. Mocked parse tests stayed green throughout.
# =========================================================================

@pytest.fixture
def isolated_cron(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir so create_job writes to a throwaway store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    import cron.jobs as jobs_mod
    importlib.reload(jobs_mod)
    yield jobs_mod
    importlib.reload(jobs_mod)


class TestCreatePathRecurring:
    def test_timed_loop_is_recurring_interval(self, isolated_cron):
        """/loop 30m ... must create a recurring interval job, never a one-shot."""
        result = json.loads(handle_loop_command("30m check the deploy"))
        assert result["success"] is True
        job = isolated_cron.resolve_job_ref(result["job_id"])
        assert job["schedule"]["kind"] == "interval"
        assert job["schedule"]["minutes"] == 30
        assert job["loop"] is True
        assert job["loop_dynamic"] is False

    def test_every_form_is_recurring_interval(self, isolated_cron):
        result = json.loads(handle_loop_command("every 2h monitor disk"))
        assert result["success"] is True
        job = isolated_cron.resolve_job_ref(result["job_id"])
        assert job["schedule"]["kind"] == "interval"
        assert job["schedule"]["minutes"] == 120

    def test_dynamic_loop_starts_at_5m_interval(self, isolated_cron):
        result = json.loads(handle_loop_command("check if tests pass"))
        assert result["success"] is True
        job = isolated_cron.resolve_job_ref(result["job_id"])
        assert job["schedule"]["kind"] == "interval"
        assert job["schedule"]["minutes"] == 5
        assert job["loop_dynamic"] is True

    def test_subminute_loop_rejected_at_create(self, isolated_cron):
        result = json.loads(handle_loop_command("30s check deploy"))
        assert result["success"] is False
        assert "Sub-minute" in result["error"]


# =========================================================================
# _build_job_prompt — feedback injection rendering
#
# Guards that injected context uses real newlines, not literal "\n" text.
# =========================================================================

class TestBuildJobPromptInjection:
    def test_verify_error_uses_real_newlines(self):
        job = _make_loop_job(
            loop_last_verify_error="verify command failed (exit 1)\nstderr: boom",
        )
        built = _build_job_prompt(job)
        # The verify error must appear with real newlines, not the literal
        # two-character sequence backslash-n.
        assert "\\n" not in built
        assert "boom" in built
        assert "Post-run verification failure" in built

    def test_prev_response_uses_real_newlines(self):
        job = _make_loop_job(loop_last_response="line one\nline two")
        built = _build_job_prompt(job)
        assert "\\n" not in built
        assert "line one" in built
        assert "Previous tick response" in built

    def test_dynamic_cadence_injected(self):
        job = _make_loop_job(
            loop_dynamic=True,
            schedule={"kind": "interval", "minutes": 7, "display": "every 7m"},
        )
        built = _build_job_prompt(job)
        assert "Loop cadence" in built
        assert "7m" in built
