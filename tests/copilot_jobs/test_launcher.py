"""Tests for copilot_jobs.launcher — command building, output parsing, and launch."""

import json
import threading
import time

import pytest

from copilot_jobs.launcher import (
    build_copilot_command,
    build_resume_command,
    launch_copilot,
    parse_copilot_output,
)
from copilot_jobs.models import RepoEntry


# ---------------------------------------------------------------------------
# parse_copilot_output
# ---------------------------------------------------------------------------

class TestParseCopilotOutput:
    def test_parses_session_id_jsonl(self):
        output = '{"sessionId": "ses_abc123"}\n{"type":"done"}\n'
        result = parse_copilot_output(output)
        assert result["session_id"] == "ses_abc123"

    def test_parses_session_id_snake_case_json(self):
        output = '{"session_id": "ses_xyz"}\n'
        result = parse_copilot_output(output)
        assert result["session_id"] == "ses_xyz"

    def test_fallback_regex(self):
        output = "Starting...\nsession_id: abc123-def456\nReady."
        result = parse_copilot_output(output)
        assert result["session_id"] == "abc123-def456"

    def test_no_match_returns_none(self):
        result = parse_copilot_output("just some random output")
        assert result["session_id"] is None

    def test_case_insensitive_regex(self):
        output = "Session_ID: UPPER_CASE_123"
        result = parse_copilot_output(output)
        assert result["session_id"] == "UPPER_CASE_123"

    def test_jsonl_takes_precedence(self):
        """JSONL match should be returned even if regex would also match."""
        output = '{"sessionId": "from_json"}\nsession_id: from_regex'
        result = parse_copilot_output(output)
        assert result["session_id"] == "from_json"

    def test_empty_string(self):
        result = parse_copilot_output("")
        assert result["session_id"] is None


# ---------------------------------------------------------------------------
# build_copilot_command
# ---------------------------------------------------------------------------

class TestBuildCopilotCommand:
    def test_basic_command(self):
        cmd = build_copilot_command("fix the tests")
        assert cmd[0] == "copilot"
        assert "-p" in cmd
        assert "fix the tests" in cmd
        assert "--allow-all" in cmd
        assert "--silent" in cmd
        assert "--no-auto-update" in cmd
        assert "--no-ask-user" in cmd

    def test_json_output_default(self):
        cmd = build_copilot_command("test")
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "json"

    def test_no_json_output(self):
        cmd = build_copilot_command("test", json_output=False)
        assert "--output-format" not in cmd

    def test_custom_model(self):
        cmd = build_copilot_command("test", model="claude-sonnet-4.6")
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4.6"

    def test_custom_binary(self):
        cmd = build_copilot_command("test", copilot_bin="/usr/bin/copilot")
        assert cmd[0] == "/usr/bin/copilot"


# ---------------------------------------------------------------------------
# build_resume_command
# ---------------------------------------------------------------------------

class TestBuildResumeCommand:
    def test_basic(self):
        cmd = build_resume_command("ses_abc")
        assert cmd == "copilot --resume=ses_abc"

    def test_custom_bin(self):
        cmd = build_resume_command("ses_abc", copilot_bin="/opt/copilot")
        assert cmd == "/opt/copilot --resume=ses_abc"


# ---------------------------------------------------------------------------
# launch_copilot
# ---------------------------------------------------------------------------

class TestLaunchCopilot:
    @pytest.fixture()
    def db(self, tmp_path):
        from hermes_state import SessionDB
        return SessionDB(db_path=tmp_path / "test.db")

    def _create_pending(self, db, job_id="cj_test"):
        db.create_copilot_job(
            job_id=job_id, repo_slug="test-repo", repo_path="/test"
        )
        return RepoEntry(slug="test-repo", path="/test")

    def test_dry_run(self, db):
        repo = self._create_pending(db)
        result = launch_copilot(
            db, "cj_test", repo, "test prompt", dry_run=True
        )

        assert result["pid"] == 0
        assert result["session_id"].startswith("dry-run-")
        assert "copilot" in result["cmd"][0]

        job = db.get_copilot_job("cj_test")
        assert job["state"] == "running"

    def test_spawn_hook(self, db):
        """_spawn hook should be used instead of real Popen."""
        repo = self._create_pending(db)

        class FakeProc:
            pid = 42
            returncode = 0
            def communicate(self):
                return ('{"sessionId": "ses_fake"}\n', "")

        spawned = []
        def fake_spawn(cmd, cwd):
            spawned.append((cmd, cwd))
            return FakeProc()

        result = launch_copilot(
            db, "cj_test", repo, "test prompt", _spawn=fake_spawn
        )
        assert result["pid"] == 42
        assert len(spawned) == 1
        assert spawned[0][1] == "/test"

        job = db.get_copilot_job("cj_test")
        assert job["state"] == "running"

    def test_monitor_thread_marks_idle_on_success(self, db):
        """When the process exits 0, the monitor thread should mark the job idle."""
        repo = self._create_pending(db)

        class FakeProc:
            pid = 99
            returncode = 0
            def communicate(self):
                return ("done\n", "")

        launch_copilot(db, "cj_test", repo, "test", _spawn=lambda c, d: FakeProc())

        # Give the monitor thread time to finish.
        for _ in range(50):
            job = db.get_copilot_job("cj_test")
            if job["state"] == "idle":
                break
            time.sleep(0.05)

        job = db.get_copilot_job("cj_test")
        assert job["state"] == "idle"

    def test_monitor_thread_marks_failed_on_error(self, db):
        """When the process exits non-zero, job should be marked failed."""
        repo = self._create_pending(db)

        class FakeProc:
            pid = 100
            returncode = 1
            def communicate(self):
                return ("error output\n", "")

        launch_copilot(
            db, "cj_test", repo, "test", _spawn=lambda c, d: FakeProc()
        )

        for _ in range(50):
            job = db.get_copilot_job("cj_test")
            if job["state"] in ("idle", "failed"):
                break
            time.sleep(0.05)

        job = db.get_copilot_job("cj_test")
        assert job["state"] == "failed"
        assert "exited with code 1" in (job.get("error_text") or "")

    def test_launch_records_events(self, db):
        """Launch should record a 'launched' event with command details."""
        repo = self._create_pending(db)

        class FakeProc:
            pid = 55
            returncode = 0
            def communicate(self):
                return ("", "")

        launch_copilot(
            db, "cj_test", repo, "test", _spawn=lambda c, d: FakeProc()
        )

        events = db.get_copilot_job_events("cj_test")
        event_types = [e["event_type"] for e in events]
        assert "launched" in event_types

    def test_spawn_failure_marks_failed(self, db):
        """If _spawn raises, job should be marked failed."""
        repo = self._create_pending(db)

        def bad_spawn(cmd, cwd):
            raise OSError("copilot not found")

        with pytest.raises(OSError):
            launch_copilot(
                db, "cj_test", repo, "test", _spawn=bad_spawn
            )

        job = db.get_copilot_job("cj_test")
        assert job["state"] == "failed"
        assert "copilot not found" in (job.get("error_text") or "")

    def test_monitor_captures_session_id(self, db):
        """Monitor thread should persist session_id from JSONL output."""
        repo = self._create_pending(db)

        class FakeProc:
            pid = 77
            returncode = 0
            def communicate(self):
                return ('{"sessionId": "ses_from_output"}\n', "")

        launch_copilot(
            db, "cj_test", repo, "test", _spawn=lambda c, d: FakeProc()
        )

        for _ in range(50):
            job = db.get_copilot_job("cj_test")
            if job["state"] == "idle":
                break
            time.sleep(0.05)

        job = db.get_copilot_job("cj_test")
        assert job["state"] == "idle"
        assert job["copilot_session_id"] == "ses_from_output"
