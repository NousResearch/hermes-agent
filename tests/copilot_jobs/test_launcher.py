"""Tests for copilot_jobs.launcher — output parsing and attach command building."""

import pytest

from copilot_jobs.launcher import (
    parse_copilot_output,
    build_attach_command,
)


class TestParseCopilotOutput:
    def test_parses_session_id(self):
        output = "Starting remote session...\nsession_id: abc123-def456\nReady."
        result = parse_copilot_output(output)
        assert result["session_id"] == "abc123-def456"

    def test_parses_remote_name(self):
        output = "remote_name: copilot-remote-1\nsession_id: xyz"
        result = parse_copilot_output(output)
        assert result["remote_name"] == "copilot-remote-1"
        assert result["session_id"] == "xyz"

    def test_no_match_returns_none(self):
        result = parse_copilot_output("just some random output")
        assert result["session_id"] is None
        assert result["remote_name"] is None

    def test_case_insensitive(self):
        output = "Session_ID: UPPER_CASE_123"
        result = parse_copilot_output(output)
        assert result["session_id"] == "UPPER_CASE_123"

    def test_colon_and_space_variations(self):
        output = "session id: with-spaces\nremote name: remote.test"
        result = parse_copilot_output(output)
        assert result["session_id"] == "with-spaces"
        assert result["remote_name"] == "remote.test"


class TestBuildAttachCommand:
    def test_basic(self):
        cmd = build_attach_command("ryanwalden-ryanwalden", "ses_abc")
        assert cmd == "docker exec -it ryanwalden-ryanwalden copilot --connect=ses_abc"

    def test_different_container(self):
        cmd = build_attach_command("eugenecho-eugenecho", "ses_xyz")
        assert cmd == "docker exec -it eugenecho-eugenecho copilot --connect=ses_xyz"


class TestLaunchDryRun:
    def test_dry_run_creates_placeholder(self, tmp_path):
        """Dry run should transition to running and persist placeholder handles."""
        from hermes_state import SessionDB
        from copilot_jobs.launcher import launch_copilot_remote
        from copilot_jobs.models import RepoEntry

        db = SessionDB(db_path=tmp_path / "test.db")
        db.create_copilot_job(
            job_id="cj_dry",
            repo_slug="test-repo",
            repo_path="/test",
        )

        repo = RepoEntry(slug="test-repo", path="/test")
        result = launch_copilot_remote(
            db=db,
            job_id="cj_dry",
            repo=repo,
            prompt="test prompt",
            dry_run=True,
        )

        assert result["session_id"].startswith("dry-run-")
        assert result["remote_name"] == "dry-run"
        assert "copilot --connect=" in result["attach_command"]
        assert result["pid"] == 0

        # Verify DB was updated
        job = db.get_copilot_job("cj_dry")
        assert job["state"] == "running"
        assert job["copilot_session_id"].startswith("dry-run-")
        assert job["attach_command"] is not None
        db.close()
