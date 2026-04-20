"""Tests for copilot_jobs.launcher — command building, output parsing, and launch."""

import json

import pytest

from copilot_jobs.launcher import (
    build_copilot_command,
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
        assert "--remote" in cmd
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
# launch_copilot
# ---------------------------------------------------------------------------

class TestLaunchCopilot:
    def test_dry_run(self):
        repo = RepoEntry(slug="test-repo", path="/test")
        result = launch_copilot(repo, "test prompt", dry_run=True)

        assert result["exit_code"] == 0
        assert result["session_id"].startswith("dry-run-")
        assert "copilot" in result["cmd"][0]

    def test_spawn_hook_success(self):
        """_spawn hook should be used instead of real Popen."""
        repo = RepoEntry(slug="test-repo", path="/test")

        class FakeProc:
            returncode = 0
            def communicate(self):
                return ('{"sessionId": "ses_fake"}\n', "")

        spawned = []
        def fake_spawn(cmd, cwd):
            spawned.append((cmd, cwd))
            return FakeProc()

        result = launch_copilot(repo, "test prompt", _spawn=fake_spawn)
        assert result["session_id"] == "ses_fake"
        assert result["exit_code"] == 0
        assert len(spawned) == 1
        assert spawned[0][1] == "/test"

    def test_exit_nonzero(self):
        """Non-zero exit code should be captured."""
        repo = RepoEntry(slug="test-repo", path="/test")

        class FakeProc:
            returncode = 1
            def communicate(self):
                return ("error output\n", "")

        result = launch_copilot(repo, "test", _spawn=lambda c, d: FakeProc())
        assert result["exit_code"] == 1
        assert result["session_id"] is None

    def test_captures_session_id_from_output(self):
        """Session ID from JSONL output should be returned."""
        repo = RepoEntry(slug="test-repo", path="/test")

        class FakeProc:
            returncode = 0
            def communicate(self):
                return ('{"sessionId": "ses_from_output"}\n', "")

        result = launch_copilot(repo, "test", _spawn=lambda c, d: FakeProc())
        assert result["session_id"] == "ses_from_output"

    def test_spawn_failure_raises(self):
        """If _spawn raises, the exception should propagate."""
        repo = RepoEntry(slug="test-repo", path="/test")

        def bad_spawn(cmd, cwd):
            raise OSError("copilot not found")

        with pytest.raises(OSError, match="copilot not found"):
            launch_copilot(repo, "test", _spawn=bad_spawn)
