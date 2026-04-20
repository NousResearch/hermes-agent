"""Tests for copilot_jobs.launcher — command building, output parsing, and launch."""

import json
import os
import threading

import pytest

from copilot_jobs.launcher import (
    build_copilot_command,
    launch_copilot,
    parse_copilot_output,
)
from copilot_jobs.models import RepoEntry


# ---------------------------------------------------------------------------
# Helpers for fake subprocesses with real pipes.
# ---------------------------------------------------------------------------

def _make_fake_proc(stdout_text: str, returncode: int = 0):
    """Create a FakeProc with a real pipe for stdout.

    The pipe is pre-loaded with *stdout_text* and the write end is closed,
    so ``readline()`` / ``read()`` work with selectors.
    """
    r_fd, w_fd = os.pipe()
    os.write(w_fd, stdout_text.encode())
    os.close(w_fd)

    class FakeProc:
        def __init__(self):
            self.stdout = os.fdopen(r_fd, "r")
            self.returncode = returncode

        def wait(self):
            return self.returncode

    return FakeProc()


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

    def test_session_id_adds_resume(self):
        cmd = build_copilot_command("test", session_id="abc-123")
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == "abc-123"

    def test_no_session_id_no_resume(self):
        cmd = build_copilot_command("test")
        assert "--resume" not in cmd


# ---------------------------------------------------------------------------
# launch_copilot
# ---------------------------------------------------------------------------

_TEST_SID = "test-sid-00000000-0000-0000-0000-000000000000"


class TestLaunchCopilot:
    def test_dry_run(self):
        repo = RepoEntry(slug="test-repo", path="/test")
        result = launch_copilot(repo, "test prompt", session_id=_TEST_SID, dry_run=True)

        assert result["exit_code"] == 0
        assert result["session_id"] == _TEST_SID
        assert "copilot" in result["cmd"][0]

    def test_dry_run_on_complete(self):
        """on_complete is called synchronously for dry_run."""
        repo = RepoEntry(slug="test-repo", path="/test")
        cb_args = {}

        def on_cb(sid, code):
            cb_args["sid"] = sid
            cb_args["code"] = code

        result = launch_copilot(repo, "test", session_id=_TEST_SID, dry_run=True, on_complete=on_cb)
        assert cb_args["sid"] == _TEST_SID
        assert cb_args["code"] == 0

    def test_spawn_hook_success(self):
        """_spawn hook should be used instead of real Popen."""
        repo = RepoEntry(slug="test-repo", path="/test")

        completed = threading.Event()
        cb_args = {}

        def on_cb(sid, code):
            cb_args["sid"] = sid
            cb_args["code"] = code
            completed.set()

        spawned = []

        def fake_spawn(cmd, cwd):
            spawned.append((cmd, cwd))
            return _make_fake_proc('some output\n', returncode=0)

        result = launch_copilot(
            repo, "test prompt", session_id=_TEST_SID, _spawn=fake_spawn, on_complete=on_cb
        )
        assert result["session_id"] == _TEST_SID
        assert len(spawned) == 1
        assert spawned[0][1] == "/test"
        # --resume should be in the command
        assert "--resume" in result["cmd"]

        # Wait for background thread to finish.
        completed.wait(timeout=5)
        assert cb_args["sid"] == _TEST_SID
        assert cb_args["code"] == 0

    def test_exit_nonzero(self):
        """Non-zero exit code should be captured via on_complete."""
        repo = RepoEntry(slug="test-repo", path="/test")

        completed = threading.Event()
        cb_args = {}

        def on_cb(sid, code):
            cb_args["sid"] = sid
            cb_args["code"] = code
            completed.set()

        result = launch_copilot(
            repo, "test",
            session_id=_TEST_SID,
            _spawn=lambda c, d: _make_fake_proc("error output\n", returncode=1),
            on_complete=on_cb,
        )
        # Session ID is always known upfront
        assert result["session_id"] == _TEST_SID

        completed.wait(timeout=5)
        assert cb_args["code"] == 1

    def test_spawn_failure_raises(self):
        """If _spawn raises, the exception should propagate."""
        repo = RepoEntry(slug="test-repo", path="/test")

        def bad_spawn(cmd, cwd):
            raise OSError("copilot not found")

        with pytest.raises(OSError, match="copilot not found"):
            launch_copilot(repo, "test", session_id=_TEST_SID, _spawn=bad_spawn)
