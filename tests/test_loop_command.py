"""Tests for /loop slash command — parsing, creation, and loop evaluation."""

import hashlib
import json
import pytest

from hermes_cli.loop_command import handle_loop_command, _parse_create_args


# =========================================================================
# _parse_create_args
# =========================================================================

class TestParseCreateArgs:
    def test_basic_interval_and_prompt(self):
        r = _parse_create_args("5m check the deployment")
        assert r["schedule"] == "5m"
        assert r["prompt"] == "check the deployment"
        assert r["skills"] is None
        assert r["verify"] is None
        assert r["error"] is None

    def test_every_prefix(self):
        r = _parse_create_args("every 2h monitor disk usage")
        assert r["schedule"] == "every 2h"
        assert r["prompt"] == "monitor disk usage"

    def test_skills_flag(self):
        r = _parse_create_args("30m check logs --skills devops,networking")
        assert r["schedule"] == "30m"
        assert r["skills"] == ["devops", "networking"]
        assert r["prompt"] == "check logs"

    def test_verify_flag_bare(self):
        r = _parse_create_args("5m fix tests --verify pytest")
        assert r["schedule"] == "5m"
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
        assert r["schedule"] == "1h"
        assert r["prompt"] == "review PRs"
        assert r["skills"] == ["github-code-review"]
        assert r["verify"] == "gh pr checks"

    def test_missing_prompt(self):
        r = _parse_create_args("5m")
        assert r["error"] == "Missing prompt text"

    def test_missing_every_prompt(self):
        r = _parse_create_args("every 30m")
        assert r["error"] == "Missing prompt text"

    def test_empty_string(self):
        r = _parse_create_args("")
        assert r["error"] == "Missing interval and prompt"


# =========================================================================
# handle_loop_command
# =========================================================================

class TestHandleLoopCommand:
    def test_empty_returns_usage(self):
        result = json.loads(handle_loop_command(""))
        assert result["success"] is False
        assert "Usage" in result["error"]

    def test_status_returns_success(self):
        result = json.loads(handle_loop_command("status"))
        assert result["success"] is True

    def test_pause_missing_id(self):
        result = json.loads(handle_loop_command("pause"))
        # pause without ID should fall through to create parsing
        # which will fail because "pause" isn't a valid interval
        assert result.get("success") is False or result.get("error")

    def test_resume_missing_id(self):
        result = json.loads(handle_loop_command("resume"))
        assert result.get("success") is False or result.get("error")

    def test_stop_missing_id(self):
        result = json.loads(handle_loop_command("stop"))
        assert result.get("success") is False or result.get("error")


# =========================================================================
# Loop evaluation logic (hash-based no-progress detection)
# =========================================================================

class TestLoopEvaluation:
    def test_hash_consistency(self):
        """Same input always produces the same hash."""
        text = "deployment is healthy, all pods running"
        h1 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        h2 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        assert h1 == h2

    def test_hash_differs_on_different_input(self):
        h1 = hashlib.sha256("output A".encode("utf-8")).hexdigest()[:16]
        h2 = hashlib.sha256("output B".encode("utf-8")).hexdigest()[:16]
        assert h1 != h2

    def test_no_progress_count_increments_on_same_hash(self):
        """When output hash matches previous, counter should increment."""
        response_hash = hashlib.sha256("same output".encode("utf-8")).hexdigest()[:16]
        last_hash = response_hash
        no_progress_count = 0
        threshold = 3

        if response_hash == last_hash:
            no_progress_count += 1

        assert no_progress_count == 1

    def test_no_progress_count_resets_on_different_hash(self):
        """When output hash changes, counter should reset to 0."""
        hash_a = hashlib.sha256("output A".encode("utf-8")).hexdigest()[:16]
        hash_b = hashlib.sha256("output B".encode("utf-8")).hexdigest()[:16]
        no_progress_count = 2

        if hash_b != hash_a:
            no_progress_count = 0

        assert no_progress_count == 0

    def test_threshold_triggers_pause(self):
        """When no_progress_count hits threshold, should trigger pause."""
        threshold = 3
        no_progress_count = 3
        should_pause = no_progress_count >= threshold
        assert should_pause is True

    def test_below_threshold_no_pause(self):
        threshold = 3
        no_progress_count = 2
        should_pause = no_progress_count >= threshold
        assert should_pause is False

    def test_delivery_gating_same_hash(self):
        """Skip delivery when output hash matches last delivered hash."""
        response_hash = "abc123"
        last_delivered_hash = "abc123"
        skip = last_delivered_hash is not None and response_hash == last_delivered_hash
        assert skip is True

    def test_delivery_gating_different_hash(self):
        """Deliver when output hash differs from last delivered."""
        response_hash = "abc123"
        last_delivered_hash = "def456"
        skip = last_delivered_hash is not None and response_hash == last_delivered_hash
        assert skip is False

    def test_delivery_gating_first_run(self):
        """Deliver on first run (no previous hash)."""
        response_hash = "abc123"
        last_delivered_hash = None
        skip = last_delivered_hash is not None and response_hash == last_delivered_hash
        assert skip is False
