"""Tests for agent/downstream_rate_guard.py — cross-session rate limit guard
for aggregator-style custom providers (e.g. a multi-account router)."""

import json
import os
import time

import pytest


@pytest.fixture
def rate_guard_env(tmp_path, monkeypatch):
    """Isolate rate guard state to a temp directory."""
    hermes_home = str(tmp_path / ".hermes")
    os.makedirs(hermes_home, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", hermes_home)
    return hermes_home


class TestExtractUpstreamTag:
    def test_matches_router_bracket_shape(self):
        from agent.downstream_rate_guard import extract_upstream_tag

        text = (
            'HTTP 429: [antigravity/gemini-pro-agent] [429]: {\n'
            '  "code": 429,\n  "message": "..."\n}'
        )
        assert extract_upstream_tag(text) == "antigravity"

    def test_matches_different_upstream_and_model_shape(self):
        from agent.downstream_rate_guard import extract_upstream_tag

        text = (
            'HTTP 429: [claude/claude-opus-5] [429]: '
            '{"type":"error","error":{"type":"rate_limit_error"}}'
        )
        assert extract_upstream_tag(text) == "claude"

    def test_ignores_non_429_brackets(self):
        from agent.downstream_rate_guard import extract_upstream_tag

        text = 'HTTP 404: [kiro/claude-sonnet-4.5] [404]: not found'
        assert extract_upstream_tag(text) is None

    def test_returns_none_for_plain_error(self):
        from agent.downstream_rate_guard import extract_upstream_tag

        assert extract_upstream_tag("Connection reset by peer") is None
        assert extract_upstream_tag("") is None
        assert extract_upstream_tag(None) is None


class TestRecordAndCheckCooldown:
    def test_records_and_reads_back_remaining(self, rate_guard_env):
        from agent.downstream_rate_guard import (
            record_downstream_rate_limit,
            downstream_rate_limit_remaining,
        )

        record_downstream_rate_limit("custom:antigravity", seconds=63.0)
        remaining = downstream_rate_limit_remaining("custom:antigravity")
        assert remaining is not None
        assert 58 < remaining <= 63

    def test_different_keys_are_independent(self, rate_guard_env):
        from agent.downstream_rate_guard import (
            record_downstream_rate_limit,
            downstream_rate_limit_remaining,
        )

        record_downstream_rate_limit("custom:antigravity", seconds=100.0)
        assert downstream_rate_limit_remaining("custom:claude") is None

    def test_no_cooldown_returns_none(self, rate_guard_env):
        from agent.downstream_rate_guard import downstream_rate_limit_remaining

        assert downstream_rate_limit_remaining("custom:never-seen") is None

    def test_expired_cooldown_returns_none_and_cleans_up(self, rate_guard_env):
        from agent.downstream_rate_guard import (
            downstream_rate_limit_remaining,
            _state_path,
        )

        path = _state_path("custom:antigravity")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"reset_at": time.time() - 10, "recorded_at": time.time() - 100}, f)

        assert downstream_rate_limit_remaining("custom:antigravity") is None
        assert not os.path.exists(path)

    def test_non_positive_seconds_is_a_noop(self, rate_guard_env):
        from agent.downstream_rate_guard import (
            record_downstream_rate_limit,
            downstream_rate_limit_remaining,
        )

        record_downstream_rate_limit("custom:antigravity", seconds=0)
        assert downstream_rate_limit_remaining("custom:antigravity") is None

    def test_cooldown_capped_at_max(self, rate_guard_env):
        from agent.downstream_rate_guard import (
            record_downstream_rate_limit,
            downstream_rate_limit_remaining,
            _MAX_COOLDOWN_SECONDS,
        )

        record_downstream_rate_limit("custom:antigravity", seconds=999999)
        remaining = downstream_rate_limit_remaining("custom:antigravity")
        assert remaining <= _MAX_COOLDOWN_SECONDS

    def test_key_is_sanitized_for_filesystem(self, rate_guard_env):
        from agent.downstream_rate_guard import (
            record_downstream_rate_limit,
            downstream_rate_limit_remaining,
        )

        # Provider/tag values could theoretically contain path separators —
        # must not escape the rate_limits directory or raise.
        record_downstream_rate_limit("custom:../../evil", seconds=30.0)
        assert downstream_rate_limit_remaining("custom:../../evil") is not None


class TestClearDownstreamRateLimit:
    def test_clears_existing_file(self, rate_guard_env):
        from agent.downstream_rate_guard import (
            record_downstream_rate_limit,
            clear_downstream_rate_limit,
            downstream_rate_limit_remaining,
        )

        record_downstream_rate_limit("custom:antigravity", seconds=60.0)
        assert downstream_rate_limit_remaining("custom:antigravity") is not None

        clear_downstream_rate_limit("custom:antigravity")
        assert downstream_rate_limit_remaining("custom:antigravity") is None

    def test_clear_when_no_file_does_not_raise(self, rate_guard_env):
        from agent.downstream_rate_guard import clear_downstream_rate_limit

        clear_downstream_rate_limit("custom:never-recorded")
