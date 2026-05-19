"""Tests for agent/nous_rate_guard.py — cross-session Nous Portal rate limit guard."""

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
    # Clear any cached module-level imports
    return hermes_home


class TestRecordNousRateLimit:
    """Test recording rate limit state."""

    def test_records_with_header_reset(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        headers = {"x-ratelimit-reset-requests-1h": "1800"}
        record_nous_rate_limit(headers=headers)

        path = _state_path()
        assert os.path.exists(path)
        with open(path) as f:
            state = json.load(f)
        assert state["reset_seconds"] == pytest.approx(1800, abs=2)
        assert state["reset_at"] > time.time()

    def test_records_with_per_minute_header(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        headers = {"x-ratelimit-reset-requests": "45"}
        record_nous_rate_limit(headers=headers)

        with open(_state_path()) as f:
            state = json.load(f)
        assert state["reset_seconds"] == pytest.approx(45, abs=2)

    def test_records_with_retry_after_header(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        headers = {"retry-after": "60"}
        record_nous_rate_limit(headers=headers)

        with open(_state_path()) as f:
            state = json.load(f)
        assert state["reset_seconds"] == pytest.approx(60, abs=2)

    def test_prefers_hourly_over_per_minute(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        headers = {
            "x-ratelimit-reset-requests-1h": "1800",
            "x-ratelimit-reset-requests": "45",
        }
        record_nous_rate_limit(headers=headers)

        with open(_state_path()) as f:
            state = json.load(f)
        # Should use the hourly value, not the per-minute one
        assert state["reset_seconds"] == pytest.approx(1800, abs=2)

    def test_falls_back_to_error_context_reset_at(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        future_reset = time.time() + 900
        record_nous_rate_limit(
            headers=None,
            error_context={"reset_at": future_reset},
        )

        with open(_state_path()) as f:
            state = json.load(f)
        assert state["reset_at"] == pytest.approx(future_reset, abs=1)

    def test_falls_back_to_default_cooldown(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        record_nous_rate_limit(headers=None)

        with open(_state_path()) as f:
            state = json.load(f)
        # Default is 300 seconds (5 minutes)
        assert state["reset_seconds"] == pytest.approx(300, abs=2)

    def test_custom_default_cooldown(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        record_nous_rate_limit(headers=None, default_cooldown=120.0)

        with open(_state_path()) as f:
            state = json.load(f)
        assert state["reset_seconds"] == pytest.approx(120, abs=2)

    def test_creates_directory_if_missing(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, _state_path

        record_nous_rate_limit(headers={"retry-after": "10"})
        assert os.path.exists(_state_path())


class TestNousRateLimitRemaining:
    """Test checking remaining rate limit time."""

    def test_returns_none_when_no_file(self, rate_guard_env):
        from agent.nous_rate_guard import nous_rate_limit_remaining

        assert nous_rate_limit_remaining() is None

    def test_returns_remaining_seconds_when_active(self, rate_guard_env):
        from agent.nous_rate_guard import record_nous_rate_limit, nous_rate_limit_remaining

        record_nous_rate_limit(headers={"x-ratelimit-reset-requests-1h": "600"})
        remaining = nous_rate_limit_remaining()
        assert remaining is not None
        assert 595 < remaining <= 605  # ~600 seconds, allowing for test execution time

    def test_returns_none_when_expired(self, rate_guard_env):
        from agent.nous_rate_guard import nous_rate_limit_remaining, _state_path

        # Write an already-expired state
        state_dir = os.path.dirname(_state_path())
        os.makedirs(state_dir, exist_ok=True)
        with open(_state_path(), "w") as f:
            json.dump({"reset_at": time.time() - 10, "recorded_at": time.time() - 100}, f)

        assert nous_rate_limit_remaining() is None
        # File should be cleaned up
        assert not os.path.exists(_state_path())

    def test_handles_corrupt_file(self, rate_guard_env):
        from agent.nous_rate_guard import nous_rate_limit_remaining, _state_path

        state_dir = os.path.dirname(_state_path())
        os.makedirs(state_dir, exist_ok=True)
        with open(_state_path(), "w") as f:
            f.write("not valid json{{{")

        assert nous_rate_limit_remaining() is None


class TestClearNousRateLimit:
    """Test clearing rate limit state."""

    def test_clears_existing_file(self, rate_guard_env):
        from agent.nous_rate_guard import (
            record_nous_rate_limit,
            clear_nous_rate_limit,
            nous_rate_limit_remaining,
            _state_path,
        )

        record_nous_rate_limit(headers={"retry-after": "600"})
        assert nous_rate_limit_remaining() is not None

        clear_nous_rate_limit()
        assert nous_rate_limit_remaining() is None
        assert not os.path.exists(_state_path())

    def test_clear_when_no_file(self, rate_guard_env):
        from agent.nous_rate_guard import clear_nous_rate_limit

        # Should not raise
        clear_nous_rate_limit()


class TestFormatRemaining:
    """Test human-readable duration formatting."""

    def test_seconds(self):
        from agent.nous_rate_guard import format_remaining

        assert format_remaining(30) == "30s"

    def test_minutes(self):
        from agent.nous_rate_guard import format_remaining

        assert format_remaining(125) == "2m 5s"

    def test_exact_minutes(self):
        from agent.nous_rate_guard import format_remaining

        assert format_remaining(120) == "2m"

    def test_hours(self):
        from agent.nous_rate_guard import format_remaining

        assert format_remaining(3720) == "1h 2m"


class TestParseResetSeconds:
    """Test header parsing for reset times."""

    def test_case_insensitive_headers(self, rate_guard_env):
        from agent.nous_rate_guard import _parse_reset_seconds

        headers = {"X-Ratelimit-Reset-Requests-1h": "1200"}
        assert _parse_reset_seconds(headers) == 1200.0

    def test_returns_none_for_empty_headers(self):
        from agent.nous_rate_guard import _parse_reset_seconds

        assert _parse_reset_seconds(None) is None
        assert _parse_reset_seconds({}) is None

    def test_ignores_zero_values(self):
        from agent.nous_rate_guard import _parse_reset_seconds

        headers = {"x-ratelimit-reset-requests-1h": "0"}
        assert _parse_reset_seconds(headers) is None

    def test_ignores_invalid_values(self):
        from agent.nous_rate_guard import _parse_reset_seconds

        headers = {"x-ratelimit-reset-requests-1h": "not-a-number"}
        assert _parse_reset_seconds(headers) is None


class TestAuxiliaryClientIntegration:
    """Test that the auxiliary client respects the rate guard."""

    def test_try_nous_skips_when_rate_limited(self, rate_guard_env, monkeypatch):
        from agent.nous_rate_guard import record_nous_rate_limit

        # Record a rate limit
        record_nous_rate_limit(headers={"retry-after": "600"})

        # Mock _read_nous_auth to return valid creds (would normally succeed)
        import agent.auxiliary_client as aux
        monkeypatch.setattr(aux, "_read_nous_auth", lambda: {
            "access_token": "test-token",
            "inference_base_url": "https://api.nous.test/v1",
        })

        result = aux._try_nous()
        assert result == (None, None)

    def test_try_nous_works_when_not_rate_limited(self, rate_guard_env, monkeypatch):
        import agent.auxiliary_client as aux

        # No rate limit recorded — _try_nous should proceed normally
        # (will return None because no real creds, but won't be blocked
        # by the rate guard)
        monkeypatch.setattr(aux, "_read_nous_auth", lambda: None)
        result = aux._try_nous()
        assert result == (None, None)


class TestIsGenuineNousRateLimit:
    """Tell a real account-level 429 apart from an upstream-capacity 429.

    Nous Portal multiplexes upstreams (DeepSeek, Kimi, MiMo, Hermes).
    A 429 from an upstream out of capacity should NOT trip the
    cross-session breaker; a real user-quota 429 should.
    """

    def test_exhausted_hourly_bucket_in_429_headers_is_genuine(self):
        from agent.nous_rate_guard import is_genuine_nous_rate_limit

        headers = {
            "x-ratelimit-limit-requests-1h": "800",
            "x-ratelimit-remaining-requests-1h": "0",
            "x-ratelimit-reset-requests-1h": "3100",
            "x-ratelimit-limit-requests": "200",
            "x-ratelimit-remaining-requests": "198",
            "x-ratelimit-reset-requests": "40",
        }
        assert is_genuine_nous_rate_limit(headers=headers) is True

    def test_exhausted_tokens_bucket_is_genuine(self):
        from agent.nous_rate_guard import is_genuine_nous_rate_limit

        headers = {
            "x-ratelimit-limit-tokens": "800000",
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "45",  # < 60s threshold -> not genuine
            "x-ratelimit-limit-tokens-1h": "8000000",
            "x-ratelimit-remaining-tokens-1h": "0",
            "x-ratelimit-reset-tokens-1h": "1800",  # >= 60s threshold -> genuine
        }
        assert is_genuine_nous_rate_limit(headers=headers) is True

    def test_healthy_headers_on_429_are_upstream_capacity(self):
        # Classic upstream-capacity symptom: Nous edge reports plenty of
        # headroom on every bucket, but returns 429 anyway because
        # upstream (DeepSeek / Kimi / ...) is out of capacity.
        from agent.nous_rate_guard import is_genuine_nous_rate_limit

        headers = {
            "x-ratelimit-limit-requests": "200",
            "x-ratelimit-remaining-requests": "198",
            "x-ratelimit-reset-requests": "40",
            "x-ratelimit-limit-requests-1h": "800",
            "x-ratelimit-remaining-requests-1h": "750",
            "x-ratelimit-reset-requests-1h": "3100",
            "x-ratelimit-limit-tokens": "800000",
            "x-ratelimit-remaining-tokens": "790000",
            "x-ratelimit-reset-tokens": "40",
            "x-ratelimit-limit-tokens-1h": "8000000",
            "x-ratelimit-remaining-tokens-1h": "7800000",
            "x-ratelimit-reset-tokens-1h": "3100",
        }
        assert is_genuine_nous_rate_limit(headers=headers) is False

    def test_bare_429_with_no_headers_is_upstream(self):
        from agent.nous_rate_guard import is_genuine_nous_rate_limit

        assert is_genuine_nous_rate_limit(headers=None) is False
        assert is_genuine_nous_rate_limit(headers={}) is False
        assert is_genuine_nous_rate_limit(
            headers={"content-type": "application/json"}
        ) is False

    def test_exhausted_bucket_with_short_reset_is_not_genuine(self):
        # remaining == 0 but reset in < 60s: almost certainly a
        # secondary per-minute throttle that will clear immediately --
        # not worth tripping the cross-session breaker.
        from agent.nous_rate_guard import is_genuine_nous_rate_limit

        headers = {
            "x-ratelimit-limit-requests": "200",
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "30",
        }
        assert is_genuine_nous_rate_limit(headers=headers) is False

    def test_last_known_state_with_exhausted_bucket_triggers_genuine(self):
        # Headers on the 429 lack rate-limit info, but the previous
        # successful response already showed the hourly bucket
        # exhausted -- the 429 is almost certainly that limit
        # continuing.
        from agent.nous_rate_guard import is_genuine_nous_rate_limit
        from agent.rate_limit_tracker import parse_rate_limit_headers

        prior_headers = {
            "x-ratelimit-limit-requests-1h": "800",
            "x-ratelimit-remaining-requests-1h": "0",
            "x-ratelimit-reset-requests-1h": "2000",
            "x-ratelimit-limit-requests": "200",
            "x-ratelimit-remaining-requests": "100",
            "x-ratelimit-reset-requests": "30",
            "x-ratelimit-limit-tokens": "800000",
            "x-ratelimit-remaining-tokens": "700000",
            "x-ratelimit-reset-tokens": "30",
            "x-ratelimit-limit-tokens-1h": "8000000",
            "x-ratelimit-remaining-tokens-1h": "7000000",
            "x-ratelimit-reset-tokens-1h": "2000",
        }
        last_state = parse_rate_limit_headers(prior_headers, provider="nous")
        assert is_genuine_nous_rate_limit(
            headers=None, last_known_state=last_state
        ) is True

    def test_last_known_state_all_healthy_stays_upstream(self):
        # Prior state was healthy; bare 429 arrives; should be treated
        # as upstream capacity.
        from agent.nous_rate_guard import is_genuine_nous_rate_limit
        from agent.rate_limit_tracker import parse_rate_limit_headers

        prior_headers = {
            "x-ratelimit-limit-requests-1h": "800",
            "x-ratelimit-remaining-requests-1h": "750",
            "x-ratelimit-reset-requests-1h": "2000",
            "x-ratelimit-limit-requests": "200",
            "x-ratelimit-remaining-requests": "180",
            "x-ratelimit-reset-requests": "30",
            "x-ratelimit-limit-tokens": "800000",
            "x-ratelimit-remaining-tokens": "790000",
            "x-ratelimit-reset-tokens": "30",
            "x-ratelimit-limit-tokens-1h": "8000000",
            "x-ratelimit-remaining-tokens-1h": "7900000",
            "x-ratelimit-reset-tokens-1h": "2000",
        }
        last_state = parse_rate_limit_headers(prior_headers, provider="nous")
        assert is_genuine_nous_rate_limit(
            headers=None, last_known_state=last_state
        ) is False

    def test_none_last_state_and_no_headers_is_upstream(self):
        from agent.nous_rate_guard import is_genuine_nous_rate_limit

        assert is_genuine_nous_rate_limit(
            headers=None, last_known_state=None
        ) is False
