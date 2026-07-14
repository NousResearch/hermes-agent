"""Tests for credential pool exponential backoff on repeated exhaustion (issue #15296)."""

from __future__ import annotations

import json
import time

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _make_pool_entry(
    entry_id: str = "cred-1",
    label: str = "primary",
    last_status: str | None = None,
    last_status_at: float | None = None,
    last_error_code: int | None = None,
    consecutive_failures: int = 0,
) -> dict:
    return {
        "id": entry_id,
        "label": label,
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "***",
        "last_status": last_status,
        "last_status_at": last_status_at,
        "last_error_code": last_error_code,
        "consecutive_failures": consecutive_failures,
    }


class TestExhaustedTtlBackoff:
    """Unit tests for _exhausted_ttl with exponential backoff."""

    def test_base_ttl_no_failures(self):
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_TTL_429_SECONDS

        assert _exhausted_ttl(429, 0) == EXHAUSTED_TTL_429_SECONDS

    def test_base_ttl_single_failure(self):
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_TTL_429_SECONDS

        assert _exhausted_ttl(429, 1) == EXHAUSTED_TTL_429_SECONDS

    def test_double_on_second_failure(self):
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_TTL_429_SECONDS

        expected = EXHAUSTED_TTL_429_SECONDS * 2
        assert _exhausted_ttl(429, 2) == expected

    def test_quadruple_on_third_failure(self):
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_TTL_429_SECONDS

        expected = EXHAUSTED_TTL_429_SECONDS * 4
        assert _exhausted_ttl(429, 3) == expected

    def test_capped_at_max(self):
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_BACKOFF_CAP_SECONDS

        assert _exhausted_ttl(429, 100) == EXHAUSTED_BACKOFF_CAP_SECONDS

    def test_401_backoff(self):
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_TTL_401_SECONDS

        assert _exhausted_ttl(401, 0) == EXHAUSTED_TTL_401_SECONDS
        assert _exhausted_ttl(401, 2) == EXHAUSTED_TTL_401_SECONDS * 2

    def test_default_error_code_backoff(self):
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_TTL_DEFAULT_SECONDS

        assert _exhausted_ttl(None, 0) == EXHAUSTED_TTL_DEFAULT_SECONDS
        assert _exhausted_ttl(None, 2) == EXHAUSTED_TTL_DEFAULT_SECONDS * 2

    def test_backoff_never_exceeds_cap(self):
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_BACKOFF_CAP_SECONDS

        for code in (429, 401, 500, None):
            for failures in range(0, 20):
                ttl = _exhausted_ttl(code, failures)
                assert ttl <= EXHAUSTED_BACKOFF_CAP_SECONDS, (
                    f"TTL {ttl} exceeds cap for code={code} failures={failures}"
                )


class TestConsecutiveFailuresTracking:
    """Tests that consecutive_failures is tracked and reset correctly."""

    def test_mark_exhausted_increments_consecutive_failures(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        _write_auth_store(
            tmp_path,
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        _make_pool_entry(
                            entry_id="c1",
                            last_status=None,
                            consecutive_failures=0,
                        ),
                    ]
                },
            },
        )

        from agent.credential_pool import load_pool

        pool = load_pool("anthropic")
        entry = pool.select()
        assert entry is not None

        pool._mark_exhausted(entry, 429)
        entries = pool.entries()
        assert entries[0].consecutive_failures == 1

        # Second exhaustion
        pool._mark_exhausted(entries[0], 429)
        assert pool.entries()[0].consecutive_failures == 2

    def test_recovery_resets_consecutive_failures(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        _write_auth_store(
            tmp_path,
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        _make_pool_entry(
                            entry_id="c1",
                            last_status="exhausted",
                            last_status_at=time.time(),
                            last_error_code=429,
                            consecutive_failures=3,
                        ),
                    ]
                },
            },
        )

        from agent.credential_pool import load_pool

        pool = load_pool("anthropic")
        # reset_statuses should clear consecutive_failures
        pool.reset_statuses()
        entries = pool.entries()
        assert entries[0].consecutive_failures == 0

    def test_select_after_ttl_expiry_resets_consecutive_failures(self, tmp_path, monkeypatch):
        """When select() clears an exhausted entry whose TTL has expired, consecutive_failures resets."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        past = time.time() - 7200  # 2 hours ago, past default TTL
        _write_auth_store(
            tmp_path,
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        _make_pool_entry(
                            entry_id="c1",
                            last_status="exhausted",
                            last_status_at=past,
                            last_error_code=429,
                            consecutive_failures=2,
                        ),
                    ]
                },
            },
        )

        from agent.credential_pool import load_pool

        pool = load_pool("anthropic")
        entry = pool.select()
        assert entry is not None
        assert entry.consecutive_failures == 0

    def test_consecutive_failures_persists_across_load(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        _write_auth_store(
            tmp_path,
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        _make_pool_entry(
                            entry_id="c1",
                            last_status="exhausted",
                            last_status_at=time.time(),
                            last_error_code=429,
                            consecutive_failures=5,
                        ),
                    ]
                },
            },
        )

        from agent.credential_pool import load_pool

        pool = load_pool("anthropic")
        entries = pool.entries()
        assert entries[0].consecutive_failures == 5


class TestBackoffProgression:
    """Integration test: verify the 1h→2h→4h→8h progression for 429s."""

    def test_429_backoff_progression(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        _write_auth_store(
            tmp_path,
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        _make_pool_entry(entry_id="c1"),
                    ]
                },
            },
        )

        from agent.credential_pool import load_pool

        pool = load_pool("anthropic")

        # Simulate repeated exhaustion cycles
        entry = pool.select()
        assert entry is not None

        # 1st failure: base TTL (1h)
        pool._mark_exhausted(entry, 429)
        entries = pool.entries()
        assert entries[0].consecutive_failures == 1

        # 2nd failure: 2x TTL
        pool._mark_exhausted(entries[0], 429)
        entries = pool.entries()
        assert entries[0].consecutive_failures == 2

        # 3rd failure: 4x TTL
        pool._mark_exhausted(entries[0], 429)
        entries = pool.entries()
        assert entries[0].consecutive_failures == 3

        # 4th failure: 8x TTL (cap)
        pool._mark_exhausted(entries[0], 429)
        entries = pool.entries()
        assert entries[0].consecutive_failures == 4

        # Verify TTLs
        from agent.credential_pool import _exhausted_ttl, EXHAUSTED_TTL_429_SECONDS, EXHAUSTED_BACKOFF_CAP_SECONDS

        assert _exhausted_ttl(429, 1) == 3600  # 1h
        assert _exhausted_ttl(429, 2) == 7200  # 2h
        assert _exhausted_ttl(429, 3) == 14400  # 4h
        assert _exhausted_ttl(429, 4) == 28800  # 8h (cap)
        assert EXHAUSTED_TTL_429_SECONDS == 3600
        assert EXHAUSTED_BACKOFF_CAP_SECONDS == 28800
