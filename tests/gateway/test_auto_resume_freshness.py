"""Unit tests for the extracted gateway auto-resume freshness seam."""

from datetime import datetime

import pytest

from gateway.runtime.auto_resume_freshness import (
    auto_continue_freshness_window,
    coerce_gateway_timestamp,
    is_fresh_gateway_interruption,
    last_transcript_timestamp,
    startup_auto_resume_max,
)


def test_coerce_gateway_timestamp_preserves_supported_inputs():
    dt = datetime.fromisoformat("2026-04-18T12:00:00+00:00")

    assert coerce_gateway_timestamp(dt) == pytest.approx(dt.timestamp(), abs=1e-3)
    assert coerce_gateway_timestamp(1_700_000_000) == 1_700_000_000.0
    assert coerce_gateway_timestamp(1_700_000_000_000) == 1_700_000_000.0
    assert coerce_gateway_timestamp("1700000000") == 1_700_000_000.0
    assert coerce_gateway_timestamp("2026-04-18T12:00:00Z") == pytest.approx(
        dt.timestamp(), abs=1e-3
    )


def test_coerce_gateway_timestamp_legacy_unknown_values_are_none():
    assert coerce_gateway_timestamp(None) is None
    assert coerce_gateway_timestamp("") is None
    assert coerce_gateway_timestamp("not-a-timestamp") is None
    assert coerce_gateway_timestamp(True) is None
    assert coerce_gateway_timestamp([1, 2, 3]) is None


def test_is_fresh_gateway_interruption_matches_existing_policy():
    now = 1_700_000_000.0

    assert is_fresh_gateway_interruption(None, now=now, window_secs=3600) is True
    assert is_fresh_gateway_interruption(now - 1800, now=now, window_secs=3600) is True
    assert is_fresh_gateway_interruption(now - 3600, now=now, window_secs=3600) is True
    assert is_fresh_gateway_interruption(now - 7200, now=now, window_secs=3600) is False
    assert is_fresh_gateway_interruption(0.0, now=now, window_secs=0) is True


def test_last_transcript_timestamp_skips_meta_and_preserves_legacy_freshness():
    assert last_transcript_timestamp(None) is None
    assert last_transcript_timestamp([]) is None
    assert (
        last_transcript_timestamp(
            [
                {"role": "user", "timestamp": 100.0},
                {"role": "assistant", "timestamp": 200.0},
                {"role": "session_meta", "timestamp": 999.0},
                {"role": "system", "timestamp": 998.0},
            ]
        )
        == 200.0
    )
    assert last_transcript_timestamp([{"role": "assistant", "content": "legacy"}]) is None


def test_env_config_helpers_preserve_defaults_and_malformed_fallback(monkeypatch):
    monkeypatch.delenv("HERMES_AUTO_CONTINUE_FRESHNESS", raising=False)
    monkeypatch.delenv("HERMES_STARTUP_AUTO_RESUME_MAX", raising=False)
    assert auto_continue_freshness_window() == 3600.0
    assert startup_auto_resume_max() == 1

    monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "7200")
    monkeypatch.setenv("HERMES_STARTUP_AUTO_RESUME_MAX", "3")
    assert auto_continue_freshness_window() == 7200.0
    assert startup_auto_resume_max() == 3

    monkeypatch.setenv("HERMES_AUTO_CONTINUE_FRESHNESS", "oops")
    monkeypatch.setenv("HERMES_STARTUP_AUTO_RESUME_MAX", "oops")
    assert auto_continue_freshness_window() == 3600.0
    assert startup_auto_resume_max() == 1
