"""Tests for agent/codex_rate_guard.py — Codex OAuth retry-storm breaker."""

import json
import os
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def codex_guard_env(tmp_path, monkeypatch):
    hermes_home = str(tmp_path / ".hermes")
    os.makedirs(hermes_home, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", hermes_home)
    return hermes_home


def test_records_explicit_usage_limit_with_resets_in_seconds(codex_guard_env):
    from agent.codex_rate_guard import (
        _state_path,
        codex_rate_limit_remaining,
        record_codex_rate_limit,
    )

    remaining = record_codex_rate_limit(
        error_context={
            "reason": "usage_limit_reached",
            "message": "The usage limit has been reached",
            "resets_in_seconds": 900,
        }
    )

    assert remaining == pytest.approx(900, abs=2)
    assert codex_rate_limit_remaining() == pytest.approx(900, abs=2)
    with open(_state_path(), encoding="utf-8") as f:
        state = json.load(f)
    assert state["reason"] == "usage_limit_reached"


def test_records_explicit_usage_limit_with_default_cooldown(codex_guard_env):
    from agent.codex_rate_guard import codex_rate_limit_remaining, record_codex_rate_limit

    remaining = record_codex_rate_limit(
        error_context={
            "reason": "usage_limit_reached",
            "message": "The usage limit has been reached",
        },
        default_cooldown=120,
    )

    assert remaining == pytest.approx(120, abs=2)
    assert codex_rate_limit_remaining() == pytest.approx(120, abs=2)


def test_ignores_short_transient_retry_after(codex_guard_env):
    from agent.codex_rate_guard import codex_rate_limit_remaining, record_codex_rate_limit

    remaining = record_codex_rate_limit(headers={"Retry-After": "2.5"})

    assert remaining is None
    assert codex_rate_limit_remaining() is None


def test_records_long_retry_after(codex_guard_env):
    from agent.codex_rate_guard import codex_rate_limit_remaining, record_codex_rate_limit

    remaining = record_codex_rate_limit(headers={"Retry-After": "180"})

    assert remaining == pytest.approx(180, abs=2)
    assert codex_rate_limit_remaining() == pytest.approx(180, abs=2)


def test_expired_state_is_cleared(codex_guard_env):
    from agent.codex_rate_guard import _state_path, codex_rate_limit_remaining

    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"reset_at": time.time() - 1}, f)

    assert codex_rate_limit_remaining() is None
    assert not os.path.exists(path)


def test_extracts_context_from_sdk_exception_body(codex_guard_env):
    from agent.codex_rate_guard import (
        codex_rate_limit_remaining,
        record_codex_rate_limit_from_exception,
    )

    class FakeSDKError(Exception):
        body = {
            "error": {
                "type": "usage_limit_reached",
                "message": "The usage limit has been reached",
                "resets_in_seconds": 300,
            }
        }
        response = SimpleNamespace(headers={})

    remaining = record_codex_rate_limit_from_exception(FakeSDKError("rate limit"))

    assert remaining == pytest.approx(300, abs=2)
    assert codex_rate_limit_remaining() == pytest.approx(300, abs=2)


def test_spark_and_standard_usage_limits_are_isolated(codex_guard_env):
    from agent.codex_rate_guard import codex_rate_limit_remaining, record_codex_rate_limit

    remaining = record_codex_rate_limit(
        error_context={
            "reason": "usage_limit_reached",
            "message": "Spark usage limit reached",
            "resets_in_seconds": 600,
        },
        model="gpt-5.3-codex-spark",
    )

    assert remaining == pytest.approx(600, abs=2)
    assert codex_rate_limit_remaining(model="gpt-5.3-codex-spark") == pytest.approx(600, abs=2)
    assert codex_rate_limit_remaining(model="gpt-5.5") is None


def test_global_usage_limit_blocks_every_lane(codex_guard_env):
    from agent.codex_rate_guard import codex_rate_limit_remaining, record_codex_rate_limit

    remaining = record_codex_rate_limit(
        error_context={
            "reason": "usage_limit_reached",
            "message": "Generic Codex usage limit reached",
            "resets_in_seconds": 480,
        }
    )

    assert remaining == pytest.approx(480, abs=2)
    assert codex_rate_limit_remaining(model="gpt-5.3-codex-spark") == pytest.approx(480, abs=2)
    assert codex_rate_limit_remaining(model="gpt-5.5") == pytest.approx(480, abs=2)
