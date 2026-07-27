"""Unit tests for the scale-to-zero idle-detection pure logic (Phase 0).

Behaviour-contract tests (AGENTS.md): each conjunct of the idle predicate and
each clause of the arm-gate is exercised independently, not frozen against a
snapshot. The pure helpers in gateway/scale_to_zero.py take plain inputs so they
test without a live gateway.
"""

from __future__ import annotations

import pytest

from gateway.scale_to_zero import (
    DEFAULT_IDLE_TIMEOUT_MINUTES,
    SCALE_TO_ZERO_ENV,
    is_idle,
    messaging_is_relay_only_or_absent,
    parse_idle_timeout_seconds,
    scale_to_zero_enabled,
    should_arm,
)


# ── scale_to_zero_enabled (the Labs HERMES_SCALE_TO_ZERO stamp, D11/Q8=A) ────


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_enabled_truthy_values(value):
    assert scale_to_zero_enabled({SCALE_TO_ZERO_ENV: value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "nope"])
def test_enabled_falsey_values(value):
    assert scale_to_zero_enabled({SCALE_TO_ZERO_ENV: value}) is False


def test_enabled_absent_is_false():
    # Fail-safe default OFF when the stamp is absent (a non-opted instance).
    assert scale_to_zero_enabled({}) is False


# ── parse_idle_timeout_seconds (config.yaml, D2) ─────────────────────────────


def test_timeout_parses_minutes_to_seconds():
    assert parse_idle_timeout_seconds(5) == 300.0
    assert parse_idle_timeout_seconds(10) == 600.0
    assert parse_idle_timeout_seconds("5") == 300.0


@pytest.mark.parametrize("bad", [None, "", "abc", {}, [], object()])
def test_timeout_degrades_to_default_on_garbage(bad):
    assert parse_idle_timeout_seconds(bad) == DEFAULT_IDLE_TIMEOUT_MINUTES * 60.0


@pytest.mark.parametrize("nonpos", [0, -1, -30, "0", "-5"])
def test_timeout_rejects_nonpositive(nonpos):
    # A zero/negative timeout would go dormant instantly — never the intent.
    assert parse_idle_timeout_seconds(nonpos) == DEFAULT_IDLE_TIMEOUT_MINUTES * 60.0


# ── messaging_is_relay_only_or_absent (F6/D1) ────────────────────────────────


class _P:
    """Stand-in for a Platform enum member with a ``.value``."""

    def __init__(self, value):
        self.value = value


def test_relay_only_is_true():
    assert messaging_is_relay_only_or_absent([_P("relay")]) is True


def test_no_platform_is_true():
    # A Chronos-only / no-messaging-platform agent can scale to zero.
    assert messaging_is_relay_only_or_absent([]) is True


def test_direct_socket_platform_disarms():
    assert messaging_is_relay_only_or_absent([_P("discord")]) is False
    assert messaging_is_relay_only_or_absent([_P("relay"), _P("telegram")]) is False


def test_accepts_bare_strings_too():
    assert messaging_is_relay_only_or_absent(["relay"]) is True
    assert messaging_is_relay_only_or_absent(["discord"]) is False


# ── should_arm (D1/D11/§3.4(1)) ──────────────────────────────────────────────


def test_arm_requires_all_three():
    assert should_arm(enabled=True, relay_only_or_absent=True, wake_url="https://x") is True


def test_arm_blocked_when_flag_off():
    assert should_arm(enabled=False, relay_only_or_absent=True, wake_url="https://x") is False


def test_arm_blocked_when_direct_socket():
    assert should_arm(enabled=True, relay_only_or_absent=False, wake_url="https://x") is False


def test_arm_blocked_without_wake_url():
    # A suspended instance with no wake target is a black hole (§3.4(1)).
    assert should_arm(enabled=True, relay_only_or_absent=True, wake_url=None) is False
    assert should_arm(enabled=True, relay_only_or_absent=True, wake_url="") is False


# ── is_idle (D2/D3/F7) — each conjunct flips the result ──────────────────────


def _idle_kwargs(**over):
    base = dict(
        running_agent_count=0,
        seconds_since_last_inbound=600.0,
        idle_timeout_seconds=300.0,
        has_live_background_work=False,
    )
    base.update(over)
    return base


def test_idle_true_when_all_quiet():
    assert is_idle(**_idle_kwargs()) is True


def test_not_idle_with_running_agent():
    assert is_idle(**_idle_kwargs(running_agent_count=1)) is False


def test_not_idle_within_timeout_window():
    assert is_idle(**_idle_kwargs(seconds_since_last_inbound=120.0)) is False


def test_idle_exactly_at_threshold():
    # >= timeout is idle (boundary).
    assert is_idle(**_idle_kwargs(seconds_since_last_inbound=300.0)) is True


def test_not_idle_with_live_background_work():
    assert is_idle(**_idle_kwargs(has_live_background_work=True)) is False
