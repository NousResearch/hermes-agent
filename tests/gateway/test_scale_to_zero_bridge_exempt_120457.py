"""Issue #120457: explicit idle_timeout_minutes 0/negative must DISABLE scale-to-zero.

The issue's own workaround sets ``idle_timeout_minutes: 0`` expecting
"Disabled - bridge stays alive", but ``parse_idle_timeout_seconds`` degrades
0/negative back to the 2-minute default, so an opted-in gateway with a
WhatsApp/Telegram bridge still goes dormant after 2 idle minutes and the
bridge connection dies. Missing/garbage config must keep the default;
only an explicit non-positive number disables.
"""

from __future__ import annotations

import math

import pytest

from gateway.scale_to_zero import (
    DEFAULT_IDLE_TIMEOUT_MINUTES,
    is_idle,
    parse_idle_timeout_seconds,
)


@pytest.mark.parametrize("value", [0, 0.0, "0", -1, -2.5])
def test_explicit_non_positive_disables(value):
    assert parse_idle_timeout_seconds(value) == math.inf


@pytest.mark.parametrize("value", [None, "", "nope"])
def test_missing_or_garbage_keeps_default(value):
    assert parse_idle_timeout_seconds(value) == DEFAULT_IDLE_TIMEOUT_MINUTES * 60.0


def test_disabled_timeout_is_never_idle():
    assert (
        is_idle(
            active_work_count=0,
            seconds_since_last_inbound=1e9,
            idle_timeout_seconds=parse_idle_timeout_seconds(0),
            has_live_background_work=False,
        )
        is False
    )


def test_positive_timeout_still_parses():
    assert parse_idle_timeout_seconds(2) == 120.0


def test_disabled_timeout_never_arms_watcher(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.scale_to_zero import SCALE_TO_ZERO_ENV

    monkeypatch.setenv(SCALE_TO_ZERO_ENV, "1")
    r = GatewayRunner.__new__(GatewayRunner)
    monkeypatch.setattr(r, "_scale_to_zero_idle_timeout_seconds", lambda: float("inf"))
    monkeypatch.setattr(r, "_scale_to_zero_active_messaging_platforms", lambda: [])
    monkeypatch.setattr(r, "_relay_wake_url_or_none", lambda: "https://x.example/sleep")
    assert r._scale_to_zero_should_arm() is False
