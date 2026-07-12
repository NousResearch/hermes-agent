"""End-to-end integration-ish tests for ``safety_redline.make_protocol``."""

from __future__ import annotations

import time

from safety_redline import make_protocol
from safety_redline.redline import SafetyConfig


def test_make_protocol_uses_conventional_defaults():
    proto = make_protocol()
    assert proto.redline.config.pause_threshold == 3
    assert proto.redline.config.hard_pause_threshold == 4
    assert proto.redline.config.cooldown_seconds == 300.0
    assert proto.redline.config.warn_threshold == 2


def test_make_protocol_propagates_custom_config():
    proto = make_protocol(
        pause_threshold=5,
        hard_pause_threshold=6,
        cooldown_seconds=10.0,
        warn_threshold=3,
    )
    cfg = proto.redline.config
    assert cfg.pause_threshold == 5
    assert cfg.hard_pause_threshold == 6
    assert cfg.cooldown_seconds == 10.0
    assert cfg.warn_threshold == 3


def test_make_protocol_uses_notifier_when_paused():
    seen = []

    def notifier(level, msg, snap):
        seen.append(level)

    proto = make_protocol(notifier=notifier)
    for _ in range(3):
        proto.handle({"type": "safety.report_failure", "body": {"reason": "x"}})
    assert "paused" in seen


def test_without_notifier_returns_clean_config():
    config = SafetyConfig(notifier=lambda *_: None)
    clean = config.without_notifier()
    assert clean.notifier is None
    assert clean.pause_threshold == config.pause_threshold


def test_redline_event_history_exposes_state_changes():
    proto = make_protocol()
    proto.handle({"type": "safety.report_failure", "body": {}})
    proto.handle({"type": "safety.report_failure", "body": {}})
    history = proto.redline.history()
    assert len(history) >= 2
    assert history[-1].state.value in {"healthy", "warn"}
