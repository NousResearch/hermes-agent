"""compression.hygiene_threshold: the gateway hygiene net's token ratio is a config key (#146216)."""

import logging
from types import SimpleNamespace

import pytest

from gateway import run_turn_hygiene
from gateway.run_turn_hygiene import GatewayTurnHygieneMixin


def _hs():
    return SimpleNamespace(
        model="m", threshold_pct=0.85, compression_enabled=True, hard_msg_limit=5000,
        timeout_seconds=30.0, total_ceiling_seconds=600.0, max_turn_hold_seconds=10.0,
        failure_cooldown_seconds=300.0, config_context_length=None, provider=None, base_url=None,
    )


def _read(compression: dict):
    hs = _hs()
    GatewayTurnHygieneMixin._hmwa_hygiene_read_config(hs, {"compression": compression})
    return hs


@pytest.fixture(autouse=True)
def _fresh_warning_ledger():
    run_turn_hygiene._HYGIENE_PREEMPT_WARNED.clear()
    yield
    run_turn_hygiene._HYGIENE_PREEMPT_WARNED.clear()


def test_default_stays_085_when_key_absent():
    assert _read({}).threshold_pct == 0.85


@pytest.mark.parametrize("raw, expected", [(0.9, 0.9), ("0.95", 0.95), (1, 1.0), (0.6, 0.6)])
def test_valid_ratio_is_applied(raw, expected):
    assert _read({"hygiene_threshold": raw}).threshold_pct == pytest.approx(expected)


@pytest.mark.parametrize("raw", [0, -0.1, 1.5, "abc", True, [0.9]])
def test_invalid_ratio_keeps_default_and_warns(raw, caplog):
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        assert _read({"hygiene_threshold": raw}).threshold_pct == 0.85
    if raw is not True:
        assert "compression.hygiene_threshold" in caplog.text


def test_net_at_or_below_agent_threshold_warns_and_changes_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        hs = _read({"hygiene_threshold": 0.8, "threshold": 0.9})
    assert hs.threshold_pct == 0.8
    assert "before the agent's own compressor runs" in caplog.text


def test_default_net_below_raised_agent_threshold_warns(caplog):
    # The #118984 situation: compression.threshold raised above the 0.85 default.
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        hs = _read({"threshold": 0.95})
    assert hs.threshold_pct == 0.85
    assert "Raise compression.hygiene_threshold" in caplog.text


def test_net_above_agent_threshold_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        hs = _read({"hygiene_threshold": 0.9, "threshold": 0.5})
    assert hs.threshold_pct == 0.9
    assert "hygiene" not in caplog.text


def test_conflict_warning_once_per_value_pair(caplog):
    # The gateway re-reads config on every turn; the same conflict must not spam the log.
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        for _ in range(3):
            _read({"hygiene_threshold": 0.7, "threshold": 0.75})
    assert caplog.text.count("before the agent's own compressor runs") == 1
