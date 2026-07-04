"""Tests for the chronoception plugin (elapsed-time sense; wall-clock only)."""
from __future__ import annotations

import time

import pytest

from plugins.chronoception import _pre_llm_call, sense
from plugins.chronoception.settings import DEFAULTS, get_settings


def _cfg(**overrides):
    cfg = dict(DEFAULTS)
    cfg.update(enabled=True)
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def _reset():
    sense.reset_for_tests()
    yield
    sense.reset_for_tests()


def test_first_turn_shows_clock_no_delta():
    out = sense.build("s1", _cfg())
    assert out is not None
    assert "clock" in out and "first turn" in out
    assert out.startswith("<turn-clock>") and out.rstrip().endswith("</turn-clock>")


def test_second_turn_shows_delta():
    sense.build("s2", _cfg())
    sense._LAST_WALL["s2"] = time.time() - 600  # 10 min ago
    out = sense.build("s2", _cfg())
    assert out is not None and "since your last turn" in out and "10 min" in out


def test_long_gap_adds_stale_notice():
    sense.build("s3", _cfg(gap_report_seconds=1800))
    sense._LAST_WALL["s3"] = time.time() - 7200  # 2 h
    out = sense.build("s3", _cfg(gap_report_seconds=1800))
    assert out is not None and "may have moved on" in out


def test_clock_off_small_gap_is_silent():
    sense.build("s4", _cfg(clock=False))
    sense._LAST_WALL["s4"] = time.time() - 60  # 1 min, below threshold
    assert sense.build("s4", _cfg(clock=False)) is None


def test_clock_off_big_gap_warns_only():
    sense.build("s5", _cfg(clock=False, gap_report_seconds=1800))
    sense._LAST_WALL["s5"] = time.time() - 7200
    out = sense.build("s5", _cfg(clock=False, gap_report_seconds=1800))
    assert out is not None
    assert "idle since your last turn" in out and "clock 2" not in out  # no clock line


def test_backward_clock_is_ignored():
    sense.build("s6", _cfg(clock=False, gap_report_seconds=60))
    sense._LAST_WALL["s6"] = time.time() + 500  # future stamp -> negative delta
    assert sense.build("s6", _cfg(clock=False, gap_report_seconds=60)) is None


def test_truncation_keeps_fence():
    sense.build("s7", _cfg())
    sense._LAST_WALL["s7"] = time.time() - 7200  # long gap -> clock + notice (long body)
    out = sense.build("s7", _cfg(max_chars=220))
    assert out is not None and len(out) <= 220
    assert out.startswith("<turn-clock>") and out.rstrip().endswith("</turn-clock>")


def test_disabled_hook_returns_none(monkeypatch):
    import hermes_cli.config as config_mod
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {})
    assert _pre_llm_call(session_id="x") is None


def test_hook_never_raises(monkeypatch):
    import hermes_cli.config as config_mod
    monkeypatch.setattr(
        config_mod, "load_config_readonly",
        lambda: {"chronoception": {"enabled": True}},
    )
    result = _pre_llm_call(session_id="s")
    assert result is None or isinstance(result, dict)


def test_config_sanitized(monkeypatch):
    import hermes_cli.config as config_mod
    monkeypatch.setattr(
        config_mod, "load_config_readonly",
        lambda: {"chronoception": {"enabled": 1, "clock": 0,
                                   "gap_report_seconds": -5, "max_chars": 3}},
    )
    cfg = get_settings()
    assert cfg["enabled"] is True and cfg["clock"] is False
    assert cfg["gap_report_seconds"] == 0 and cfg["max_chars"] == 200  # floored
