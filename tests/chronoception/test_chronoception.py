"""Tests for the chronoception plugin (elapsed-time sense; wall-clock only)."""
from __future__ import annotations

import time

import pytest

from plugins.chronoception import _post_llm_call, _pre_llm_call, sense
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
    sense.record_completion("s2", time.time() - 600)
    out = sense.build("s2", _cfg())
    assert out is not None and "since your last turn" in out and "10 min" in out


def test_long_gap_adds_stale_notice():
    sense.record_completion("s3", time.time() - 7200)
    out = sense.build("s3", _cfg(gap_report_seconds=1800))
    assert out is not None and "may have moved on" in out


def test_clock_off_small_gap_is_silent():
    sense.record_completion("s4", time.time() - 60)
    assert sense.build("s4", _cfg(clock=False)) is None


def test_clock_off_big_gap_warns_only():
    sense.record_completion("s5", time.time() - 7200)
    out = sense.build("s5", _cfg(clock=False, gap_report_seconds=1800))
    assert out is not None
    assert "idle since your last turn" in out and "clock 2" not in out  # no clock line


def test_backward_clock_is_ignored():
    sense.record_completion("s6", time.time() + 500)
    assert sense.build("s6", _cfg(clock=False, gap_report_seconds=60)) is None


def test_truncation_keeps_fence():
    sense.record_completion("s7", time.time() - 7200)
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


def test_long_prior_turn_is_not_reported_as_idle(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"chronoception": {"enabled": True, "gap_report_seconds": 30}},
    )
    clock = iter([100.0, 200.0, 201.0])
    monkeypatch.setattr(sense.time, "time", lambda: next(clock))

    first = _pre_llm_call(session_id="lifecycle")
    assert first and "first turn" in first["context"]
    _post_llm_call(session_id="lifecycle")
    second = _pre_llm_call(session_id="lifecycle")
    assert second and "+1s since your last turn" in second["context"]
    assert "may have moved on" not in second["context"]


def test_plugin_manager_discovers_enabled_chronoception(monkeypatch):
    import hermes_cli.config as config_mod
    from hermes_cli.plugins import PluginManager

    config = {
            "plugins": {"enabled": ["chronoception"]},
            "chronoception": {"enabled": True},
        }
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: config)
    monkeypatch.setattr(config_mod, "load_config", lambda: config)
    manager = PluginManager()
    manager.discover_and_load()

    assert manager.has_hook("pre_llm_call")
    assert manager.has_hook("post_llm_call")
