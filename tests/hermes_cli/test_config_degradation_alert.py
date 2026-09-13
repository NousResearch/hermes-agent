"""A config load that falls back to DEFAULTS must leave a countable trace.

Live evidence (TONY, 2026-09-12): a blocked read dropped the whole user config while the only
signal was a WARNING naming a YAML problem that did not exist. The mis-wording is fixed; this pins
the escalation that makes the event unmistakable and countable by monitoring.

Contract:

  1. No last-known-good config in the process + unreadable config.yaml → ERROR record *and* a line
     in ``logs/config-degraded.log``.
  2. A last-known-good config keeps serving (the #31188 behaviour) and records no degradation.
"""

from __future__ import annotations

import builtins
import errno
import logging

import pytest

from hermes_cli import config as config_mod


@pytest.fixture
def config_home(tmp_path):
    """Temp config.yaml plus clean loader state for this path."""
    path = config_mod.get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config_mod._CONFIG_PARSE_FAILURES.clear()
    config_mod._CONFIG_PARSE_WARNED.clear()
    config_mod._LOAD_CONFIG_CACHE.clear()
    config_mod._RAW_CONFIG_CACHE.clear()
    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    return path


def _marker_lines(path) -> list:
    marker = path.parent / "logs" / "config-degraded.log"
    return marker.read_text(encoding="utf-8").splitlines() if marker.exists() else []


def _deny_config_reads(path, monkeypatch) -> None:
    real_open = builtins.open

    def denied(file, *args, **kwargs):
        if str(file) == str(path):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denied)


def _degradation_records(caplog) -> list:
    return [r for r in caplog.records
            if r.levelno >= logging.ERROR and "no last-known-good" in r.getMessage()]


def test_running_on_defaults_is_recorded_when_there_is_nothing_to_fall_back_to(
    config_home, monkeypatch, caplog,
):
    config_home.write_text("display:\n  skin: usertheme\n", encoding="utf-8")
    _deny_config_reads(config_home, monkeypatch)

    with caplog.at_level(logging.ERROR):
        cfg = config_mod.load_config()

    assert cfg["display"]["skin"] != "usertheme", "precondition: defaults really are in effect"
    assert _degradation_records(caplog), "the drop to DEFAULTS was not escalated"
    assert _marker_lines(config_home), "monitoring has no countable trace of the degradation"


def test_last_known_good_keeps_serving_and_records_no_degradation(
    config_home, monkeypatch, caplog,
):
    config_home.write_text("display:\n  skin: usertheme\n", encoding="utf-8")
    assert config_mod.load_config()["display"]["skin"] == "usertheme"  # seeds last-known-good

    config_mod._LOAD_CONFIG_CACHE.clear()
    _deny_config_reads(config_home, monkeypatch)

    with caplog.at_level(logging.ERROR):
        cfg = config_mod.load_config()

    assert cfg["display"]["skin"] == "usertheme", "the last-known-good config must keep serving"
    assert not _degradation_records(caplog), "a served last-known-good is not a degradation"
    assert _marker_lines(config_home) == []
