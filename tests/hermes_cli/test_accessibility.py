"""display.reduced_motion resolution (port of openai/codex#46040): explicit settings beat the
screen-reader probe, and the probe never stalls startup."""
from __future__ import annotations

import subprocess

import pytest

from hermes_cli import accessibility


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("HERMES_REDUCED_MOTION", raising=False)
    monkeypatch.delenv("HERMES_SCREEN_READER", raising=False)
    accessibility.reset_caches()
    yield
    accessibility.reset_caches()


def test_explicit_settings_beat_the_screen_reader_probe(monkeypatch):
    monkeypatch.setenv("HERMES_SCREEN_READER", "1")
    # Probe says reader → auto on.
    assert accessibility.reduced_motion_enabled({}) is True
    # display.reduced_motion: false is the user's word: animations stay.
    assert accessibility.reduced_motion_enabled({"reduced_motion": False}) is False
    # The launch-time env override beats both.
    monkeypatch.setenv("HERMES_REDUCED_MOTION", "0")
    assert accessibility.reduced_motion_enabled({"reduced_motion": True}) is False
    # No reader, nothing configured → animations on.
    monkeypatch.setenv("HERMES_SCREEN_READER", "0")
    monkeypatch.delenv("HERMES_REDUCED_MOTION")
    accessibility.reset_caches()
    assert accessibility.reduced_motion_enabled({}) is False


def test_probe_is_bounded_and_fails_closed_to_animations(monkeypatch):
    """A hung/absent probe command means 'no reader' and returns within the timeout budget."""
    monkeypatch.setattr(accessibility.sys, "platform", "linux")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/nonexistent")
    seen: list[float] = []

    def hung(argv, **kwargs):
        seen.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(accessibility.subprocess, "run", hung)
    assert accessibility.screen_reader_active() is False
    assert seen and all(t <= accessibility._PROBE_TIMEOUT_S for t in seen)
    # No display config, no reader → the per-process default keeps animations on.
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {"display": {}}
    )
    assert accessibility.reduced_motion_enabled() is False
