"""The host gateway's cron ticker visits every profile, flag or no flag.

``gateway.multiplex_profiles`` gates ADAPTERS. Cron ownership is not optional: one gateway process
per host ticks every profile's store, so a secondary profile's due job fires even when the flag is
off. Before this, ``_start_gateway_start_cron_and_housekeeping`` only passed ``profile_homes`` when
the flag was on and every non-launch profile's cron silently never fired.
"""

from __future__ import annotations

import asyncio
import threading
import types

import pytest


class _CapturedTicker:
    """Stands in for ``SupervisedTickerThread``; records the ticker's start kwargs."""

    captured: dict = {}

    def __init__(self, target, *, args=(), kwargs=None, stop_event=None, name="cron"):
        _CapturedTicker.captured = dict(kwargs or {})
        self.restarts = 0

    def start(self):
        return None

    def restart_if_dead(self):
        return None

    def is_alive(self):
        return False


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    """A scratch HOME so profile enumeration can never read or write the live install."""
    fake_home = tmp_path / "fakehome"
    hermes_home = fake_home / ".hermes"
    (hermes_home / "profiles" / "secondary").mkdir(parents=True)
    # A dir is only a profile once it carries an identity marker.
    (hermes_home / "profiles" / "secondary" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from hermes_cli import profiles as profiles_mod

    assert str(profiles_mod._get_profiles_root()).startswith(str(tmp_path))
    return hermes_home


def test_secondary_profile_is_ticked_with_multiplex_profiles_off(
    isolated_profiles, monkeypatch
):
    from cron import scheduler_thread
    from gateway import run as gateway_run

    monkeypatch.setattr(scheduler_thread, "SupervisedTickerThread", _CapturedTicker)
    monkeypatch.setattr(
        gateway_run, "_start_gateway_housekeeping", lambda *_a, **_kw: None)

    runner = types.SimpleNamespace(
        config=types.SimpleNamespace(multiplex_profiles=False),
        adapters={},
        _profile_adapters=None,
        _primary_profile_name="default",
        _draining=False,
        _external_drain_active=False,
    )

    async def _start():
        return gateway_run._start_gateway_start_cron_and_housekeeping(runner)

    cron_stop, _provider, _thread, housekeeping = asyncio.run(_start())
    cron_stop.set()
    if isinstance(housekeeping, threading.Thread):
        housekeeping.join(timeout=5)

    enumerate_homes = _CapturedTicker.captured.get("profile_homes")
    assert enumerate_homes is not None, (
        "the ticker must be given the profile set even with multiplex_profiles off"
    )
    served = {name for name, _home in enumerate_homes()}
    assert "secondary" in served, "a secondary profile's cron must still be ticked"
