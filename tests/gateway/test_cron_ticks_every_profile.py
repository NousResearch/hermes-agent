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


class _ExternalProvider:
    """A configured external ``cron.provider`` (one unscoped remote registry)."""

    name = "external-test"

    def start(self, stop_event, *, adapters=None, loop=None, interval=60):
        return None

    def stop(self):
        return None


def _started_provider(gateway_run, monkeypatch, launch_profile):
    from cron import scheduler_provider, scheduler_thread

    external = _ExternalProvider()
    monkeypatch.setattr(scheduler_provider, "resolve_cron_scheduler", lambda: external)
    monkeypatch.setattr(scheduler_thread, "SupervisedTickerThread", _CapturedTicker)
    monkeypatch.setattr(gateway_run, "_start_gateway_housekeeping", lambda *_a, **_kw: None)
    runner = types.SimpleNamespace(
        config=types.SimpleNamespace(multiplex_profiles=False),
        adapters={},
        _profile_adapters=None,
        _primary_profile_name=launch_profile,
        _draining=False,
        _external_drain_active=False,
    )

    async def _start():
        return gateway_run._start_gateway_start_cron_and_housekeeping(runner)

    cron_stop, provider, _thread, housekeeping = asyncio.run(_start())
    cron_stop.set()
    if isinstance(housekeeping, threading.Thread):
        housekeeping.join(timeout=5)
    return external, provider


def test_a_standalone_gateway_keeps_its_external_cron_provider(isolated_profiles, monkeypatch):
    """A standalone gateway ticks one home, so an external provider is not forced back to the
    built-in multiplex ticker."""
    from gateway import run as gateway_run

    solo = isolated_profiles / "profiles" / "solo"
    solo.mkdir()
    (solo / "config.yaml").write_text("gateway:\n  standalone: true\n")
    monkeypatch.setenv("HERMES_HOME", str(solo))

    external, provider = _started_provider(gateway_run, monkeypatch, "solo")
    assert provider is external


def test_the_host_gateway_still_uses_the_builtin_ticker_for_several_profiles(
    isolated_profiles, monkeypatch
):
    """Control: the host ticks default + secondary, which an external provider cannot scope."""
    from cron.scheduler_provider import InProcessCronScheduler
    from gateway import run as gateway_run

    external, provider = _started_provider(gateway_run, monkeypatch, "default")
    assert provider is not external
    assert isinstance(provider, InProcessCronScheduler)
