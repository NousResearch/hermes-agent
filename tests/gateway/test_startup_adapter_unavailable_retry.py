"""A platform with no adapter at boot is queued for retry, not stranded (#126356).

A platform plugin whose load overran its deadline during a slow boot (unclean-reboot I/O contention)
left the enabled platform with no adapter and no reconnect queue entry: the gateway logged one
WARNING and never served the platform again. These tests pin the retry contract: the platform is
queued like any other retryable startup failure, a reconnect pass re-creates the adapter once the
plugin (re)registers it, and a pass that still finds no adapter keeps the platform queued instead of
dropping it.
"""
import time

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


class _HealthyAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(monkeypatch, tmp_path, create_adapter):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_create_adapter", create_adapter)

    async def _no_secondary_profiles():
        return 0

    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", _no_secondary_profiles)
    return runner


@pytest.mark.asyncio
async def test_missing_adapter_at_boot_is_queued_for_retry(monkeypatch, tmp_path):
    """The #126356 outage shape: no adapter at boot must queue the platform with a visible
    ``retrying``/``adapter_unavailable`` status instead of one WARNING and silence."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        assert runner.adapters == {}
        info = runner._failed_platforms.get(Platform.TELEGRAM)
        assert info is not None, "platform with no adapter must be queued for reconnect"
        assert info["adapter_unavailable"] is True
        state = read_runtime_status()
        assert state["platforms"]["telegram"]["state"] == "retrying"
        assert state["platforms"]["telegram"]["error_code"] == "adapter_unavailable"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_queued_platform_heals_once_the_plugin_registers(monkeypatch, tmp_path):
    """Once adapter creation succeeds again (the loader's retry re-registered the platform), one
    reconnect pass installs the adapter and the platform leaves the queue connected."""
    available: list = [None]

    def _create(platform, cfg):
        return available[0]

    runner = _runner(monkeypatch, tmp_path, _create)
    try:
        assert await runner.start() is True
        assert Platform.TELEGRAM in runner._failed_platforms
        # The plugin finishes loading: adapter creation now works. One watcher pass heals.
        available[0] = _HealthyAdapter()
        info = runner._failed_platforms[Platform.TELEGRAM]
        info["next_retry"] = 0
        await runner._reconnect_failed_platform(Platform.TELEGRAM, time.monotonic())
        assert Platform.TELEGRAM not in runner._failed_platforms
        assert isinstance(runner.adapters.get(Platform.TELEGRAM), _HealthyAdapter)
        state = read_runtime_status()
        assert state["platforms"]["telegram"]["state"] == "connected"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_reconnect_without_adapter_stays_queued(monkeypatch, tmp_path):
    """A reconnect pass that still finds no adapter must NOT drop the platform (the pre-fix
    behaviour) — it backs off and keeps the entry for the plugin's eventual registration."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: None)
    try:
        assert await runner.start() is True
        info = runner._failed_platforms[Platform.TELEGRAM]
        info["next_retry"] = 0
        attempts_before = info["attempts"]
        await runner._reconnect_failed_platform(Platform.TELEGRAM, time.monotonic())
        assert Platform.TELEGRAM in runner._failed_platforms, "adapterless retry must not drop the platform"
        assert runner._failed_platforms[Platform.TELEGRAM]["attempts"] == attempts_before + 1
        state = read_runtime_status()
        assert state["platforms"]["telegram"]["state"] == "retrying"
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_reconnect_without_flag_still_drops_unknown_platform(monkeypatch, tmp_path):
    """Protection: a platform queued WITHOUT the adapter_unavailable marker (it had an adapter that
    later vanished, e.g. plugin uninstalled mid-run) keeps the old drop-on-None semantics."""
    runner = _runner(monkeypatch, tmp_path, lambda platform, cfg: _HealthyAdapter())
    try:
        assert await runner.start() is True
        assert isinstance(runner.adapters.get(Platform.TELEGRAM), _HealthyAdapter)
        # Simulate a runtime fatal queue entry, then the plugin disappearing.
        adapter = runner.adapters.pop(Platform.TELEGRAM)
        runner._failed_platforms[Platform.TELEGRAM] = runner._reconnect_queue_entry(
            Platform.TELEGRAM, adapter, runner.config.platforms[Platform.TELEGRAM],
            attempts=0, delay=0.0,
        )
        monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)
        await runner._reconnect_failed_platform(Platform.TELEGRAM, time.monotonic())
        assert Platform.TELEGRAM not in runner._failed_platforms
    finally:
        await runner.stop()
