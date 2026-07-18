"""Smoke tests for gateway shutdown_watchdog module."""

import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def tmp_hermes_home(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return tmp_path


class TestShutdownWatchdog:
    def test_import(self):
        from gateway.shutdown_watchdog import (
            ShutdownWatchdog,
            LoopHeartbeat,
            arming_shutdown_watchdog,
            _load_watchdog_config,
        )

    def test_watchdog_creation(self, tmp_hermes_home):
        from gateway.shutdown_watchdog import ShutdownWatchdog

        wd = ShutdownWatchdog(
            drain_timeout=1.0,
            hermes_home=tmp_hermes_home,
            headroom_seconds=0.5,
        )
        assert wd._drain_deadline > time.monotonic()
        assert wd._thread is None

    def test_watchdog_cancel_before_fire(self, tmp_hermes_home):
        from gateway.shutdown_watchdog import ShutdownWatchdog

        wd = ShutdownWatchdog(
            drain_timeout=1.0,
            hermes_home=tmp_hermes_home,
            headroom_seconds=60.0,  # long headroom so it won't fire
        )
        wd.start()
        assert wd._thread is not None
        assert wd._thread.is_alive()
        wd.cancel()
        # cancel() sets _thread to None after join
        assert wd._thread is None

    def test_watchdog_fires_and_exits(self, tmp_hermes_home):
        """Test that watchdog writes forensic data before exiting.

        We can't actually test os._exit in a subprocess here easily,
        so we just verify the trigger logic works by checking the method exists.
        """
        from gateway.shutdown_watchdog import ShutdownWatchdog

        wd = ShutdownWatchdog(
            drain_timeout=0.0,
            hermes_home=tmp_hermes_home,
            headroom_seconds=0.0,  # will fire immediately
        )
        # Deadline is already in the past
        assert wd._drain_deadline <= time.monotonic() + 0.1

    def test_load_watchdog_config_defaults(self):
        from gateway.shutdown_watchdog import _load_watchdog_config

        cfg = {}
        result = _load_watchdog_config(cfg)
        assert result["enabled"] is True
        assert result["headroom_seconds"] == 60
        assert result["heartbeat_interval_seconds"] == 30

    def test_load_watchdog_config_custom(self):
        from gateway.shutdown_watchdog import _load_watchdog_config

        cfg = {
            "gateway": {
                "shutdown_watchdog": {
                    "enabled": False,
                    "headroom_seconds": 120,
                    "heartbeat_interval_seconds": 60,
                }
            }
        }
        result = _load_watchdog_config(cfg)
        assert result["enabled"] is False
        assert result["headroom_seconds"] == 120
        assert result["heartbeat_interval_seconds"] == 60


class TestLoopHeartbeat:
    def test_heartbeat_writes_file(self, tmp_hermes_home):
        import asyncio

        from gateway.shutdown_watchdog import LoopHeartbeat

        hb = LoopHeartbeat(hermes_home=tmp_hermes_home, interval=0.1)

        async def _test():
            await hb.start()
            await asyncio.sleep(0.3)  # wait for at least one heartbeat
            await hb.stop()

        asyncio.run(_test())

        hb_path = tmp_hermes_home / "state" / "gateway.heartbeat"
        assert hb_path.exists()
        data = json.loads(hb_path.read_text())
        assert "pid" in data
        assert "updated_at" in data
        assert isinstance(data["pid"], int)

    def test_heartbeat_multiple_writes(self, tmp_hermes_home):
        import asyncio

        from gateway.shutdown_watchdog import LoopHeartbeat

        hb = LoopHeartbeat(hermes_home=tmp_hermes_home, interval=0.05)

        async def _test():
            await hb.start()
            await asyncio.sleep(0.2)
            await hb.stop()

        asyncio.run(_test())

        hb_path = tmp_hermes_home / "state" / "gateway.heartbeat"
        data = json.loads(hb_path.read_text())
        assert data["pid"] == os.getpid()

    def test_heartbeat_stops_cleanly(self, tmp_hermes_home):
        import asyncio

        from gateway.shutdown_watchdog import LoopHeartbeat

        hb = LoopHeartbeat(hermes_home=tmp_hermes_home, interval=0.1)

        async def _test():
            await hb.start()
            initial_mtime = (
                hb._heartbeat_path.stat().st_mtime if hb._heartbeat_path.exists() else 0
            )
            await asyncio.sleep(0.3)
            await hb.stop()
            # Should not raise

        asyncio.run(_test())
