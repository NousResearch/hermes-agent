"""Tests for Windows Gateway Fragility Fixes (#48820 / #48852)."""

import os
import sys
import time
import pytest
import subprocess
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import aiohttp
import httpx

from gateway.config import load_gateway_config, _apply_env_overrides, GatewayConfig, PlatformConfig
from gateway.platforms.base import trust_env_for_gateway, resolve_proxy_url
from hermes_cli.gateway_windows import run_watchdog, _assert_windows
from gateway.config import Platform


# ---------------------------------------------------------------------------
# Bug 1: Watchdog tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
class TestWatchdogBehavior:
    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch, tmp_path):
        monkeypatch.setattr(os, "environ", {"HERMES_HOME": str(tmp_path)})
        self.tmp_path = tmp_path
        self.working_dir = str(tmp_path)
        self.argv = ["pythonw.exe", "-m", "hermes_cli.main", "gateway", "run"]
        self.env_overlay = {"HERMES_HOME": str(tmp_path)}

        # Mock standard Windows Job Object API calls
        self.mock_create_job = MagicMock(return_value=123)
        self.mock_set_job = MagicMock(return_value=True)
        self.mock_assign_job = MagicMock(return_value=True)
        
        # Inject mocks
        import ctypes
        monkeypatch.setattr(ctypes.windll.kernel32, "CreateJobObjectW", self.mock_create_job)
        monkeypatch.setattr(ctypes.windll.kernel32, "SetInformationJobObject", self.mock_set_job)
        monkeypatch.setattr(ctypes.windll.kernel32, "AssignProcessToJobObject", self.mock_assign_job)

        # Mock time tracking and sleep
        self.sleep_calls = []
        monkeypatch.setattr(time, "sleep", lambda s: self.sleep_calls.append(s))
        
        # Mock sys.exit to intercept terminations
        self.exit_code = None
        def mock_exit(code):
            self.exit_code = code
            raise SystemExit(code)
        monkeypatch.setattr(sys, "exit", mock_exit)

    def test_watchdog_clean_exit_on_zero(self, monkeypatch):
        """Watchdog must stop cleanly when child exits with code 0."""
        class MockProc:
            pid = 9999
            _handle = 8888
            def wait(self):
                return 0

        mock_popen = MagicMock(return_value=MockProc())
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        with pytest.raises(SystemExit) as excinfo:
            run_watchdog(self.working_dir, self.argv, self.env_overlay)

        assert excinfo.value.code == 0
        assert mock_popen.call_count == 1
        assert self.mock_assign_job.call_count == 1

    def test_watchdog_immediate_restart_on_75(self, monkeypatch):
        """Watchdog must restart immediately without backoff delay or breaker fail count on 75."""
        exits = [75, 0]  # First run: restart requested. Second run: clean exit.
        
        class MockProc:
            pid = 9999
            _handle = 8888
            def __init__(self):
                self.calls = 0
            def wait(self):
                return exits.pop(0)

        mock_proc_instance = MockProc()
        mock_popen = MagicMock(return_value=mock_proc_instance)
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        with pytest.raises(SystemExit) as excinfo:
            run_watchdog(self.working_dir, self.argv, self.env_overlay)

        assert excinfo.value.code == 0
        assert len(self.sleep_calls) == 0  # No delay on 75
        assert mock_popen.call_count == 2

    def test_watchdog_exponential_backoff(self, monkeypatch):
        """Watchdog respawn delay must double exponentially on failures (1s -> 2s -> 4s...)."""
        exits = [1, 1, 0]  # Fail twice, then succeed
        
        class MockProc:
            pid = 9999
            _handle = 8888
            def wait(self):
                return exits.pop(0)

        mock_popen = MagicMock(return_value=MockProc())
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        with pytest.raises(SystemExit) as excinfo:
            run_watchdog(self.working_dir, self.argv, self.env_overlay)

        assert excinfo.value.code == 0
        assert self.sleep_calls == [1.0, 2.0]
        assert mock_popen.call_count == 3

    def test_watchdog_backoff_reset_on_healthy_run(self, monkeypatch):
        """Backoff delay must reset to 1s after a healthy run >= 30s."""
        exits = [1, 1, 0]
        now = [100.0]  # Simulated times
        
        def mock_time():
            return now[0]

        monkeypatch.setattr(time, "time", mock_time)

        class MockProc:
            pid = 9999
            _handle = 8888
            def wait(self):
                # Simulate running for 35 seconds on the second run
                if len(exits) == 2:  # After first failure, before second
                    now[0] += 35.0
                return exits.pop(0)

        mock_popen = MagicMock(return_value=MockProc())
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        with pytest.raises(SystemExit) as excinfo:
            run_watchdog(self.working_dir, self.argv, self.env_overlay)

        assert excinfo.value.code == 0
        # First failure: delay is 1s, backoff becomes 2s
        # Healthy run: delay is 2s, but run duration >= 30s triggers reset, delay goes back to 1s
        assert self.sleep_calls == [1.0, 1.0]

    def test_watchdog_circuit_breaker(self, monkeypatch):
        """Watchdog must exit with code 1 after 5 failures in < 60 seconds."""
        class MockProc:
            pid = 9999
            _handle = 8888
            def wait(self):
                return 1

        mock_popen = MagicMock(return_value=MockProc())
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        # Force time to not advance so all failures happen instantly
        monkeypatch.setattr(time, "time", lambda: 100.0)

        with pytest.raises(SystemExit) as excinfo:
            run_watchdog(self.working_dir, self.argv, self.env_overlay)

        assert excinfo.value.code == 1
        # It should exit on the 5th failure
        assert mock_popen.call_count == 5

    def test_watchdog_log_rotation(self, monkeypatch):
        """Watchdog must rotate gateway-stdio.log to .old if size exceeds 5MB before child spawn."""
        log_dir = self.tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "gateway-stdio.log"
        
        # Write > 5MB of dummy data
        dummy_data = b"x" * (5 * 1024 * 1024 + 100)
        log_file.write_bytes(dummy_data)

        class MockProc:
            pid = 9999
            _handle = 8888
            def wait(self):
                return 0

        mock_popen = MagicMock(return_value=MockProc())
        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        with pytest.raises(SystemExit) as excinfo:
            run_watchdog(self.working_dir, self.argv, self.env_overlay)

        assert excinfo.value.code == 0
        assert (log_dir / "gateway-stdio.log.old").exists()
        assert (log_dir / "gateway-stdio.log.old").stat().st_size > 5 * 1024 * 1024
        assert log_file.exists()
        assert log_file.stat().st_size < 1000


# ---------------------------------------------------------------------------
# Bug 2: enabled: false overrides
# ---------------------------------------------------------------------------

class TestEnabledFalseOverrides:
    def test_explicit_enabled_false_slack(self, monkeypatch):
        """Explicit enabled: false in config must remain disabled even with env tokens present."""
        config = GatewayConfig()
        config.platforms[Platform.SLACK] = PlatformConfig(enabled=False)
        config.platforms[Platform.SLACK].extra["_enabled_explicit"] = True

        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake-token")
        _apply_env_overrides(config)

        assert config.platforms[Platform.SLACK].enabled is False
        assert config.platforms[Platform.SLACK].token == "xoxb-fake-token"

    def test_explicit_enabled_false_generalization(self, monkeypatch):
        """Generalize enabled: false override to all platform configurations."""
        config = GatewayConfig()
        config.platforms[Platform.DISCORD] = PlatformConfig(enabled=False)
        config.platforms[Platform.DISCORD].extra["_enabled_explicit"] = True

        monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-fake-token")
        _apply_env_overrides(config)

        assert config.platforms[Platform.DISCORD].enabled is False
        assert config.platforms[Platform.DISCORD].token == "discord-fake-token"


# ---------------------------------------------------------------------------
# Bug 3: trust_env and proxy isolation
# ---------------------------------------------------------------------------

class TestProxyTrustIsolation:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        for key in ["GATEWAY_TRUST_PROXY", "GATEWAY_TRUST_ENV", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"]:
            monkeypatch.delenv(key, raising=False)

    def test_trust_env_for_gateway_defaults_to_false(self):
        """trust_env_for_gateway must default to False when no env/config overrides exist."""
        assert trust_env_for_gateway() is False

    def test_trust_env_for_gateway_activated_via_env(self, monkeypatch):
        """trust_env_for_gateway must return True when GATEWAY_TRUST_PROXY is set."""
        monkeypatch.setenv("GATEWAY_TRUST_PROXY", "true")
        assert trust_env_for_gateway() is True

    def test_trust_env_for_gateway_activated_via_config(self):
        """trust_env_for_gateway must return True when config has trust_proxy=True."""
        config = GatewayConfig()
        config.trust_proxy = True
        assert trust_env_for_gateway(config) is True

    def test_resolve_proxy_url_ignores_generic_proxies_unless_trusted(self, monkeypatch):
        """resolve_proxy_url must ignore HTTP_PROXY etc. if trust_env_for_gateway is False."""
        monkeypatch.setenv("HTTPS_PROXY", "http://generic-proxy:8080")
        
        # Untrusted
        with patch("gateway.platforms.base.should_trust_env", return_value=False):
            assert resolve_proxy_url() is None

        # Trusted via env override
        monkeypatch.setenv("GATEWAY_TRUST_PROXY", "true")
        with patch("gateway.platforms.base.should_trust_env", return_value=True):
            assert resolve_proxy_url() == "http://generic-proxy:8080"

    def test_resolve_proxy_url_respects_explicit_platform_proxy(self, monkeypatch):
        """resolve_proxy_url must respect explicit platform proxies regardless of trust_env."""
        monkeypatch.setenv("DISCORD_PROXY", "http://discord-explicit-proxy:8080")
        monkeypatch.setenv("HTTPS_PROXY", "http://generic-proxy:8080")

        # Even if trust_env is False, explicit platform proxy is honored
        with patch("gateway.platforms.base.should_trust_env", return_value=False):
            assert resolve_proxy_url("DISCORD_PROXY") == "http://discord-explicit-proxy:8080"

    def test_llm_clients_unpatched(self):
        """Verify LLM clients/other sessions created outside the gateway remain unaffected by overrides."""
        # Standard ClientSession and AsyncClient must default to standard library behaviors,
        # verifying the global monkeypatching was correctly removed.
        import asyncio
        async def run():
            async with aiohttp.ClientSession() as sess:
                assert sess._trust_env is False
        asyncio.run(run())

        cli = httpx.AsyncClient()
        assert cli.trust_env is True  # Standard default for httpx
