"""Regression tests for the three Windows gateway fixes (issue #48820).

Bug 2: ``platforms.<x>.enabled: false`` in config.yaml must win over a
       matching env var (the env var must NOT force-enable the platform).
Bug 3: gateway HTTP clients must not inherit ``HTTP_PROXY``/``HTTPS_PROXY``
       from the process environment unless explicitly opted in
       (``trust_env_for_gateway()`` / ``resolve_proxy_url`` gating).
Bug 1: the Windows watchdog must respawn the worker with capped backoff,
       a crash-loop circuit breaker, and exit-code handling (0 = stop,
       75 = restart now).
"""

from __future__ import annotations

import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms import base as base_mod
from gateway.platforms.base import resolve_proxy_url, trust_env_for_gateway


@pytest.fixture(autouse=True)
def _clear_trust_proxy_cache():
    """trust_env_for_gateway() caches the config read; clear it around each
    test so config-patching tests don't leak state to (or from) others."""
    from gateway.platforms.base import _config_trust_proxy
    _config_trust_proxy.cache_clear()
    yield
    _config_trust_proxy.cache_clear()


# ---------------------------------------------------------------------------
# Bug 2 — explicit enabled: false wins over env vars (all platforms)
# ---------------------------------------------------------------------------
class TestEnabledFalseWinsOverEnv:
    def test_explicit_disable_blocks_env_enable(self, monkeypatch, caplog):
        """A platform marked enabled: false in YAML stays disabled even when
        its env token is present; the credential is still stored."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=False, extra={"_enabled_explicit": True}
                ),
            },
        )
        with caplog.at_level(logging.WARNING):
            _apply_env_overrides(config)

        tg = config.platforms[Platform.TELEGRAM]
        assert tg.enabled is False
        # Token still stored so skills can use it without the adapter running.
        assert tg.token == "env-token"
        assert any("remains" in r.message and "disabled" in r.message
                   for r in caplog.records)

    def test_env_enables_when_not_explicitly_disabled(self, monkeypatch):
        """Without an explicit disable marker, the env token enables it."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
        config = GatewayConfig(platforms={})
        _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].enabled is True

    def test_explicit_disable_blocks_env_enable_non_slack(self, monkeypatch):
        """Bug 2 was previously Slack-only; verify it now holds for an
        arbitrary other platform (Home Assistant)."""
        monkeypatch.setenv("HASS_TOKEN", "hass-tok")
        config = GatewayConfig(
            platforms={
                Platform.HOMEASSISTANT: PlatformConfig(
                    enabled=False, extra={"_enabled_explicit": True}
                ),
            },
        )
        _apply_env_overrides(config)
        assert config.platforms[Platform.HOMEASSISTANT].enabled is False


# ---------------------------------------------------------------------------
# Bug 3 — proxy env inheritance is gated (default off)
# ---------------------------------------------------------------------------
class TestProxyTrustGating:
    def _clear(self, monkeypatch):
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy",
                  "http_proxy", "all_proxy", "GATEWAY_TRUST_PROXY",
                  "GATEWAY_TRUST_ENV", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(k, raising=False)

    def test_generic_env_proxy_ignored_by_default(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
        # No opt-in → must not return the inherited proxy.
        assert resolve_proxy_url() is None
        assert trust_env_for_gateway() is False

    def test_env_optin_restores_proxy(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
        monkeypatch.setenv("GATEWAY_TRUST_PROXY", "true")
        assert trust_env_for_gateway() is True
        assert resolve_proxy_url() == "http://127.0.0.1:7890"

    def test_config_optin_restores_proxy(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")

        # _config_trust_proxy is lru_cached; clear it so the patch takes effect.
        from gateway.platforms.base import _config_trust_proxy
        _config_trust_proxy.cache_clear()
        import gateway.config as gw_config
        monkeypatch.setattr(gw_config, "load_gateway_config",
                            lambda: GatewayConfig(trust_proxy=True))
        try:
            assert trust_env_for_gateway() is True
            assert resolve_proxy_url() == "http://127.0.0.1:7890"
        finally:
            _config_trust_proxy.cache_clear()

    def test_explicit_platform_proxy_var_always_honored(self, monkeypatch):
        """Explicit per-platform proxy vars bypass the gate — they are a
        deliberate user choice, not implicit inheritance."""
        self._clear(monkeypatch)
        monkeypatch.setenv("DISCORD_PROXY", "http://127.0.0.1:1080")
        # No GATEWAY_TRUST_* opt-in, yet the explicit var is honored.
        assert resolve_proxy_url("DISCORD_PROXY") == "http://127.0.0.1:1080"

    def test_config_roundtrip_preserves_trust_proxy(self):
        cfg = GatewayConfig(trust_proxy=True)
        restored = GatewayConfig.from_dict(cfg.to_dict())
        assert restored.trust_proxy is True

    def test_legacy_trust_env_alias_maps_to_trust_proxy(self):
        # Back-compat: a config that wrote the old ``trust_env`` key still opts in.
        restored = GatewayConfig.from_dict({"trust_env": True})
        assert restored.trust_proxy is True

    def test_trust_flags_default_false(self):
        assert GatewayConfig().trust_proxy is False


# ---------------------------------------------------------------------------
# Bug 1 — Windows watchdog supervision logic
# ---------------------------------------------------------------------------
class _FakeProc:
    def __init__(self, exit_code, pid=4321):
        self._exit_code = exit_code
        self.pid = pid

    def wait(self):
        return self._exit_code

    def terminate(self):
        pass


class TestWatchdog:
    def _patch_common(self, monkeypatch, tmp_path):
        from hermes_cli import gateway_windows as gw

        monkeypatch.setattr(gw, "_watchdog_assign_job_object", lambda pid: None)
        sleeps: list[float] = []
        monkeypatch.setattr(gw.time, "sleep", lambda s: sleeps.append(s))
        return gw, sleeps

    def test_clean_exit_stops_watchdog(self, monkeypatch, tmp_path):
        gw, sleeps = self._patch_common(monkeypatch, tmp_path)
        spawned = []

        def fake_popen(argv, **kwargs):
            spawned.append(argv)
            return _FakeProc(0)

        monkeypatch.setattr(gw.subprocess, "Popen", fake_popen)
        rc = gw.run_watchdog(str(tmp_path), [r"C:\py\pythonw.exe", "-m",
                             "hermes_cli.main", "gateway", "run"],
                             {"HERMES_HOME": str(tmp_path)})
        assert rc == 0
        assert len(spawned) == 1  # spawned once, exited clean, no respawn
        # python.exe substituted for pythonw.exe for the worker.
        assert spawned[0][0].endswith("python.exe")
        assert not sleeps

    def test_restart_code_respawns_immediately(self, monkeypatch, tmp_path):
        gw, sleeps = self._patch_common(monkeypatch, tmp_path)
        results = iter([75, 0])  # restart-request then clean stop

        def fake_popen(argv, **kwargs):
            return _FakeProc(next(results))

        monkeypatch.setattr(gw.subprocess, "Popen", fake_popen)
        rc = gw.run_watchdog(str(tmp_path), ["pythonw.exe", "-m",
                             "hermes_cli.main", "gateway", "run"], {})
        assert rc == 0
        # Restart (75) must NOT incur a backoff sleep.
        assert not sleeps

    def test_crash_loop_circuit_breaker(self, monkeypatch, tmp_path):
        gw, sleeps = self._patch_common(monkeypatch, tmp_path)
        spawned = {"n": 0}

        def fake_popen(argv, **kwargs):
            spawned["n"] += 1
            return _FakeProc(1)  # always crashes immediately

        monkeypatch.setattr(gw.subprocess, "Popen", fake_popen)
        # monotonic stays effectively still → all crashes land in one window
        # and ran_for stays under the healthy threshold.
        monkeypatch.setattr(gw.time, "monotonic", lambda: 100.0)

        rc = gw.run_watchdog(str(tmp_path), ["pythonw.exe", "-m",
                             "hermes_cli.main", "gateway", "run"], {})
        # Gives up after the burst limit instead of looping forever.
        assert rc == 1
        assert spawned["n"] == gw._WATCHDOG_BURST_LIMIT + 1

    def test_backoff_is_capped_and_exponential(self, monkeypatch, tmp_path):
        gw, sleeps = self._patch_common(monkeypatch, tmp_path)
        # run_watchdog reads monotonic() 3x per failed iteration:
        #   started_at, ran_for(=now-started_at), now(crash-window stamp).
        # Keep all three equal within an iteration → ran_for == 0 (unhealthy,
        # so backoff grows); jump +200s between iterations → crashes stay
        # outside the 60s burst window so the breaker never trips. The final
        # clean-exit iteration reads monotonic once (started_at) then returns.
        clock = {"t": 0.0}
        reads = iter([
            0, 0, 0,          # crash 1
            200, 200, 200,    # crash 2
            400, 400, 400,    # crash 3
            600, 600, 600,    # crash 4
        ])

        def _mono():
            return float(next(reads, 9999.0))

        monkeypatch.setattr(gw.time, "monotonic", _mono)
        results = iter([1, 1, 1, 1, 0])

        def fake_popen(argv, **kwargs):
            return _FakeProc(next(results))

        monkeypatch.setattr(gw.subprocess, "Popen", fake_popen)
        gw.run_watchdog(str(tmp_path), ["pythonw.exe", "-m",
                        "hermes_cli.main", "gateway", "run"], {})
        # Four crashes → four backoff sleeps, exponential and capped.
        assert sleeps == [1.0, 2.0, 4.0, 8.0]
        assert all(s <= gw._WATCHDOG_BACKOFF_CAP_S for s in sleeps)


def test_watchdog_main_rejects_bad_args(monkeypatch):
    from hermes_cli import gateway_windows as gw
    # invalid JSON → return code 2 (no spawn)
    rc = gw._watchdog_main(["watchdog", "--working-dir", ".",
                            "--argv", "not-json", "--env", "{}"])
    assert rc == 2
