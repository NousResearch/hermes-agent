"""Regression tests for the ``gateway.non_conversational`` startup gate.

See #85792. A ``gateway.non_conversational: true`` profile is declared to
have no chat surface — it exists to tick cron only. Any enabled
``platforms.*`` entry contradicts that contract, so the gateway must fail
closed at startup rather than quietly expose a chat channel on a profile
the operator intended to be execution-only.

The gate has two entry points:

* :func:`gateway.run.GatewayRunner.start` — the single-profile path
  (``hermes gateway run``). Must set ``exit_code`` to the fatal-config
  code, request a clean exit, and never actually start any adapter.
* :func:`gateway.run.GatewayRunner._start_one_profile_adapters` — the
  multiplex secondary-profile path. Must raise
  :class:`gateway.run.MultiplexConfigError` so the whole multiplexer
  aborts (rather than half-honor the guard by silently skipping the
  offending profile).

Both entry points call a shared helper,
:func:`gateway.run._non_conversational_startup_violation`, which the
first block of tests covers as a pure function.
"""

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
from gateway.run import (
    GatewayRunner,
    _non_conversational_startup_violation,
)


# ---------------------------------------------------------------------------
# Pure helper: _non_conversational_startup_violation
# ---------------------------------------------------------------------------


class TestNonConversationalViolationHelper:
    """Contract of the pure helper both start paths call."""

    def test_flag_off_returns_none_regardless_of_platforms(self):
        cfg = GatewayConfig(
            non_conversational=False,
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
        )
        assert _non_conversational_startup_violation(cfg) is None

    def test_flag_on_with_no_enabled_platform_returns_none(self):
        cfg = GatewayConfig(
            non_conversational=True,
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=False)},
        )
        assert _non_conversational_startup_violation(cfg) is None

    def test_flag_on_with_empty_platforms_returns_none(self):
        cfg = GatewayConfig(non_conversational=True, platforms={})
        assert _non_conversational_startup_violation(cfg) is None

    def test_flag_on_single_enabled_platform_names_it(self):
        cfg = GatewayConfig(
            non_conversational=True,
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
        )
        reason = _non_conversational_startup_violation(cfg)
        assert reason is not None
        assert "telegram" in reason

    def test_flag_on_multiple_enabled_platforms_sorted_deterministically(self):
        cfg = GatewayConfig(
            non_conversational=True,
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True),
                Platform.DISCORD: PlatformConfig(enabled=True),
                Platform.SLACK: PlatformConfig(enabled=False),
            },
        )
        reason = _non_conversational_startup_violation(cfg)
        assert reason is not None
        # Sorted alphabetically so the message is stable in logs / status.
        assert "discord, telegram" in reason
        assert "slack" not in reason


# ---------------------------------------------------------------------------
# Single-profile path: GatewayRunner.start
# ---------------------------------------------------------------------------


class TestStartRefusesNonConversationalWithPlatforms:
    """The single-profile ``gateway run`` path must fail closed."""

    @pytest.mark.asyncio
    async def test_start_refuses_when_platform_enabled(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = GatewayConfig(
            non_conversational=True,
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
            sessions_dir=tmp_path / "sessions",
        )
        runner = GatewayRunner(config)

        ok = await runner.start()

        assert ok is True
        assert runner.should_exit_cleanly is True
        assert runner.exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
        reason = (runner.exit_reason or "").lower()
        assert "non_conversational" in reason
        assert "telegram" in reason

    @pytest.mark.asyncio
    async def test_start_allows_flag_off_with_platforms(
        self, monkeypatch, tmp_path,
    ):
        """The guard must not fire when the flag is off — otherwise every
        existing conversational profile would break."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = GatewayConfig(
            non_conversational=False,
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=False)},
            sessions_dir=tmp_path / "sessions",
        )
        runner = GatewayRunner(config)

        ok = await runner.start()

        # The runner may still exit for other reasons (no adapters connected,
        # etc.) but must NOT be aborting for the non_conversational violation.
        reason = (runner.exit_reason or "").lower()
        assert "non_conversational" not in reason

    @pytest.mark.asyncio
    async def test_start_allows_flag_on_with_all_platforms_disabled(
        self, monkeypatch, tmp_path,
    ):
        """A cron-only profile with no enabled platform is the intended
        happy path — must not trip the guard."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config = GatewayConfig(
            non_conversational=True,
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=False)},
            sessions_dir=tmp_path / "sessions",
        )
        runner = GatewayRunner(config)

        ok = await runner.start()

        # Ditto the previous test — no non_conversational violation, whatever
        # else the runner does with a platform-less config.
        reason = (runner.exit_reason or "").lower()
        assert "non_conversational" not in reason


# ---------------------------------------------------------------------------
# Multiplex secondary path: _start_one_profile_adapters
# ---------------------------------------------------------------------------


class TestMultiplexSecondaryProfileGuard:
    """The multiplexer must abort — not just skip — the offending profile."""

    @pytest.mark.asyncio
    async def test_secondary_profile_non_conversational_with_platform_raises(
        self, monkeypatch, tmp_path,
    ):
        """A secondary profile setting ``non_conversational: true`` while
        enabling a platform must raise ``MultiplexConfigError``.

        Aborting the whole multiplexer forces the config to be fixed rather
        than running a gateway that half-honors the guard.
        """
        from gateway.run import MultiplexConfigError

        profile_home = tmp_path / "profiles" / "bad"
        profile_home.mkdir(parents=True)

        secondary_cfg = GatewayConfig(
            non_conversational=True,
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
        )

        # Bypass the disk-backed config load and the plugin discovery / runtime
        # scope machinery — the guard only reads the loaded GatewayConfig, and
        # this test only cares that it raises on the offending config.
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: secondary_cfg,
        )
        monkeypatch.setattr(
            "gateway.run._load_gateway_runtime_config", lambda: None,
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.discover_plugins", lambda: None,
        )

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.adapters = {}
        runner._profile_adapters = {}
        runner._snapshot_profile_busy_modes = lambda *a, **kw: None

        with pytest.raises(MultiplexConfigError) as excinfo:
            await runner._start_one_profile_adapters(
                "bad", profile_home, claimed={},
            )

        msg = str(excinfo.value)
        assert "bad" in msg
        assert "non_conversational" in msg
        assert "telegram" in msg
