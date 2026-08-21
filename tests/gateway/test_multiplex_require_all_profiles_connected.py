"""gateway.require_all_profiles_connected: fail startup on an offline profile."""
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner, ProfileConnectivityError


def _runner(**config_kwargs) -> GatewayRunner:
    """A GatewayRunner with just the attributes the startup path reads."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True, **config_kwargs)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._profile_startup_failures = {}
    runner._profile_relay_served = set()
    runner._failed_platforms = {}
    return runner


class TestAssertAllProfilesConnected:
    def test_default_off_tolerates_offline_profile(self):
        runner = _runner()
        runner._profile_startup_failures = {"reviewer": ["telegram: failed to connect"]}

        runner._assert_all_profiles_connected(["reviewer"], "default")

    def test_opt_in_raises_naming_the_failed_profile(self):
        runner = _runner(require_all_profiles_connected=True)
        runner.adapters = {Platform.TELEGRAM: object()}
        runner._profile_adapters = {"ops": {Platform.DISCORD: object()}}
        runner._profile_startup_failures = {"reviewer": ["telegram: failed to connect"]}

        with pytest.raises(ProfileConnectivityError) as excinfo:
            runner._assert_all_profiles_connected(["reviewer", "ops"], "default")

        message = str(excinfo.value)
        assert "reviewer" in message
        assert "telegram: failed to connect" in message
        assert "1 of 3 served profile(s)" in message
        # The remedy has to be in the message: this aborts the gateway.
        assert "require_all_profiles_connected" in message
        # A profile that came up must not be blamed.
        assert "ops (" not in message

    def test_opt_in_passes_when_every_profile_connected(self, caplog):
        runner = _runner(require_all_profiles_connected=True)
        runner.adapters = {Platform.TELEGRAM: object()}
        runner._profile_adapters = {"reviewer": {Platform.DISCORD: object()}}

        caplog.set_level(logging.INFO, logger="gateway.run")
        runner._assert_all_profiles_connected(["reviewer"], "default")

        assert "all 2 served profile(s) connected" in caplog.text

    def test_opt_in_flags_profile_with_zero_adapters(self):
        runner = _runner(require_all_profiles_connected=True)
        runner.adapters = {Platform.TELEGRAM: object()}

        with pytest.raises(ProfileConnectivityError, match="no platform adapter"):
            runner._assert_all_profiles_connected(["reviewer"], "default")

    def test_relay_only_profile_is_not_offline(self):
        """Multiplex skips a secondary's Relay adapter by design, not by failure."""
        runner = _runner(require_all_profiles_connected=True)
        runner.adapters = {Platform.TELEGRAM: object()}
        runner._profile_relay_served = {"reviewer"}

        runner._assert_all_profiles_connected(["reviewer"], "default")

    def test_offline_active_profile_is_caught(self):
        """The active profile is served too; its adapters start elsewhere."""
        runner = _runner(require_all_profiles_connected=True)
        runner._profile_adapters = {"reviewer": {Platform.DISCORD: object()}}

        with pytest.raises(ProfileConnectivityError) as excinfo:
            runner._assert_all_profiles_connected(["reviewer"], "default")

        assert "default (no platform adapter" in str(excinfo.value)

    def test_active_profile_queued_for_retry_counts_as_offline(self):
        runner = _runner(require_all_profiles_connected=True)
        runner.adapters = {Platform.TELEGRAM: object()}
        runner._failed_platforms = {Platform.DISCORD: {}}

        with pytest.raises(ProfileConnectivityError, match="queued for background"):
            runner._assert_all_profiles_connected([], "default")

    def test_truthy_non_bool_does_not_arm_the_abort(self):
        """A MagicMock config must not abort a gateway it never opted in."""
        runner = _runner()
        runner.config = MagicMock()

        runner._assert_all_profiles_connected(["reviewer"], "default")


class TestSecondaryStartupIntegration:
    @pytest.mark.asyncio
    async def test_default_off_still_skips_and_warns(self, monkeypatch, caplog):
        """The pre-existing log-and-continue path is byte-identical when off."""
        from gateway.run import SecondaryPortBindingConfigError

        runner = _runner()
        runner.pairing_stores = {"default": MagicMock(), "bad": MagicMock()}
        runner.pairing_store = runner.pairing_stores["default"]

        async def fake_start_one(profile_name, profile_home, claimed):
            raise SecondaryPortBindingConfigError("bad enables webhook")

        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist=None: [
                ("default", Path("/tmp/default")),
                ("bad", Path("/tmp/bad")),
            ],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )
        monkeypatch.setattr(runner, "_start_one_profile_adapters", fake_start_one)
        monkeypatch.setattr(
            "gateway.status.write_runtime_status", lambda **kwargs: None
        )

        caplog.set_level(logging.WARNING, logger="gateway.run")
        assert await runner._start_secondary_profile_adapters() == 0
        assert "Skipping secondary profile 'bad'" in caplog.text

    @pytest.mark.asyncio
    async def test_opt_in_turns_the_same_skip_into_a_fatal(self, monkeypatch):
        from gateway.run import SecondaryPortBindingConfigError

        runner = _runner(require_all_profiles_connected=True)
        runner.adapters = {Platform.TELEGRAM: object()}
        runner.pairing_stores = {"default": MagicMock(), "bad": MagicMock()}
        runner.pairing_store = runner.pairing_stores["default"]

        async def fake_start_one(profile_name, profile_home, claimed):
            raise SecondaryPortBindingConfigError("bad enables webhook")

        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist=None: [
                ("default", Path("/tmp/default")),
                ("bad", Path("/tmp/bad")),
            ],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "default"
        )
        monkeypatch.setattr(runner, "_start_one_profile_adapters", fake_start_one)

        with pytest.raises(ProfileConnectivityError, match="port-binding config error"):
            await runner._start_secondary_profile_adapters()


class TestConfigRoundTrip:
    def test_defaults_off(self):
        assert GatewayConfig().require_all_profiles_connected is False
        assert (
            GatewayConfig.from_dict({}).require_all_profiles_connected is False
        )

    def test_top_level_key(self):
        cfg = GatewayConfig.from_dict({"require_all_profiles_connected": True})
        assert cfg.require_all_profiles_connected is True

    def test_nested_gateway_key(self):
        """`hermes config set gateway.<key> true` writes the nested form."""
        cfg = GatewayConfig.from_dict(
            {"gateway": {"require_all_profiles_connected": True}}
        )
        assert cfg.require_all_profiles_connected is True

    def test_top_level_wins_over_nested(self):
        cfg = GatewayConfig.from_dict(
            {
                "require_all_profiles_connected": False,
                "gateway": {"require_all_profiles_connected": True},
            }
        )
        assert cfg.require_all_profiles_connected is False

    def test_yaml_string_is_coerced(self):
        cfg = GatewayConfig.from_dict({"require_all_profiles_connected": "true"})
        assert cfg.require_all_profiles_connected is True

    def test_to_dict_round_trip(self):
        cfg = GatewayConfig(require_all_profiles_connected=True)
        assert cfg.to_dict()["require_all_profiles_connected"] is True
        assert (
            GatewayConfig.from_dict(cfg.to_dict()).require_all_profiles_connected
            is True
        )
