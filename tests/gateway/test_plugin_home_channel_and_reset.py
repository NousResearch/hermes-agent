"""Tests for plugin-declared home_channel + default_reset_policy bootstrap.

Built-in platforms hard-code their `<PLATFORM>_HOME_CHANNEL` env-var
loading in ``_apply_env_overrides()``. Plugin platforms get the same
treatment via two ``PlatformEntry`` fields:

* ``home_channel_env`` — env var name to read for the default chat ID.
  ``<env>_NAME`` and ``<env>_THREAD_ID`` siblings are also read.
* ``default_reset_policy`` — applied to ``config.reset_by_platform`` when
  the user has not configured one for this platform.
"""

import pytest


@pytest.fixture
def isolated_registry(monkeypatch):
    """Replace the module-level platform_registry with a fresh instance.

    Prevents cross-test pollution and keeps real bundled platform plugins
    (irc, teams) from interfering with these tests.
    """
    from gateway import platform_registry as pr_module

    fresh = pr_module.PlatformRegistry()
    monkeypatch.setattr(pr_module, "platform_registry", fresh)
    return fresh


def _register_test_platform(
    registry,
    *,
    name: str,
    home_channel_env: str = "",
    default_reset_policy=None,
    check_ok: bool = True,
):
    from gateway.platform_registry import PlatformEntry

    registry.register(
        PlatformEntry(
            name=name,
            label=name.title(),
            adapter_factory=lambda cfg: object(),
            check_fn=lambda: check_ok,
            source="plugin",
            home_channel_env=home_channel_env,
            default_reset_policy=default_reset_policy,
        )
    )


class TestPluginHomeChannelBootstrap:
    def test_home_channel_loaded_from_env(self, monkeypatch, isolated_registry):
        from gateway.config import (
            GatewayConfig,
            Platform,
            _apply_env_overrides,
        )

        _register_test_platform(
            isolated_registry,
            name="acme",
            home_channel_env="ACME_HOME_CHANNEL",
        )
        monkeypatch.setenv("ACME_HOME_CHANNEL", "room-42")
        monkeypatch.setenv("ACME_HOME_CHANNEL_NAME", "ACME Lobby")
        monkeypatch.setenv("ACME_HOME_CHANNEL_THREAD_ID", "thread-7")

        config = GatewayConfig()
        _apply_env_overrides(config)

        platform = Platform("acme")
        assert platform in config.platforms
        hc = config.platforms[platform].home_channel
        assert hc is not None
        assert hc.chat_id == "room-42"
        assert hc.name == "ACME Lobby"
        assert hc.thread_id == "thread-7"

    def test_no_home_channel_env_var_no_op(self, monkeypatch, isolated_registry):
        """Plugin declares the env var but it isn't set — no home_channel."""
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        _register_test_platform(
            isolated_registry,
            name="bravo",
            home_channel_env="BRAVO_HOME_CHANNEL",
        )
        monkeypatch.delenv("BRAVO_HOME_CHANNEL", raising=False)

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert config.platforms[Platform("bravo")].home_channel is None

    def test_no_env_field_skips_bootstrap(self, monkeypatch, isolated_registry):
        """Plugin omits home_channel_env entirely — bootstrap is a no-op."""
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        _register_test_platform(isolated_registry, name="charlie")
        monkeypatch.setenv("CHARLIE_HOME_CHANNEL", "would-be-loaded")

        config = GatewayConfig()
        _apply_env_overrides(config)
        # No env field declared, so the env var is ignored.
        assert config.platforms[Platform("charlie")].home_channel is None

    def test_user_set_home_channel_not_overwritten(
        self, monkeypatch, isolated_registry
    ):
        """If config.yaml already populated home_channel, env doesn't clobber it."""
        from gateway.config import (
            GatewayConfig,
            HomeChannel,
            Platform,
            PlatformConfig,
            _apply_env_overrides,
        )

        _register_test_platform(
            isolated_registry,
            name="delta",
            home_channel_env="DELTA_HOME_CHANNEL",
        )
        monkeypatch.setenv("DELTA_HOME_CHANNEL", "from-env")

        config = GatewayConfig()
        platform = Platform("delta")
        config.platforms[platform] = PlatformConfig()
        config.platforms[platform].home_channel = HomeChannel(
            platform=platform, chat_id="from-yaml", name="user choice"
        )
        _apply_env_overrides(config)
        assert config.platforms[platform].home_channel.chat_id == "from-yaml"


class TestPluginResetPolicyBootstrap:
    def test_default_reset_policy_applied(self, monkeypatch, isolated_registry):
        from gateway.config import (
            GatewayConfig,
            Platform,
            SessionResetPolicy,
            _apply_env_overrides,
        )

        _register_test_platform(
            isolated_registry,
            name="echo",
            default_reset_policy=SessionResetPolicy(mode="none"),
        )

        config = GatewayConfig()
        _apply_env_overrides(config)
        platform = Platform("echo")
        assert platform in config.reset_by_platform
        assert config.reset_by_platform[platform].mode == "none"

    def test_user_reset_policy_not_overwritten(
        self, monkeypatch, isolated_registry
    ):
        """User-supplied config.yaml policy beats plugin default."""
        from gateway.config import (
            GatewayConfig,
            Platform,
            SessionResetPolicy,
            _apply_env_overrides,
        )

        _register_test_platform(
            isolated_registry,
            name="foxtrot",
            default_reset_policy=SessionResetPolicy(mode="none"),
        )

        config = GatewayConfig()
        platform = Platform("foxtrot")
        config.reset_by_platform[platform] = SessionResetPolicy(mode="daily")
        _apply_env_overrides(config)
        # User's explicit "daily" wins over plugin default "none".
        assert config.reset_by_platform[platform].mode == "daily"

    def test_no_default_no_entry(self, monkeypatch, isolated_registry):
        """Plugin omits default_reset_policy — nothing added to reset_by_platform."""
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        _register_test_platform(isolated_registry, name="golf")
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform("golf") not in config.reset_by_platform


class TestCheckFnGate:
    def test_check_fn_false_skips_bootstrap(self, monkeypatch, isolated_registry):
        """When check_fn returns False, neither home_channel nor reset are applied."""
        from gateway.config import (
            GatewayConfig,
            Platform,
            SessionResetPolicy,
            _apply_env_overrides,
        )

        _register_test_platform(
            isolated_registry,
            name="hotel",
            home_channel_env="HOTEL_HOME_CHANNEL",
            default_reset_policy=SessionResetPolicy(mode="none"),
            check_ok=False,
        )
        monkeypatch.setenv("HOTEL_HOME_CHANNEL", "should-not-load")

        config = GatewayConfig()
        _apply_env_overrides(config)
        platform = Platform("hotel")
        assert platform not in config.platforms
        assert platform not in config.reset_by_platform
