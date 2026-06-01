"""Gateway plugin discovery regression tests."""


def test_gateway_startup_forces_plugin_rescan_for_bundled_platforms(monkeypatch):
    """A stale plugin-manager cache must not hide bundled platform adapters.

    Regression coverage for gateway boots where plugin discovery had already
    been marked complete before bundled platform adapters registered. Without a
    forced startup rescan, Discord stays absent from platform_registry and the
    gateway logs "No adapter available for discord" despite valid Discord
    dependencies/config.
    """
    from gateway.platform_registry import platform_registry
    from gateway.run import _discover_gateway_plugins_for_startup
    import hermes_cli.plugins as plugins_mod

    stale_manager = plugins_mod.PluginManager()
    stale_manager._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", stale_manager)
    platform_registry.unregister("discord")

    assert not platform_registry.is_registered("discord")

    _discover_gateway_plugins_for_startup()

    assert platform_registry.is_registered("discord")


def test_create_adapter_recovers_when_platform_registry_cache_is_stale(monkeypatch):
    """Direct adapter creation should also recover from stale discovery state."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platform_registry import platform_registry
    from gateway.run import GatewayRunner
    import hermes_cli.plugins as plugins_mod

    stale_manager = plugins_mod.PluginManager()
    stale_manager._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", stale_manager)
    platform_registry.unregister("discord")

    runner = GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="test-token"),
            }
        )
    )

    adapter = runner._create_adapter(
        Platform.DISCORD,
        PlatformConfig(enabled=True, token="test-token"),
    )

    assert adapter is not None
    assert type(adapter).__name__ == "DiscordAdapter"
    assert platform_registry.is_registered("discord")
