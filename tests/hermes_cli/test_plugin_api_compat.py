"""Behavior-contract compatibility tests for native Hermes plugins."""

from pathlib import Path
import shutil

import hermes_yaml as yaml
import pytest

from hermes_cli.plugins import PluginManager


LEGACY_PLUGIN = Path(__file__).parent / "fixtures" / "plugin_compat_legacy"


def test_legacy_plugin_loads_and_ignores_additive_hook_and_manifest_fields(
    tmp_path, monkeypatch
):
    """A frozen plugin keeps working as manifests and hook payloads grow."""
    hermes_home = tmp_path / "hermes-home"
    plugins_dir = hermes_home / "plugins"
    plugins_dir.mkdir(parents=True)
    shutil.copytree(LEGACY_PLUGIN, plugins_dir / "legacy-contract-fixture")
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"plugins": {"enabled": ["legacy-contract-fixture"]}}
        ),
        encoding="utf-8",
    )
    empty_bundled = tmp_path / "bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["legacy-contract-fixture"]
    assert loaded.enabled is True
    assert loaded.error is None
    assert loaded.module is not None
    assert loaded.manifest.version == "0.1.0"
    assert "on_session_start" in loaded.hooks_registered

    results = manager.invoke_hook(
        "on_session_start",
        session_id="legacy-session",
        resumed=True,
        future_additive_field={"nested": "value"},
    )

    assert results == [{"legacy_session_id": "legacy-session"}]
    assert loaded.module.received_sessions == ["legacy-session"]


@pytest.mark.asyncio
async def test_frozen_handoff_override_creates_a_thread_after_plugin_discovery(tmp_path, monkeypatch):
    import hermes_state
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platform_registry import platform_registry
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionStore
    from tests.gateway.test_branch_thread_command import _discord_channel_source, _runner, _seed

    home = tmp_path / "hermes-home"
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures" / "plugin_compat_handoff"
    shutil.copytree(fixture, plugins_dir / "legacy-handoff-fixture")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["legacy-handoff-fixture"]}}))
    empty_bundled = tmp_path / "bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    manager = PluginManager()
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        manager.discover_and_load()
        loaded = manager._plugins["legacy-handoff-fixture"]
        assert loaded.enabled and loaded.error is None
        adapter = platform_registry.create_adapter("discord", PlatformConfig())
        assert adapter is not None
        source = _discord_channel_source()
        original = _seed(store, source)
        runner = _runner(store, adapter)
        reply = await runner._handle_branch_command(MessageEvent(text="/branch side quest", source=source))
        assert adapter.calls == [("123", "side quest")], reply
        assert store.get_or_create_session(source).session_id == original.session_id
        assert "<#777000>" in reply
    finally:
        store.close_all_db_handles()
        manager.unload()
