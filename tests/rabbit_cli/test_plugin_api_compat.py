"""Behavior-contract compatibility tests for native Rabbit plugins."""

from pathlib import Path
import shutil

import rabbit_yaml as yaml

from rabbit_cli.plugins import PluginManager


LEGACY_PLUGIN = Path(__file__).parent / "fixtures" / "plugin_compat_legacy"


def test_legacy_plugin_loads_and_ignores_additive_hook_and_manifest_fields(
    tmp_path, monkeypatch
):
    """A frozen plugin keeps working as manifests and hook payloads grow."""
    rabbit_home = tmp_path / "rabbit-home"
    plugins_dir = rabbit_home / "plugins"
    plugins_dir.mkdir(parents=True)
    shutil.copytree(LEGACY_PLUGIN, plugins_dir / "legacy-contract-fixture")
    (rabbit_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"plugins": {"enabled": ["legacy-contract-fixture"]}}
        ),
        encoding="utf-8",
    )
    empty_bundled = tmp_path / "bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
    monkeypatch.setenv("RABBIT_BUNDLED_PLUGINS", str(empty_bundled))

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
