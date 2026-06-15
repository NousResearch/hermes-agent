from pathlib import Path


def _write_plugin(root: Path, rel: str, *, name: str, logical_id: str) -> None:
    plugin_dir = root / rel
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"""
name: {name}
version: 1.0.0
description: duplicate test
logical_id: {logical_id}
provides_hooks:
  - on_session_start
""".strip() + "\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    def cb(**kwargs):\n"
        "        return None\n"
        "    ctx.register_hook('on_session_start', cb)\n",
        encoding="utf-8",
    )


def test_plugin_discovery_dedupes_by_logical_id(monkeypatch, tmp_path):
    from hermes_cli.plugins import PluginManager

    bundled = tmp_path / "bundled"
    user = tmp_path / "user" / "plugins"
    _write_plugin(bundled, "reasoning-display", name="reasoning-display", logical_id="statusbar.reasoning-display")
    _write_plugin(user, "status/reasoning-display", name="reasoning-display", logical_id="statusbar.reasoning-display")

    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr("hermes_cli.plugins.get_hermes_home", lambda: tmp_path / "user")
    monkeypatch.setattr("hermes_cli.plugins._get_enabled_plugins", lambda: {"reasoning-display", "status/reasoning-display"})
    monkeypatch.setattr("hermes_cli.plugins._get_disabled_plugins", lambda: set())

    manager = PluginManager()
    manager.discover_and_load(force=True)

    loaded = [p for p in manager.list_plugins() if p["enabled"] and p["name"] == "reasoning-display"]
    assert len(loaded) == 1
    assert loaded[0]["source"] == "user"
