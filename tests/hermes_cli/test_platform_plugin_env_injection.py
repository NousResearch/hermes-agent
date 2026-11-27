"""User-installed platform plugins feed OPTIONAL_ENV_VARS like bundled ones (#46600, redo of #46964).

The Desktop Gateway form and ``hermes config`` render env fields from ``OPTIONAL_ENV_VARS``; before
this, only ``plugins/platforms/*`` in the repo was scanned, so a third-party platform's
``requires_env`` prompts/descriptions/password flags never reached the UI.
"""

import hermes_cli.config as config_mod


def _manifest(path, text):
    path.mkdir(parents=True)
    (path / "plugin.yaml").write_text(text, encoding="utf-8")


def test_user_platform_plugins_inject_env_metadata_but_non_platforms_do_not(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    _manifest(home / "plugins" / "demo-platform", (
        "name: demo-platform\nkind: platform\nlabel: Demo Platform\n"
        "requires_env:\n  - name: DEMO_PLATFORM_TOKEN\n    description: Token for the demo platform\n"
        "    prompt: Demo token\n    url: https://example.invalid/demo\n"
        "optional_env:\n  - name: DEMO_PLATFORM_ROOM\n    password: false\n"))
    _manifest(home / "plugins" / "platforms" / "nested-platform", (
        "name: nested-platform\nrequires_env:\n  - NESTED_PLATFORM_SECRET\n"))
    _manifest(home / "plugins" / "ignore-me", "name: ignore-me\nkind: backend\nrequires_env:\n  - IGNORE_ME_TOKEN\n")
    keys = ["DEMO_PLATFORM_TOKEN", "DEMO_PLATFORM_ROOM", "NESTED_PLATFORM_SECRET", "IGNORE_ME_TOKEN"]
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)
    for key in keys:
        monkeypatch.delitem(config_mod.OPTIONAL_ENV_VARS, key, raising=False)

    config_mod._inject_platform_plugin_env_vars()

    try:
        assert config_mod.OPTIONAL_ENV_VARS["DEMO_PLATFORM_TOKEN"] == {
            "description": "Token for the demo platform", "prompt": "Demo token",
            "url": "https://example.invalid/demo", "password": True, "category": "messaging"}
        assert config_mod.OPTIONAL_ENV_VARS["DEMO_PLATFORM_ROOM"]["password"] is False
        assert config_mod.OPTIONAL_ENV_VARS["NESTED_PLATFORM_SECRET"]["password"] is True  # category dir needs no kind
        assert "IGNORE_ME_TOKEN" not in config_mod.OPTIONAL_ENV_VARS
    finally:
        for key in keys:
            config_mod.OPTIONAL_ENV_VARS.pop(key, None)


def test_reload_drops_a_removed_line_for_a_platform_plugin_loaded_after_import(tmp_path, monkeypatch):
    """A platform plugin installed in-process (its manifest was not on disk when hermes_cli.config read them at
    import) gets its declared env names into reload_env's known set when it loads, so /reload drops a removed
    .env line for them as it did when core shipped QQ Bot (QQ_SANDBOX, QQ_PORTAL_HOST)."""
    import os
    from hermes_cli import config
    from hermes_cli.plugins import PluginManager
    home = tmp_path / "home"
    plugin = home / "plugins" / "lateplat"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: lateplat\nkind: platform\nrequires_env: [LATEPLAT_TOKEN]\noptional_env:\n  - name: LATEPLAT_SANDBOX\n")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n")
    (home / "config.yaml").write_text("plugins:\n  enabled: [lateplat]\n")
    (home / ".env").write_text("LATEPLAT_TOKEN=t\nLATEPLAT_SANDBOX=true\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    PluginManager().discover_and_load()
    for key in ("LATEPLAT_TOKEN", "LATEPLAT_SANDBOX"):
        monkeypatch.delenv(key, raising=False)
    config.reload_env()
    assert os.environ.get("LATEPLAT_SANDBOX") == "true"
    (home / ".env").write_text("LATEPLAT_TOKEN=t\n")
    config.reload_env()
    assert "LATEPLAT_SANDBOX" not in os.environ and os.environ.get("LATEPLAT_TOKEN") == "t"


def test_a_platform_manifest_with_an_empty_requires_env_key_still_loads_with_the_others(tmp_path, monkeypatch):
    """``requires_env:`` with no value reads as no env names, as the loader always treated it: noting a loading
    platform's env names for /reload must not raise on it and take the rest of the discovery down."""
    from hermes_cli.plugins import PluginManager
    home = tmp_path / "home"
    for name, env in (("emptyenv", "requires_env:\n"), ("otherplat", "requires_env: [OTHERPLAT_TOKEN]\n")):
        plugin = home / "plugins" / name
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(f"name: {name}\nkind: platform\n{env}", encoding="utf-8")
        (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled: [emptyenv, otherplat]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    assert {"emptyenv", "otherplat"} <= {key for key, plugin in manager._plugins.items() if not plugin.error}


def test_a_platform_manifest_with_non_list_env_fields_still_loads_with_the_others(tmp_path, monkeypatch):
    """``optional_env: 5`` / ``requires_env: true`` read as no env names where the manifest is parsed: noting a
    loading platform's env names for /reload must not raise on them and take the rest of the discovery down."""
    from hermes_cli.plugins import PluginManager
    home = tmp_path / "home"
    manifests = (("numenv", "optional_env: 5\n"), ("boolenv", "requires_env: true\noptional_env: true\n"),
                 ("otherplat", "requires_env: [OTHERPLAT_TOKEN]\n"))
    for name, env in manifests:
        plugin = home / "plugins" / name
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(f"name: {name}\nkind: platform\n{env}", encoding="utf-8")
        (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled: [numenv, boolenv, otherplat]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = PluginManager()
    manager.discover_and_load()
    assert {"numenv", "boolenv", "otherplat"} <= {key for key, plugin in manager._plugins.items() if not plugin.error}
