"""A provider plugin the Hermes plugin catalog recalled after it was installed stops loading.

The general plugin loader refuses an installed plugin the catalog removal list names; provider
discovery imports user provider plugins on its own path, so it applies the same offline check (the
in-tree list plus the cached live copy) before importing one, in both layouts, honouring an install
made with ``--allow-removed``. Bundled providers are never checked.
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap
from pathlib import Path

import hermes_yaml as yaml
import pytest

_PROFILE_SOURCE = textwrap.dedent(
    """
    open({marker!r}, "a").close()
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="{name}", base_url="acp://{name}", auth_type="external_process"))
    """
)


def _clear_provider_caches():
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    for mod in list(sys.modules):
        if mod.startswith(("plugins.model_providers", "_hermes_user_provider")):
            del sys.modules[mod]


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    _clear_provider_caches()
    yield home
    _clear_provider_caches()


def _write_plugin(directory: Path, name: str) -> Path:
    """A provider plugin that leaves a marker file when its code runs."""
    directory.mkdir(parents=True)
    marker = directory.parent.parent / f"ran-{name}"
    (directory / "plugin.yaml").write_text(f"name: {name}\nkind: model-provider\n", encoding="utf-8")
    (directory / "__init__.py").write_text(_PROFILE_SOURCE.format(marker=str(marker), name=name), encoding="utf-8")
    return marker


def _recall(home: Path, *names: str) -> None:
    """The cached live catalog copy, as the last catalog fetch leaves it, naming *names* as removed."""
    (home / "cache").mkdir(exist_ok=True)
    removed = [{"name": name, "reason": "withdrawn by its author"} for name in names]
    (home / "cache" / "plugin-catalog.json").write_text(json.dumps({"entries": [], "removed": removed}),
                                                        encoding="utf-8")


def _layouts(home: Path, name: str) -> dict[str, Path]:
    return {"flat": home / "plugins" / name, "nested": home / "plugins" / "model-providers" / name}


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_a_recalled_provider_plugin_is_not_imported(hermes_home, caplog, layout):
    marker = _write_plugin(_layouts(hermes_home, "recalledprov")[layout], "recalledprov")
    kept = _write_plugin(_layouts(hermes_home, "keptprov")[layout], "keptprov")
    _recall(hermes_home, "recalledprov")
    import providers

    with caplog.at_level(logging.WARNING, logger="providers"):
        assert providers.get_provider_profile("recalledprov") is None

    assert not marker.exists()
    assert providers.plugin_load_error("recalledprov") == "removed from the Hermes plugin catalog: withdrawn by its author"
    assert any("Refusing to load provider plugin 'recalledprov'" in r.getMessage() for r in caplog.records)
    # A plugin the list does not name still loads without a plugins.enabled entry.
    assert providers.get_provider_profile("keptprov") is not None and kept.exists()
    assert providers.plugin_load_error("keptprov") == ""


def test_an_install_made_with_allow_removed_keeps_loading(hermes_home):
    from hermes_cli import plugins_cmd

    marker = _write_plugin(hermes_home / "plugins" / "exemptprov", "exemptprov")
    plugins_cmd._write_install_metadata({"exemptprov": {"source": "file:///src/exemptprov", "allow_removed": True}})
    _recall(hermes_home, "exemptprov")
    import providers

    assert providers.get_provider_profile("exemptprov") is not None
    assert marker.exists()


def test_bundled_providers_are_not_checked(hermes_home):
    _recall(hermes_home, "anthropic", "openrouter")
    import providers

    assert providers.get_provider_profile("anthropic") is not None
    assert providers.get_provider_profile("openrouter") is not None


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_a_recalled_provider_plugin_never_runs_in_the_plugin_host(hermes_home):
    from hermes_cli import plugins as plugins_mod

    marker = _write_plugin(hermes_home / "plugins" / "hostedrecalled", "hostedrecalled")
    kept = _write_plugin(hermes_home / "plugins" / "hostedkept", "hostedkept")
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"isolation": "host"}}), encoding="utf-8")
    _recall(hermes_home, "hostedrecalled")
    import providers

    try:
        assert providers.get_provider_profile("hostedrecalled") is None
        assert not marker.exists()
        # The host path itself works: the plugin the list does not name ran there and registered.
        assert providers.get_provider_profile("hostedkept") is not None and kept.exists()
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()


# ── A cached catalog copy that is not a JSON object is ignored ────────────────
# The last fetch only ever writes an object; a hand-edited or damaged file holding any other JSON
# value counts as no cached copy, so the lookup goes on with the in-tree removal list alone.

_NOT_AN_OBJECT = {"list": "[1]", "string": '"text"', "number": "1", "null": "null"}


def _write_cache(home: Path, text: str) -> None:
    (home / "cache").mkdir(exist_ok=True)
    (home / "cache" / "plugin-catalog.json").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("value", sorted(_NOT_AN_OBJECT))
def test_a_cached_catalog_that_is_not_an_object_does_not_stop_provider_lookup(hermes_home, value):
    layouts = [_layouts(hermes_home, name)["nested"] for name in ("firstprov", "secondprov")]
    markers = [_write_plugin(directory, directory.name) for directory in layouts]
    _write_cache(hermes_home, _NOT_AN_OBJECT[value])
    import providers

    assert providers.get_provider_profile("firstprov") is not None
    assert providers.get_provider_profile("secondprov") is not None
    assert all(marker.exists() for marker in markers)
    assert providers.plugin_load_error("firstprov") == ""


@pytest.mark.parametrize("value", sorted(_NOT_AN_OBJECT))
def test_the_general_loader_loads_an_enabled_plugin_beside_such_a_cache(hermes_home, value):
    from hermes_cli.plugins import PluginManager

    plugin_dir = hermes_home / "plugins" / "generalplug"
    plugin_dir.mkdir(parents=True)
    marker = hermes_home / "ran-generalplug"
    (plugin_dir / "plugin.yaml").write_text("name: generalplug\nversion: 0.1.0\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(f"def register(ctx):\n    open({str(marker)!r}, 'a').close()\n",
                                            encoding="utf-8")
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["generalplug"]}}),
                                             encoding="utf-8")
    _write_cache(hermes_home, _NOT_AN_OBJECT[value])

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["generalplug"]
    assert (loaded.enabled, loaded.error) == (True, None)
    assert marker.exists()
