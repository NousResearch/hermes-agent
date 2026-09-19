"""A provider installed by ``hermes plugins install`` must actually be found —
and only while the user has it enabled.

The installer clones into ``$HERMES_HOME/plugins/<name>/`` (flat), provider
discovery only scanned ``plugins/model-providers/<name>/``, and PluginManager
skips ``kind: model-provider`` on purpose — so the documented install path
reported success and registered nothing. These tests pin the join, that
discovery keeps its hands off every other plugin in that directory, and that
the join respects ``plugins.enabled`` / ``plugins.disabled``: importing a
plugin executes its code, so "installed" must not mean "loaded".
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_PROFILE_SOURCE = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="{name}", aliases=("{name}-alias",),
                                      base_url="acp://{name}", auth_type="external_process"))
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


def _write_plugin(directory: Path, *, name: str, manifest: str | None):
    directory.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (directory / "plugin.yaml").write_text(manifest, encoding="utf-8")
    # Registers a provider on import, so an unwanted import is *visible*.
    (directory / "__init__.py").write_text(_PROFILE_SOURCE.format(name=name), encoding="utf-8")


def _write_plugins_config(home: Path, *, enabled=(), disabled=()):
    """Write the real ``config.yaml`` keys ``hermes plugins enable|disable`` write."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "plugins:\n"
        f"  enabled: [{', '.join(enabled)}]\n"
        f"  disabled: [{', '.join(disabled)}]\n",
        encoding="utf-8",
    )


def _imported_plugin_modules() -> set:
    return {m for m in sys.modules if m.startswith("_hermes_user_provider")}


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_caches()
    yield tmp_path
    _clear_provider_caches()


def test_flat_installed_model_provider_plugins_are_discovered_alongside_nested_ones(hermes_home):
    _write_plugin(hermes_home / "plugins" / "installed-acp", name="installed-acp",
                  manifest='name: installed-acp\nkind: "model-provider"\n')
    _write_plugin(hermes_home / "plugins" / "model-providers" / "nested-acp", name="nested-acp",
                  manifest="name: nested-acp\nkind: model-provider\n")
    _write_plugins_config(hermes_home, enabled=["installed-acp"])
    from providers import get_provider_profile

    assert get_provider_profile("installed-acp").base_url == "acp://installed-acp"
    assert get_provider_profile("installed-acp-alias") is not None
    assert get_provider_profile("nested-acp") is not None


def test_other_plugins_in_the_flat_directory_are_left_to_the_plugin_manager(hermes_home):
    _write_plugin(hermes_home / "plugins" / "other-standalone", name="other-standalone",
                  manifest="name: other-standalone\nkind: standalone\n")
    _write_plugin(hermes_home / "plugins" / "manifestless", name="manifestless", manifest=None)
    _write_plugin(hermes_home / "plugins" / "broken-manifest", name="broken-manifest",
                  manifest="kind: [this is: not valid\n")
    from providers import get_provider_profile, list_providers

    assert not [p for p in list_providers() if p.name in ("other-standalone", "manifestless", "broken-manifest")]
    assert get_provider_profile("copilot-acp") is not None  # bundled set still intact


def test_a_disabled_installed_provider_plugin_is_never_imported(hermes_home):
    """`hermes plugins disable` has to stop the import, not just the listing.

    The plugin registers on import, so a leaked import is visible both as a
    registered profile and as a module in ``sys.modules``.
    """
    _write_plugin(hermes_home / "plugins" / "off-acp", name="off-acp",
                  manifest="name: off-acp\nkind: model-provider\n")
    _write_plugin(hermes_home / "plugins" / "on-acp", name="on-acp",
                  manifest="name: on-acp\nkind: model-provider\n")
    _write_plugins_config(hermes_home, enabled=["off-acp", "on-acp"], disabled=["off-acp"])
    from providers import get_provider_profile

    assert get_provider_profile("on-acp") is not None
    assert get_provider_profile("off-acp") is None
    assert get_provider_profile("off-acp-alias") is None
    assert "_hermes_user_provider_off_acp" not in _imported_plugin_modules()


def test_an_installed_provider_plugin_is_not_imported_until_it_is_enabled(hermes_home):
    """Installed is not loaded — the contract the pip entry-point scan enforces.

    ``hermes plugins install`` prompts "Enable now?" and, on a decline or a
    non-interactive run, tells the user to run `hermes plugins enable`; until
    then the clone must not execute.
    """
    _write_plugin(hermes_home / "plugins" / "unenabled-acp", name="unenabled-acp",
                  manifest="name: unenabled-acp\nkind: model-provider\n")
    _write_plugins_config(hermes_home, enabled=["some-other-plugin"])
    from providers import get_provider_profile

    assert get_provider_profile("unenabled-acp") is None
    assert "_hermes_user_provider_unenabled_acp" not in _imported_plugin_modules()


def test_a_disabled_nested_user_provider_plugin_is_never_imported(hermes_home):
    """Same contract for $HERMES_HOME/plugins/model-providers/<name>/.

    PluginManager lists those under the path-derived key
    ``model-providers/<dir>``, which is what `hermes plugins disable` writes,
    so discovery must match that key too. Dropping the directory there is
    still the opt-in, so an absent allow-list entry keeps loading it.
    """
    _write_plugin(hermes_home / "plugins" / "model-providers" / "nested-off", name="nested-off",
                  manifest="name: nested-off\nkind: model-provider\n")
    _write_plugin(hermes_home / "plugins" / "model-providers" / "nested-on", name="nested-on",
                  manifest="name: nested-on\nkind: model-provider\n")
    _write_plugins_config(hermes_home, disabled=["model-providers/nested-off"])
    from providers import get_provider_profile

    assert get_provider_profile("nested-on") is not None
    assert get_provider_profile("nested-off") is None
    assert "_hermes_user_provider_nested_off" not in _imported_plugin_modules()
