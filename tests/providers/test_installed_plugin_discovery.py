"""A provider installed by ``hermes plugins install`` must actually be found.

The installer clones into ``$HERMES_HOME/plugins/<name>/`` (flat), provider
discovery only scanned ``plugins/model-providers/<name>/``, and PluginManager
skips ``kind: model-provider`` on purpose — so the documented install path
reported success and registered nothing. These tests pin the join, and that
discovery keeps its hands off every other plugin in that directory.
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
