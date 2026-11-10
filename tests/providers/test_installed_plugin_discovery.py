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
    assert get_provider_profile("openrouter") is not None  # bundled set still intact


# The hosted code fence: a hosted process loads provider code only from what its
# deployment image ships (the tree bundled with Hermes, or an installed package's entry point)


@pytest.fixture
def hosted(monkeypatch):
    """A frozen hosted verdict with the process-wide code fence installed."""
    from hermes_cli import hosted_mode
    from hermes_cli.hosted_code_fence import install_hosted_code_fence

    monkeypatch.setitem(hosted_mode._VERDICT, "hosted", True)
    install_hosted_code_fence()


def test_a_home_dir_provider_is_refused_before_import_in_a_hosted_process(hermes_home, hosted, caplog):
    """The home's plugin directories are tenant-writable, so a hosted process never imports a
    provider from them: neither a nested model-providers dir nor a flat catalog install. The
    code fence refuses the import itself; the module never lands in sys.modules and the
    failure is logged with the reason."""
    from providers import get_provider_profile

    _write_plugin(hermes_home / "plugins" / "model-providers" / "nested-acp", name="nested-acp",
                  manifest="name: nested-acp\nkind: model-provider\n")
    _write_plugin(hermes_home / "plugins" / "installed-acp", name="installed-acp",
                  manifest='name: installed-acp\nkind: "model-provider"\n')

    with caplog.at_level("WARNING"):
        assert get_provider_profile("nested-acp") is None
        assert get_provider_profile("installed-acp") is None

    assert not [m for m in sys.modules if m.startswith("_hermes_user_provider")]
    assert "deployment image" in caplog.text  # the fence's refusal names the rule


def test_bundled_providers_still_load_in_a_hosted_process(hermes_home, hosted):
    """The tree bundled with Hermes is the image's own: hosted or not, its providers register
    exactly as before (the refusal above touches only the home's directories)."""
    from providers import get_provider_profile

    assert get_provider_profile("openrouter") is not None


def test_a_symlink_planted_in_the_bundled_provider_tree_is_refused_in_a_hosted_process(
        hermes_home, monkeypatch, hosted, caplog):
    """The bundled label is not provenance either: a symlink in the tree bundled with Hermes
    whose realpath leaves that tree — targeting the deployment home — is refused by the code
    fence exactly like a home directory, whichever bundled executor reaches it (provider
    discovery and the descriptive catalog alike). Its code never runs."""
    import hermes_cli.plugins as plugins_mod
    import providers

    home_plugin = hermes_home / "plugins" / "model-providers" / "symlinked-home-provider"
    home_plugin.mkdir(parents=True)
    marker = hermes_home / "executed"
    (home_plugin / "__init__.py").write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).write_text("home code ran", encoding="utf-8")\n'
        + _PROFILE_SOURCE.format(name="symlinked-home-provider"),
        encoding="utf-8")
    # The image's bundled tree, isolated for the test (planting the symlink in the checkout's
    # own tree would race every parallel test process that discovers providers).
    image_root = hermes_home / "image-tree" / "plugins"
    bundled = image_root / "model-providers"
    bundled.mkdir(parents=True)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: image_root)
    monkeypatch.setattr(providers, "_BUNDLED_PLUGINS_DIR", bundled)
    link = bundled / "rv3-symlink-probe"
    link.symlink_to(home_plugin, target_is_directory=True)
    module_name = f"plugins.model_providers.{link.name.replace('-', '_')}"
    catalog_name = f"_hermes_provider_catalog_{link.name.replace('-', '_')}"
    try:
        with caplog.at_level("WARNING"):
            providers._import_plugin_dir(link, "bundled")
        assert not marker.exists()  # the home's code never ran
        assert module_name not in sys.modules
        assert "deployment image" in caplog.text  # the fence's refusal names the rule
        # the descriptive-catalog executor answers the same fence
        assert not [p for p in providers.bundled_profiles() if p.name == "symlinked-home-provider"]
        assert not marker.exists()
        assert catalog_name not in sys.modules
    finally:
        link.unlink(missing_ok=True)
        sys.modules.pop(module_name, None)
        sys.modules.pop(catalog_name, None)
