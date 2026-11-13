"""The aux provider alias table follows the bound profile home and later plugin installs.

It merges the plugin aliases of the home bound when it is built; memoized once per process, the
first home's aliases leaked into every other profile and a plugin installed while the process ran
never reached it (R4 H2).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_PLUGIN = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="aliased-fake", aliases=("aliasy",), auth_type="api_key",
                                      env_vars=("ALIASED_FAKE_KEY",), base_url="https://aliased.invalid/v1"))
    """
)


def _install(home: Path) -> None:
    plugin = home / "plugins" / "model-providers" / "aliased-fake"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import providers
    from hermes_cli import auth_plugin_providers as app
    from hermes_cli.auth import PROVIDER_REGISTRY

    plugin_home, bare_home = tmp_path / "with_plugin", tmp_path / "bare"
    plugin_home.mkdir()
    bare_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(bare_home))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})
    monkeypatch.setattr(providers, "_PLUGIN_DIR_STAMP_TTL_SECONDS", 0.0)
    monkeypatch.setattr(app, "PLUGIN_MIRRORED_PROVIDERS", set(app.PLUGIN_MIRRORED_PROVIDERS))
    rows = list(dict.items(PROVIDER_REGISTRY))
    yield plugin_home, bare_home
    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(rows)
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _normalize_in(home: Path, name: str) -> str:
    from agent.auxiliary_client import _normalize_aux_provider

    token = set_hermes_home_override(home)
    try:
        return _normalize_aux_provider(name)
    finally:
        reset_hermes_home_override(token)


def test_alias_table_is_per_home_in_either_order(homes):
    plugin_home, bare_home = homes
    _install(plugin_home)

    assert _normalize_in(bare_home, "aliasy") == "aliasy"
    assert _normalize_in(plugin_home, "aliasy") == "aliased-fake"
    assert _normalize_in(bare_home, "aliasy") == "aliasy"
    assert _normalize_in(plugin_home, "aliasy") == "aliased-fake"


def test_plugin_installed_while_running_reaches_the_alias_table(homes):
    _plugin_home, bare_home = homes

    assert _normalize_in(bare_home, "aliasy") == "aliasy"
    _install(bare_home)
    assert _normalize_in(bare_home, "aliasy") == "aliased-fake"
