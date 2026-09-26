"""``$HERMES_HOME`` model-provider plugins resolve for the profile home bound at lookup time (#88143).

One process serves several profiles (multiplex gateway, Desktop ``serve``); discovery used to read the
plugins of whichever home was bound first and never look again, so a plugin installed in a secondary
profile was ``Unknown provider`` from Desktop while ``hermes -p <profile>`` in a terminal worked.
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

    register_provider(ProviderProfile(name="{name}", aliases=("{name}-alias",), auth_type="external_process",
                                      base_url="process://{name}", api_mode="chat_completions"))
    """
)


def _install(home: Path, name: str) -> None:
    plugin = home / "plugins" / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(f"name: {name}\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_PLUGIN.format(name=name), encoding="utf-8")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import providers

    launch = tmp_path / "launch"
    secondary = tmp_path / "profiles" / "scaleup"
    launch.mkdir()
    secondary.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {}, raising=False)
    yield launch, secondary
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _bound(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_secondary_profile_plugin_resolves_for_its_home_only(homes):
    import providers
    from hermes_cli.auth import resolve_provider

    launch, secondary = homes
    _install(secondary, "scaleup-only")

    assert providers.get_provider_profile("scaleup-only") is None  # launch home discovers first

    assert _bound(secondary, lambda: providers.get_provider_profile("scaleup-only")) is not None
    assert _bound(secondary, lambda: providers.get_provider_profile("scaleup-only-alias")) is not None
    assert _bound(secondary, lambda: providers.provider_source("scaleup-only")) == "user"
    assert "scaleup-only" in _bound(secondary, lambda: {p.name for p in providers.list_providers()})
    # The agent-build gate Desktop hits (``Unknown provider`` came from here).
    assert _bound(secondary, lambda: resolve_provider("scaleup-only")) == "scaleup-only"

    # Profiles are islands: the launch home still does not see the secondary's install.
    assert providers.get_provider_profile("scaleup-only") is None
    assert "scaleup-only" not in {p.name for p in providers.list_providers()}


def test_plugin_installed_after_discovery_is_found_without_a_restart(homes):
    import providers

    launch, _ = homes
    assert providers.get_provider_profile("late-install") is None

    _install(launch, "late-install")

    assert providers.get_provider_profile("late-install") is not None
    assert "late-install" in {p.name for p in providers.list_providers()}


_REENTRANT_PLUGIN = textwrap.dedent(
    """
    import reentry_log
    from providers import list_providers, register_provider
    from providers.base import ProviderProfile

    reentry_log.LOG.append("{name}:start")
    list_providers()  # stands in for a module whose body calls list_providers() (hermes_cli.models)
    reentry_log.LOG.append("{name}:end")
    register_provider(ProviderProfile(name="{name}", auth_type="external_process",
                                      base_url="process://{name}", api_mode="chat_completions"))
    """
)


def test_lookup_during_a_scan_does_not_start_a_nested_scan(homes, tmp_path, monkeypatch):
    """A plugin whose import calls list_providers() must not make the scan import the NEXT plugin
    while the first is still half-imported: that nested scan ran claude-bpx against a partially
    initialized hermes_cli.models and every one of its pins failed with a circular import."""
    import providers

    launch, _ = homes
    helper = tmp_path / "helper"
    helper.mkdir()
    (helper / "reentry_log.py").write_text("LOG = []\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(helper))
    monkeypatch.delitem(sys.modules, "reentry_log", raising=False)

    _install(launch, "b-second")  # sorted after a-first, so the outer scan imports it second
    first = launch / "plugins" / "a-first"
    first.mkdir(parents=True)
    (first / "plugin.yaml").write_text("name: a-first\nkind: model-provider\n", encoding="utf-8")
    (first / "__init__.py").write_text(_REENTRANT_PLUGIN.format(name="a-first"), encoding="utf-8")
    (launch / "plugins" / "b-second" / "__init__.py").write_text(
        "import reentry_log\nreentry_log.LOG.append('b-second:start')\n" + _PLUGIN.format(name="b-second"),
        encoding="utf-8",
    )

    names = {p.name for p in providers.list_providers()}

    import reentry_log

    assert reentry_log.LOG == ["a-first:start", "a-first:end", "b-second:start"]
    assert {"a-first", "b-second"} <= names
