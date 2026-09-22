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


def test_home_discovery_can_sync_auth_without_recursively_rescanning(homes, monkeypatch):
    import providers
    import hermes_cli.auth as auth
    from agent import secret_scope

    launch, secondary = homes
    # Complete process-wide discovery before observing the per-home import boundary.
    providers.list_providers()
    _install(launch, "launch-sync")
    _install(secondary, "secondary-sync")
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", dict(auth.PROVIDER_REGISTRY))
    real_sync = auth.sync_plugin_provider_registry
    depth = 0
    max_depth = 0

    def observe_sync():
        nonlocal depth, max_depth
        depth += 1
        max_depth = max(max_depth, depth)
        try:
            # The real auth synchronizer enumerates providers again. That read
            # must see the completed scan rather than start another sync.
            return real_sync()
        finally:
            depth -= 1

    monkeypatch.setattr(auth, "sync_plugin_provider_registry", observe_sync)
    was_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope({})
    secret_scope.set_multiplex_active(True)
    try:
        for home, name in ((launch, "launch-sync"), (secondary, "secondary-sync"), (launch, "launch-sync")):
            profile = _bound(home, lambda: providers.get_provider_profile(name))
            assert profile is not None
            assert _bound(home, lambda: auth.resolve_provider(name)) == name
            assert auth.PROVIDER_REGISTRY[name].inference_base_url == profile.base_url
    finally:
        secret_scope.set_multiplex_active(was_multiplex)
        secret_scope.reset_secret_scope(token)
    assert max_depth == 1, "provider discovery recursively re-entered auth synchronization"
