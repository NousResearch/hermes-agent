"""Repeated profile misses honour the plugin-dir stamp TTL (R4 H4).

Registry ids with no profile (``openai-api``, ``xai-oauth`` …) are looked up by per-request hooks;
each miss forced a plugin-dir re-stat, defeating the 1 s TTL. A new name still forces a check, so a
provider installed while the process runs is found on its first request.
"""

from __future__ import annotations

import sys
import textwrap
import types

import pytest

_PLUGIN = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(name="{name}", auth_type="api_key", env_vars=("X_KEY",),
                                      base_url="https://{name}.invalid/v1"))
    """
)


@pytest.fixture
def layer_env(tmp_path, monkeypatch):
    import providers

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})
    clock = types.SimpleNamespace(now=1000.0)
    monkeypatch.setattr(providers, "time", types.SimpleNamespace(monotonic=lambda: clock.now))
    stats = []
    real = providers._plugin_dir_stamps
    monkeypatch.setattr(providers, "_plugin_dir_stamps", lambda h: stats.append(1) or real(h))
    yield home, clock, stats
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _install(home, name):
    plugin = home / "plugins" / "model-providers" / name
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(_PLUGIN.format(name=name), encoding="utf-8")


def test_repeated_misses_do_not_restat_within_the_ttl(layer_env):
    import providers

    _home, clock, stats = layer_env
    assert providers.get_provider_profile("no-profile-id") is None  # first lookup: periodic check
    stats.clear()
    clock.now += 0.1
    for _ in range(50):
        assert providers.get_provider_profile("no-profile-id") is None
    assert len(stats) <= 1  # the first miss of this name may force one check, never one per call

    clock.now += providers._PLUGIN_DIR_STAMP_TTL_SECONDS
    stats.clear()
    assert providers.get_provider_profile("no-profile-id") is None
    assert len(stats) == 1  # TTL elapsed: checked again


def test_new_install_found_by_new_name_and_after_the_ttl(layer_env):
    import providers

    home, clock, _stats = layer_env
    assert providers.get_provider_profile("late-a") is None
    clock.now += 0.1
    assert providers.get_provider_profile("late-a") is None

    _install(home, "late-b")
    clock.now += 0.1
    assert providers.get_provider_profile("late-b") is not None  # a new name forces a check

    _install(home, "late-a")
    clock.now += providers._PLUGIN_DIR_STAMP_TTL_SECONDS
    assert providers.get_provider_profile("late-a") is not None


def test_runtime_registration_is_found_despite_a_cached_miss(layer_env):
    import providers
    from providers.base import ProviderProfile

    assert providers.get_provider_profile("late-runtime") is None
    providers.register_provider(ProviderProfile(name="late-runtime", auth_type="api_key"))
    assert providers.get_provider_profile("late-runtime") is not None
