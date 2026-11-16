"""Repeated profile misses honour the plugin-dir stamp TTL (R4 H4).

Registry ids with no profile (``openai-api``, ``xai-oauth`` …) are looked up by per-request hooks;
each miss forced a plugin-dir re-stat, defeating the 1 s TTL. A new name still forces a check, so a
provider installed while the process runs is found on its first request.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time
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
    clock = types.SimpleNamespace(now=1000.0, ns=time.time_ns())
    monkeypatch.setattr(providers, "time", types.SimpleNamespace(monotonic=lambda: clock.now, time_ns=lambda: clock.ns))
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


def test_install_in_the_same_timestamp_tick_is_found_after_the_ttl(layer_env, monkeypatch):
    """A kernel with a coarse timestamp clock (or FAT/HFS+ rounding) gives every directory change
    within one tick the same mtime, so the stamp read at the last scan can equal the stamp after an
    install: the new plugin must still be found once the TTL has passed, even when the scan itself
    outlasts the tick."""
    import providers

    home, clock, _stats = layer_env
    real_scan = providers._scan_home_layer

    def slow_scan(*args):
        real_scan(*args)
        clock.ns += 3 * 10**9  # plugin imports took longer than one mtime tick

    monkeypatch.setattr(providers, "_scan_home_layer", slow_scan)
    _install(home, "tick-b")
    model_providers = home / "plugins" / "model-providers"
    for path in (home / "plugins", model_providers):  # fresh: inside the current tick
        os.utime(path, ns=(clock.ns, clock.ns))
    assert providers.get_provider_profile("tick-b") is not None
    before = model_providers.stat()

    _install(home, "tick-a")
    os.utime(model_providers, ns=(before.st_atime_ns, before.st_mtime_ns))  # same tick, same mtime
    clock.now += providers._PLUGIN_DIR_STAMP_TTL_SECONDS
    assert providers.get_provider_profile("tick-a") is not None


def test_settled_unchanged_plugin_dirs_are_not_rescanned(layer_env, monkeypatch):
    import providers

    home, clock, _stats = layer_env
    _install(home, "settled")
    hour_ago = clock.ns - 3600 * 10**9
    for path in (home / "plugins", home / "plugins" / "model-providers"):
        os.utime(path, ns=(hour_ago, hour_ago))
    scans = []
    real_scan = providers._scan_home_layer
    monkeypatch.setattr(providers, "_scan_home_layer", lambda *a: scans.append(1) or real_scan(*a))

    assert providers.get_provider_profile("settled") is not None
    for _ in range(3):
        clock.now += providers._PLUGIN_DIR_STAMP_TTL_SECONDS
        assert providers.get_provider_profile("settled") is not None
        assert providers.get_provider_profile("no-profile-id") is None
    assert len(scans) == 1  # stamps older than the filesystem's timestamp tick are trusted
