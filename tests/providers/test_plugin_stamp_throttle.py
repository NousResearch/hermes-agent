"""The per-lookup plugin-dir stamp check is throttled to one round per home per TTL (#119950).

Every ``get_provider_profile()`` used to ``os.stat`` ``$HERMES_HOME/plugins`` and
``plugins/model-providers`` again, so a large-catalog picker build paid two stats per model
row. The check now runs at most once per TTL window — while a home whose plugin dirs do not
exist yet is never throttled, so the first ``hermes plugins install`` stays visible to the
very next lookup (#88143).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

from providers import (
    _plugin_dir_stamps,
    _STAMP_CHECKS,
    _STAMP_CHECK_TTL_SECONDS,
    get_provider_profile,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    import providers

    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {}, raising=False)
    monkeypatch.setattr(providers, "_STAMP_CHECKS", {}, raising=False)
    return launch


def test_repeated_lookups_stat_the_plugin_dirs_once_per_ttl(home, monkeypatch):
    (home / "plugins" / "model-providers").mkdir(parents=True)
    stats = []
    real_stat = os.stat

    def counting_stat(path, *args, **kwargs):
        stats.append(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)
    get_provider_profile("openai")  # warmup: first lookup scans the dirs once
    stats.clear()
    for _ in range(50):
        get_provider_profile("openai")
    watched = {str(home / "plugins"), str(home / "plugins" / "model-providers")}
    plugin_stats = [p for p in stats if p in watched]
    assert plugin_stats == []  # every later lookup is served from the cache


def test_dir_mtime_change_after_ttl_triggers_a_rescan(home, monkeypatch):
    import providers

    plugin = home / "plugins" / "model-providers" / "fresh"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        'register_provider(ProviderProfile(name="fresh", base_url="process://fresh", '
        'auth_type="external_process", api_mode="chat_completions"))\n',
        encoding="utf-8",
    )

    assert get_provider_profile("fresh") is not None

    later = home / "plugins" / "model-providers" / "later"
    later.mkdir()
    (later / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        'register_provider(ProviderProfile(name="later", base_url="process://later", '
        'auth_type="external_process", api_mode="chat_completions"))\n',
        encoding="utf-8",
    )

    # Inside the TTL window the second install is deliberately not seen...
    assert get_provider_profile("later") is None
    # ...and once the window has passed, the next lookup picks it up.
    import providers as providers_module

    checks = providers_module._STAMP_CHECKS
    checked_at, _ = checks[next(k for k in checks if k.endswith("launch"))]
    monkeypatch.setattr(
        "providers.time.monotonic", lambda: checked_at + _STAMP_CHECK_TTL_SECONDS + 1
    )
    assert get_provider_profile("later") is not None


def test_missing_plugin_dirs_are_never_throttled(home, monkeypatch):
    stats = []
    real_stat = os.stat

    def counting_stat(path, *args, **kwargs):
        stats.append(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)
    for _ in range(3):
        _plugin_dir_stamps(home)
    plugin_stats = [p for p in stats if p.endswith("/plugins")]
    assert len(plugin_stats) == 3


def test_first_plugin_install_is_visible_immediately(home):
    # No plugins dir at all yet: the stamp check runs unthrottled, so the layer
    # re-scans and the just-created install resolves on the very next lookup.
    assert get_provider_profile("first-ever") is None

    plugin = home / "plugins" / "model-providers" / "first-ever"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        'register_provider(ProviderProfile(name="first-ever", base_url="process://first", '
        'auth_type="external_process", api_mode="chat_completions"))\n',
        encoding="utf-8",
    )

    assert get_provider_profile("first-ever") is not None
