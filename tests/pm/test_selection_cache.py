"""Config selection caching: cache hit skips YAML re-parsing.

Regression for #130797 (slow CLI startup parses every profile's config.yaml
on every command).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import pm.plugins_state as pstate


@pytest.fixture
def homes(tmp_path, monkeypatch):
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "work"
    default_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: default_home)
    monkeypatch.setattr(pstate, "_profiles_root", lambda: tmp_path / "profiles")
    pstate.clear_selection_cache()
    yield default_home, profile_home
    pstate.clear_selection_cache()


def _write_config(home: Path, enabled: list) -> None:
    import hermes_yaml as yaml
    config = {"plugins": {"enabled": enabled}} if enabled else {"plugins": {}}
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def test_cache_hit_skips_yaml_reparse(homes, monkeypatch):
    default_home, profile_home = homes
    _write_config(default_home, ["a-plug"])
    _write_config(profile_home, ["c-plug"])

    import utils
    calls: list = []
    real = utils.fast_safe_load

    def counting(stream):
        calls.append(stream)
        return real(stream)

    monkeypatch.setattr(utils, "fast_safe_load", counting)
    first = pstate.enabled_plugins_ordered()
    assert len(calls) == 2
    calls.clear()
    second = pstate.enabled_plugins_ordered()
    assert second == first
    assert calls == []


def test_cache_invalidates_on_config_change(homes, monkeypatch):
    default_home, profile_home = homes
    _write_config(default_home, ["a-plug"])
    _write_config(profile_home, ["c-plug"])
    assert pstate.enabled_plugins_ordered()[default_home / "plugins"] == ["a-plug"]

    import utils
    calls: list = []
    real = utils.fast_safe_load

    def counting(stream):
        calls.append(stream)
        return real(stream)

    monkeypatch.setattr(utils, "fast_safe_load", counting)
    _write_config(default_home, ["a-plug", "b-plug"])
    result = pstate.enabled_plugins_ordered()
    assert result[default_home / "plugins"] == ["a-plug", "b-plug"]
    assert len(calls) >= 1


def test_parked_profiles_skipped_from_union(homes):
    default_home, profile_home = homes
    _write_config(default_home, ["a-plug"])
    _write_config(profile_home, ["parked-plug"])
    (profile_home / "gateway.parked").write_text("offline\n", encoding="utf-8")
    pstate.clear_selection_cache()

    assert profile_home not in pstate.dependency_homes()
    by_root = pstate.enabled_plugins_ordered()
    assert set(by_root) == {default_home / "plugins"}

    # Opt-in still sees parked homes.
    assert profile_home in pstate.dependency_homes(include_parked=True)
    by_root_all = pstate.enabled_plugins_ordered(include_parked=True)
    assert by_root_all.get(profile_home / "plugins") == ["parked-plug"]
