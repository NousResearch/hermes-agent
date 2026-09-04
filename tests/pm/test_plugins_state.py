"""pm.plugins_state: order-preserving enabled reads + disable write-back.

The union is cross-profile and recency-ordered; the bisect writes its
disable decisions back through the same config.yaml the plugins CLI
owns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pm.plugins_state as pstate


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Default home + one profile, each with a config.yaml."""
    default_home = tmp_path / "default-home"
    profile_home = tmp_path / "profiles" / "work"
    default_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)

    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "get_default_hermes_root", lambda: default_home
    )
    monkeypatch.setattr(pstate, "_profiles_root", lambda: tmp_path / "profiles")
    return default_home, profile_home


def _write_config(home: Path, enabled: list) -> None:
    import yaml

    config = {"plugins": {"enabled": enabled}} if enabled else {"plugins": {}}
    with (home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def test_enabled_plugins_ordered_reads_all_homes(homes):
    default_home, profile_home = homes
    _write_config(default_home, ["a-plug", "b-plug"])
    _write_config(profile_home, ["c-plug"])

    by_root = pstate.enabled_plugins_ordered()
    assert by_root.get(default_home / "plugins") == ["a-plug", "b-plug"]
    assert by_root.get(profile_home / "plugins") == ["c-plug"]


def test_enabled_list_preserves_config_order(homes):
    default_home, _ = homes
    # NOT alphabetical: recency order must survive the read
    _write_config(default_home, ["z-first-enabled", "a-second"])
    by_root = pstate.enabled_plugins_ordered()
    assert by_root[default_home / "plugins"] == ["z-first-enabled", "a-second"]


def test_disable_plugins_removes_across_homes(homes):
    default_home, profile_home = homes
    _write_config(default_home, ["bad-plug", "keep-plug"])
    _write_config(profile_home, ["bad-plug", "other"])

    removed = pstate.disable_plugins(["bad-plug"])
    assert removed[str(default_home)] == ["bad-plug"]
    assert removed[str(profile_home)] == ["bad-plug"]

    by_root = pstate.enabled_plugins_ordered()
    assert by_root[default_home / "plugins"] == ["keep-plug"]
    assert by_root[profile_home / "plugins"] == ["other"]


def test_disable_plugins_noop_when_not_enabled(homes):
    default_home, _ = homes
    _write_config(default_home, ["keep-plug"])
    removed = pstate.disable_plugins(["not-there"])
    assert removed == {}
    # config untouched
    by_root = pstate.enabled_plugins_ordered()
    assert by_root[default_home / "plugins"] == ["keep-plug"]


def test_enabled_read_survives_garbage_config(homes):
    default_home, _ = homes
    (default_home / "config.yaml").write_text("{ not yaml", encoding="utf-8")
    assert pstate.enabled_plugins_ordered() == {}


def test_active_memory_provider_joins_union(homes, tmp_path):
    """The mnemosyne path: a provider installed via memory.provider (not
    plugins.enabled) must join the union — deps ride the lock either way."""
    default_home, _ = homes
    provider_dir = default_home / "plugins" / "mnemosyne-like"
    provider_dir.mkdir(parents=True)
    (provider_dir / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    import yaml

    with (default_home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"plugins": {"enabled": ["regular-plug"]},
             "memory": {"provider": "mnemosyne-like"}},
            f,
        )

    by_root = pstate.enabled_plugins_ordered()
    assert by_root[default_home / "plugins"] == ["regular-plug", "mnemosyne-like"]


def test_memory_provider_without_dir_is_skipped(homes):
    """memory.provider set but no plugin dir on disk — not a member."""
    default_home, _ = homes
    import yaml

    with (default_home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"memory": {"provider": "ghost-provider"}}, f)

    assert pstate.enabled_plugins_ordered() == {}


def test_memory_provider_already_enabled_not_duplicated(homes):
    default_home, _ = homes
    (default_home / "plugins" / "dual").mkdir(parents=True)
    import yaml

    with (default_home / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"plugins": {"enabled": ["dual"]}, "memory": {"provider": "dual"}}, f
        )

    by_root = pstate.enabled_plugins_ordered()
    assert by_root[default_home / "plugins"] == ["dual"]  # once, not twice
