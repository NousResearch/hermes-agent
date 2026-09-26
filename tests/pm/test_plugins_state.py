"""pm.plugins_state: complete, order-preserving cross-profile discovery."""

from __future__ import annotations

import re
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
    import hermes_yaml as yaml

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


def test_only_live_profiles_join_the_dependency_union(homes):
    from hermes_constants import mark_named_profile_deleted

    default_home, profile_home = homes
    profiles = profile_home.parent
    _write_config(default_home, ["a-plug"])
    _write_config(profile_home, ["live-plug"])
    for name in (".work.staging-123", "Bad Name", "retired", "ghost"):
        (profiles / name).mkdir()
    for name in (".work.staging-123", "Bad Name", "retired"):
        _write_config(profiles / name, [f"{name}-plug"])
    mark_named_profile_deleted(profiles / "retired")
    (profiles / "ghost" / "plugins").mkdir()  # runtime side-effect dir, no identity marker

    assert pstate.dependency_homes() == [default_home, profile_home]
    by_root = pstate.enabled_plugins_ordered()
    assert set(by_root) == {default_home / "plugins", profile_home / "plugins"}


@pytest.mark.parametrize("boundary", ["profile-listing", "profile-stat", "plugin-stat", "manifest-read", "manifest-stat", "provider-stat"])
def test_unreadable_profile_state_is_not_an_empty_selection(homes, monkeypatch, boundary):
    from pm.workspace import enabled_member_dirs

    default_home, profile_home = homes
    _write_config(profile_home, ["keep-plug"])
    plugin = profile_home / "plugins" / "keep-plug"
    plugin.mkdir(parents=True)
    manifest = plugin / "plugin.yaml"
    manifest.write_text("python_dependencies: [fixture-dep]\n", encoding="utf-8")
    if boundary == "provider-stat":
        (profile_home / "config.yaml").write_text("memory:\n  provider: keep-plug\n", encoding="utf-8")
    method, target = {
        "profile-listing": ("iterdir", profile_home.parent),
        "profile-stat": ("stat", profile_home),
        "plugin-stat": ("stat", plugin),
        "manifest-read": ("read_text", manifest),
        "manifest-stat": ("stat", manifest),
        "provider-stat": ("stat", plugin),
    }[boundary]
    original = getattr(Path, method)

    def unreadable(path, *args, **kwargs):
        if path == target:
            raise PermissionError("access denied by fixture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, method, unreadable)
    with pytest.raises(ValueError, match=re.escape(str(target))):
        enabled_member_dirs()


def test_missing_config_parser_is_not_an_empty_plugin_selection(homes, monkeypatch):
    import sys

    default_home, _ = homes
    _write_config(default_home, ["keep-plug"])
    before = (default_home / "config.yaml").read_bytes()
    monkeypatch.setitem(sys.modules, "utils", None)
    with pytest.raises(ImportError):
        pstate.enabled_plugins_ordered()
    assert (default_home / "config.yaml").read_bytes() == before



@pytest.mark.parametrize("content", ["{ not yaml", "[]", "plugins: wrong", "plugins:\n  enabled: wrong", "memory: wrong"])
def test_enabled_read_refuses_invalid_existing_config(homes, content):
    default_home, _ = homes
    config = default_home / "config.yaml"
    config.write_text(content, encoding="utf-8")
    before = config.read_bytes()
    with pytest.raises(ValueError, match=re.escape(str(config))):
        pstate.enabled_plugins_ordered()
    assert config.read_bytes() == before

def test_verify_skips_bad_secondary_but_admission_refuses_it(homes, caplog):
    from pm.workspace import enabled_member_dirs, enabled_plugin_dirs

    default_home, profile_home = homes
    plugin = default_home / "plugins" / "working"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: working\npython_dependencies: [requests]\n", encoding="utf-8")
    _write_config(default_home, ["working"])
    bad = profile_home / "config.yaml"
    bad.write_text("plugins: [invalid]\n", encoding="utf-8")

    assert enabled_member_dirs() == [plugin]
    assert str(bad) in caplog.text
    with pytest.raises(ValueError, match=re.escape(str(bad))):
        enabled_plugin_dirs(proposed_home=default_home, enabled=["working"])
    bad.write_text("plugins: {enabled: []}\n", encoding="utf-8")
    assert enabled_member_dirs() == [plugin]


@pytest.mark.parametrize("content", ["", "# empty config\n", "null", "{}", "plugins: {}", "plugins:\n  enabled: []"])
def test_empty_config_is_an_explicit_empty_selection(homes, content):
    default_home, _ = homes
    (default_home / "config.yaml").write_text(content, encoding="utf-8")
    assert pstate.enabled_plugins_ordered() == {}


def test_legacy_false_provider_reads_as_unset(homes):
    """Older config writers emitted `memory.provider: false` to mean unset; the runtime reader
    coerces falsy to no-provider, so the PM validator must not reject the literal — that blocks
    `hermes pm repair`, the only tool that could heal such a config."""
    default_home, _ = homes
    (default_home / "config.yaml").write_text(
        "memory:\n  memory_enabled: true\n  provider: false\n", encoding="utf-8"
    )

    config = pstate.read_home_selection(default_home)

    assert config == {"memory": {"memory_enabled": True, "provider": False}}


def test_memory_provider_must_still_be_a_name(homes):
    """Anything else non-string (a mapping, a number) stays a hard error: it cannot be traced
    to a writer Hermes ever shipped."""
    import pytest

    default_home, _ = homes
    (default_home / "config.yaml").write_text(
        "memory:\n  provider: 5\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="memory.provider must be a name"):
        pstate.read_home_selection(default_home)


@pytest.mark.parametrize("enabled, provider, exists, expected", [
    (["z-first", "a-second"], "provider", True, ["z-first", "a-second", "provider"]),
    (["provider"], "provider", True, ["provider"]),
    ([], "ghost", False, []),
])
def test_memory_provider_joins_ordered_selection(homes, enabled, provider, exists, expected):
    import hermes_yaml as yaml
    home, sibling = homes
    if exists:
        (home / "plugins" / provider).mkdir(parents=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": enabled}, "memory": {"provider": provider}}))
    _write_config(sibling, ["sibling"])
    by_root = pstate.enabled_plugins_ordered()
    assert by_root.get(home / "plugins", []) == expected
    assert by_root[sibling / "plugins"] == ["sibling"]


def test_member_discovery_uses_real_enabled_config(homes):
    from pm.workspace import enabled_member_dirs
    home, _ = homes
    manifests = {
        "modern/pyproject.toml": "[project]\n",
        "legacy/plugin.yaml": 'name: legacy\npip_dependencies: ["requests>=2"]\n',
        "plain/plugin.yaml": "name: plain\n",
        "orphan/pyproject.toml": "[project]\n",
    }
    for relative, body in manifests.items():
        path = home / "plugins" / relative
        path.parent.mkdir(parents=True)
        path.write_text(body)
    _write_config(home, ["legacy", "modern", "plain"])
    assert enabled_member_dirs() == [home / "plugins/legacy", home / "plugins/modern"]
    _write_config(home, [])
    assert enabled_member_dirs() == []




def test_read_parses_config_once_per_home(homes, monkeypatch):
    """enabled_plugins_ordered must parse each home's config.yaml once,
    not once for plugins.enabled and again for memory.provider."""
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
    pstate.enabled_plugins_ordered()
    assert len(calls) == 2  # one per home, not one per query
