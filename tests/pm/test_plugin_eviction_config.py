"""Config edits made by PluginEviction keep plugins.enabled and plugins.disabled consistent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import hermes_yaml as yaml
import hermes_cli.plugins_cmd as _plugins_cmd
import pm.publication as publication
from pm.plugin_eviction import PluginEviction, _discard_aliases


_REAL_ALIASES = _plugins_cmd._plugin_aliases


@pytest.fixture(autouse=True)
def _no_alias_discovery(monkeypatch):
    """Alias discovery scans installed plugins; these tests only need key + bare leaf."""
    import hermes_cli.plugins_cmd as plugins_cmd

    monkeypatch.setattr(plugins_cmd, "_plugin_aliases", lambda key, entries=None: {key, key.split("/")[-1]})
    monkeypatch.setattr(publication, "selection_snapshot", lambda: {})


def _entry(home: Path, name: str) -> tuple[Path, str, Path]:
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    return home / "plugins", name, plugin_dir


def _evict(home: Path, names: list[str], text: str | bytes | None) -> dict:
    config = home / "config.yaml"
    if text is not None:
        config.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    entries = [_entry(home, name) for name in names]
    reasons = {plugin_dir.resolve(): "too old" for _d, _n, plugin_dir in entries}
    eviction = PluginEviction(entries, reasons)
    (_path, _previous, proposed), = eviction.edits
    return yaml.safe_load(proposed.decode("utf-8")), proposed.decode("utf-8")


def test_evicted_plugin_leaves_enabled_and_joins_disabled(tmp_path):
    cfg, _ = _evict(tmp_path, ["plug-b"], "plugins:\n  enabled: [plug-a, plug-b]\n")
    assert cfg["plugins"]["enabled"] == ["plug-a"]
    assert cfg["plugins"]["disabled"] == ["plug-b"]


def test_only_enabled_case_creates_disabled_list(tmp_path):
    cfg, _ = _evict(tmp_path, ["plug-a"], "plugins:\n  enabled: [plug-a]\n")
    assert cfg["plugins"]["enabled"] == []
    assert cfg["plugins"]["disabled"] == ["plug-a"]


def test_already_in_both_lists_is_repaired_without_duplicates(tmp_path):
    cfg, _ = _evict(tmp_path, ["plug-a"], "plugins:\n  enabled: [plug-a, plug-b]\n  disabled: [plug-a]\n")
    assert cfg["plugins"]["enabled"] == ["plug-b"]
    assert cfg["plugins"]["disabled"] == ["plug-a"]


def test_missing_enabled_list_is_not_created(tmp_path):
    cfg, _ = _evict(tmp_path, ["plug-a"], "plugins:\n  disabled: [other]\n")
    assert "enabled" not in cfg["plugins"]
    assert cfg["plugins"]["disabled"] == ["other", "plug-a"]


def test_namespaced_and_bare_spellings_are_both_removed(tmp_path):
    cfg, _ = _evict(tmp_path, ["web/firecrawl"], "plugins:\n  enabled: [web/firecrawl, firecrawl, keep]\n")
    assert cfg["plugins"]["enabled"] == ["keep"]
    assert cfg["plugins"]["disabled"] == ["web/firecrawl"]


def test_comments_and_order_survive(tmp_path):
    text = "# operator note\nplugins:\n  enabled:\n    - first  # keep me\n    - plug-b\n    - last\n"
    cfg, out = _evict(tmp_path, ["plug-b"], text)
    assert "# operator note" in out
    assert "# keep me" in out
    assert cfg["plugins"]["enabled"] == ["first", "last"]


def test_running_twice_is_idempotent(tmp_path):
    _cfg, first = _evict(tmp_path, ["plug-b"], "plugins:\n  enabled: [plug-a, plug-b]\n")
    cfg, second = _evict(tmp_path, ["plug-b"], first)
    assert second == first
    assert cfg["plugins"]["disabled"] == ["plug-b"]


def test_memory_provider_cleared_only_for_evicted_plugin(tmp_path):
    cfg, _ = _evict(tmp_path, ["mem"], "plugins:\n  enabled: [mem]\nmemory:\n  provider: mem\n")
    assert cfg["memory"]["provider"] == ""
    other = tmp_path / "other"
    other.mkdir()
    cfg, _ = _evict(other, ["plug-a"], "plugins:\n  enabled: [plug-a]\nmemory:\n  provider: mem\n")
    assert cfg["memory"]["provider"] == "mem"


def test_utf8_bom_config_round_trips(tmp_path):
    cfg, out = _evict(tmp_path, ["plug-b"], "\ufeffplugins:\n  enabled: [plug-a, plug-b]\n".encode("utf-8"))
    assert cfg["plugins"]["enabled"] == ["plug-a"]
    assert not out.startswith("\ufeff")


def test_secondary_home_is_edited_independently(tmp_path):
    root, profile = tmp_path / "root", tmp_path / "root" / "profiles" / "work"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("plugins:\n  enabled: [plug-a]\n", encoding="utf-8")
    (profile / "config.yaml").write_text("plugins:\n  enabled: [plug-a, keep]\n", encoding="utf-8")
    entries = [_entry(root, "plug-a"), _entry(profile, "plug-a")]
    eviction = PluginEviction(entries, {entries[1][2].resolve(): "too old"})
    (path, _previous, proposed), = eviction.edits
    assert path == profile / "config.yaml"
    assert yaml.safe_load(proposed.decode())["plugins"]["enabled"] == ["keep"]


def test_publish_records_hashes_and_writes_configs(tmp_path, monkeypatch):
    import pm.plugin_eviction as module

    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("plugins:\n  enabled: [plug-a]\n", encoding="utf-8")
    entry = _entry(home, "plug-a")
    eviction = PluginEviction([entry], {entry[2].resolve(): "too old"})
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setattr(module, "install_state_dir", lambda project: state)
    monkeypatch.setattr(module, "runtime_facts_path", lambda project: tmp_path / "facts.json")
    eviction.publish(tmp_path)

    row = json.loads((state / "publication.json").read_text())
    (_p, previous, proposed), = eviction.edits
    assert row["configs"][0]["config_after"] == hashlib.sha256(proposed).hexdigest()
    assert config.read_bytes() == proposed
    assert yaml.safe_load(config.read_text())["plugins"]["disabled"] == ["plug-a"]


def test_publish_refuses_when_selection_changed(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: [plug-a]\n", encoding="utf-8")
    entry = _entry(home, "plug-a")
    eviction = PluginEviction([entry], {entry[2].resolve(): "too old"})
    monkeypatch.setattr(publication, "selection_snapshot", lambda: {"changed": True})
    with pytest.raises(ValueError, match="changed while preparing"):
        eviction.publish(tmp_path)
    assert yaml.safe_load((home / "config.yaml").read_text())["plugins"]["enabled"] == ["plug-a"]


def test_discard_aliases_keeps_other_entries_in_order():
    names = ["a", "x", "b", "x"]
    _discard_aliases(names, {"x"})
    assert names == ["a", "b"]


def test_alias_discovery_failure_from_io_falls_back_and_is_logged(tmp_path, monkeypatch, caplog):
    import hermes_cli.plugins_cmd as plugins_cmd

    def unreadable(key, entries=None):
        raise OSError("tree unreadable")

    monkeypatch.setattr(plugins_cmd, "_plugin_aliases", unreadable)
    with caplog.at_level("WARNING", logger="pm.plugin_eviction"):
        cfg, _ = _evict(tmp_path, ["web/firecrawl"], "plugins:\n  enabled: [web/firecrawl, firecrawl, keep]\n")
    assert cfg["plugins"]["enabled"] == ["keep"]
    assert "tree unreadable" in caplog.text


def test_unexpected_alias_discovery_error_is_not_swallowed(tmp_path, monkeypatch):
    import hermes_cli.plugins_cmd as plugins_cmd

    def broken(key, entries=None):
        raise RuntimeError("half-broken plugin tree")

    monkeypatch.setattr(plugins_cmd, "_plugin_aliases", broken)
    with pytest.raises(RuntimeError, match="half-broken plugin tree"):
        _evict(tmp_path, ["plug-a"], "plugins:\n  enabled: [plug-a]\n")


def test_real_alias_discovery_skips_a_malformed_manifest_and_still_removes_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_plugins_cmd, "_plugin_aliases", _REAL_ALIASES)
    good = tmp_path / "plugins" / "web" / "firecrawl"
    good.mkdir(parents=True)
    (good / "plugin.yaml").write_text("name: web-firecrawl\n", encoding="utf-8")
    bad = tmp_path / "plugins" / "broken"
    bad.mkdir(parents=True)
    (bad / "plugin.yaml").write_text("name: [unterminated\n", encoding="utf-8")
    cfg, _ = _evict(tmp_path, ["web/firecrawl"],
                    "plugins:\n  enabled: [web/firecrawl, web-firecrawl, firecrawl, keep]\n")
    assert cfg["plugins"]["enabled"] == ["keep"]
