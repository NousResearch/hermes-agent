"""An update that evicts one plugin tree disables that tree by its own registry key, never by the entry that
enabled it: one ``plugins.enabled`` entry may name a working copy too (``hermes_cli.plugin_identity``)."""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml
import pytest

import pm.plugins_state as pstate


def _tree(home: Path, rel: str, name: str) -> Path:
    tree = home / "plugins" / rel
    tree.mkdir(parents=True)
    (tree / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    (tree / "__init__.py").write_text("", encoding="utf-8")
    return tree


@pytest.mark.parametrize("failing", ["team/acme", "acme"])
def test_evicting_one_of_two_same_named_trees_keeps_the_other_on(tmp_path, monkeypatch, failing):
    import hermes_constants
    from hermes_cli.plugins_discovery import collect_directory_manifests, gate_manifest, resolve_manifest_winners
    from pm.plugin_eviction import PluginEviction
    from pm.workspace import enabled_plugin_entries

    home = tmp_path / "home"
    catalog, fork = _tree(home, "acme", "acme"), _tree(home, "team/acme", "acme")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["acme"]}}), encoding="utf-8")
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    monkeypatch.setattr(pstate, "_profiles_root", lambda: tmp_path / "no-profiles")
    monkeypatch.setenv("HERMES_HOME", str(home))
    entries = enabled_plugin_entries()
    assert {plugin_dir for _d, _n, plugin_dir in entries} == {catalog, fork}

    broken = fork if failing == "team/acme" else catalog
    edit = PluginEviction(entries, {broken.resolve(): "fixture: does not build"}, configs=())
    plugins = yaml.safe_load(edit.edits[0][2].decode("utf-8"))["plugins"]
    loads = {Path(str(m.path)) for m in resolve_manifest_winners(collect_directory_manifests(home)).values()
             if m.source == "user" and gate_manifest(m, set(plugins.get("disabled") or ()),
                                                     set(plugins.get("enabled") or ())).action == "load"}
    assert broken not in loads
    assert plugins["disabled"] == [failing]
    if failing == "team/acme":  # the nested fork has a key of its own: the working catalog copy stays on
        assert loads == {catalog}
        assert plugins["enabled"] == ["acme"]
