"""Property: every reader of ``plugins.enabled`` / ``plugins.disabled`` gives the loader's verdict.

One rule (``hermes_cli.plugin_identity``) decides whether a list names a plugin. The corpus draws both lists
from a plugin's every spelling (catalog name, and for a plugin that left core its bundled key and old manifest
name), case variants, blank entries, a nested ``team/<x>`` key and unrelated names, over installed-tree shapes
(the catalog copy, a fork at the core-era path, a flat fork, nested copies, a folder that cannot be read, a
flat folder claiming a bundled plugin's key from another directory name), for every plugin that left core on
this tree, a plugin core never shipped and a bundled one. It asserts that the loader's gate (``gate_manifest``
over discovery's own scan and winner rule), PM's dependency selection, the default-plugin installer's
``_settled``, the left-core migrator's ``_installed_for_good`` and ``hermes plugins`` status agree per tree,
and that an update evicting one selected tree leaves every other tree's verdict as it was.
A counterexample a reviewer finds joins ``_SHAPES`` / ``_spellings`` / ``known``.
"""

from __future__ import annotations

import os
import random
import stat
from pathlib import Path

import pytest

import pm.plugins_state as pstate

_CASES = 120
_BUNDLED = "disk-cleanup"  # a flat bundled plugin: the key an impostor folder claims


def _plugins() -> list[str]:
    """Every plugin that left core on this tree, one core never shipped, and a bundled one."""
    from hermes_cli.left_core_migration import LEFT_CORE
    return [*dict.fromkeys(f.plugin for f in LEFT_CORE), "acme-tool", _BUNDLED]


def _old_names(plugin: str) -> list[str]:
    from hermes_cli.left_core_migration import LEFT_CORE
    return [n for f in LEFT_CORE if f.plugin == plugin for n in (f.plugin_key, *f.plugin_names) if n]


def _spellings(p: str) -> list[str]:
    return [p, *_old_names(p), p.upper(), p.title(), *(n.upper() for n in _old_names(p)), "", "  ",
            f"team/{p}", f"my-{p}", f"{p}-team", "unrelated", f" {p}"]


# (relative dir, manifest name) per tree; ``{p}`` is the plugin's catalog name. A ``None`` name is a tree
# without a manifest; a trailing ``!`` marks a folder nobody can read.
_SHAPES = {
    "catalog": [("{p}", "{p}")],
    "fork-at-core-path": [("platforms/{p}", "my-{p}")],
    "flat-fork": [("{p}", "my-{p}")],
    "nested-renamed": [("team/{p}", "{p}-team")],
    "nested-same-name": [("team/{p}", "{p}")],
    "catalog+fork": [("{p}", "{p}"), ("platforms/{p}", "my-{p}")],
    "catalog+nested": [("{p}", "{p}"), ("team/{p}", "{p}-team")],
    "catalog+nested-same-name": [("{p}", "{p}"), ("team/{p}", "{p}")],
    "manifestless": [("{p}", None)],
    "catalog+unreadable": [("{p}", "{p}"), ("zz-broken!", "zz-broken")],
    "impostor": [("other-{p}", "{p}")],
}


def _install(home: Path, shape: str, plugin: str) -> list[Path]:
    trees = []
    for rel, name in _SHAPES[shape]:
        tree = home / "plugins" / rel.format(p=plugin).rstrip("!")
        tree.mkdir(parents=True, exist_ok=True)
        if name is not None:
            (tree / "plugin.yaml").write_text(f"name: {name.format(p=plugin)}\n", encoding="utf-8")
        (tree / "__init__.py").write_text("", encoding="utf-8")
        if rel.endswith("!"):
            tree.chmod(0)
        trees.append(tree)
    return trees


def _loader_verdicts(home: Path, enabled: list, disabled: list) -> dict[Path, str]:
    """``load`` / ``disabled`` / ``off`` per user tree, by the plugin loader's gate over discovery's scan."""
    from hermes_cli.plugins_discovery import (
        IN_BOTH_LISTS, collect_directory_manifests, gate_manifest, resolve_manifest_winners,
    )
    verdicts = {}
    for manifest in resolve_manifest_winners(collect_directory_manifests(home)).values():
        if manifest.source != "user":
            continue
        gate = gate_manifest(manifest, set(disabled), set(enabled))
        verdicts[Path(str(manifest.path))] = ("load" if gate.action == "load" else
                                         "disabled" if gate.error in ("disabled via config", IN_BOTH_LISTS) else "off")
    return verdicts


def _case(rng: random.Random, plugin: str) -> tuple[str, list, list]:
    names = _spellings(plugin)
    return (rng.choice(sorted(_SHAPES)), rng.sample(names, rng.randint(0, 4)), rng.sample(names, rng.randint(0, 3)))


def _evicted(entries: list, tree: Path) -> tuple[list, list]:
    """``plugins.enabled`` / ``plugins.disabled`` after an update evicts *tree* alone (``PluginEviction``)."""
    import hermes_yaml as yaml
    from pm.plugin_eviction import PluginEviction
    edit = PluginEviction(entries, {tree.resolve(): "fixture: does not build"}, configs=())
    _path, _previous, proposed = edit.edits[0]
    plugins = (yaml.safe_load(proposed.decode("utf-8")) or {}).get("plugins") or {}
    return list(plugins.get("enabled") or []), list(plugins.get("disabled") or [])


def _check(home: Path, row, entry, plugin: str, enabled: list, disabled: list, trees: list) -> list:
    from hermes_cli import default_plugins, left_core_migration as lcm, plugins_cmd
    from hermes_cli.memory_provider_migration import _failure_stamp
    from hermes_cli.plugin_identity import installed_trees, registry_key
    from hermes_cli.plugins_discovery import scan_directory
    from hermes_cli.plugins_manifest import manifest_key
    from pm.workspace import enabled_plugin_entries
    failures = []
    loader = _loader_verdicts(home, enabled, disabled)
    # A tree discovery does not read (no manifest) has no loader verdict; PM resolves it by path.
    by_path = {tree for tree in trees if os.access(tree, os.R_OK) and not (tree / "plugin.yaml").exists()}
    entries = enabled_plugin_entries()
    # Each selected dir as the tree it is (a case-insensitive filesystem reaches one tree by several spellings).
    pm_selected = {next((t for t in trees if os.path.samefile(t, plugin_dir)), plugin_dir)
                   for _d, _n, plugin_dir in entries if plugin_dir.is_relative_to(home / "plugins")} - by_path
    expected_pm = {tree for tree, verdict in loader.items() if verdict == "load"}
    if pm_selected != expected_pm:
        failures.append(("pm", sorted(map(str, pm_selected)), sorted(map(str, expected_pm))))
    for manifest in scan_directory(home / "plugins", "user"):
        if Path(str(manifest.path)) not in loader:
            continue  # an impostor discovery's winner rule drops: no status of its own
        status = plugins_cmd._plugin_status(manifest.name, set(enabled), set(disabled), key=manifest_key(manifest),
                                            source="user", dir_path=manifest.path)
        want = {"load": "enabled", "disabled": "disabled", "off": "not enabled"}[loader[Path(str(manifest.path))]]
        if status != want:
            failures.append(("hermes plugins", manifest.path, status, want))
    identities = installed_trees(home / "plugins")
    for tree in sorted(pm_selected):
        after = _loader_verdicts(home, *_evicted(entries, tree))
        # Its registry key is the one entry an update can disable it by; a tree that key names too goes with it.
        key = registry_key(home / "plugins", tree)
        others = {t: v for t, v in loader.items() if t != tree and key not in identities.get(t, ())}
        if after.get(tree) == "load" or {t: after.get(t) for t in others} != others:
            failures.append(("eviction", str(tree), loader, after))
    catalog_tree = home / "plugins" / plugin
    if catalog_tree in loader:
        # Left unfinished by an automatic installer: only the user's own choice settles it.
        chose = loader[catalog_tree] != "off"
        stamp = _failure_stamp(home, default_plugins._attempt(entry))
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text("")
        if default_plugins._settled(home, entry) != chose:
            failures.append(("_settled", not chose))
        if row is not None:
            lcm._placed(home, plugin).parent.mkdir(parents=True, exist_ok=True)
            lcm._placed(home, plugin).write_text("")
            if lcm._installed_for_good(home, plugin) != chose:
                failures.append(("_installed_for_good", not chose))
    return failures


@pytest.mark.platforms("posix")  # a mode-000 folder is unreadable only on POSIX
@pytest.mark.parametrize("plugin", _plugins())
def test_every_plugin_list_reader_gives_the_loaders_verdict(tmp_path, monkeypatch, plugin):
    import hermes_constants
    import hermes_yaml as yaml
    from hermes_cli.left_core_migration import LEFT_CORE
    from hermes_cli.plugin_catalog import entry_from_mapping

    row = next((f for f in LEFT_CORE if f.plugin == plugin), None)
    entry = entry_from_mapping({"name": plugin, "repo": "https://github.com/NousResearch/hermes-official-plugins",
                                "sha": "a" * 40, "description": "d", "maintainer": "NousResearch",
                                "tier": "official", "default": True}, plugin)
    rng = random.Random(f"plugin-identity-{plugin}")
    monkeypatch.setattr(pstate, "_profiles_root", lambda: tmp_path / "no-profiles")
    p, old = plugin, (_old_names(plugin) or [plugin])
    known = [  # review counterexamples
        ("catalog", [p], [""]), ("catalog", [p], ["  "]),  # a blank disable names nothing
        ("nested-renamed", [f"team/{p}"], [old[-1]]),  # a nested copy is judged by its own names
        ("fork-at-core-path", [old[0]], [old[-1]]), ("catalog+fork", [old[0]], [p]),  # a fork at the core-era path
        ("flat-fork", [p], []), ("catalog", [old[0].upper()], []),  # no case folding
        ("catalog+unreadable", [p], []), ("catalog+unreadable", [], []),  # an unreadable unrelated folder
        ("impostor", [p], []),  # a flat folder claiming a bundled key from another directory name
        ("catalog+nested-same-name", [p], []),  # evicting one of two same-named trees keeps the other on
    ]
    failures = []
    for i in range(_CASES):
        shape, enabled, disabled = known[i] if i < len(known) else _case(rng, plugin)
        home = tmp_path / f"case{i}"
        trees = _install(home, shape, plugin)
        (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": enabled, "disabled": disabled}}))
        monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda home=home: home)
        monkeypatch.setenv("HERMES_HOME", str(home))
        try:
            failures += [(shape, enabled, disabled, *f) for f in _check(home, row, entry, plugin, enabled, disabled, trees)]
        except Exception as exc:  # health: allow BLE001 -- a reader that raises is a counterexample, reported with its case
            failures.append((shape, enabled, disabled, "raised", repr(exc)))
        finally:
            for tree in trees:
                tree.chmod(stat.S_IRWXU)
    assert not failures, (sorted({(f[0], f[3]) for f in failures}), failures[:3])
