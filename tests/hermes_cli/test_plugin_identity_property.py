"""Property: every reader of ``plugins.enabled`` / ``plugins.disabled`` gives the loader's verdict.

One rule (``hermes_cli.plugin_identity``) decides whether a list names a plugin. The corpus draws both lists
from a moved plugin's every spelling (catalog name, core-era key, core-era manifest name), case variants,
blank entries, a nested ``team/<x>`` key and unrelated names, over installed-tree shapes (the catalog copy, a
fork at the core-era path, a flat fork under the catalog dir, nested copies), and asserts that the loader's
gate (``gate_manifest`` over discovery's own scan), PM's dependency selection, the default-plugin installer's
``_settled``, the left-core migrator's ``_installed_for_good`` and ``hermes plugins`` status agree per tree.
A counterexample a reviewer finds joins ``_SHAPES`` / ``_spellings``.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import pm.plugins_state as pstate

_CASES = 160


def _moved() -> list[str]:
    """Every catalog plugin that left core under a bundled key on this tree."""
    from hermes_cli.left_core_migration import LEFT_CORE
    return [f.plugin for f in LEFT_CORE if f.plugin_key and f.plugin_names]


def _spellings(row) -> list[str]:
    p = row.plugin
    return [p, row.plugin_key, *row.plugin_names, p.upper(), p.title(), row.plugin_key.upper(), "", "  ",
            f"team/{p}", f"my-{p}", f"{p}-team", "unrelated", f" {p}"]


# (relative dir, manifest name) per tree; ``{p}`` is the moved plugin's catalog name.
_SHAPES = {
    "catalog": [("{p}", "{p}")],
    "fork-at-core-path": [("platforms/{p}", "my-{p}")],
    "flat-fork": [("{p}", "my-{p}")],
    "nested-renamed": [("team/{p}", "{p}-team")],
    "nested-same-name": [("team/{p}", "{p}")],
    "catalog+fork": [("{p}", "{p}"), ("platforms/{p}", "my-{p}")],
    "catalog+nested": [("{p}", "{p}"), ("team/{p}", "{p}-team")],
    "manifestless": [("{p}", None)],
}


def _install(home: Path, shape: str, plugin: str) -> list[Path]:
    trees = []
    for rel, name in _SHAPES[shape]:
        tree = home / "plugins" / rel.format(p=plugin)
        tree.mkdir(parents=True, exist_ok=True)
        if name is not None:
            (tree / "plugin.yaml").write_text(f"name: {name.format(p=plugin)}\n", encoding="utf-8")
        (tree / "__init__.py").write_text("", encoding="utf-8")
        trees.append(tree)
    return trees


def _loader_verdicts(home: Path, enabled: list, disabled: list) -> dict[Path, str]:
    """``load`` / ``disabled`` / ``off`` per tree, by the plugin loader's gate over discovery's scan."""
    from hermes_cli.plugins_discovery import IN_BOTH_LISTS, gate_manifest, resolve_manifest_winners, scan_directory
    verdicts = {}
    for manifest in resolve_manifest_winners(scan_directory(home / "plugins", "user")).values():
        gate = gate_manifest(manifest, set(disabled), set(enabled))
        verdicts[Path(str(manifest.path))] = ("load" if gate.action == "load" else
                                         "disabled" if gate.error in ("disabled via config", IN_BOTH_LISTS) else "off")
    return verdicts


def _case(rng: random.Random, row) -> tuple[str, list, list]:
    names = _spellings(row)
    return (rng.choice(sorted(_SHAPES)), rng.sample(names, rng.randint(0, 4)), rng.sample(names, rng.randint(0, 3)))


@pytest.mark.parametrize("plugin", _moved())
def test_every_plugin_list_reader_gives_the_loaders_verdict(tmp_path, monkeypatch, plugin):
    import hermes_constants
    import hermes_yaml as yaml
    from hermes_cli import default_plugins, left_core_migration as lcm, plugins_cmd
    from hermes_cli.plugin_catalog import entry_from_mapping
    from hermes_cli.plugins_manifest import manifest_key
    from hermes_cli.plugins_discovery import scan_directory
    from pm.workspace import enabled_plugin_dirs

    from hermes_cli.left_core_migration import LEFT_CORE
    row = next(f for f in LEFT_CORE if f.plugin == plugin)
    entry = entry_from_mapping({"name": row.plugin, "repo": "https://github.com/NousResearch/hermes-official-plugins",
                                "sha": "a" * 40, "description": "d", "maintainer": "NousResearch",
                                "tier": "official", "default": True}, row.plugin)
    rng = random.Random(f"plugin-identity-{plugin}")
    monkeypatch.setattr(pstate, "_profiles_root", lambda: tmp_path / "no-profiles")
    failures = []
    p, old_key, old_name = row.plugin, row.plugin_key, row.plugin_names[0]
    known = [  # review counterexamples: a blank disable, a nested copy, a fork at the core-era path
        ("catalog", [p], [""]), ("catalog", [p], ["  "]), ("nested-renamed", [f"team/{p}"], [old_name]),
        ("fork-at-core-path", [old_key], [old_name]), ("catalog+fork", [old_key], [p]),
        ("flat-fork", [p], []), ("catalog", [old_key.upper()], []),
    ]
    for i in range(_CASES):
        shape, enabled, disabled = known[i] if i < len(known) else _case(rng, row)
        home = tmp_path / f"case{i}"
        trees = _install(home, shape, row.plugin)
        (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": enabled, "disabled": disabled}}))
        monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda home=home: home)
        monkeypatch.setenv("HERMES_HOME", str(home))
        loader = _loader_verdicts(home, enabled, disabled)
        # A tree discovery does not read (no manifest) has no loader verdict; PM resolves it by path.
        pm_selected = {p for p in enabled_plugin_dirs() if p in loader}
        expected_pm = {tree for tree, verdict in loader.items() if verdict == "load"}
        if pm_selected != expected_pm:
            failures.append((shape, enabled, disabled, "pm", sorted(map(str, pm_selected)), sorted(map(str, expected_pm))))
        for manifest in scan_directory(home / "plugins", "user"):
            status = plugins_cmd._plugin_status(manifest.name, set(enabled), set(disabled), key=manifest_key(manifest),
                                                source="user", dir_path=manifest.path)
            want = {"load": "enabled", "disabled": "disabled", "off": "not enabled"}[loader[Path(str(manifest.path))]]
            if status != want:
                failures.append((shape, enabled, disabled, "hermes plugins", manifest.path, status, want))
        catalog_tree = home / "plugins" / row.plugin
        if catalog_tree in loader:
            # Left unfinished by both automatic installers: only the user's own choice settles it.
            (home / "cache").mkdir(exist_ok=True)
            lcm._placed(home, row.plugin).write_text("")
            chose = loader[catalog_tree] != "off"
            if default_plugins._settled(home, entry) != chose:
                failures.append((shape, enabled, disabled, "_settled", not chose))
            if lcm._installed_for_good(home, row.plugin) != chose:
                failures.append((shape, enabled, disabled, "_installed_for_good", not chose))
        assert trees
    assert not failures, failures[:5]
