"""Invariants of the plugin Python-dependency contract (hermes_cli.plugin_python_deps):

* a candidate whose declared deps conflict is REFUSED before install and nothing else changes;
* after an update, a non-memory plugin that no longer resolves is disabled loudly while memory
  providers survive.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import plugin_python_deps as deps


def _plugin(root: Path, name: str, spec_lines: str, *, memory: bool = False, external: bool = False) -> Path:
    d = root / "plugins" / name
    d.mkdir(parents=True)
    manifest = f"name: {name}\nversion: 1.0.0\n" + ("kind: exclusive\n" if memory else "")
    if external:
        manifest += "python_runtime: external\n"
    (d / "plugin.yaml").write_text(manifest, encoding="utf-8")
    (d / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (d / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "1.0"\ndependencies = [{spec_lines}]\n', encoding="utf-8")
    return d


def _home(tmp_path: Path, enabled: list[str]) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [" + ", ".join(enabled) + "]\n  disabled: []\n", encoding="utf-8")
    return home


def _fake_resolver(conflicting: set[str]):
    """Stand-in for uv: fails with a conflict whenever the spec set contains a poisoned spec."""
    calls: list[list[str]] = []

    def resolve(specs, constraints, *, dry_run, timeout=900):
        calls.append(list(specs))
        if conflicting & set(specs):
            raise deps.DependencyConflict("No solution found when resolving dependencies")

    return resolve, calls


def test_conflicting_candidate_is_refused_and_enabled_plugins_are_untouched(tmp_path, monkeypatch):
    home = _home(tmp_path, ["good"])
    _plugin(home, "good", '"tabulate>=0.9"')
    candidate = _plugin(tmp_path / "staging", "bad", '"tabulate<0.9"')
    resolve, calls = _fake_resolver({"tabulate<0.9"})
    monkeypatch.setattr(deps, "resolve", resolve)
    monkeypatch.setattr(deps, "core_constraints", lambda root: ["httpx==0.28.1"])
    monkeypatch.setattr(deps, "dependency_homes", lambda: [home])

    with pytest.raises(deps.DependencyConflict):
        deps.check_candidate(deps.read_declaration(candidate), home=home, project_root=tmp_path)

    # The dry run saw the union (candidate + enabled peer) and never ran a real install.
    assert [sorted(c) for c in calls] == [["tabulate<0.9", "tabulate>=0.9"]]
    assert (home / "plugins" / "good").is_dir()
    assert "good" in (home / "config.yaml").read_text()


def test_reapply_disables_non_memory_plugin_loudly_and_keeps_memory_provider(tmp_path, monkeypatch):
    home = _home(tmp_path, ["memory", "weather", "sidecar", "aaa-innocent"])
    _plugin(home, "memory", '"mnemosyne-memory>=3"', memory=True)
    _plugin(home, "weather", '"httpx<0.20"')
    _plugin(home, "sidecar", '"torch==99"', external=True)
    _plugin(home, "aaa-innocent", '"tabulate>=0.9"')  # sorts before the culprit; must NOT be sacrificed
    resolve, calls = _fake_resolver({"httpx<0.20"})
    monkeypatch.setattr(deps, "resolve", resolve)
    monkeypatch.setattr(deps, "core_constraints", lambda root: ["httpx==0.28.1"])
    monkeypatch.setattr(deps, "dependency_homes", lambda: [home])
    disabled: list[tuple[Path, str]] = []

    report = deps.reapply_all(project_root=tmp_path, disable=lambda h, n: disabled.append((h, n)))

    assert disabled == [(home, "weather")]
    assert report.dropped and report.dropped[0][0] == "weather"
    assert sorted(report.installed) == ["mnemosyne-memory>=3", "tabulate>=0.9"]
    # External-runtime plugin never joined the union; memory provider was the survivor.
    assert all("torch==99" not in c for c in calls)
