"""pm.workspace: the generated uv-workspace root for plugin deps.

The workspace root is a pm-GENERATED project (never the committed
pyproject.toml — sealed installs are read-only and member lists are
machine-specific). Its pyproject = core's pyproject verbatim +
``[tool.uv.workspace] members`` pointing at each enabled plugin dir via
relative ``../``-escaping paths (proven to resolve). ``uv lock`` unions
core + plugin deps into ONE lock; conflict = loud refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pm.workspace as ws


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A fake install: core repo with pyproject, plugin dirs, store."""
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        "[project]\n"
        'name = "hermes-agent"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = ["httpx==0.28.1"]\n',
        encoding="utf-8",
    )
    plugins = tmp_path / "home" / "plugins"
    plug_a = plugins / "plug-a"
    plug_a.mkdir(parents=True)
    (plug_a / "plugin.yaml").write_text("name: plug-a\n", encoding="utf-8")
    (plug_a / "pyproject.toml").write_text(
        "[project]\nname = \"plug-a\"\nversion = \"0.1.0\"\n"
        'requires-python = ">=3.11"\ndependencies = ["rich==13.9.4"]\n',
        encoding="utf-8",
    )
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    monkeypatch.setattr(ws.paths, "store_root", lambda: store)
    return tmp_path, core, plug_a, store


def test_workspace_root_lives_in_the_store(layout):
    _, _, _, store = layout
    assert ws.workspace_root() == store / ".pm-workspace"
    assert ws.workspace_root() == ws.workspace_root()  # stable


def test_build_writes_core_pyproject_verbatim(layout):
    _, core, plug_a, _ = layout
    root = ws.build_root([plug_a])
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    core_text = (core / "pyproject.toml").read_text(encoding="utf-8")
    # core's project table is carried verbatim (name, deps, requires-python)
    assert 'name = "hermes-agent"' in text
    assert 'dependencies = ["httpx==0.28.1"]' in text
    assert 'requires-python = ">=3.11"' in text
    # nothing else was invented
    for line in core_text.strip().splitlines():
        assert line in text


def test_members_are_relative_paths_escaping_the_root(layout):
    _, _, plug_a, _ = layout
    root = ws.build_root([plug_a])
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.uv.workspace]" in text
    # member must be the relative path from the root to the plugin dir
    expected = ws._member_rel(root, plug_a)
    assert f'"{expected}"' in text
    assert expected.startswith(".."), "member must escape the generated root"


def test_build_is_idempotent(layout):
    _, _, plug_a, _ = layout
    ws.build_root([plug_a])
    first = (ws.workspace_root() / "pyproject.toml").read_text(encoding="utf-8")
    ws.build_root([plug_a])
    second = (ws.workspace_root() / "pyproject.toml").read_text(encoding="utf-8")
    assert first == second


def test_zero_plugins_still_builds_a_root_with_no_members(layout):
    _, _, _, _ = layout
    root = ws.build_root([])
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "hermes-agent"' in text
    assert "[tool.uv.workspace]" not in text or "members = []" in text


def test_resolve_union_passes_through_on_clean_resolve(monkeypatch):
    a, b = Path("/x/a"), Path("/x/b")
    calls = []
    monkeypatch.setattr(
        ws, "lock_and_sync", lambda members, extras=None, **k: calls.append(members)
    )
    survivors, decisions = ws.resolve_union([a, b], ["web"])
    assert survivors == [a, b]
    assert decisions == []
    assert calls == [[a, b]]


def test_resolve_union_disables_fail_alone_plugins(monkeypatch):
    a, b, bad = Path("/x/a"), Path("/x/b"), Path("/x/bad")
    from pm.package import InstallError

    def fake_lock(members, extras=None, **k):
        if any(m == bad for m in members):
            raise InstallError("venv", "bad conflicts with core pin")
        return None

    monkeypatch.setattr(ws, "lock_and_sync", fake_lock)
    survivors, decisions = ws.resolve_union([a, bad, b])
    assert bad not in survivors
    assert [d["plugin"] for d in decisions] == ["bad"]
    assert "core pin" in decisions[0]["reason"]


def test_resolve_union_incumbent_wins_on_mutual_conflict(monkeypatch):
    old, new = Path("/x/old-plug"), Path("/x/new-plug")
    from pm.package import InstallError

    def fake_lock(members, extras=None, **k):
        if old in members and new in members:
            raise InstallError("venv", "old-plug and new-plug are incompatible")
        return None

    monkeypatch.setattr(ws, "lock_and_sync", fake_lock)
    # newest-enabled LAST (the tiebreak contract)
    survivors, decisions = ws.resolve_union([old, new])
    assert survivors == [old]  # incumbent survives
    assert [d["plugin"] for d in decisions] == ["new-plug"]
    assert "incompatible" in decisions[0]["reason"]


def test_member_stamp_hash_changes_with_plugin_set(layout):
    _, _, plug_a, _ = layout
    stamp_empty = ws.members_stamp([])
    stamp_a = ws.members_stamp([plug_a])
    stamp_b = ws.members_stamp([plug_a, plug_a])  # dedupes to same set
    assert stamp_empty != stamp_a
    assert stamp_a == stamp_b


def test_member_stamp_changes_when_pyproject_content_changes(tmp_path):
    """Task 4 contract: a pulled plugin with changed pins must move the
    stamp — path-only hashing left dep bumps invisible to the venv sync."""
    plug = tmp_path / "plug"
    plug.mkdir()
    (plug / "pyproject.toml").write_text(
        'dependencies = ["pkg==1.0.0"]\n', encoding="utf-8"
    )
    before = ws.members_stamp([plug])
    # the plugin update: same dir, new pins
    (plug / "pyproject.toml").write_text(
        'dependencies = ["pkg==2.0.0"]\n', encoding="utf-8"
    )
    after = ws.members_stamp([plug])
    assert before != after
    # no pyproject at all: still hashable (the dir identity carries it)
    bare = tmp_path / "bare"
    bare.mkdir()
    assert ws.members_stamp([bare]) != before


def test_member_stamp_missing_pyproject_does_not_crash(tmp_path):
    """A member dir whose pyproject vanished mid-scan hashes on path only."""
    plug = tmp_path / "ghost"
    plug.mkdir()
    (plug / "pyproject.toml").write_text("x\n", encoding="utf-8")
    first = ws.members_stamp([plug])
    (plug / "pyproject.toml").unlink()
    second = ws.members_stamp([plug])
    assert first != second  # content term dropped out, stamp moved


def test_enabled_member_dirs_finds_enabled_dep_plugins(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    plugins.mkdir(parents=True)

    # modern plugin: pyproject.toml
    modern = plugins / "modern-plug"
    modern.mkdir()
    (modern / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    # legacy plugin: pip_dependencies in plugin.yaml, no pyproject
    legacy = plugins / "legacy-plug"
    legacy.mkdir()
    (legacy / "plugin.yaml").write_text(
        "name: legacy-plug\npip_dependencies:\n  - \"requests>=2\"\n",
        encoding="utf-8",
    )

    # dep-less plugin: neither — not a member even when enabled
    plain = plugins / "plain-plug"
    plain.mkdir()
    (plain / "plugin.yaml").write_text("name: plain-plug\n", encoding="utf-8")

    # dep-carrying but NOT-ENABLED plugin — must not join the union
    orphan = plugins / "orphan-plug"
    orphan.mkdir()
    (orphan / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    monkeypatch.setattr(ws, "_plugin_dir_roots", lambda: {plugins})
    # enabled order = enable recency (legacy enabled first/older, modern
    # newest LAST) — order must carry through for the bisect tiebreak.
    monkeypatch.setattr(
        "pm.plugins_state.enabled_plugins_ordered",
        lambda: {plugins: ["legacy-plug", "modern-plug", "plain-plug"]},
    )
    found = ws.enabled_member_dirs()
    names = [p.name for p in found]
    assert names == ["legacy-plug", "modern-plug"]
    assert "orphan-plug" not in names


def test_enabled_member_dirs_empty_when_nothing_enabled(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    plugins.mkdir(parents=True)
    member = plugins / "member"
    member.mkdir()
    (member / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    monkeypatch.setattr(ws, "_plugin_dir_roots", lambda: {plugins})
    monkeypatch.setattr(
        "pm.plugins_state.enabled_plugins_ordered", lambda: {}
    )
    assert ws.enabled_member_dirs() == []


def test_record_disabled_plugins_writes_back(monkeypatch):
    calls = []

    def fake_disable(names):
        calls.append(names)

    monkeypatch.setattr("pm.plugins_state.disable_plugins", fake_disable)
    removed = ws.record_disabled_plugins(
        [
            {"plugin": "bad", "action": "disabled", "reason": "conflict"},
            {"plugin": "good", "action": "kept", "reason": ""},
        ]
    )
    assert removed == ["bad"]
    assert calls == [["bad"]]


@pytest.fixture
def lazy_on(monkeypatch):
    import sys

    ensure_mod = sys.modules["pm.ensure"]
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: True)


def test_materialize_legacy_pyproject_from_pip_dependencies(tmp_path, lazy_on):
    plug = tmp_path / "legacy-plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: legacy-plug\n"
        "version: 1.0.0\n"
        "pip_dependencies:\n"
        "  - \"mcp>=2,<3\"\n"
        "  - 'click==8.1.7'\n",
        encoding="utf-8",
    )

    generated = ws.materialize_legacy_pyproject(plug)
    assert generated is not None and generated.is_file()
    text = generated.read_text(encoding="utf-8")
    assert "GENERATED by pm" in text
    assert 'name = "legacy-plug"' in text
    assert "'mcp>=2,<3'" in text
    assert "'click==8.1.7'" in text

    # idempotent: regenerating with same specs is byte-identical
    again = ws.materialize_legacy_pyproject(plug)
    assert again.read_text(encoding="utf-8") == text

    # spec change rewrites
    (plug / "plugin.yaml").write_text(
        "name: legacy-plug\npip_dependencies:\n  - \"requests>=2\"\n",
        encoding="utf-8",
    )
    ws.materialize_legacy_pyproject(plug)
    assert "'requests>=2'" in generated.read_text(encoding="utf-8")


def test_materialize_skips_modern_and_depless_plugins(tmp_path):
    # modern plugin: has its own pyproject — untouched
    modern = tmp_path / "modern"
    modern.mkdir()
    (modern / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (modern / "plugin.yaml").write_text(
        "name: modern\npython_dependencies:\n  - \"x\"\n", encoding="utf-8"
    )
    assert ws.materialize_legacy_pyproject(modern) is None
    assert '[project]' == (modern / "pyproject.toml").read_text(encoding="utf-8").strip()

    # dep-less plugin: nothing to bridge
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "plugin.yaml").write_text("name: plain\n", encoding="utf-8")
    assert ws.materialize_legacy_pyproject(plain) is None

    # no manifest at all
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ws.materialize_legacy_pyproject(empty) is None


def test_scan_plugin_classifies_dep_surfaces(tmp_path):
    full = tmp_path / "full-plug"
    full.mkdir()
    for name in ("pyproject.toml", "package.json", "packages.py", "plugin.yaml"):
        (full / name).write_text("x\n", encoding="utf-8")
    scan = ws.scan_plugin(full)
    assert scan["pyproject"] and scan["package_json"] and scan["packages_py"]
    assert not scan["legacy_deps"]

    legacy = tmp_path / "legacy-plug"
    legacy.mkdir()
    (legacy / "plugin.yaml").write_text(
        "name: legacy\npip_dependencies:\n  - \"x>=1\"\n", encoding="utf-8"
    )
    scan = ws.scan_plugin(legacy)
    assert scan["legacy_deps"] and not scan["pyproject"]


def test_enabled_member_dirs_survives_unreadable_roots(tmp_path, monkeypatch):
    # an OSError mid-scan (dangling junction etc.) must not lose other roots
    good = tmp_path / "good-plugins"
    good.mkdir()
    member = good / "member"
    member.mkdir()
    (member / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    class _Broken:
        def is_dir(self):
            raise OSError("dangling junction")

    monkeypatch.setattr(ws, "_plugin_dir_roots", lambda: {good, _Broken()})
    monkeypatch.setattr(
        "pm.plugins_state.enabled_plugins_ordered",
        lambda: {good: ["member"]},
    )
    found = ws.enabled_member_dirs()
    assert [p.name for p in found] == ["member"]


def test_materialize_never_when_lazy_off(tmp_path, monkeypatch):
    import sys

    ensure_mod = sys.modules["pm.ensure"]
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: False)
    plug = tmp_path / "legacy-plug"
    plug.mkdir()
    (plug / "plugin.yaml").write_text(
        "name: legacy-plug\npip_dependencies:\n  - \"requests>=2\"\n",
        encoding="utf-8",
    )
    assert ws.materialize_legacy_pyproject(plug) is None
    assert not (plug / "pyproject.toml").exists()
