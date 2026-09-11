"""pm.workspace: the generated uv-workspace root for plugin deps.

The workspace root is a pm-GENERATED project (never the committed
pyproject.toml — sealed installs are read-only and member lists are
machine-specific). Its pyproject = core's pyproject verbatim +
``[tool.uv.workspace] members`` pointing at each enabled plugin dir via
relative ``../``-escaping paths (proven to resolve). ``uv lock`` unions
core + plugin deps into ONE lock; conflict = loud refusal.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import pm.workspace as ws


@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


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


def test_workspace_root_is_per_install_not_in_the_store(layout):
    from hermes_cli.runtime_paths import install_state_dir
    _, core, _, store = layout
    assert ws.workspace_root() == install_state_dir(core) / ".pm-workspace"
    assert not ws.workspace_root().is_relative_to(store)


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


def test_members_keep_their_source_with_the_generation(layout):
    import tomllib

    _, _, plug_a, _ = layout
    root = ws.build_root([plug_a])
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    [relative] = document["tool"]["uv"]["workspace"]["members"]
    copied = root / relative / "pyproject.toml"
    assert copied.resolve().is_relative_to(root.resolve())
    before = copied.read_bytes()
    assert before == (plug_a / "pyproject.toml").read_bytes()
    (plug_a / "pyproject.toml").write_bytes(b"changed after publication")
    assert copied.read_bytes() == before


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

    monkeypatch.setattr(
        "pm.plugins_state.enabled_plugins_ordered", lambda: {}
    )
    assert ws.enabled_member_dirs() == []


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


def test_enabled_member_dirs_ignores_non_profile_entries(tmp_path, monkeypatch):
    import pm.plugins_state as pstate

    home = tmp_path / "home"
    profiles = home / "profiles"
    profiles.mkdir(parents=True)
    member = profiles / "work" / "plugins" / "member"
    member.mkdir(parents=True)
    (member / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (profiles / "work" / "config.yaml").write_text(
        "plugins:\n  enabled: [member]\n", encoding="utf-8",
    )
    (profiles / "README.txt").write_text("not a profile", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.runtime_paths.dependency_home_root", lambda: home)

    assert pstate._all_homes() == [home, profiles / "work"]
    assert ws.enabled_member_dirs() == [member]


# --- classified failures + staging surface (FINAL-RUNTIME-CONTRACT) ---


def test_classify_resolver_conflict_is_resolutionconflict():
    from pm.package import InstallError
    from pm.workspace import ResolutionConflict, classify_uv_failure

    err = classify_uv_failure(
        "lock", 1,
        "  x No solution found for `hermes-agent>=0.1.0` because only the "
        "following versions are available:\n",
    )
    assert isinstance(err, ResolutionConflict)
    assert isinstance(err, InstallError)
    assert "no solution found" in str(err).lower()


def test_classify_network_failure_stays_generic():
    from pm.workspace import ResolutionConflict, classify_uv_failure

    err = classify_uv_failure("lock", 1, "error: Failed to fetch https://pypi.org (timed out)")
    assert not isinstance(err, ResolutionConflict)


def test_sync_failure_is_never_a_conflict(tmp_path, monkeypatch):
    """--frozen: the lock already resolved, so a sync failure (download,
    build, tooling) must stay generic — it must not disable plugins."""
    from pm.package import InstallError
    from pm.workspace import ResolutionConflict

    class FakeProc:
        returncode = 1
        stderr = "error: Failed to download wheel (connection reset)"
        stdout = ""

    monkeypatch.setattr(ws, "_generate_pyproject", lambda *a, **k: (Path("/x/ws"), False))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path("uv"), Path("pm-python")))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(ws.subprocess, "run", fake_run)
    with pytest.raises(InstallError) as excinfo:
        ws.lock_and_sync([], [], venv_dir=tmp_path / "candidate")
    assert not isinstance(excinfo.value, ResolutionConflict)


def test_staging_root_and_env_are_honored_without_live_mutation(monkeypatch, tmp_path):
    """lock_and_sync must resolve into the PARENT-SUPPLIED staging root +
    venv, pass a COPY of the environment (never mutate the live one), and
    leave the default generated root untouched."""
    staging = tmp_path / "staging-ws"
    staging.mkdir()

    seen = {}

    class FakeProc:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd, cwd=None, env=None, **kwargs):
        seen["cwd"] = cwd
        seen["env"] = env
        return FakeProc()

    monkeypatch.setattr(ws, "_generate_pyproject", lambda *a, **k: (staging, False))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path("uv"), Path("pm-python")))
    monkeypatch.setattr("pm.packages.uv_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(ws.subprocess, "run", fake_run)

    live_key = "PM_WORKSPACE_TEST_SENTINEL"
    os.environ[live_key] = "live"
    try:
        ws.lock_and_sync(
            [], [], venv_dir=tmp_path / "staging-venv", root=staging,
            env={"PATH": "/staged/bin", live_key: "staged"},
        )

        assert Path(seen["cwd"]) == staging
        assert seen["env"][live_key] == "staged"          # staged env wins
        assert seen["env"]["UV_CACHE_DIR"] == str(tmp_path / "cache")
        assert seen["env"]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "staging-venv")
        assert seen["env"]["UV_PYTHON"] == "pm-python"
        assert os.environ[live_key] == "live"             # live env untouched
    finally:
        del os.environ[live_key]


def test_changed_root_seeds_from_committed_lock_unchanged_keeps_extended(
    layout, monkeypatch, tmp_path
):
    """Seed the CURRENT resolution without writing shipped bytes: a fresh
    root seeds from the committed lock; an existing root seeds from its
    own extended lock; an explicit parent-supplied seed_lock wins."""
    _, core, plug_a, store = layout
    (core / "uv.lock").write_bytes(b"committed-bytes\n")

    class FakeProc:
        returncode = 0
        stderr = ""
        stdout = ""

    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path("uv"), Path("pm-python")))
    monkeypatch.setattr(ws.subprocess, "run", lambda cmd, **k: FakeProc())

    root = ws.workspace_root()
    venv = tmp_path / "venv"

    # First sync: fresh root -> seeded from the COMMITTED lock.
    ws.lock_and_sync([plug_a], [], venv_dir=venv)
    assert (root / "uv.lock").read_bytes() == b"committed-bytes\n"

    # Changed surface (new member), root's own EXTENDED lock present ->
    # the current extended resolution is the seed, not the committed one.
    (root / "uv.lock").write_bytes(b"extended-bytes\n")
    plug_b = layout[0] / "home" / "plugins" / "plug-b"
    plug_b.mkdir(parents=True)
    (plug_b / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    ws.lock_and_sync([plug_a, plug_b], [], venv_dir=venv)
    assert (root / "uv.lock").read_bytes() == b"extended-bytes\n"

    # Parent-supplied seed_lock wins over both.
    other = tmp_path / "parent-extended.lock"
    other.write_bytes(b"parent-bytes\n")
    plug_c = layout[0] / "home" / "plugins" / "plug-c"
    plug_c.mkdir(parents=True)
    (plug_c / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    ws.lock_and_sync([plug_a, plug_b, plug_c], [], venv_dir=venv, seed_lock=other)
    assert (root / "uv.lock").read_bytes() == b"parent-bytes\n"

    # Shipped core lock untouched throughout.
    assert (core / "uv.lock").read_bytes() == b"committed-bytes\n"
