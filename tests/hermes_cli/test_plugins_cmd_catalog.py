"""Catalog-aware ``hermes plugins`` surface (hermes_cli/plugins_cmd_catalog.py): a bare catalog name installs
the PINNED sha and records provenance; the kill list blocks every install path (CLI needs an explicit
bypass, dashboard/TUI have none); ``update`` re-pins instead of pulling. Real git, file:// repos."""

from __future__ import annotations

import json
import os
import shutil
import subprocess as sp
from pathlib import Path

import pytest

from hermes_cli import plugin_catalog as pc_cat
from hermes_cli import plugins_cmd as pc
from hermes_cli import plugins_cmd_catalog as cat
from pm.filesystem import is_junction
from tests.pm._fixtures import client, isolated_python  # noqa: F401

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

_GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def _commit(repo: Path, msg: str) -> str:
    sp.run(["git", "add", "-A"], cwd=repo, check=True, env=_GIT_ENV)
    sp.run(["git", "commit", "-q", "-m", msg], cwd=repo, check=True, env=_GIT_ENV)
    return sp.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def world(client, tmp_path, monkeypatch):
    """A file:// plugin repo with two commits, a catalog pinned to the FIRST, an isolated plugins dir."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "plugin.yaml").write_text("name: cat-plugin\nversion: 1.0.0\ndescription: d\n")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n")
    sp.run(["git", "init", "-q"], cwd=repo, check=True, env=_GIT_ENV)
    sha1 = _commit(repo, "v1")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass  # v2\n")
    sha2 = _commit(repo, "v2")

    home = tmp_path / "home"
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: False)
    monkeypatch.setattr(pc, "_console", lambda: type("C", (), {"print": lambda *a, **k: None})())

    def publish_without_environment(*_args, plugins=None, **_kwargs):
        from hermes_cli.runtime_state import finish_publication
        from pm import paths
        from pm.plugin_inputs import Selection, StagedUpdate
        from pm.publication import PluginSelection, StagedPlugin

        if isinstance(plugins, StagedUpdate):
            change = StagedPlugin(dict(plugins.data))
        else:
            assert isinstance(plugins, Selection)
            change = PluginSelection(dict(plugins.data))
        change.publish(paths.repo_root())
        finish_publication(paths.repo_root())

    # Catalog behavior is independent of dependency-environment construction.
    monkeypatch.setattr("pm.client.sync_venv", publish_without_environment)

    # Catalog: one entry pinned to sha1, mutable via state["pin"]; kill list via state["removed"]. The
    # real loader is https-only, so the fixture entry is built directly (file:// repo).
    state = {"pin": sha1, "removed": []}

    def _entries():
        return [pc_cat.PluginCatalogEntry(name="cat-plugin", repo=repo.as_uri(), sha=state["pin"],
                                          description="d", maintainer="t")]

    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: _entries())
    monkeypatch.setattr(pc_cat, "fetch_live_catalog", lambda **_: None)  # in-tree only, no network
    monkeypatch.setattr(pc_cat, "load_removed_list", lambda catalog_dir=None: list(state["removed"]))
    return {"repo": repo, "sha1": sha1, "sha2": sha2, "plugins_dir": plugins_dir, "state": state}


def _head(path: Path) -> str:
    return sp.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True).stdout.strip()


def test_catalog_platform_mismatch_refuses_before_install(world):
    from hermes_platform.host.facts import os_family
    host = os_family()
    other = "linux" if host == "windows" else "windows"
    entry = pc_cat.PluginCatalogEntry(
        name="cat-plugin", repo=world["repo"].as_uri(), sha=world["sha1"],
        description="d", maintainer="t", platforms=[other],
    )

    with pytest.raises(pc.PluginOperationError, match=f"cat-plugin.*{host}.*{other}"):
        cat.install_catalog_entry(entry, force=False)
    assert not (world["plugins_dir"] / "cat-plugin").exists()


def test_catalog_name_installs_pinned_sha_with_sidecar_then_update_repins(world, monkeypatch):
    entry = pc_cat.get_live_catalog_entry("cat-plugin")
    assert entry is not None
    target, _m, name = cat.install_catalog_entry(entry, force=False)
    assert name == "cat-plugin"
    assert _head(target) == world["sha1"] != world["sha2"]  # pinned, not HEAD
    sidecar = cat.catalog_install_record(target)
    assert (sidecar["catalog_name"], sidecar["sha"]) == ("cat-plugin", world["sha1"])
    assert cat.catalog_annotation(target) == f"catalog:community@{world['sha1'][:8]}"

    # Dashboard update on a catalog install = re-pin. Pin unchanged → no-op.
    assert pc.dashboard_update_user_plugin("cat-plugin") == {
        "ok": True, "name": "cat-plugin", "sha": world["sha1"], "unchanged": True,
        "python_dependencies": [], "warnings": []}
    # Bump the catalog pin → the checkout moves to exactly that sha.
    world["state"]["pin"] = world["sha2"]
    assert pc.dashboard_update_user_plugin("cat-plugin")["unchanged"] is False
    assert _head(world["plugins_dir"] / "cat-plugin") == world["sha2"]


def test_kill_list_blocks_cli_dashboard_and_tui_paths(world, monkeypatch):
    world["state"]["removed"].append(
        pc_cat.RemovedEntry(name="cat-plugin", repo=world["repo"].as_uri(), reason="malware"))
    # Dashboard/TUI: catalog name AND raw repo URL both refused, no bypass parameter exists.
    assert "malware" in pc.dashboard_install_plugin("", force=False, enable=False, catalog_name="cat-plugin")["error"]
    assert "malware" in pc.dashboard_install_plugin(world["repo"].as_uri(), force=False, enable=False)["error"]
    assert not (world["plugins_dir"] / "cat-plugin").exists()
    # CLI: refused by default, `--allow-removed` installs anyway.
    with pytest.raises(SystemExit):
        pc.cmd_install("cat-plugin", enable=False)
    pc.cmd_install("cat-plugin", enable=False, allow_removed=True)
    assert cat.catalog_install_record(world["plugins_dir"] / "cat-plugin") is not None
    assert cat.removed_annotation("cat-plugin", world["plugins_dir"] / "cat-plugin",
                                  cat.resolved_removed_entries()) == "malware"




def test_owner_repo_hash_subdir_shorthand_resolves_like_the_catalog_spelling():
    from hermes_cli.plugins_cmd import _resolve_git_url
    assert _resolve_git_url("plastic-labs/honcho#hermes-plugin-honcho") == (
        "https://github.com/plastic-labs/honcho.git", "hermes-plugin-honcho")
    assert _resolve_git_url("owner/repo") == ("https://github.com/owner/repo.git", None)


def _install_url(repo: Path, name: str, files: dict) -> Path:
    """A URL (non-catalog) install of a fresh repo carrying *files*."""
    repo.mkdir()
    (repo / "plugin.yaml").write_text(f"name: {name}\nversion: 1.0.0\ndescription: d\n")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n")
    for rel, text in files.items():
        (repo / rel).write_text(text)
    sp.run(["git", "init", "-q"], cwd=repo, check=True, env=_GIT_ENV)
    _commit(repo, "v1")
    return pc._install_plugin_core(repo.as_uri(), force=False)[0]


def test_in_tree_sidecar_cannot_forge_catalog_provenance(world, tmp_path):
    """Provenance is the installer's metadata record, never a file the repo ships: a URL install carrying
    its own .hermes-catalog.json is not a catalog install and does not mark the real entry installed."""
    forged = json.dumps({"catalog_name": "cat-plugin", "tier": "official", "sha": "0" * 40, "repo": "x"})
    target = _install_url(tmp_path / "evil", "evil-plugin", {cat.CATALOG_SIDECAR: forged})
    assert (target / cat.CATALOG_SIDECAR).exists()  # the file is there, and inert
    assert cat.read_catalog_sidecar(target) is None
    assert cat.catalog_annotation(target) is None
    assert cat.catalog_row_fields(target, cat.catalog_pins()) == {}
    state = cat.installed_catalog_state({"evil-plugin": {"dir": str(target), "runtime_status": None}})
    assert state["entries"][0]["installed"] is False
    # Control: a real catalog install is still recognised through the metadata record.
    real, _m, _n = cat.install_catalog_entry(pc_cat.get_live_catalog_entry("cat-plugin"), force=False)
    assert cat.catalog_annotation(real) == f"catalog:community@{world['sha1'][:8]}"


def test_ref_install_records_installed_sha_so_update_is_offered(world):
    """`install NAME --ref X` checks out X, not the reviewed pin; provenance must say X so list/TUI flag the
    drift and `update` re-pins instead of answering 'already at catalog pin'."""
    entry = pc_cat.get_live_catalog_entry("cat-plugin")
    target, _m, _n = cat.install_catalog_entry(entry, force=False, ref=world["sha2"])
    assert _head(target) == world["sha2"]
    assert cat.read_catalog_sidecar(target)["sha"] == world["sha2"]
    assert cat.catalog_row_fields(target, cat.catalog_pins())["update_available"] is True
    result = pc.dashboard_update_user_plugin("cat-plugin")
    assert result["unchanged"] is False and _head(target) == world["sha1"]


def test_repin_keeps_local_files_backs_up_edits_and_follows_manifest_rename(world, monkeypatch):
    entry = pc_cat.get_live_catalog_entry("cat-plugin")
    target, _m, _n = cat.install_catalog_entry(entry, force=False)
    (target / "config.yaml").write_text("api_key: real\n")                      # installer/user data
    (target / "__init__.py").write_text("def register(ctx):\n    pass  # mine\n")  # tracked edit
    pc._write_config_value("plugins", "enabled", ["cat-plugin"])
    # New pin renames the manifest.
    repo = world["repo"]
    (repo / "plugin.yaml").write_text("name: cat-plugin-v2\nversion: 2.0.0\ndescription: d\n")
    world["state"]["pin"] = _commit(repo, "rename")
    result = pc.dashboard_update_user_plugin("cat-plugin")
    new_target = world["plugins_dir"] / "cat-plugin-v2"
    assert result["ok"] and result["name"] == "cat-plugin-v2" and _head(new_target) == world["state"]["pin"]
    assert (new_target / "config.yaml").read_text() == "api_key: real\n"
    assert not target.exists() and "cat-plugin" not in pc._read_install_metadata()
    assert pc._get_enabled_set() == {"cat-plugin-v2"}
    backups = list((world["plugins_dir"].parent / "plugins-backup").glob("cat-plugin-*/__init__.py"))
    assert backups and "# mine" in backups[0].read_text()
    assert any("plugins-backup" in w for w in result["warnings"]) and any("renamed" in w for w in result["warnings"])


def test_carry_user_files_without_git_preserves_data_but_not_old_code(tmp_path):
    """No-git fallback keeps user state without resurrecting removed executable/control surfaces."""
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()

    (old / "config.yaml").write_text("endpoint: mine\n")
    (old / "data").mkdir()
    (old / "data" / "state.db").write_text("user data")
    (old / "legacy.py").write_text("OLD = True\n")
    (old / ".git").write_text("gitdir: /tmp/foreign-worktree\\n")
    (old / "desktop").mkdir()
    (old / "desktop" / "plugin.js").write_text('export default { id: "stale" }\n')
    (old / "skills").mkdir()
    (old / "skills" / "stale").mkdir()
    (old / "skills" / "stale" / "SKILL.md").write_text("# stale\n")
    (old / "mcp.json").write_text('{"mcpServers":{"stale":{"type":"stdio","command":"./stale"}}}\n')
    (old / "pyproject.toml").write_text('[project]\nname="stale"\nversion="1"\n')
    (old / "package.json").write_text('{"name":"stale"}\n')

    cat._carry_user_files(old, new, None)

    assert (new / "config.yaml").read_text() == "endpoint: mine\n"
    assert (new / "data" / "state.db").read_text() == "user data"
    assert not (new / "legacy.py").exists()
    assert not (new / ".git").exists()
    assert not (new / "desktop").exists()
    assert not (new / "skills").exists()
    assert not (new / "mcp.json").exists()
    assert not (new / "pyproject.toml").exists()
    assert not (new / "package.json").exists()


@pytest.mark.parametrize("shape", ["old-file-new-dir", "old-dir-new-file"])
def test_carry_user_files_fails_closed_on_type_clashes(tmp_path, shape):
    """An update never drops user state just because the new revision changed a path's type."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    if shape == "old-file-new-dir":
        (old / "data").write_text("user data")
        (new / "data").mkdir()
    else:
        (old / "data" / "db").mkdir(parents=True)
        (old / "data" / "db" / "index.db").write_text("user data")
        (new / "data").write_text("new upstream file")

    with pytest.raises(pc.PluginOperationError, match="Cannot preserve user file"):
        cat._carry_user_files(old, new, None)

    if shape == "old-file-new-dir":
        assert (old / "data").read_text() == "user data"
        assert (new / "data").is_dir()
    else:
        assert (old / "data" / "db" / "index.db").read_text() == "user data"
        assert (new / "data").read_text() == "new upstream file"


def test_carry_user_files_fails_closed_on_staged_symlink_parent(tmp_path):
    """A staged symlink cannot redirect carried user data outside the replacement transaction."""
    old, new, outside = tmp_path / "old", tmp_path / "new", tmp_path / "outside"
    (old / "data").mkdir(parents=True)
    new.mkdir()
    outside.mkdir()
    (old / "data" / "index.db").write_text("user data")
    try:
        (new / "data").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(pc.PluginOperationError, match="Cannot preserve user file"):
        cat._carry_user_files(old, new, None)

    assert not (outside / "index.db").exists()
    assert (old / "data" / "index.db").read_text() == "user data"


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory modes")
def test_carry_user_files_preserves_mode_of_created_directories(tmp_path):
    """Directories created solely for carried state keep the source directory's restrictive mode."""
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "data").mkdir(parents=True, mode=0o700)
    (old / "data").chmod(0o700)
    new.mkdir()
    (old / "data" / "state.db").write_text("user data")

    previous_umask = os.umask(0o022)
    try:
        cat._carry_user_files(old, new, None)
    finally:
        os.umask(previous_umask)

    assert (new / "data" / "state.db").read_text() == "user data"
    assert (new / "data").stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs unavailable on this platform")
def test_carry_user_files_skips_runtime_special_files(tmp_path):
    """Runtime pipes are transient state and must not make an otherwise valid update fail."""
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "data").mkdir(parents=True)
    new.mkdir()
    (old / "data" / "state.db").write_text("user data")
    os.mkfifo(old / "data" / "events.fifo")

    cat._carry_user_files(old, new, None)

    assert (new / "data" / "state.db").read_text() == "user data"
    assert not os.path.lexists(new / "data" / "events.fifo")


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-only")
def test_carry_user_files_does_not_follow_source_junction(tmp_path):
    """A source junction cannot pull files from outside the installed plugin into an update."""
    old, new, outside = tmp_path / "old", tmp_path / "new", tmp_path / "outside"
    old.mkdir()
    new.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside")
    junction = old / "data"
    sp.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], check=True, capture_output=True, text=True)
    assert is_junction(junction)

    cat._carry_user_files(old, new, None)

    assert not os.path.lexists(new / "data")
    assert not (new / "secret.txt").exists()


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-only")
def test_carry_user_files_fails_closed_on_staged_junction_parent(tmp_path):
    """A staged junction cannot redirect carried user data outside the replacement tree."""
    old, new, outside = tmp_path / "old", tmp_path / "new", tmp_path / "outside"
    (old / "data").mkdir(parents=True)
    new.mkdir()
    outside.mkdir()
    (old / "data" / "index.db").write_text("user data")
    junction = new / "data"
    sp.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], check=True, capture_output=True, text=True)
    assert is_junction(junction)

    with pytest.raises(pc.PluginOperationError, match="Cannot preserve user file"):
        cat._carry_user_files(old, new, None)

    assert not (outside / "index.db").exists()
    assert (old / "data" / "index.db").read_text() == "user data"


def test_url_subdir_reclone_revalidates_carried_code_before_publication(world, tmp_path, monkeypatch):
    """The URL subdir path must run both admission gates again after carrying user state."""
    from hermes_cli import plugins_cmd_install as install_cmd

    mono = tmp_path / "mono-rescan"
    src = mono / "plugins" / "sub-plugin"
    src.mkdir(parents=True)
    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 1.0.0\ndescription: d\n")
    (src / "__init__.py").write_text("def register(ctx):\n    pass\n")
    (src / "mcp.json").write_text(
        '{"mcpServers":{"demo":{"type":"stdio","command":"${PLUGIN_ROOT}/server.js"}}}\n'
    )
    (src / "server.js").write_text('console.log("old revision")\n')
    sp.run(["git", "init", "-q"], cwd=mono, check=True, env=_GIT_ENV)
    _commit(mono, "v1")

    target = pc._install_plugin_core(f"{mono.as_uri()}#plugins/sub-plugin", force=False)[0]
    assert (target / "server.js").is_file()
    (src / "server.js").unlink()
    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 2.0.0\ndescription: d\n")
    _commit(mono, "v2")

    scans, portable_checks = [], []

    def scan_gate(tree, *_args, **_kwargs):
        scans.append((Path(tree) / "server.js").exists())

    def portable_gate(_plugin_name, tree):
        has_server = (Path(tree) / "server.js").exists()
        portable_checks.append(has_server)
        if has_server:
            raise pc.PluginOperationError("carried server.js failed final admission")

    monkeypatch.setattr(pc, "_scan_plugin_tree", scan_gate)
    monkeypatch.setattr(install_cmd, "_refuse_unavailable_portable_plugin", portable_gate)
    result = pc.dashboard_update_user_plugin("sub-plugin")

    assert result["ok"] is False
    assert "carried server.js failed final admission" in result["error"]
    assert scans == [False, True]
    assert portable_checks == [False, True]
    assert (target / "server.js").is_file()
    assert "version: 1.0.0" in (target / "plugin.yaml").read_text()


def test_carry_user_files_fails_closed_when_source_tree_cannot_be_walked(tmp_path, monkeypatch):
    """Unreadable user state aborts replacement instead of being silently omitted."""
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()

    def denied_walk(_path, *, onerror=None, **_kwargs):
        assert onerror is not None
        onerror(PermissionError("denied"))
        return ()

    monkeypatch.setattr(cat.os, "walk", denied_walk)
    with pytest.raises(pc.PluginOperationError, match="Could not preserve user files.*denied"):
        cat._carry_user_files(old, new, None)


def test_git_checkout_update_fails_closed_when_local_changes_cannot_be_inspected(world, monkeypatch):
    """A destructive re-pin must not guess ownership when a real git checkout cannot be inspected."""
    target = cat.install_catalog_entry(pc_cat.get_live_catalog_entry("cat-plugin"), force=False)[0]
    assert (target / ".git").exists()
    monkeypatch.setattr(pc, "_resolve_git_executable", lambda: None)
    world["state"]["pin"] = world["sha2"]

    with pytest.raises(pc.PluginOperationError, match="git executable is unavailable"):
        cat.repin_catalog_plugin(target, cat.read_catalog_sidecar(target))

    assert _head(target) == world["sha1"]


@pytest.mark.parametrize("via", ["url", "catalog"])
def test_update_of_a_subdir_install_keeps_files_the_user_created_or_edited(world, tmp_path, monkeypatch, via):
    """A subdirectory install carries no ``.git``; both update paths must preserve user config/data."""
    mono = tmp_path / "mono"
    src = mono / "plugins" / "sub-plugin"
    src.mkdir(parents=True)
    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 1.0.0\ndescription: d\n")
    (src / "__init__.py").write_text("def register(ctx):\n    pass\n")
    (src / "config.yaml.example").write_text("endpoint: default\n")
    (src / "desktop").mkdir()
    (src / "desktop" / "plugin.js").write_text("export default { id: \"v1\" }\n")
    sp.run(["git", "init", "-q"], cwd=mono, check=True, env=_GIT_ENV)
    pin = {"sha": _commit(mono, "v1")}

    def entry():
        return pc_cat.PluginCatalogEntry(
            name="sub-plugin",
            repo=mono.as_uri(),
            sha=pin["sha"],
            description="d",
            maintainer="t",
            subdir="plugins/sub-plugin",
        )

    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: [entry()])
    if via == "catalog":
        target = cat.install_catalog_entry(entry(), force=False)[0]
    else:
        target = pc._install_plugin_core(f"{mono.as_uri()}#plugins/sub-plugin", force=False)[0]
    assert not (target / ".git").exists()

    (target / "config.yaml").write_text("endpoint: mine\n")
    (target / "data").mkdir()
    (target / "data" / "state.json").write_text("{}")

    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 2.0.0\ndescription: d\n")
    (src / "config.yaml.example").write_text("endpoint: new-default\n")
    shutil.rmtree(src / "desktop")
    pin["sha"] = _commit(mono, "v2")
    assert pc.dashboard_update_user_plugin("sub-plugin")["ok"] is True

    assert "version: 2.0.0" in (target / "plugin.yaml").read_text()
    assert (target / "config.yaml").read_text() == "endpoint: mine\n"
    assert (target / "data" / "state.json").read_text() == "{}"
    assert not (target / "desktop").exists()


def test_repin_keeps_a_wholly_ignored_data_dir_in_a_git_checkout(world):
    """A single ``!! data/`` status entry must preserve every file below that ignored directory."""
    repo = world["repo"]
    (repo / ".gitignore").write_text("data/\n")
    world["state"]["pin"] = _commit(repo, "ignore data")
    target = cat.install_catalog_entry(pc_cat.get_live_catalog_entry("cat-plugin"), force=False)[0]
    assert (target / ".git").exists()

    (target / "data" / "db").mkdir(parents=True)
    (target / "data" / "db" / "index.db").write_text("user data")

    (repo / "__init__.py").write_text("def register(ctx):\n    pass  # v3\n")
    world["state"]["pin"] = _commit(repo, "v3")
    assert pc.dashboard_update_user_plugin("cat-plugin")["unchanged"] is False

    assert _head(target) == world["state"]["pin"]
    assert (target / "data" / "db" / "index.db").read_text() == "user data"


def test_kill_list_covers_update_enable_and_load_of_an_installed_plugin(world, tmp_path, monkeypatch):
    """A URL install whose name lands on the kill list AFTER install must stop pulling, cannot be enabled
    and is refused at load; an install made with --allow-removed keeps working."""
    from hermes_cli.plugins_discovery import gate_manifest
    from hermes_cli.plugins_manifest import PluginManifest
    target = _install_url(tmp_path / "later-killed", "killed", {})
    world["state"]["removed"].append(pc_cat.RemovedEntry(name="killed", reason="backdoor"))
    monkeypatch.setattr(pc_cat, "_live_cache_path", lambda: tmp_path / "no-cache.json")
    assert "backdoor" in pc.dashboard_update_user_plugin("killed")["error"]
    with pytest.raises(SystemExit):
        pc.cmd_enable("killed")
    assert "killed" not in pc._get_enabled_set()
    manifest = PluginManifest(name="killed", source="user", path=str(target), key="killed")
    assert gate_manifest(manifest, set(), {"killed"}).action != "load"
    # Explicit bypass at install time is remembered.
    pc.cmd_install((tmp_path / "later-killed").as_uri(), force=True, enable=False, allow_removed=True)
    assert gate_manifest(manifest, set(), {"killed"}).action == "load"


def test_annotated_tag_pin_keeps_reviewed_trust_and_reads_as_at_pin(world, monkeypatch):
    """A pin recorded as `git rev-parse <tag>` names the TAG object; HEAD can only ever be the commit it
    points at. Trust (scan skips the caution prompt) and the at-pin check must both use the peeled commit,
    or every tag-pinned entry prompts at install and shows 'update available' forever."""
    repo = world["repo"]
    sp.run(["git", "tag", "-a", "v1", world["sha1"], "-m", "v1"], cwd=repo, check=True, env=_GIT_ENV)
    tag_obj = sp.run(["git", "rev-parse", "v1"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    assert tag_obj != world["sha1"]
    world["state"]["pin"] = tag_obj
    seen = {}
    real_scan = pc._scan_plugin_tree
    monkeypatch.setattr(pc, "_scan_plugin_tree", lambda *a, **k: seen.update(k) or real_scan(*a, **k))
    entry = pc_cat.get_live_catalog_entry("cat-plugin")
    target, _m, _n = cat.install_catalog_entry(entry, force=False)
    assert _head(target) == world["sha1"] and seen["reviewed_pin"] is True
    sidecar = cat.read_catalog_sidecar(target)
    assert (sidecar["sha"], sidecar["pin"]) == (world["sha1"], tag_obj)
    assert cat.catalog_row_fields(target, cat.catalog_pins())["update_available"] is False
    assert cat.installed_catalog_state({"cat-plugin": {"dir": str(target), "runtime_status": None}})["entries"][0]["update_available"] is False
    assert pc.dashboard_update_user_plugin("cat-plugin")["unchanged"] is True
    # A bump to a commit sha is still an update.
    world["state"]["pin"] = world["sha2"]
    assert cat.catalog_row_fields(target, cat.catalog_pins())["update_available"] is True


def test_repin_that_widens_the_plugin_requires_consent_on_every_surface(world, monkeypatch):
    """A new pin adding tools / a Desktop half is a new grant: the dashboard/TUI answer consent_required
    with the delta and touch nothing; a retry with consent applies it; the CLI asks y/N and a decline
    leaves the tree at the old pin. A pin that widens nothing (sha2) needs no consent."""
    entry = pc_cat.get_live_catalog_entry("cat-plugin")
    target, _m, _n = cat.install_catalog_entry(entry, force=False)
    repo = world["repo"]
    (repo / "plugin.yaml").write_text("name: cat-plugin\nversion: 3.0.0\ndescription: d\nprovides_tools: [shell_out]\n")
    (repo / "desktop").mkdir()
    (repo / "desktop" / "plugin.js").write_text("export default {}\n")
    world["state"]["pin"] = wide = _commit(repo, "widen")
    result = pc.dashboard_update_user_plugin("cat-plugin")
    assert result["ok"] is False and result["consent_required"] is True
    assert result["delta"] == {"tools": ["shell_out"], "desktop": ["desktop/plugin.js"]}
    assert _head(target) == world["sha1"]  # nothing moved
    # CLI: non-interactive (or 'n') → refused, tree untouched.
    monkeypatch.setattr(pc, "_is_tty", lambda: True)
    monkeypatch.setattr(pc, "_ask_yes", lambda *a, **k: False)
    with pytest.raises(SystemExit):
        pc.cmd_update("cat-plugin")
    assert _head(target) == world["sha1"]
    # Consent given → applied.
    assert pc.dashboard_update_user_plugin("cat-plugin", accept_capabilities=True)["unchanged"] is False
    assert _head(target) == wide
