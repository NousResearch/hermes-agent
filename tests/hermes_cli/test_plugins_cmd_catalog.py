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


@pytest.mark.parametrize("via", ["url", "catalog", "url-old-revision-gone"])
def test_update_of_a_subdir_install_keeps_user_files_and_drops_removed_code(world, tmp_path, monkeypatch, via):
    """A subdirectory install carries no ``.git``, so ``update`` re-installs and swaps the whole tree —
    through the catalog re-pin or the URL re-clone. The user's files survive, code the new version removed
    is not resurrected, and whatever cannot be carried (an edit to a shipped file, a file where the new
    tree has a directory or the reverse) is copied to ``plugins-backup/`` rather than lost — also when
    the installed revision can no longer be fetched to tell user files from the old version's."""
    mono = tmp_path / "mono"
    src = mono / "plugins" / "sub-plugin"
    (src / "utils").mkdir(parents=True)
    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 1.0.0\ndescription: d\n")
    (src / "__init__.py").write_text("def register(ctx):\n    pass\n")
    (src / "config.yaml.example").write_text("endpoint: default\n")
    (src / "settings.py").write_text("LIMIT = 1\n")
    (src / "utils" / "__init__.py").write_text("OLD = True\n")
    sp.run(["git", "init", "-q"], cwd=mono, check=True, env=_GIT_ENV)
    pin = {"sha": _commit(mono, "v1")}

    def entry():
        return pc_cat.PluginCatalogEntry(name="sub-plugin", repo=mono.as_uri(), sha=pin["sha"],
                                         description="d", maintainer="t", subdir="plugins/sub-plugin")

    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: [entry()])
    if via == "catalog":
        target = cat.install_catalog_entry(entry(), force=False)[0]
    else:
        target = pc._install_plugin_core(f"{mono.as_uri()}#plugins/sub-plugin", force=False)[0]
    assert not (target / ".git").exists()
    (target / "config.yaml").write_text("endpoint: mine\n")
    (target / "data").mkdir()
    (target / "data" / "state.json").write_text("{}")
    (target / "settings.py").write_text("LIMIT = 99\n")
    (target / "extras").write_text("mine")
    (target / "cache").mkdir()
    (target / "cache" / "blob").write_text("mine")

    # v2: the utils/ package becomes utils.py; extras and cache ship with the type the user did not use.
    shutil.rmtree(src / "utils")
    (src / "utils.py").write_text("NEW = True\n")
    (src / "extras").mkdir()
    (src / "extras" / "a.txt").write_text("upstream")
    (src / "cache").write_text("upstream")
    (src / "settings.py").write_text("LIMIT = 2\n")
    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 2.0.0\ndescription: d\n")
    (src / "config.yaml.example").write_text("endpoint: new-default\n")
    if via == "url-old-revision-gone":  # upstream rewrote history: the installed revision is unfetchable
        sp.run(["git", "add", "-A"], cwd=mono, check=True, env=_GIT_ENV)
        sp.run(["git", "commit", "-q", "--amend", "-m", "v2"], cwd=mono, check=True, env=_GIT_ENV)
        sp.run(["git", "reflog", "expire", "--expire=now", "--all"], cwd=mono, check=True, env=_GIT_ENV)
        sp.run(["git", "gc", "-q", "--prune=now"], cwd=mono, check=True, env=_GIT_ENV)
    else:
        pin["sha"] = _commit(mono, "v2")
    assert pc.dashboard_update_user_plugin("sub-plugin")["ok"] is True

    assert "version: 2.0.0" in (target / "plugin.yaml").read_text()
    assert (target / "config.yaml").read_text() == "endpoint: mine\n"
    assert (target / "data" / "state.json").read_text() == "{}"
    assert not (target / "utils").exists() and (target / "utils.py").read_text() == "NEW = True\n"
    assert (target / "settings.py").read_text() == "LIMIT = 2\n"
    assert (target / "extras" / "a.txt").read_text() == "upstream"
    assert (target / "cache").read_text() == "upstream"
    backup, = (world["plugins_dir"].parent / "plugins-backup").iterdir()
    assert (backup / "settings.py").read_text() == "LIMIT = 99\n"
    assert (backup / "extras").read_text() == "mine" and (backup / "cache" / "blob").read_text() == "mine"
    # The old version's own code is told apart from the user's files whenever its revision is fetchable.
    assert (backup / "utils" / "__init__.py").exists() is (via == "url-old-revision-gone")


def test_repin_keeps_a_wholly_ignored_data_dir_in_a_git_checkout(world):
    """``git status --ignored=matching`` reports an ignored dir as ONE ``data/`` entry; its files must
    still be carried into the re-pinned tree."""
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
