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
from tests.pm._fixtures import client, isolated_python

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
    return sp.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False).stdout.strip()


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


def test_the_catalog_name_you_installed_with_works_for_every_later_verb(world, tmp_path):
    """17 catalog entries install under a different manifest name (`hermes-memory-wiki` lands as
    `plugins/memory-wiki`). Every later verb must accept the name the user typed at install, resolved
    through the installer's own record; a URL install shipping a forged sidecar must not claim it."""
    repo = world["repo"]
    (repo / "plugin.yaml").write_text("name: renamed-plugin\nversion: 1.0.0\ndescription: d\n")
    world["state"]["pin"] = _commit(repo, "manifest name differs from catalog name")
    forged = json.dumps({"catalog_name": "cat-plugin", "tier": "official", "sha": "0" * 40, "repo": "x"})
    _install_url(tmp_path / "evil", "evil-plugin", {cat.CATALOG_SIDECAR: forged})
    assert pc._find_plugin_entry("cat-plugin") is None

    pc.cmd_install("cat-plugin", enable=False)
    target = world["plugins_dir"] / "renamed-plugin"
    assert target.is_dir() and pc._get_enabled_set() == set()

    pc.cmd_enable("cat-plugin")
    assert pc._get_enabled_set() == {"renamed-plugin"}
    pc.cmd_show("cat-plugin")
    pc.cmd_capabilities("cat-plugin")
    pc.cmd_disable("cat-plugin")
    assert pc._get_disabled_set() == {"renamed-plugin"}
    assert pc.dashboard_set_agent_plugin_enabled("cat-plugin", enabled=True)["name"] == "renamed-plugin"
    assert pc.dashboard_update_user_plugin("cat-plugin")["ok"] is True
    assert pc.dashboard_remove_user_plugin("cat-plugin")["ok"] is True
    assert not target.exists()
    assert "renamed-plugin" not in pc._get_enabled_set()


def test_custom_install_records_an_anonymous_extension_install_and_reinstall_none(world, monkeypatch):
    import hermes_cli.observability.shared_metrics_events as events

    calls = []
    monkeypatch.setattr(events, "record_extension_install", lambda **kw: calls.append(kw))
    assert pc.dashboard_install_plugin(world["repo"].as_uri(), force=False, enable=False)["ok"]
    assert pc.dashboard_install_plugin("", force=True, enable=False, catalog_name="cat-plugin")["ok"]

    assert calls == [{"kind": "plugin", "source": "local", "name": None, "outcome": "success"}]


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


def test_url_install_of_an_archived_handoff_repo_is_repinned_to_its_catalog_entry(world, tmp_path, monkeypatch):
    """A provider that left core was first handed off through a Nous repo that is now archived; a URL install of
    it has no catalog record, so `update` would keep pulling the archive. It is adopted as the catalog entry that
    superseded it, and `update` re-pins it to the maintained repo at the reviewed sha."""
    archived = tmp_path / "archived"
    target = _install_url(archived, "cat-plugin", {"ARCHIVED.md": "moved\n"})
    assert cat.read_catalog_sidecar(target) is None  # control: an unknown repo stays a plain URL install
    monkeypatch.setitem(cat._HANDOFF_REPOS, pc_cat._normalize_repo(archived.as_uri()), "cat-plugin")
    assert cat.read_catalog_sidecar(target)["catalog_name"] == "cat-plugin"
    seen = {}
    real_scan = pc._scan_plugin_tree
    monkeypatch.setattr(pc, "_scan_plugin_tree", lambda *a, **k: seen.update(k) or real_scan(*a, **k))
    result = pc.dashboard_update_user_plugin("cat-plugin")
    assert result["ok"] and result["unchanged"] is False and _head(target) == world["sha1"]
    assert seen["reviewed_pin"] is True  # the re-pinned tree IS the reviewed sha: caution findings don't block it
    record = pc._read_install_metadata()["cat-plugin"]
    assert record["source"] == world["repo"].as_uri() and record["catalog"]["repo"] == world["repo"].as_uri()


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
    (src / "server.js").write_text("console.log('v1')\n")
    (src / "dashboard").mkdir()
    (src / "dashboard" / "manifest.json").write_text("{}")
    (src / "hooks").mkdir()
    (src / "hooks" / "run.cjs").write_text("module.exports = 1\n")
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
    (src / "server.js").unlink()
    shutil.rmtree(src / "dashboard")
    shutil.rmtree(src / "hooks")
    pin["sha"] = _commit(mono, "v2")
    assert pc.dashboard_update_user_plugin("sub-plugin")["ok"] is True

    assert "version: 2.0.0" in (target / "plugin.yaml").read_text()
    assert (target / "config.yaml").read_text() == "endpoint: mine\n"
    assert (target / "data" / "state.json").read_text() == "{}"
    assert not (target / "desktop").exists()
    assert not (target / "server.js").exists()
    assert not (target / "dashboard").exists() and not (target / "hooks" / "run.cjs").exists()


def test_repin_keeps_a_wholly_ignored_data_dir_in_a_git_checkout(world):
    """A single ``!! data/`` status entry must preserve every file below that ignored directory."""
    repo = world["repo"]
    (repo / ".gitignore").write_text("data/\n.venv/\nnode_modules/\n")
    world["state"]["pin"] = _commit(repo, "ignore data")
    target = cat.install_catalog_entry(pc_cat.get_live_catalog_entry("cat-plugin"), force=False)[0]
    assert (target / ".git").exists()

    (target / "data" / "db").mkdir(parents=True)
    (target / "data" / "db" / "index.db").write_text("user data")
    # An ignored venv always holds symlinks (bin/python); it is a reproducible artefact, not user state.
    (target / ".venv" / "bin").mkdir(parents=True)
    (target / ".venv" / "bin" / "python").symlink_to("/usr/bin/python3")
    (target / ".venv" / "pyvenv.cfg").write_text("home = /usr/bin")
    (target / "node_modules" / "x").mkdir(parents=True)
    (target / "node_modules" / "x" / "index.js").write_text("module.exports = 1")

    (repo / "__init__.py").write_text("def register(ctx):\n    pass  # v3\n")
    world["state"]["pin"] = _commit(repo, "v3")
    assert pc.dashboard_update_user_plugin("cat-plugin")["unchanged"] is False

    assert _head(target) == world["state"]["pin"]
    assert (target / "data" / "db" / "index.db").read_text() == "user data"
    # Excluded dependency dirs are not carried at all: a partial copy would be a broken install.
    assert not (target / ".venv").exists() and not (target / "node_modules").exists()

    # A symlink in the ignored set is never followed into the update: it fails closed, naming the path,
    # and the live plugin stays at its current revision with its user data.
    published = world["state"]["pin"]
    outside = repo.parent / "outside.yaml"
    outside.write_text("secret")
    (target / "data" / "link.yaml").symlink_to(outside)
    (repo / "__init__.py").write_text("def register(ctx):\n    pass  # v4\n")
    world["state"]["pin"] = _commit(repo, "v4")
    result = pc.dashboard_update_user_plugin("cat-plugin")
    assert result["ok"] is False and "data/link.yaml" in result["error"]
    assert _head(target) == published
    assert (target / "data" / "link.yaml").is_symlink()
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


_HOOK = "def register(ctx):\n    ctx.register_hook('pre_llm_call', print)\n"


_PROVIDER = "def register(ctx):\n    ctx.register_memory_provider(object())\n"


@pytest.mark.parametrize("init,enable", [
    (_PROVIDER, True),
    (_PROVIDER, False),
    (_HOOK, True),
    ('"""Recall hook. Not a MemoryProvider: it only injects context."""\n' + _HOOK, True),
])
def test_memory_category_install_activates_via_memory_provider_not_plugins_enable(world, monkeypatch,
                                                                                  init, enable):
    """A memory provider activates through memory.provider alone — the loader never consults
    plugins.enabled — so "Enable now?" must select the provider, and a decline must point at
    `hermes memory setup`, never at the dead-end `plugins enable` hint. Catalog category "memory" also
    holds hook plugins: those enable normally and leave the user's memory.provider alone, even when their
    docstring names MemoryProvider (the loader would return None for them)."""
    repo = world["repo"]
    provider = init is _PROVIDER
    (repo / "__init__.py").write_text(init)
    world["state"]["pin"] = _commit(repo, "memory category plugin")

    def _memory_entries():
        return [pc_cat.PluginCatalogEntry(name="cat-plugin", repo=repo.as_uri(), sha=world["state"]["pin"],
                                          description="d", maintainer="t", category="memory")]

    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: _memory_entries())
    printed: list[str] = []

    def _capture(*args, **_kwargs):
        printed.extend(str(a) for a in args)

    monkeypatch.setattr(pc, "_console",
                        lambda: type("C", (), {"print": staticmethod(_capture)})())

    if not provider:
        pc._save_memory_provider("honcho")
        pc.cmd_install("cat-plugin", enable=enable)
        assert (pc._get_current_memory_provider(), pc._get_enabled_set()) == ("honcho", {"cat-plugin"})
        return
    pc.cmd_install("cat-plugin", enable=enable)
    assert pc._get_current_memory_provider() == ("cat-plugin" if enable else "")
    assert pc._get_enabled_set() == set()
    assert any("hermes memory setup" in line for line in printed)
    assert not any("plugins enable cat-plugin" in line for line in printed)


@pytest.mark.parametrize("prior", [None, "disabled", "enabled"])
def test_url_install_of_a_memory_provider_dir_gets_the_provider_hint(world, tmp_path, monkeypatch, prior):
    """A URL (non-catalog) install whose tree satisfies the memory-provider contract gets the same
    provider activation path — the catalog category is a shortcut, not the only trigger. An explicit
    --enable must leave the provider loadable: a plugins.disabled entry would make the loader refuse it,
    and a reinstall must repair the old state that listed the provider in plugins.enabled instead."""
    from hermes_cli.config import load_config, save_config
    from plugins.memory import _explicitly_disabled, find_provider_dir

    repo = tmp_path / "mem-repo"
    repo.mkdir()
    (repo / "plugin.yaml").write_text("name: mem-plugin\nversion: 1.0.0\ndescription: d\n")
    (repo / "__init__.py").write_text("from agent.memory_provider import MemoryProvider\n\n\n"
                                      "class Mem(MemoryProvider):\n    pass\n")
    sp.run(["git", "init", "-q"], cwd=repo, check=True, env=_GIT_ENV)
    _commit(repo, "v1")

    printed: list[str] = []

    def _capture(*args, **_kwargs):
        printed.extend(str(a) for a in args)

    monkeypatch.setattr(pc, "_console",
                        lambda: type("C", (), {"print": staticmethod(_capture)})())

    if prior == "enabled":
        pc.cmd_install(repo.as_uri(), enable=False)
    if prior:
        cfg = load_config()
        cfg.setdefault("plugins", {})[prior] = ["mem-plugin"]
        save_config(cfg)
    pc.cmd_install(repo.as_uri(), enable=True, force=prior == "enabled")
    assert pc._get_current_memory_provider() == "mem-plugin"
    assert pc._get_enabled_set() == ({"mem-plugin"} if prior == "enabled" else set())
    assert not _explicitly_disabled("mem-plugin", find_provider_dir("mem-plugin"))
    assert any("memory.provider" in line for line in printed)


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


def test_catalog_rows_maps_resolves_the_live_catalog_once(monkeypatch):
    """``_plugin_rows`` needs pins, versions and titles together; taking them via the three single-map
    helpers pays the whole ``load_catalog_live()`` pass — git probe, ~300 catalog YAMLs, prefer-in-tree
    merges — three times per inventory request (#125683). ``catalog_rows_maps`` must resolve once and
    stay best effort like the per-map helpers: an empty triple on failure."""
    calls = []

    def fake_live():
        calls.append(1)
        return [pc_cat.PluginCatalogEntry(name="a", repo="https://github.com/o/r", sha="a" * 40,
                                          description="d", maintainer="t", version="1.2.3", title="Alpha"),
                pc_cat.PluginCatalogEntry(name="b", repo="https://github.com/o/r2", sha="b" * 40,
                                          description="d", maintainer="t")]

    monkeypatch.setattr(cat, "load_catalog_live", fake_live)
    assert cat.catalog_rows_maps() == ({"a": "a" * 40, "b": "b" * 40}, {"a": "1.2.3"}, {"a": "Alpha"})
    assert len(calls) == 1

    def boom():
        raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(cat, "load_catalog_live", boom)
    assert cat.catalog_rows_maps() == ({}, {}, {})


def test_same_version_off_pin_update_is_labeled_as_a_sync_to_pin(world, monkeypatch):
    """Update detection stays SHA-based: an off-pin install carrying the pin's own version still gets
    the update affordance (the pin may have moved commits without a version bump). Label honesty: the
    version label drops so surfaces show the target commit ref instead of "Update to <same version>".
    A real version bump reads as a version update, and an unknown installed version keeps the
    SHA-only signal with the label intact."""
    world["state"]["version"] = "1.0.0"
    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: [
        pc_cat.PluginCatalogEntry(name="cat-plugin", repo=world["repo"].as_uri(), sha=world["state"]["pin"],
                                  description="d", maintainer="t", version=world["state"]["version"])])
    entry = pc_cat.get_live_catalog_entry("cat-plugin")
    target, _m, _n = cat.install_catalog_entry(entry, force=False, ref=world["sha2"])
    assert _head(target) == world["sha2"]  # off-pin by a commit, same declared version
    row = cat.catalog_row_fields(target, cat.catalog_pins(), cat.catalog_versions(),
                                 installed_version="1.0.0")
    assert row["update_available"] is True
    assert row["catalog_version"] != "1.0.0"  # presented target differs from the installed version
    # A pin advance that keeps the same version label keeps the affordance too (honcho pattern: a new
    # commit, same declared version).
    repo = world["repo"]
    (repo / "plugin.yaml").write_text("name: cat-plugin\nversion: 1.0.0\ndescription: d (rev)\n")
    world["state"]["pin"] = _commit(repo, "advance")
    row = cat.catalog_row_fields(target, cat.catalog_pins(), cat.catalog_versions(),
                                 installed_version="1.0.0")
    assert row["update_available"] is True and row["catalog_version"] != "1.0.0"
    # The pin moving to a genuinely newer version is labeled as a version update.
    repo = world["repo"]
    (repo / "plugin.yaml").write_text("name: cat-plugin\nversion: 2.0.0\ndescription: d\n")
    world["state"]["pin"] = _commit(repo, "bump")
    world["state"]["version"] = "2.0.0"
    row = cat.catalog_row_fields(target, cat.catalog_pins(), cat.catalog_versions(),
                                 installed_version="1.0.0")
    assert row["update_available"] is True and row["catalog_version"] == "2.0.0"
    # Unknown installed version: the SHA-only signal stands, label untouched.
    row = cat.catalog_row_fields(target, cat.catalog_pins(), cat.catalog_versions())
    assert row["update_available"] is True and row["catalog_version"] == "2.0.0"
