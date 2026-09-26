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
    (src / "mcp.json").write_text('{"mcpServers": {}}')
    (src / "skills" / "old").mkdir(parents=True)
    (src / "skills" / "old" / "SKILL.md").write_text("# old\n")
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
    (target / "data").chmod(0o700)
    (target / "settings.py").write_text("LIMIT = 99\n")
    (target / "extras").write_text("mine")
    (target / "cache").mkdir()
    (target / "cache" / "blob").write_text("mine")

    # v2: the utils/ package becomes utils.py; extras and cache ship with the type the user did not use.
    shutil.rmtree(src / "utils")
    shutil.rmtree(src / "skills")
    (src / "mcp.json").unlink()
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
    if os.name != "nt":
        assert (target / "data").stat().st_mode & 0o777 == 0o700
    assert not (target / "utils").exists() and (target / "utils.py").read_text() == "NEW = True\n"
    assert not (target / "mcp.json").exists() and not (target / "skills").exists()
    assert (target / "settings.py").read_text() == "LIMIT = 2\n"
    assert (target / "extras" / "a.txt").read_text() == "upstream"
    assert (target / "cache").read_text() == "upstream"
    backup, = (world["plugins_dir"].parent / "plugins-backup").iterdir()
    assert (backup / "settings.py").read_text() == "LIMIT = 99\n"
    assert (backup / "extras").read_text() == "mine" and (backup / "cache" / "blob").read_text() == "mine"
    # The old version's own code is told apart from the user's files whenever its revision is fetchable.
    assert (backup / "utils" / "__init__.py").exists() is (via == "url-old-revision-gone")


def _installed_subdir_plugin(tmp_path, monkeypatch, via, prepare=None):
    """A monorepo subdirectory install (no ``.git``) through *via*; returns ``(src, target, release)``,
    ``release()`` committing the next version and moving the catalog pin to it."""
    mono = tmp_path / "mono"
    src = mono / "plugins" / "sub-plugin"
    src.mkdir(parents=True)
    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 1.0.0\ndescription: d\n")
    (src / "__init__.py").write_text("def register(ctx):\n    pass\n")
    if prepare is not None:
        prepare(src)
    sp.run(["git", "init", "-q"], cwd=mono, check=True, env=_GIT_ENV)
    pin = {"sha": _commit(mono, "v1")}

    def entry():
        return pc_cat.PluginCatalogEntry(name="sub-plugin", repo=mono.as_uri(), sha=pin["sha"],
                                         description="d", maintainer="t", subdir="plugins/sub-plugin")

    def release():
        (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 2.0.0\ndescription: d\n")
        pin["sha"] = _commit(mono, "v2")

    monkeypatch.setattr(pc_cat, "load_catalog", lambda catalog_dir=None: [entry()])
    if via == "catalog":
        target = cat.install_catalog_entry(entry(), force=False)[0]
    else:
        target = pc._install_plugin_core(f"{mono.as_uri()}#plugins/sub-plugin", force=False)[0]
    assert not (target / ".git").exists()
    return src, target, release


@pytest.mark.parametrize("via", ["url", "catalog"])
def test_update_scans_the_user_files_it_carries(world, tmp_path, monkeypatch, via):
    """The carry adds files to a tree the installer already scanned; a carried file the scanner blocks
    must block the update instead of being published under the fresh clone's clean verdict."""
    _src, target, release = _installed_subdir_plugin(tmp_path, monkeypatch, via)
    (target / "evil.py").write_text("open('/etc/passwd').read()\n")
    release()
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: True)

    assert pc.dashboard_update_user_plugin("sub-plugin")["ok"] is False
    assert "version: 1.0.0" in (target / "plugin.yaml").read_text()


def test_url_subdir_reclone_revalidates_carried_files_before_publication(world, tmp_path, monkeypatch):
    """The URL re-clone admits the fresh clone (scan, portable-package check) before the carry; the
    tree the user's files were carried into must pass both gates again."""
    from hermes_cli import plugins_cmd_install as install_cmd

    _src, target, release = _installed_subdir_plugin(tmp_path, monkeypatch, "url")
    (target / "server.js").write_text('console.log("mine")\n')
    release()
    scans, portable_checks = [], []

    def scan_gate(tree, *_args, **_kwargs):
        scans.append((Path(tree) / "server.js").exists())

    def portable_gate(_plugin_name, tree):
        portable_checks.append((Path(tree) / "server.js").exists())
        if portable_checks[-1]:
            raise pc.PluginOperationError("carried server.js failed final admission")

    monkeypatch.setattr(pc, "_scan_plugin_tree", scan_gate)
    monkeypatch.setattr(install_cmd, "_refuse_unavailable_portable_plugin", portable_gate)
    result = pc.dashboard_update_user_plugin("sub-plugin")

    assert result["ok"] is False and "carried server.js failed final admission" in result["error"]
    assert scans == [False, True] and portable_checks == [False, True]
    assert (target / "server.js").is_file() and "version: 1.0.0" in (target / "plugin.yaml").read_text()


@pytest.mark.platforms("posix")  # symlink creation needs privileges on Windows
@pytest.mark.parametrize("via", ["url", "catalog"])
def test_update_backs_up_a_repointed_shipped_symlink(world, tmp_path, monkeypatch, via):
    """A shipped symlink the user re-pointed is an edit to a shipped file: the new version's link is
    published and the user's link is kept under ``plugins-backup/`` as a link, not lost."""
    def prepare(src):
        (src / "default.txt").write_text("default")
        (src / "mine.txt").write_text("mine")
        (src / "active").symlink_to("default.txt")

    _src, target, release = _installed_subdir_plugin(tmp_path, monkeypatch, via, prepare)
    (target / "active").unlink()
    (target / "active").symlink_to("mine.txt")
    release()

    assert pc.dashboard_update_user_plugin("sub-plugin")["ok"] is True
    assert os.readlink(target / "active") == "default.txt"
    backup, = (world["plugins_dir"].parent / "plugins-backup").iterdir()
    assert os.readlink(backup / "active") == "mine.txt"


@pytest.mark.platforms("posix")  # symlinks and the execute bit
@pytest.mark.parametrize("edit", ["link-to-equal-bytes", "dangling-link", "link-to-dir", "execute-bit"])
def test_update_without_the_installed_revision_backs_up_every_changed_shipped_entry(world, tmp_path, monkeypatch,
                                                                                     edit):
    """Once the installed revision is unfetchable, the old tree is compared with the new one entry by entry.
    What differs as git would record it (kind, link target, bytes, execute bit) is the user's edit and goes
    to ``plugins-backup/``: a link re-pointed at a file with the same bytes, a dangling link, a link to a
    directory, a ``chmod +x``."""
    def prepare(src):
        for name in ("default.txt", "same.txt", "adir/f", "tool.sh"):
            (src / name).parent.mkdir(exist_ok=True)
            (src / name).write_text("same")
        (src / "tool.sh").chmod(0o644)
        (src / "active").symlink_to("default.txt")

    src, target, _release = _installed_subdir_plugin(tmp_path, monkeypatch, "url", prepare)
    link = {"link-to-equal-bytes": "same.txt", "dangling-link": "gone", "link-to-dir": "adir"}.get(edit)
    if link is None:
        (target / "tool.sh").chmod(0o755)
    else:
        (target / "active").unlink()
        (target / "active").symlink_to(link)
    mono = src.parents[1]
    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 2.0.0\ndescription: d\n")
    sp.run(["git", "commit", "-q", "-a", "--amend", "-m", "v2"], cwd=mono, check=True, env=_GIT_ENV)
    sp.run(["git", "reflog", "expire", "--expire=now", "--all"], cwd=mono, check=True, env=_GIT_ENV)
    sp.run(["git", "gc", "-q", "--prune=now"], cwd=mono, check=True, env=_GIT_ENV)

    assert pc.dashboard_update_user_plugin("sub-plugin")["ok"] is True
    assert "version: 2.0.0" in (target / "plugin.yaml").read_text()
    assert os.readlink(target / "active") == "default.txt" and not (target / "tool.sh").stat().st_mode & 0o111
    backup, = (world["plugins_dir"].parent / "plugins-backup").iterdir()
    if link is None:
        assert (backup / "tool.sh").stat().st_mode & 0o100
    else:
        assert os.readlink(backup / "active") == link


@pytest.mark.platforms("posix")  # the execute bit
@pytest.mark.parametrize("via", ["url", "catalog"])
def test_update_backs_up_an_execute_bit_edit_to_a_shipped_file(world, tmp_path, monkeypatch, via):
    """Git records a file's execute bit (``100644``/``100755``), so a ``chmod`` of a shipped file is an edit:
    the new version's mode is published and the user's copy goes to ``plugins-backup/``. A shipped
    executable the user left alone is not an edit."""
    def prepare(src):
        for name, mode in (("tool.sh", 0o644), ("run.sh", 0o755), ("kept.sh", 0o755)):
            (src / name).write_text("#!/bin/sh\n")
            (src / name).chmod(mode)

    _src, target, release = _installed_subdir_plugin(tmp_path, monkeypatch, via, prepare)
    (target / "tool.sh").chmod(0o755)
    (target / "run.sh").chmod(0o644)
    release()

    assert pc.dashboard_update_user_plugin("sub-plugin")["ok"] is True
    assert "version: 2.0.0" in (target / "plugin.yaml").read_text()
    assert not (target / "tool.sh").stat().st_mode & 0o111 and (target / "run.sh").stat().st_mode & 0o100
    backup, = (world["plugins_dir"].parent / "plugins-backup").iterdir()
    assert (backup / "tool.sh").stat().st_mode & 0o100 and not (backup / "run.sh").stat().st_mode & 0o111
    assert not (backup / "kept.sh").exists()


@pytest.mark.parametrize("via", ["url", "catalog", "catalog-git-checkout"])
def test_update_refuses_when_the_plugin_writes_after_its_files_were_classified(world, tmp_path, monkeypatch, via):
    """What to carry is decided from the installed tree as classified. A file the live plugin writes after
    that (e.g. during the clone) is not in that plan, so the update must refuse to replace the tree rather
    than delete the file; a retry classifies it and carries it. Classifying does not itself count as a
    change (``git status`` refreshing the index of a touched checkout)."""
    if via == "catalog-git-checkout":
        name = "cat-plugin"
        target = cat.install_catalog_entry(pc_cat.get_live_catalog_entry(name), force=False)[0]
        assert (target / ".git").exists()

        def release():
            world["state"]["pin"] = world["sha2"]
    else:
        name = "sub-plugin"
        _src, target, release = _installed_subdir_plugin(tmp_path, monkeypatch, via)
    release()
    classify = cat._local_changes

    def classify_then_plugin_writes(tree):
        changes = classify(tree)
        (tree / "late.json").write_text("{}")
        return changes

    monkeypatch.setattr(cat, "_local_changes", classify_then_plugin_writes)
    result = pc.dashboard_update_user_plugin(name)
    assert result["ok"] is False and "retry" in result["error"]
    assert (target / "late.json").read_text() == "{}"

    monkeypatch.setattr(cat, "_local_changes", classify)
    os.utime(target / "__init__.py", (1_000_000_000, 1_000_000_000))  # same bytes, stale index stat
    assert pc.dashboard_update_user_plugin(name)["ok"] is True
    assert (target / "late.json").read_text() == "{}"


@pytest.mark.platforms("posix")  # symlinks and the execute bit
@pytest.mark.parametrize("via", ["url", "catalog"])
@pytest.mark.parametrize("edit", ["execute-bit", "file-to-link"])
def test_update_refuses_when_an_entry_mode_or_kind_changes_before_publication(world, tmp_path, monkeypatch, via,
                                                                            edit):
    """The publication baseline is the tree as classified, including what the byte digest cannot see: a
    ``chmod +x``, or a file swapped for a link whose target text equals its bytes, must refuse the update
    instead of being published over."""
    from hermes_cli import plugins_transaction as tx

    _src, target, release = _installed_subdir_plugin(tmp_path, monkeypatch, via)
    (target / "notes").write_text("x")
    release()
    publish = tx.publish_plugin

    def plugin_changes_then_publish(*args, **kwargs):
        if edit == "execute-bit":
            (target / "notes").chmod(0o755)
        else:
            (target / "notes").unlink()
            (target / "notes").symlink_to("x")
        return publish(*args, **kwargs)

    monkeypatch.setattr(tx, "publish_plugin", plugin_changes_then_publish)
    result = pc.dashboard_update_user_plugin("sub-plugin")

    assert result["ok"] is False and "retry" in result["error"]
    assert "version: 1.0.0" in (target / "plugin.yaml").read_text()
    if edit == "execute-bit":
        assert (target / "notes").stat().st_mode & 0o100
    else:
        assert os.readlink(target / "notes") == "x"


def test_repin_rename_keeps_the_old_tree_when_it_changed_after_classification(world, monkeypatch):
    """A manifest rename publishes under the new name, then removes the old directory. A file the live plugin
    wrote there after its files were classified was never considered, so that directory is kept, not deleted."""
    from hermes_cli import plugins_transaction as tx

    target = cat.install_catalog_entry(pc_cat.get_live_catalog_entry("cat-plugin"), force=False)[0]
    repo = world["repo"]
    (repo / "plugin.yaml").write_text("name: cat-plugin-v2\nversion: 2.0.0\ndescription: d\n")
    world["state"]["pin"] = _commit(repo, "rename")
    publish = tx.publish_plugin

    def plugin_writes_then_publish(*args, **kwargs):
        (target / "late.json").write_text("{}")
        return publish(*args, **kwargs)

    monkeypatch.setattr(tx, "publish_plugin", plugin_writes_then_publish)
    result = pc.dashboard_update_user_plugin("cat-plugin")

    assert result["ok"] and (world["plugins_dir"] / "cat-plugin-v2" / "plugin.yaml").is_file()
    assert (target / "late.json").read_text() == "{}"
    assert any("kept" in w for w in result["warnings"])


@pytest.mark.platforms("posix")  # the execute bit
def test_carry_without_the_installed_revision_does_not_resurrect_removed_scripts(tmp_path):
    """Without the installed revision, what the new version no longer ships is carried only when it cannot be
    the old version's code: scripts and executables go to the backup, not into the new tree."""
    old, new, backup = tmp_path / "old", tmp_path / "new", tmp_path / "backup"
    old.mkdir()
    new.mkdir()
    for name, mode in (("setup.sh", 0o755), ("install.ps1", 0o644), ("run-me", 0o755), ("tool.exe", 0o644),
                       ("lib.dll", 0o644), ("notes.txt", 0o644)):
        (old / name).write_text("#!/bin/sh\n")
        (old / name).chmod(mode)

    set_aside = cat._carry_user_files(old, new, None, backup)

    assert sorted(set_aside) == ["install.ps1", "lib.dll", "run-me", "setup.sh", "tool.exe"]
    assert sorted(p.name for p in new.iterdir()) == ["notes.txt"]
    assert (backup / "setup.sh").stat().st_mode & 0o100


def test_update_without_the_installed_revision_does_not_resurrect_a_removed_binary(world, tmp_path, monkeypatch):
    """A shipped binary has no source suffix and, as git records a Windows executable, no execute bit. Once
    the installed revision is unfetchable, a ``tool.exe`` the new version removed goes to the backup, not
    into the new tree: the URL re-clone rescans with ``force=True``, which accepts the caution verdict a
    bundled binary earns, so a carried one would be published."""
    def prepare(src):
        (src / "tool.exe").write_bytes(b"MZ\0\0old")
        (src / "tool.exe").chmod(0o644)

    src, target, _release = _installed_subdir_plugin(tmp_path, monkeypatch, "url", prepare)
    (target / "config.yaml").write_text("endpoint: mine\n")
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: True)
    mono = src.parents[1]
    (src / "tool.exe").unlink()
    (src / "plugin.yaml").write_text("name: sub-plugin\nversion: 2.0.0\ndescription: d\n")
    sp.run(["git", "add", "-A"], cwd=mono, check=True, env=_GIT_ENV)
    sp.run(["git", "commit", "-q", "--amend", "-m", "v2"], cwd=mono, check=True, env=_GIT_ENV)
    sp.run(["git", "reflog", "expire", "--expire=now", "--all"], cwd=mono, check=True, env=_GIT_ENV)
    sp.run(["git", "gc", "-q", "--prune=now"], cwd=mono, check=True, env=_GIT_ENV)

    assert pc.dashboard_update_user_plugin("sub-plugin")["ok"] is True
    assert "version: 2.0.0" in (target / "plugin.yaml").read_text()
    assert not (target / "tool.exe").exists()
    assert (target / "config.yaml").read_text() == "endpoint: mine\n"
    backup, = (world["plugins_dir"].parent / "plugins-backup").iterdir()
    assert (backup / "tool.exe").read_bytes() == b"MZ\0\0old"


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
