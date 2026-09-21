"""Installer-owned payloads survive offline cloning, not arbitrary runtime trees."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextlib.contextmanager
def scoped(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, env={**os.environ, "GIT_AUTHOR_NAME": "Snapshot Test",
                        "GIT_AUTHOR_EMAIL": "snapshot@example.invalid",
                        "GIT_COMMITTER_NAME": "Snapshot Test",
                        "GIT_COMMITTER_EMAIL": "snapshot@example.invalid"},
    ).stdout.strip()


def installed(tmp_path, monkeypatch, subdir):
    from hermes_cli import plugins_cmd as cmd, plugins_cmd_catalog as catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry
    home = tmp_path / "A"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    repo = tmp_path / "upstream"
    package = repo / subdir
    package.mkdir(parents=True)
    # A provider-shaped fixture must never execute during install or clone.
    (package / "__init__.py").write_text('raise AssertionError("provider imported")\n')
    (package / "plugin.yaml").write_text("name: snapshot-provider\n")
    (package / "value.txt").write_text("original\n")
    (package / ".env.example").write_text("PLUGIN_KEY=\n")
    (package / "data").mkdir()
    (package / "data" / "asset.json").write_text('{"source_asset": true}\n')
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "source payload")
    revision = git(repo, "rev-parse", "HEAD")
    entry = PluginCatalogEntry(name="snapshot-catalog", repo=repo.as_uri(), sha=revision,
                               subdir=subdir or None, description="Local test", maintainer="Fixture")
    cache = home / "cache" / "plugin-catalog.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps({"entries": [entry.to_dict()], "removed": []}))
    config = {"plugins": {"enabled": ["snapshot-provider"], "disabled": ["snapshot-provider"]},
              "memory": {"provider": "unchanged"}}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    def forbidden(*args, **kwargs):
        raise AssertionError("No networking or dependency installation in this test")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(cmd, "_refuse_conflicting_python_deps", forbidden)
    monkeypatch.setattr(cmd, "_install_python_dependencies", forbidden)
    with scoped(home):
        target, _, _ = catalog.install_catalog_entry(entry, force=False, python_deps=False)
    return home, target, repo, entry


def clone(source, destination):
    from hermes_cli.profile_plugins import copy_profile_plugins
    from hermes_cli.plugin_installation import plugin_installation_lock
    destination.mkdir()
    with plugin_installation_lock(source):
        return copy_profile_plugins(source, destination)


def records(home):
    return json.loads((home / "plugins" / ".install-metadata.json").read_text())


@pytest.mark.parametrize("subdir", ["", "examples/hermes-plugin"])
def test_installer_payload_clones_offline_and_updates_independently(tmp_path, monkeypatch, subdir):
    from hermes_cli import plugins_cmd as cmd, plugins_cmd_catalog as catalog
    home, target, repo, entry = installed(tmp_path, monkeypatch, subdir)
    before = records(home)
    inventory = before[target.name]["files"]
    assert set(inventory["entries"]) == {"__init__.py", "plugin.yaml", "value.txt", ".env.example", "data/asset.json"}
    assert (target / ".env").exists()  # generated after capture
    assert before[target.name]["source"] == entry.install_identifier
    assert before[target.name]["revision"] == entry.sha
    config_before = (home / "config.yaml").read_bytes()
    sidecar_before = (target / ".hermes-catalog.json").read_bytes()
    (target / "private-notes.txt").write_text("not source")
    (target / "runtime").symlink_to(tmp_path, target_is_directory=True)
    second, third = tmp_path / "B", tmp_path / "C"
    imported = set(sys.modules)
    assert clone(home, second)["copied"] == [target.name]
    assert records(second) == before
    shutil.rmtree(home / "plugins")
    assert clone(second, third)["copied"] == [target.name]
    assert records(third) == before
    for owner in (second, third):
        package = owner / "plugins" / target.name
        assert set(p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()) == set(inventory["entries"]) | {".hermes-catalog.json"}
        assert not (package / "runtime").exists()
        assert (package / ".hermes-catalog.json").read_bytes() == sidecar_before
        for name, info in inventory["entries"].items():
            assert hashlib.sha256((package / name).read_bytes()).hexdigest() == info["sha256"]
            assert (package / name).stat().st_ino != (third if owner == second else second).joinpath("plugins", target.name, name).stat().st_ino
    assert (home / "config.yaml").read_bytes() == config_before
    assert not (second / "config.yaml").exists()  # host clone owns settings, never this helper
    assert not any(str(getattr(sys.modules[m], "__file__", "")).startswith(str(home / "plugins")) for m in set(sys.modules) - imported)
    # Catalog updates still work without .git, and never mutate C's older copy.
    (repo / subdir / "value.txt").write_text("updated\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "new payload")
    entry.sha = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(catalog, "get_live_catalog_entry", lambda name: entry)
    monkeypatch.setattr(catalog, "raise_if_removed", lambda *args: None)
    real_install = catalog.install_catalog_entry
    monkeypatch.setattr(catalog, "install_catalog_entry", lambda *args, **kwargs: real_install(*args, **kwargs, python_deps=False))
    b_package = second / "plugins" / target.name
    with scoped(second):
        sha, changed = catalog.repin_catalog_plugin(b_package, catalog.read_catalog_sidecar(b_package))
        assert changed and sha == entry.sha
        assert records(second)[target.name]["files"] != inventory
        assert (b_package / "value.txt").read_text() == "updated\n"
        assert records(third) == before
        cmd._remove_plugin_core(b_package)
    assert not b_package.exists() and not records(second)
    assert (third / "plugins" / target.name / "value.txt").read_text() == "original\n"


@pytest.mark.parametrize("failure", ["legacy", "dirty", "missing", "mode", "symlink", "ancestor", "hardlink", "traversal", "reserved", "version", "sidecar", "removed", "metadata", "mutation"])
def test_invalid_owned_payload_refuses_without_publication_and_exact_reinstall_repairs(tmp_path, monkeypatch, failure):
    from hermes_cli import profile_plugins as snapshots, plugins_cmd as cmd, plugin_catalog
    home, target, repo, entry = installed(tmp_path, monkeypatch, "examples/hermes-plugin")
    metadata_path = home / "plugins" / ".install-metadata.json"
    original = records(home)
    broken = records(home)
    owned = target / "value.txt"
    if failure == "legacy":
        broken[target.name].pop("files", None)
    elif failure == "dirty":
        owned.write_text("edited")
    elif failure == "missing":
        owned.unlink()
    elif failure == "mode":
        owned.chmod(0o444)
    elif failure == "symlink":
        owned.unlink()
        owned.symlink_to(repo / entry.subdir / "value.txt")
    elif failure == "ancestor":
        shutil.rmtree(target / "data")
        (target / "data").symlink_to(repo / entry.subdir / "data", target_is_directory=True)
    elif failure == "hardlink":
        os.link(owned, tmp_path / "linked")
    elif failure in {"traversal", "reserved"}:
        entries = broken[target.name]["files"]["entries"]
        entries["../escape" if failure == "traversal" else ".git/config"] = entries["value.txt"]
    elif failure == "version":
        broken[target.name]["files"]["version"] = 999
    elif failure == "sidecar":
        (target / ".hermes-catalog.json").write_text("{}")
    elif failure == "removed":
        directory = tmp_path / "catalog"
        directory.mkdir()
        (directory / "removed.yaml").write_text(yaml.safe_dump({"removed": [{"name": "different", "repo": entry.repo}]}))
        monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: directory)
    elif failure == "metadata":
        broken[target.name]["revision"] = "a" * 40
    elif failure == "mutation":
        from hermes_cli import plugin_inventory
        real_copy = plugin_inventory.copy_install_inventory
        def mutate(*args, **kwargs):
            real_copy(*args, **kwargs)
            owned.write_text("racing change")
        monkeypatch.setattr(plugin_inventory, "copy_install_inventory", mutate)
    metadata_path.write_text(json.dumps(broken))
    staging = tmp_path / "B"
    with pytest.raises(snapshots.PluginSnapshotError, match="(?i)(repair|reinstall|removed)") as error:
        clone(home, staging)
    assert not (staging / "plugins").exists()
    if failure == "legacy":
        assert entry.install_identifier in str(error.value) and entry.sha in str(error.value)
        assert "back up" in str(error.value).lower()
        # Repair uses saved exact source INCLUDING subdirectory, not repository root.
        with scoped(home):
            cmd._install_plugin_core(original[target.name]["source"], force=True,
                                     ref=original[target.name]["revision"], python_deps=False)
        assert records(home) == original
        assert clone(home, tmp_path / "repaired")["copied"] == [target.name]


@pytest.mark.parametrize("malformed", ["manifest", "removals", "cache"])
def test_snapshot_parse_errors_never_expose_source_canaries(tmp_path, monkeypatch, malformed):
    from hermes_cli import profile_plugins as snapshots, plugin_catalog

    home, target, _, _ = installed(tmp_path, monkeypatch, "")
    canary = "CANARY_7f3a"
    if malformed == "manifest":
        path = target / "plugin.yaml"
        path.write_text(f"name: [{canary}\n")
        metadata = records(home)
        metadata[target.name]["files"]["entries"]["plugin.yaml"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (home / "plugins" / ".install-metadata.json").write_text(json.dumps(metadata))
    elif malformed == "removals":
        directory = tmp_path / "catalog"
        directory.mkdir()
        path = directory / "removed.yaml"
        path.write_text(f"removed: [{canary}\n")
        monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: directory)
    else:
        path = home / "cache" / "plugin-catalog.json"
        path.write_bytes(b'{"removed": "' + canary.encode() + b'\xff"}')
    before = path.read_bytes()
    destination = tmp_path / "B"
    with pytest.raises(snapshots.PluginSnapshotError) as error:
        clone(home, destination)
    import traceback
    rendered = "".join(traceback.format_exception(error.value))
    assert canary not in str(error.value)
    assert canary not in rendered
    assert "Repair the source installation" in str(error.value)
    assert path.read_bytes() == before
    assert not (destination / "plugins").exists()
    assert not list(destination.glob(".plugin-snapshot-*"))


@pytest.mark.windows_only
@pytest.mark.parametrize("subdir", ["", "examples/hermes-plugin"])
def test_windows_catalog_install_and_clone_keep_git_modes(tmp_path, monkeypatch, subdir):
    from hermes_cli import plugins_cmd_catalog as catalog
    from hermes_cli.plugin_inventory import validate_install_inventory, PluginInventoryError

    home, target, repo, entry = installed(tmp_path, monkeypatch, subdir)
    script = repo / subdir / "run.sh"
    script.write_bytes(b"#!/bin/sh\nprintf 'fixture\\n'\n")
    git(repo, "add", ".")
    git(repo, "update-index", "--chmod=+x", str(script.relative_to(repo)))
    git(repo, "commit", "-qm", "executable payload")
    entry.sha = git(repo, "rev-parse", "HEAD")
    with scoped(home):
        catalog.install_catalog_entry(entry, force=True, python_deps=False)
    inventory = records(home)[target.name]["files"]
    assert inventory["entries"]["run.sh"]["mode"] == 0o755
    assert inventory["entries"]["value.txt"]["mode"] == 0o644
    before = (home / "config.yaml").read_bytes()
    for source, destination in ((home, tmp_path / "B"), (tmp_path / "B", tmp_path / "C")):
        assert clone(source, destination)["copied"] == [target.name]
        assert records(destination) == records(home)
        package = destination / "plugins" / target.name
        validate_install_inventory(package, inventory)
        for name, record in inventory["entries"].items():
            assert (package / name).read_bytes() == (target / name).read_bytes()
            assert (package / name).stat().st_mode & stat.S_IWRITE
    assert (home / "config.yaml").read_bytes() == before
    owned = target / "value.txt"
    owned.chmod(stat.S_IREAD)
    try:
        with pytest.raises(PluginInventoryError, match="mode"):
            validate_install_inventory(target, inventory)
    finally:
        owned.chmod(stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.windows_only
@pytest.mark.parametrize("junction_at", ["package", "ancestor"])
def test_windows_clone_rejects_real_junction_without_pathlib_helper(tmp_path, monkeypatch, junction_at):
    from hermes_cli import profile_plugins as snapshots

    home, target, _, _ = installed(tmp_path, monkeypatch, "")
    junction = target if junction_at == "package" else target / "data"
    outside = tmp_path / "outside"
    shutil.move(str(junction), str(outside))
    subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
                   check=True, capture_output=True)
    # Python 3.11 has no Path.is_junction: exercise the lstat contract on newer hosts too.
    monkeypatch.delattr(Path, "is_junction", raising=False)
    before = {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()}
    try:
        with pytest.raises(snapshots.PluginSnapshotError):
            clone(home, tmp_path / "B")
        assert not (tmp_path / "B" / "plugins").exists()
        assert {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()} == before
    finally:
        junction.rmdir()
