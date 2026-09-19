"""Pristine inventories retain checkout transforms, never reapplied local edits."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, env={**os.environ, "GIT_AUTHOR_NAME": "Fixture",
                          "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_NAME": "Fixture",
                          "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}).stdout.strip()


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("transform", ["plain", "crlf", "encoding"])
def test_staged_git_update_refreshes_pristine_inventory_without_adopting_local_edits(tmp_path, monkeypatch, dirty, transform):
    from hermes_cli import plugins_cmd as cmd
    from hermes_cli.plugin_inventory import capture_install_inventory, validate_install_inventory, PluginInventoryError
    home, repo = tmp_path / "home", tmp_path / "repo"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    repo.mkdir()
    git(repo, "init", "-b", "main")
    newline = "\r\n" if transform == "crlf" else "\n"
    encoding = "utf-16-le" if transform == "encoding" else "utf-8"
    attributes = "*.txt text eol=crlf\n" if transform == "crlf" else "*.txt text eol=lf working-tree-encoding=UTF-16LE\n"
    if transform != "plain":
        (repo / ".gitattributes").write_text(attributes)
    def payload(text):
        return text.replace("\n", newline).encode(encoding)
    (repo / "plugin.yaml").write_text("name: inventory-update\n")
    (repo / "payload.txt").write_bytes(payload("old\n"))
    (repo / "local.txt").write_bytes(payload("pristine\n"))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "initial")
    target, _, name = cmd._install_plugin_core(repo.as_uri(), force=False, python_deps=False)
    # Existing inventory-bearing installs can use the normal Git updater too.
    metadata = cmd._read_install_metadata()
    metadata[name]["files"] = capture_install_inventory(target, target)
    cmd._write_install_metadata(metadata)
    if dirty:
        (target / "local.txt").write_bytes(payload("local edits\n"))
        (target / "private.txt").write_text("private untracked\n")
        if transform == "crlf":
            (target / ".gitattributes").write_text("*.txt text eol=lf\n")
    (repo / "payload.txt").write_bytes(payload("new\n"))
    (repo / "added.txt").write_bytes(payload("new asset\n"))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "new revision")
    revision = git(repo, "rev-parse", "HEAD")
    cmd._pull_plugin_update(target, lambda record: "pinned", lambda: "no git")
    updated = cmd._read_install_metadata()[name]
    assert updated["revision"] == revision
    assert updated["files"] != metadata[name]["files"]
    entries = updated["files"]["entries"]
    assert entries["payload.txt"]["sha256"] == hashlib.sha256(payload("new\n")).hexdigest()
    assert entries["local.txt"]["sha256"] == hashlib.sha256(payload("pristine\n")).hexdigest()
    assert "added.txt" in entries and "private.txt" not in entries
    # Reapplying locally edited attributes legitimately changes checkout EOL.
    expected_local = b"local edits\n" if dirty and transform == "crlf" else payload("local edits\n" if dirty else "pristine\n")
    assert (target / "local.txt").read_bytes() == expected_local
    if dirty:
        assert (target / "private.txt").read_text() == "private untracked\n"
        if transform == "crlf":
            assert (target / ".gitattributes").read_text() == "*.txt text eol=lf\n"
        with pytest.raises(PluginInventoryError):
            validate_install_inventory(target, updated["files"])
    else:
        validate_install_inventory(target, updated["files"])


@pytest.mark.parametrize("subdir", ["", "packages/selected"])
def test_catalog_checkout_transformations_survive_repeated_profile_copies(tmp_path, monkeypatch, subdir):
    from hermes_cli import plugins_cmd as cmd, plugins_cmd_catalog as catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry
    from hermes_cli.plugin_installation import plugin_installation_lock
    from hermes_cli.plugin_inventory import capture_install_inventory
    from hermes_cli.profile_plugins import copy_profile_plugins

    home, repo = tmp_path / "A", tmp_path / "repo"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    package = repo / subdir
    package.mkdir(parents=True)
    (repo / ".gitattributes").write_text("*.txt text eol=crlf filter=fixture\n*.utf16 text eol=lf working-tree-encoding=UTF-16LE\n")
    (package / "plugin.yaml").write_text("name: transformed-package\n")
    (package / "payload.txt").write_bytes(b"pristine\r\n")
    (package / "asset.utf16").write_bytes("pristine\n".encode("utf-16-le"))
    (package / ".env.example").write_text("FIXTURE_KEY=\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "transformed payload")
    entry = PluginCatalogEntry(name="transformed-catalog", repo=repo.as_uri(),
                               sha=git(repo, "rev-parse", "HEAD"), subdir=subdir,
                               description="Local fixture", maintainer="Fixture")
    def forbidden(*args, **kwargs):
        raise AssertionError("Dependency actions are forbidden")
    monkeypatch.setattr(cmd, "_refuse_conflicting_python_deps", forbidden)
    monkeypatch.setattr(cmd, "_install_python_dependencies", forbidden)
    target, _, name = catalog.install_catalog_entry(entry, force=False, python_deps=False)
    metadata = cmd._read_install_metadata()
    entries = metadata[name]["files"]["entries"]
    if not subdir:
        # Inventory conversion must never execute configured process drivers.
        import shlex
        import sys
        marker = tmp_path / "driver-ran"
        script = tmp_path / "driver.py"
        script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(1)\n")
        command = f'"{sys.executable}" "{script}"' if os.name == "nt" else shlex.join([sys.executable, str(script)])
        for kind in ("smudge", "process"):
            git(target, "config", f"filter.fixture.{kind}", command)
        git(target, "config", "filter.fixture.required", "true")
        assert capture_install_inventory(target, target) == metadata[name]["files"]
        assert not marker.exists()
    assert entries["payload.txt"]["sha256"] == hashlib.sha256(b"pristine\r\n").hexdigest()
    assert entries["asset.utf16"]["sha256"] == hashlib.sha256("pristine\n".encode("utf-16-le")).hexdigest()
    assert (target / ".env").exists() and ".env" not in entries
    source = home
    for destination in (tmp_path / "B", tmp_path / "C"):
        destination.mkdir()
        with plugin_installation_lock(source):
            assert copy_profile_plugins(source, destination)["copied"] == [name]
        copied = destination / "plugins" / name
        assert (copied / "payload.txt").read_bytes() == b"pristine\r\n"
        assert (copied / "asset.utf16").read_bytes() == "pristine\n".encode("utf-16-le")
        assert not (copied / ".env").exists()
        assert json.loads((destination / "plugins" / ".install-metadata.json").read_text()) == metadata
        source = destination


@pytest.mark.parametrize("bad", ["symlink", "submodule", "reserved", "modified", "outside"])
def test_capture_rejects_non_pristine_or_non_regular_selected_payload(tmp_path, bad):
    from hermes_cli.plugin_inventory import capture_install_inventory, PluginInventoryError
    repo = tmp_path / "repo"
    package = repo / "selected"
    package.mkdir(parents=True)
    git(repo, "init", "-q")
    (package / "asset.txt").write_text("pristine\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "initial")
    if bad == "symlink":
        (package / "link").symlink_to(tmp_path)
        git(repo, "add", ".")
    elif bad == "submodule":
        git(repo, "update-index", "--add", "--cacheinfo", "160000," + git(repo, "rev-parse", "HEAD") + ",selected/submodule")
    elif bad == "reserved":
        (package / ".hermes-catalog.json").write_text("{}")
        git(repo, "add", ".")
    elif bad == "modified":
        (package / "asset.txt").write_text("dirty\n")
    else:
        package = tmp_path
    if bad in {"symlink", "submodule", "reserved"}:
        git(repo, "commit", "-qm", "unsafe")
    with pytest.raises((PluginInventoryError, ValueError)):
        capture_install_inventory(repo, package)


@pytest.mark.parametrize("attributes,tag", [(0x400, 0), (0, 0xA0000003)])
def test_checked_stat_rejects_reparse_metadata_without_pathlib_helper(tmp_path, monkeypatch, attributes, tag):
    from types import SimpleNamespace
    import stat
    from hermes_cli.plugin_inventory import checked_stat, PluginInventoryError

    # Stat records are data: no emulation of the interpreter's host OS.
    info = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755,
                           st_file_attributes=attributes, st_reparse_tag=tag)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "lstat", lambda self: info)
        patch.delattr(Path, "is_junction", raising=False)
        with pytest.raises(PluginInventoryError, match="(?i)(reparse|junction)"):
            checked_stat(tmp_path)


@pytest.mark.linux_only
def test_linux_copy_preserves_git_modes_and_rejects_permission_drift(tmp_path):
    _check_posix_copy_modes(tmp_path)


@pytest.mark.macos_only
def test_macos_copy_preserves_git_modes_and_rejects_permission_drift(tmp_path):
    _check_posix_copy_modes(tmp_path)


def _check_posix_copy_modes(tmp_path):
    from hermes_cli.plugin_inventory import (
        capture_install_inventory, copy_install_inventory, validate_install_inventory, PluginInventoryError,
    )
    import stat

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for name, mode in (("regular", 0o644), ("executable", 0o755)):
        (repo / name).write_bytes(b"fixture bytes\n")
        (repo / name).chmod(mode)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "payload")
    inventory = capture_install_inventory(repo, repo)
    destination = tmp_path / "copy"
    copy_install_inventory(repo, destination, inventory)
    for name, record in inventory["entries"].items():
        assert stat.S_IMODE((destination / name).stat().st_mode) == record["mode"]
        assert (destination / name).read_bytes() == (repo / name).read_bytes()
    for mode in (0o600, 0o666, 0o755):
        (repo / "regular").chmod(mode)
        with pytest.raises(PluginInventoryError, match="mode"):
            validate_install_inventory(repo, inventory)
    (repo / "regular").chmod(0o644)
    for name, invalid_modes in (("regular", (0o400, 0o200, 0o755, 0o610, 0o4600)),
                                 ("executable", (0o600, 0o644, 0o4700))):
        for mode in invalid_modes:
            (repo / name).chmod(mode)
            with pytest.raises(PluginInventoryError, match="mode"):
                capture_install_inventory(repo, repo)
        (repo / name).chmod(inventory["entries"][name]["mode"])
    for invalid in (0, 0o400, 0o200, 0o610, 0o4644, 0o2755, 0o1755, -1, True):
        malformed = {"version": 1, "entries": {"regular": {
            **inventory["entries"]["regular"], "mode": invalid}}}
        with pytest.raises(PluginInventoryError, match="mode"):
            copy_install_inventory(repo, tmp_path / "invalid-copy", malformed)
        assert not (tmp_path / "invalid-copy").exists()


@pytest.mark.linux_only
@pytest.mark.parametrize("mask", [0o022, 0o077])
@pytest.mark.parametrize("subdir", ["", "packages/selected"])
def test_linux_catalog_checkout_umask_preserves_permissions(tmp_path, mask, subdir):
    _run_checkout_umask_probe(tmp_path, mask, subdir)


@pytest.mark.macos_only
@pytest.mark.parametrize("mask", [0o022, 0o077])
@pytest.mark.parametrize("subdir", ["", "packages/selected"])
def test_macos_catalog_checkout_umask_preserves_permissions(tmp_path, mask, subdir):
    _run_checkout_umask_probe(tmp_path, mask, subdir)


def _run_checkout_umask_probe(tmp_path, mask, subdir):
    import sys

    # Set umask only in a disposable child, never in the threaded test host.
    result = subprocess.run(
        [sys.executable, "-c",
         "import runpy, sys; from pathlib import Path; "
         "runpy.run_path(sys.argv[1])['_check_checkout_umask'](Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4])",
         str(Path(__file__).resolve()), str(tmp_path), str(mask), subdir],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "A"),
             "GIT_ALLOW_PROTOCOL": "file"})
    assert result.returncode == 0, result.stdout + result.stderr


def _check_checkout_umask(tmp_path, mask, subdir):
    import stat
    from dataclasses import replace
    from hermes_cli import plugins_cmd as cmd, plugins_cmd_catalog as catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry
    from hermes_cli.plugin_inventory import validate_install_inventory, PluginInventoryError
    from hermes_cli.plugin_installation import plugin_installation_lock
    from hermes_cli.profile_plugins import copy_profile_plugins, PluginSnapshotError

    os.umask(mask)
    home, repo = tmp_path / "A", tmp_path / "repo"
    package = repo / subdir
    package.mkdir(parents=True)
    (package / "plugin.yaml").write_text("name: umask-package\n")
    (package / "regular").write_bytes(b"original\n")
    (package / "executable").write_bytes(b"executable fixture\n")
    (package / "executable").chmod(0o777 & ~mask)
    git(repo, "init", "-b", "main")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "fixture")
    entry = PluginCatalogEntry(name="umask-catalog", repo=repo.as_uri(),
                               sha=git(repo, "rev-parse", "HEAD"), subdir=subdir,
                               description="Local fixture", maintainer="Fixture")
    def forbidden(*args, **kwargs):
        raise AssertionError("Dependency operations forbidden")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cmd, "_refuse_conflicting_python_deps", forbidden)
        patch.setattr(cmd, "_install_python_dependencies", forbidden)
        target, _, name = catalog.install_catalog_entry(entry, force=False, python_deps=False)
        for generation in range(2):
            if generation:
                (package / "regular").write_bytes(b"updated\n")
                git(repo, "add", ".")
                git(repo, "commit", "-qm", "update")
                if subdir:
                    catalog.install_catalog_entry(replace(entry, sha=git(repo, "rev-parse", "HEAD")),
                                                  force=True, python_deps=False)
                else:
                    metadata = cmd._read_install_metadata()
                    metadata[name]["pinned"] = False
                    cmd._write_install_metadata(metadata)
                    original_mode = stat.S_IMODE((target / "regular").stat().st_mode)
                    (target / "regular").chmod(original_mode ^ 0o040)
                    with pytest.raises(cmd.PluginOperationError, match="mode"):
                        cmd._pull_plugin_update(target, lambda record: "pinned", lambda: "no git")
                    assert cmd._read_install_metadata() == metadata
                    assert (target / "regular").read_bytes() == b"original\n"
                    (target / "regular").chmod(original_mode)
                    pristine_executable = (target / "executable").read_bytes()
                    (target / "executable").write_bytes(b"local edit\n")
                    cmd._pull_plugin_update(target, lambda record: "pinned", lambda: "no git")
                    refreshed = cmd._read_install_metadata()[name]["files"]
                    assert isinstance(refreshed, dict)
                    assert (target / "executable").read_bytes() == b"local edit\n"
                    with pytest.raises(PluginInventoryError, match="file differs"):
                        validate_install_inventory(target, refreshed)
                    (target / "executable").write_bytes(pristine_executable)
            inventory = cmd._read_install_metadata()[name]["files"]
            assert isinstance(inventory, dict)
            for filename, base_mode in (("regular", 0o666), ("executable", 0o777)):
                expected = base_mode & ~mask
                assert inventory["entries"][filename]["mode"] == expected
                assert stat.S_IMODE((target / filename).stat().st_mode) == expected
            validate_install_inventory(target, inventory)
            # The ordinary updater has no catalog re-pin; retain fixture identity
            # for the package-only copier after that direct update.
            if generation and not subdir:
                catalog.write_catalog_sidecar(target, replace(entry, sha=git(repo, "rev-parse", "HEAD")))
            destination = tmp_path / f"copy-{generation}"
            destination.mkdir()
            with plugin_installation_lock(home):
                assert copy_profile_plugins(home, destination)["copied"] == [name]
            copied = destination / "plugins" / name
            validate_install_inventory(copied, inventory)
            assert (copied / "regular").read_bytes() == (package / "regular").read_bytes()
            for filename, changed_modes in (("regular", (0o600, 0o644, 0o660, 0o755, 0o4600)),
                                             ("executable", (0o600, 0o700, 0o755, 0o770, 0o4700))):
                original = inventory["entries"][filename]["mode"]
                for changed in changed_modes:
                    if changed == original:
                        continue
                    (target / filename).chmod(changed)
                    actual = stat.S_IMODE((target / filename).stat().st_mode)
                    assert actual != original, (filename, oct(changed), oct(actual))
                    with pytest.raises(PluginInventoryError, match="mode"):
                        validate_install_inventory(target, inventory)
                    refused = tmp_path / f"refused-{generation}-{filename}-{changed}"
                    refused.mkdir()
                    with plugin_installation_lock(home), pytest.raises(PluginSnapshotError):
                        copy_profile_plugins(home, refused)
                    assert not (refused / "plugins").exists()
                (target / filename).chmod(original)


@pytest.mark.windows_only
def test_windows_install_capture_rejects_real_junction_without_pathlib_helper(tmp_path, monkeypatch):
    import shutil
    from hermes_cli.plugin_inventory import capture_install_inventory, PluginInventoryError

    repo = tmp_path / "repo"
    selected = repo / "selected"
    selected.mkdir(parents=True)
    (selected / "payload").write_bytes(b"pristine\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "payload")
    outside = tmp_path / "outside"
    shutil.move(str(selected), str(outside))
    subprocess.run(["cmd", "/c", "mklink", "/J", str(selected), str(outside)],
                   check=True, capture_output=True)
    monkeypatch.delattr(Path, "is_junction", raising=False)
    try:
        with pytest.raises(PluginInventoryError, match="reparse"):
            capture_install_inventory(repo, selected)
        assert (outside / "payload").read_bytes() == b"pristine\n"
    finally:
        selected.rmdir()
