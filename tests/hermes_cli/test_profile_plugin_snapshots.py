"""Verified package snapshots are code-only, offline and profile-independent."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextlib.contextmanager
def scoped(home):
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    token = set_hermes_home_override(home)
    secret_token = set_secret_scope(build_profile_secret_scope(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(token)


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, env={**os.environ, "GIT_AUTHOR_NAME": "Snapshot Test",
                        "GIT_AUTHOR_EMAIL": "snapshot@example.invalid",
                        "GIT_COMMITTER_NAME": "Snapshot Test",
                        "GIT_COMMITTER_EMAIL": "snapshot@example.invalid"},
    ).stdout.strip()


PROVIDER = '''from pathlib import Path
from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home
from .value import VALUE
class Provider(MemoryProvider):
    name = "snapshot_provider"
    def is_available(self): return True
    def initialize(self, session_id, **kwargs): pass
    def get_tool_schemas(self): return []
    def prefetch(self, query, *, session_id=""):
        return VALUE + ":" + str(get_hermes_home())
'''


def installed(tmp_path, monkeypatch, *, general=False):
    from hermes_cli.plugins_cmd import _install_plugin_core
    home = tmp_path / "A"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    repo = tmp_path / "upstream"
    repo.mkdir()
    (repo / "__init__.py").write_text(PROVIDER, encoding="utf-8")
    (repo / "value.py").write_text('VALUE = "original"\n', encoding="utf-8")
    (repo / ".env.example").write_text("PLUGIN_KEY=\n", encoding="utf-8")
    (repo / ".env.template").write_text("PLUGIN_KEY=\n", encoding="utf-8")
    (repo / ".gitignore").write_text(".env\n.venv/\n", encoding="utf-8")
    if general:
        (repo / "plugin.yaml").write_text("name: snapshot_provider\n", encoding="utf-8")
    git(repo, "init")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "fixture")
    revision = git(repo, "rev-parse", "HEAD")
    with scoped(home):
        target, _, _ = _install_plugin_core(repo.as_uri(), force=False, ref=revision,
                                            python_deps=False)
    assert (target / ".env").read_text() == "PLUGIN_KEY=\n"
    return home, target, revision


@pytest.mark.parametrize("layout", ["manifestless", "category", "general", "memory-category"])
@pytest.mark.parametrize("provenance", ["matching", "historical", "none"])
def test_verified_snapshot_preserves_package_without_activation(tmp_path, monkeypatch, layout, provenance):
    home, target, revision = installed(tmp_path, monkeypatch, general=layout != "manifestless")
    from hermes_cli.profile_plugins import copy_profile_plugins
    from hermes_cli.plugin_installation import plugin_installation_lock
    from plugins.memory import load_memory_provider

    import socket
    def no_network(*args, **kwargs):
        raise AssertionError("snapshot must stay offline")
    monkeypatch.setattr(socket.socket, "connect", no_network)

    root = home / "plugins"
    metadata = json.loads((root / ".install-metadata.json").read_text())
    name = target.name
    if layout in {"category", "memory-category"}:
        category = "memory" if layout == "memory-category" else "tools"
        nested = root / category / name
        nested.parent.mkdir()
        target.rename(nested)
        target = nested
        metadata[category + "/" + name] = metadata.pop(name)
        name = category + "/" + name
        (root / ".install-metadata.json").write_text(json.dumps(metadata))
    # Local runtime data is never package payload, even if .gitignore hides it.
    import venv
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(target / ".venv")
    (target / ".env").write_text("PRIVATE=source-only\n")
    (target / "__pycache__").mkdir()
    (target / "__pycache__" / "value.pyc").write_bytes(b"runtime")
    (target / ".git" / "hooks" / "pre-commit").write_text("must not run or copy")
    sidecar = {"catalog_name": "historical", "repo": "https://example.invalid/old.git",
               "sha": "f" * 40, "tier": "community", "installed_at": "2026-01-01T00:00:00Z"}
    if provenance == "matching":
        sidecar.update(repo=metadata[name]["source"], sha=revision)
    if provenance != "none":
        (target / ".hermes-catalog.json").write_text(json.dumps(sidecar))
    source_status = git(target, "status", "--porcelain", "--untracked-files=all")
    source_metadata = (root / ".install-metadata.json").read_bytes()
    before = set(sys.modules)
    staging = tmp_path / "B"
    staging.mkdir()
    with plugin_installation_lock(home):
        report = copy_profile_plugins(home, staging)
    dest = staging / "plugins" / name
    assert set(report["copied"]) == {name}
    assert not (dest / ".git").exists()
    assert not (dest / ".env").exists()
    assert (dest / ".env.example").read_bytes() == (target / ".env.example").read_bytes()
    assert (dest / ".env.template").read_bytes() == (target / ".env.template").read_bytes()
    assert not (dest / "__pycache__").exists()
    assert not (dest / ".venv").exists()
    assert not (staging / "config.yaml").exists()
    assert not any(x.startswith("_hermes_user_memory") for x in set(sys.modules) - before)
    record = json.loads((staging / "plugins" / ".install-metadata.json").read_text())[name]
    assert record["revision"] == revision
    assert record["source"] == metadata[name]["source"]
    if provenance == "historical":
        assert record["catalog_provenance"] == sidecar
    else:
        assert "catalog_provenance" not in record
    if provenance == "matching":
        assert json.loads((dest / ".hermes-catalog.json").read_text()) == sidecar
    else:
        assert not (dest / ".hermes-catalog.json").exists()
    assert report["warnings"] and "reinstall" in " ".join(report["warnings"]).lower()
    assert "approval" not in " ".join(report["warnings"]).lower()
    third = tmp_path / "C"
    third.mkdir()
    with plugin_installation_lock(staging):
        repeated = copy_profile_plugins(staging, third)
    assert repeated["copied"] == report["copied"]
    assert json.loads((third / "plugins" / ".install-metadata.json").read_text())[name] == record
    for path in dest.rglob("*"):
        if path.is_file():
            assert (third / "plugins" / name / path.relative_to(dest)).read_bytes() == path.read_bytes()
    assert git(target, "status", "--porcelain", "--untracked-files=all") == source_status
    assert (root / ".install-metadata.json").read_bytes() == source_metadata
    assert (target / ".env").read_text() == "PRIVATE=source-only\n"
    assert (target / ".venv" / "pyvenv.cfg").is_file()
    if layout == "manifestless":
        from agent import secret_scope
        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        modules = []
        for current in (home, staging, home):
            with scoped(current):
                provider = load_memory_provider(name, register_skills=False)
                assert provider is not None
                assert provider.prefetch("probe") == "original:" + str(current)
                modules.append(type(provider).__module__)
        assert modules[0] == modules[2] != modules[1]
        assert "__source_" in modules[0] and "__source_" in modules[1]
        # A snapshot remains loadable after its original installation is gone.
        import shutil
        shutil.rmtree(target)
        with scoped(third):
            provider = load_memory_provider(name, register_skills=False)
            assert provider is not None
            assert type(provider).__module__ not in modules
            assert provider.prefetch("probe") == "original:" + str(third)


@pytest.mark.parametrize("env_path", ["generated.env", "settings/local.env", "GENERATED.ENV"])
def test_tracked_env_suffix_refuses_snapshot(tmp_path, monkeypatch, env_path):
    from hermes_cli.profile_plugins import PluginSnapshotError, copy_profile_plugins
    from hermes_cli.plugin_installation import plugin_installation_lock

    home, target, _ = installed(tmp_path, monkeypatch)
    private = target / env_path
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("PRIVATE=source-only\n")
    git(target, "add", env_path)
    git(target, "commit", "-m", "tracked environment")
    metadata_path = home / "plugins" / ".install-metadata.json"
    records = json.loads(metadata_path.read_text())
    records[target.name]["revision"] = git(target, "rev-parse", "HEAD")
    metadata_path.write_text(json.dumps(records))
    before = metadata_path.read_bytes()
    staging = tmp_path / "B"
    staging.mkdir()

    with plugin_installation_lock(home), pytest.raises(
        PluginSnapshotError, match="tracked runtime/secret-like payload"
    ):
        copy_profile_plugins(home, staging)
    assert not (staging / "plugins").exists()
    assert metadata_path.read_bytes() == before
    assert private.read_text() == "PRIVATE=source-only\n"


@pytest.mark.parametrize("env_path", ["generated.env", "settings/local.env", "GENERATED.ENV"])
@pytest.mark.parametrize("source_kind", ["git", "snapshot"])
def test_untracked_env_suffix_excluded_across_clone_generations(
    tmp_path, monkeypatch, env_path, source_kind
):
    from hermes_cli.profile_plugins import copy_profile_plugins
    from hermes_cli.plugin_installation import plugin_installation_lock

    home, target, _ = installed(tmp_path, monkeypatch)
    if source_kind == "snapshot":
        snapshot_home = tmp_path / "snapshot-source"
        snapshot_home.mkdir()
        with plugin_installation_lock(home):
            copy_profile_plugins(home, snapshot_home)
        home = snapshot_home
        target = home / "plugins" / target.name
    private = target / env_path
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("PRIVATE=source-only\n")
    metadata_before = (home / "plugins" / ".install-metadata.json").read_bytes()

    source = home
    manifests = []
    for generation in ("B", "C"):
        destination = tmp_path / generation
        destination.mkdir()
        with plugin_installation_lock(source):
            report = copy_profile_plugins(source, destination)
        package = destination / "plugins" / target.name
        assert report["copied"] == [target.name]
        assert not (package / env_path).exists()
        manifest = json.loads((package / ".hermes-snapshot.json").read_text())
        assert env_path not in manifest["files"]
        for template in (".env.example", ".env.template"):
            assert (package / template).read_bytes() == (target / template).read_bytes()
            assert template in manifest["files"]
        manifests.append(manifest)
        source = destination
    assert manifests[0] == manifests[1]
    assert (home / "plugins" / ".install-metadata.json").read_bytes() == metadata_before
    assert private.read_text() == "PRIVATE=source-only\n"


@pytest.mark.parametrize("failure", [
    "dirty", "mode", "unowned", "metadata", "revision", "tracked-runtime", "symlink",
    "root-symlink", "worktree", "shipped-removal", "cached-removal", "bad-kill-list",
    "mutation", "runtime-root-mutation", "inventory-mutation", "manifest-removal", "no-git", "no-metadata",
    "staged", "origin", "tracked-symlink", "excluded-symlink", "bad-sidecar",
    "nested-git", "tracked-venv", "bad-cache", "metadata-secret", "git-hardlink",
])
def test_unverifiable_snapshot_fails_closed(tmp_path, monkeypatch, failure):
    home, target, revision = installed(tmp_path, monkeypatch)
    from hermes_cli import profile_plugins as snapshots
    from hermes_cli.plugin_installation import plugin_installation_lock
    from hermes_cli import plugin_catalog
    root = home / "plugins"
    metadata_path = root / ".install-metadata.json"
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "removed.yaml").write_text("removed: []\n")
    monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: catalog)
    if failure == "dirty":
        (target / "value.py").write_text("VALUE = 'changed'\n")
    elif failure == "mode":
        (target / "value.py").chmod(0o755)
    elif failure == "unowned":
        (target / "unknown.txt").write_text("unowned")
    elif failure == "metadata":
        metadata_path.write_text("[]")
    elif failure == "revision":
        records = json.loads(metadata_path.read_text())
        records[target.name]["revision"] = "a" * 40
        metadata_path.write_text(json.dumps(records))
    elif failure == "tracked-runtime":
        (target / "state.db").write_bytes(b"private runtime")
        git(target, "add", "state.db")
        git(target, "commit", "-m", "runtime")
        records = json.loads(metadata_path.read_text())
        records[target.name]["revision"] = git(target, "rev-parse", "HEAD")
        metadata_path.write_text(json.dumps(records))
    elif failure == "symlink":
        (target / "escape").symlink_to(tmp_path)
    elif failure == "root-symlink":
        actual = home / "actual"
        root.rename(actual)
        root.symlink_to(actual, target_is_directory=True)
    elif failure == "worktree":
        import shutil
        shutil.rmtree(target / ".git")
        (target / ".git").write_text("gitdir: /untrusted\n")
    elif failure in {"shipped-removal", "cached-removal"}:
        removal = {"removed": [{"name": target.name, "reason": "security"}]}
        if failure == "shipped-removal":
            (catalog / "removed.yaml").write_text(json.dumps(removal))
        else:
            (home / "cache").mkdir()
            (home / "cache" / "plugin-catalog.json").write_text(json.dumps(removal))
    elif failure == "bad-kill-list":
        (catalog / "removed.yaml").write_text("removed: malformed\n")
    elif failure == "manifest-removal":
        (target / "plugin.yaml").write_text("name: blocked-declared-name\n")
        git(target, "add", "plugin.yaml")
        git(target, "commit", "-m", "manifest")
        records = json.loads(metadata_path.read_text())
        records[target.name]["revision"] = git(target, "rev-parse", "HEAD")
        metadata_path.write_text(json.dumps(records))
        (catalog / "removed.yaml").write_text("removed:\n- name: blocked-declared-name\n")
    elif failure == "no-git":
        import shutil
        shutil.rmtree(target / ".git")
    elif failure == "no-metadata":
        metadata_path.unlink()
    elif failure == "staged":
        (target / "value.py").write_text("VALUE = 'staged'\n")
        git(target, "add", "value.py")
    elif failure == "origin":
        git(target, "remote", "set-url", "origin", "https://example.invalid/other.git")
    elif failure in {"tracked-symlink", "excluded-symlink"}:
        link = target / ("escape" if failure == "tracked-symlink" else ".env")
        if link.exists():
            link.unlink()
        link.symlink_to(tmp_path)
        if failure == "tracked-symlink":
            git(target, "add", "escape")
    elif failure == "bad-sidecar":
        (target / ".hermes-catalog.json").write_text("[]")
    elif failure == "nested-git":
        nested = target / "vendor"
        nested.mkdir()
        (nested / "__init__.py").write_text("pass\n")
        git(target, "add", "vendor/__init__.py")
        git(target, "commit", "-m", "nested package")
        git(nested, "init")
        records = json.loads(metadata_path.read_text())
        records[target.name]["revision"] = git(target, "rev-parse", "HEAD")
        metadata_path.write_text(json.dumps(records))
    elif failure == "tracked-venv":
        (target / ".venv").mkdir()
        (target / ".venv" / "pyvenv.cfg").write_text("home = /private/runtime")
        git(target, "add", "-f", ".venv")
    elif failure == "bad-cache":
        (home / "cache").mkdir()
        (home / "cache" / "plugin-catalog.json").write_text("{}")
    elif failure == "metadata-secret":
        records = json.loads(metadata_path.read_text())
        records[target.name]["source"] = "https://user:secret@example.invalid/repo.git"
        metadata_path.write_text(json.dumps(records))
    elif failure == "git-hardlink":
        os.link(target / ".git" / "HEAD", tmp_path / "linked-head")
    else:
        if failure == "runtime-root-mutation":
            (target / ".venv").mkdir()
        original = snapshots._write_payload
        def mutate(*args, **kwargs):
            original(*args, **kwargs)
            if failure == "mutation":
                (target / "value.py").write_text("VALUE = 'racing'\n")
            elif failure == "runtime-root-mutation":
                (target / ".venv").rename(target / "venv")
            else:
                (root / "new-plugin").mkdir(exist_ok=True)
        monkeypatch.setattr(snapshots, "_write_payload", mutate)
    staging = tmp_path / "B"
    staging.mkdir()
    with plugin_installation_lock(home), pytest.raises(
        snapshots.PluginSnapshotError, match="(?i)(repair|reinstall|removed)"
    ):
        snapshots.copy_profile_plugins(home, staging)
    assert not (staging / "plugins").exists()


@pytest.mark.parametrize("change", [
    "bytes", "mode", "missing", "extra", "manifest", "manifest-mode", "manifest-hash", "record",
    "manifest-mutation", "payload-mutation", "removed", "sidecar",
])
def test_reclone_validates_snapshot_ownership(tmp_path, monkeypatch, change):
    from hermes_cli import profile_plugins as snapshots, plugin_catalog
    from hermes_cli.plugin_installation import plugin_installation_lock
    home, target, _ = installed(tmp_path, monkeypatch)
    second = tmp_path / "B"
    second.mkdir()
    with plugin_installation_lock(home):
        snapshots.copy_profile_plugins(home, second)
    package = second / "plugins" / target.name
    manifest = package / ".hermes-snapshot.json"
    record_path = second / "plugins" / ".install-metadata.json"
    if change == "bytes":
        (package / "value.py").write_text("changed")
    elif change == "mode":
        (package / "value.py").chmod(0o755)
    elif change == "missing":
        (package / "value.py").unlink()
    elif change == "extra":
        (package / "unknown.py").write_text("pass")
    elif change == "manifest":
        manifest.write_text("{}")
    elif change == "manifest-mode":
        manifest.chmod(0o755)
    elif change in {"record", "manifest-hash"}:
        records = json.loads(record_path.read_text())
        records[target.name]["revision" if change == "record" else "snapshot_sha256"] = "a" * (40 if change == "record" else 64)
        record_path.write_text(json.dumps(records))
    elif change == "sidecar":
        (package / ".hermes-catalog.json").write_text("{}")
    elif change == "removed":
        catalog = tmp_path / "catalog"
        catalog.mkdir()
        (catalog / "removed.yaml").write_text("removed:\n- name: " + target.name + "\n")
        monkeypatch.setattr(plugin_catalog, "get_catalog_dir", lambda: catalog)
    else:
        write = snapshots._write_payload
        def mutate(*args, **kwargs):
            write(*args, **kwargs)
            (manifest if change == "manifest-mutation" else package / "value.py").write_text("changed")
        monkeypatch.setattr(snapshots, "_write_payload", mutate)
    third = tmp_path / "C"
    third.mkdir()
    with plugin_installation_lock(second), pytest.raises(snapshots.PluginSnapshotError):
        snapshots.copy_profile_plugins(second, third)
    assert not (third / "plugins").exists()


def test_nested_repair_guidance_matches_flat_installer(tmp_path, monkeypatch):
    from hermes_cli import profile_plugins as snapshots
    from hermes_cli.plugin_installation import plugin_installation_lock
    from hermes_cli.plugins_cmd import _install_plugin_core
    home, target, revision = installed(tmp_path, monkeypatch)
    root = home / "plugins"
    nested = root / "tools" / target.name
    nested.parent.mkdir()
    target.rename(nested)
    record_path = root / ".install-metadata.json"
    records = json.loads(record_path.read_text())
    record = records.pop(target.name)
    key = "tools/" + target.name
    records[key] = record
    record_path.write_text(json.dumps(records))
    (nested / "value.py").write_text("local edits")
    second = tmp_path / "B"
    second.mkdir()
    with plugin_installation_lock(home), pytest.raises(snapshots.PluginSnapshotError) as error:
        snapshots.copy_profile_plugins(home, second)
    message = str(error.value)
    assert "flat" in message and "nested" in message and ".install-metadata.json" in message
    # Apply the documented steps, preserving the original private/editable tree.
    nested.rename(tmp_path / "backup")
    nested.parent.rmdir()
    records.pop(key)
    record_path.write_text(json.dumps(records))
    with scoped(home):
        restored, _, _ = _install_plugin_core(record["source"], force=True, ref=revision, python_deps=False)
    assert restored == target
    with plugin_installation_lock(home):
        snapshots.copy_profile_plugins(home, second)
    assert (tmp_path / "backup" / "value.py").read_text() == "local edits"


def test_subdirectory_repair_does_not_reuse_subdirectory_selector(tmp_path, monkeypatch):
    from hermes_cli import profile_plugins as snapshots
    from hermes_cli.plugin_installation import plugin_installation_lock
    from hermes_cli.plugins_cmd import _install_plugin_core

    home, target, _ = installed(tmp_path, monkeypatch)
    upstream = tmp_path / "upstream"
    nested = upstream / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("# subdirectory plugin\n")
    git(upstream, "add", "nested")
    git(upstream, "commit", "-m", "subdirectory plugin")
    revision = git(upstream, "rev-parse", "HEAD")
    with scoped(home):
        subdir, _, _ = _install_plugin_core(upstream.as_uri() + "#nested", force=False,
                                          ref=revision, python_deps=False)
    metadata_path = home / "plugins" / ".install-metadata.json"
    before = metadata_path.read_bytes()
    second = tmp_path / "B"
    second.mkdir()
    with plugin_installation_lock(home), pytest.raises(snapshots.PluginSnapshotError) as error:
        snapshots.copy_profile_plugins(home, second)
    message = str(error.value)
    assert "omit the #subdir" in message
    assert "repository root must itself" in message.lower()
    assert metadata_path.read_bytes() == before
    assert not (second / "plugins").exists()
    # Follow the repair only because this fixture also ships a root plugin.
    subdir.rename(tmp_path / "subdir-backup")
    records = json.loads(before)
    record = records.pop(subdir.name)
    metadata_path.write_text(json.dumps(records))
    with scoped(home):
        restored, _, _ = _install_plugin_core(record["source"].partition("#")[0],
                                            force=True, ref=revision, python_deps=False)
    assert restored == target
    with plugin_installation_lock(home):
        snapshots.copy_profile_plugins(home, second)
    assert (tmp_path / "subdir-backup" / "__init__.py").is_file()
