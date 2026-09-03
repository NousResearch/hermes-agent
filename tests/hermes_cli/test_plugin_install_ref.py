"""Exact-commit plugin installation and source metadata."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from hermes_cli.plugins import SUPPORTED_MANIFEST_VERSION
from hermes_cli.subcommands.plugins import build_plugins_parser


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str, marker: str) -> str:
    (repo / "marker.txt").write_text(marker, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _plugin_repo(root: Path, name: str = "demo") -> tuple[Path, str, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture")
    (repo / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "1.0.0"}), encoding="utf-8"
    )
    old_sha = _commit(repo, "old", "old")
    new_sha = _commit(repo, "new", "new")
    return repo, old_sha, new_sha


def _versioned_plugin_repo(root: Path, manifest_version: int) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture")
    (repo / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "versioned-demo",
                "version": "1.0.0",
                "manifest_version": manifest_version,
            }
        ),
        encoding="utf-8",
    )
    (repo / "__init__.py").write_text(
        "def register(ctx):\n    pass\n", encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "versioned plugin")
    return repo, _git(repo, "rev-parse", "HEAD")


def _commit_manifest_version(repo: Path, manifest_version: int) -> str:
    manifest = yaml.safe_load((repo / "plugin.yaml").read_text(encoding="utf-8"))
    manifest["manifest_version"] = manifest_version
    (repo / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    _git(repo, "add", "plugin.yaml")
    _git(repo, "commit", "-qm", f"manifest v{manifest_version}")
    return _git(repo, "rev-parse", "HEAD")


def _metadata(home: Path) -> dict:
    return json.loads((home / "plugins" / ".install-metadata.json").read_text())


def test_parser_accepts_only_explicit_install_ref_option():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_plugins_parser(subparsers, cmd_plugins=lambda _args: None)

    args = parser.parse_args(["plugins", "install", "owner/repo", "--ref", "a" * 40])

    assert args.ref == "a" * 40


def test_canonical_source_never_persists_http_credentials():
    from hermes_cli.plugins_cmd import _canonical_source

    assert (
        _canonical_source("https://user:token@example.com/owner/repo.git", None)
        == "https://example.com/owner/repo.git"
    )
    assert (
        _canonical_source("https://example.com/owner/repo.git?token=secret", None)
        == "https://example.com/owner/repo.git"
    )


def test_cloned_origin_never_persists_http_credentials(tmp_path):
    from hermes_cli.plugins_cmd import _scrub_cloned_origin

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://user:secret@example.com/owner/repo.git?token=secret",
    )

    _scrub_cloned_origin(
        repo,
        "git",
        "https://user:secret@example.com/owner/repo.git?token=secret",
    )

    assert _git(repo, "remote", "get-url", "origin") == (
        "https://example.com/owner/repo.git"
    )
    assert "secret" not in (repo / ".git" / "config").read_text(encoding="utf-8")


def test_git_errors_never_echo_source_credentials():
    from hermes_cli.plugins_cmd import _safe_git_error

    source = "https://user:secret@example.com/owner/repo.git?token=secret"
    result = subprocess.CompletedProcess(
        args=["git", "clone"],
        returncode=1,
        stdout="",
        stderr=f"fatal: unable to access '{source}': connection failed",
    )

    error = _safe_git_error(result, source)

    assert "secret" not in error
    assert "user:" not in error
    assert "https://example.com/owner/repo.git" in error


def test_exact_ref_installs_old_commit_and_normalizes_uppercase(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo, old_sha, new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    target, _manifest, name = _install_plugin_core(
        repo.as_uri(), force=False, ref=old_sha.upper()
    )

    assert name == "demo"
    assert _git(target, "rev-parse", "HEAD") == old_sha
    assert old_sha != new_sha
    assert (target / "marker.txt").read_text() == "old"
    assert _metadata(home) == {
        "demo": {"pinned": True, "revision": old_sha, "source": repo.as_uri()}
    }


@pytest.mark.parametrize("ref", ["", "main", "abc", "g" * 40, "a" * 39, "a" * 41])
def test_invalid_ref_is_rejected_before_any_install_state(monkeypatch, tmp_path, ref):
    from hermes_cli.plugins_cmd import PluginOperationError, _install_plugin_core

    repo, _old_sha, _new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    with pytest.raises(PluginOperationError, match="40-character commit SHA"):
        _install_plugin_core(repo.as_uri(), force=False, ref=ref)

    assert not (home / "plugins" / "demo").exists()
    assert not (home / "plugins" / ".install-metadata.json").exists()


def test_subdir_pin_records_source_identity_and_installs_requested_tree(
    monkeypatch, tmp_path
):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo = tmp_path / "monorepo"
    plugin = repo / "extensions" / "demo"
    plugin.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "Fixture")
    (plugin / "plugin.yaml").write_text("name: nested-demo\n", encoding="utf-8")
    (plugin / "value.txt").write_text("old", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "old")
    old_sha = _git(repo, "rev-parse", "HEAD")
    (plugin / "value.txt").write_text("new", encoding="utf-8")
    _git(repo, "commit", "-qam", "new")
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    identifier = f"{repo.as_uri()}#extensions/demo"

    target, _manifest, _name = _install_plugin_core(
        identifier, force=False, ref=old_sha
    )

    assert (target / "value.txt").read_text() == "old"
    assert _metadata(home)["nested-demo"] == {
        "pinned": True,
        "revision": old_sha,
        "source": identifier,
    }


def test_force_reinstall_does_not_drift_pin_without_explicit_new_ref(
    monkeypatch, tmp_path
):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo, old_sha, new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_plugin_core(repo.as_uri(), force=False, ref=old_sha)

    target, _manifest, _name = _install_plugin_core(repo.as_uri(), force=True)
    assert _git(target, "rev-parse", "HEAD") == old_sha
    assert _metadata(home)["demo"]["pinned"] is True

    target, _manifest, _name = _install_plugin_core(
        repo.as_uri(), force=True, ref=new_sha
    )
    assert _git(target, "rev-parse", "HEAD") == new_sha
    assert _metadata(home)["demo"]["revision"] == new_sha


def test_unpinned_install_and_force_reinstall_keep_tracking_head(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo, _old_sha, first_head = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    target, _manifest, _name = _install_plugin_core(repo.as_uri(), force=False)
    assert _git(target, "rev-parse", "HEAD") == first_head
    assert _metadata(home)["demo"]["pinned"] is False

    next_head = _commit(repo, "later", "later")
    target, _manifest, _name = _install_plugin_core(repo.as_uri(), force=True)
    assert _git(target, "rev-parse", "HEAD") == next_head
    assert _metadata(home)["demo"]["revision"] == next_head
    assert _metadata(home)["demo"]["pinned"] is False


def test_metadata_is_profile_local_and_read_from_disk_each_time(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import _install_plugin_core, _read_install_metadata

    repo, old_sha, _new_sha = _plugin_repo(tmp_path)
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    _install_plugin_core(repo.as_uri(), force=False, ref=old_sha)
    assert _read_install_metadata()["demo"]["revision"] == old_sha

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert _read_install_metadata() == {}

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert _read_install_metadata()["demo"]["source"] == repo.as_uri()


def test_pinned_plugin_update_refuses_to_drift(monkeypatch, tmp_path, capsys):
    from hermes_cli.plugins_cmd import _install_plugin_core, cmd_update

    repo, old_sha, _new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_plugin_core(repo.as_uri(), force=False, ref=old_sha)

    with pytest.raises(SystemExit) as exc:
        cmd_update("demo")

    assert exc.value.code == 1
    assert "pinned" in capsys.readouterr().out.lower()
    assert _git(home / "plugins" / "demo", "rev-parse", "HEAD") == old_sha


def test_dashboard_update_also_refuses_to_drift_pin(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import (
        _install_plugin_core,
        dashboard_update_user_plugin,
    )

    repo, old_sha, _new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_plugin_core(repo.as_uri(), force=False, ref=old_sha)

    result = dashboard_update_user_plugin("demo")

    assert result["ok"] is False
    assert "pinned" in result["error"]
    assert _git(home / "plugins" / "demo", "rev-parse", "HEAD") == old_sha


def test_failed_force_reinstall_keeps_existing_plugin_and_metadata(
    monkeypatch, tmp_path
):
    from hermes_cli.plugins_cmd import PluginOperationError, _install_plugin_core

    repo, old_sha, _new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target, _manifest, _name = _install_plugin_core(
        repo.as_uri(), force=False, ref=old_sha
    )
    before = _metadata(home)

    with pytest.raises(PluginOperationError):
        _install_plugin_core(repo.as_uri(), force=True, ref="f" * 40)

    assert target.exists()
    assert _git(target, "rev-parse", "HEAD") == old_sha
    assert _metadata(home) == before


def test_checkout_mismatch_is_rejected(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import PluginOperationError, _checkout_exact_revision

    repo, old_sha, new_sha = _plugin_repo(tmp_path)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", repo.as_uri(), str(clone)], check=True)
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._git_head_revision", lambda _repo, _git: new_sha
    )

    with pytest.raises(PluginOperationError, match="does not match requested"):
        _checkout_exact_revision(clone, "git", old_sha)


def test_metadata_write_failure_rolls_back_new_install(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo, old_sha, _new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._write_install_metadata",
        lambda _metadata: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        _install_plugin_core(repo.as_uri(), force=False, ref=old_sha)

    assert not (home / "plugins" / "demo").exists()
    assert not (home / "plugins" / ".install-metadata.json").exists()


def test_metadata_write_failure_rolls_back_removal(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import _install_plugin_core, _remove_plugin_core

    repo, old_sha, _new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target, _manifest, _name = _install_plugin_core(
        repo.as_uri(), force=False, ref=old_sha
    )
    before = _metadata(home)
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._write_install_metadata",
        lambda _metadata: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        _remove_plugin_core(target)

    assert target.exists()
    assert _git(target, "rev-parse", "HEAD") == old_sha
    assert _metadata(home) == before
    assert list(target.parent.glob(".demo.remove-*")) == []


def test_reinstall_after_manual_directory_removal_retains_pin(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo, old_sha, _new_sha = _plugin_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target, _manifest, _name = _install_plugin_core(
        repo.as_uri(), force=False, ref=old_sha
    )
    shutil.rmtree(target)

    target, _manifest, _name = _install_plugin_core(repo.as_uri(), force=False)

    assert _git(target, "rev-parse", "HEAD") == old_sha
    assert _metadata(home)["demo"]["pinned"] is True


@pytest.mark.parametrize("manifest_version", [1, SUPPORTED_MANIFEST_VERSION])
def test_exact_ref_installs_runtime_supported_manifest_versions(
    monkeypatch, tmp_path, manifest_version
):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo, revision = _versioned_plugin_repo(tmp_path, manifest_version)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    target, manifest, name = _install_plugin_core(
        repo.as_uri(), force=False, ref=revision
    )

    assert target == (home / "plugins" / "versioned-demo").resolve()
    assert name == "versioned-demo"
    assert manifest["manifest_version"] == manifest_version
    assert _git(target, "rev-parse", "HEAD") == revision


def test_force_reinstall_accepts_v2_manifest(monkeypatch, tmp_path):
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo, v1_revision = _versioned_plugin_repo(tmp_path, 1)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _install_plugin_core(repo.as_uri(), force=False, ref=v1_revision)
    v2_revision = _commit_manifest_version(repo, 2)

    target, manifest, _name = _install_plugin_core(
        repo.as_uri(), force=True, ref=v2_revision
    )

    assert manifest["manifest_version"] == 2
    assert _git(target, "rev-parse", "HEAD") == v2_revision
    assert _metadata(home)["versioned-demo"]["revision"] == v2_revision


def test_future_force_reinstall_preserves_installed_plugin(monkeypatch, tmp_path):
    from hermes_cli.plugins import SUPPORTED_MANIFEST_VERSION
    from hermes_cli.plugins_cmd import PluginOperationError, _install_plugin_core

    repo, supported_revision = _versioned_plugin_repo(
        tmp_path, SUPPORTED_MANIFEST_VERSION
    )
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target, _manifest, _name = _install_plugin_core(
        repo.as_uri(), force=False, ref=supported_revision
    )
    future_revision = _commit_manifest_version(
        repo, SUPPORTED_MANIFEST_VERSION + 1
    )
    before = _metadata(home)

    with pytest.raises(PluginOperationError, match="requires manifest_version"):
        _install_plugin_core(repo.as_uri(), force=True, ref=future_revision)

    assert _git(target, "rev-parse", "HEAD") == supported_revision
    assert _metadata(home) == before


def test_installed_v2_plugin_passes_runtime_doctor(monkeypatch, tmp_path):
    from hermes_cli.plugin_dev import doctor_plugin
    from hermes_cli.plugins_cmd import _install_plugin_core

    repo, revision = _versioned_plugin_repo(tmp_path, 2)
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    target, _manifest, _name = _install_plugin_core(
        repo.as_uri(), force=False, ref=revision
    )

    report = doctor_plugin(target)

    assert report.ok
    assert report.manifest.manifest_version == 2


def test_exact_ref_rejects_future_native_manifest(monkeypatch, tmp_path):
    from hermes_cli.plugins import SUPPORTED_MANIFEST_VERSION
    from hermes_cli.plugins_cmd import PluginOperationError, _install_plugin_core

    repo, revision = _versioned_plugin_repo(
        tmp_path, SUPPORTED_MANIFEST_VERSION + 1
    )
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    with pytest.raises(PluginOperationError, match="requires manifest_version"):
        _install_plugin_core(repo.as_uri(), force=False, ref=revision)

    assert not (home / "plugins" / "versioned-demo").exists()
    assert not (home / "plugins" / ".install-metadata.json").exists()
