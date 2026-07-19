from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import plugins_cmd as pc
from hermes_cli.plugin_supply_chain import LOCK_FILENAME, read_provenance_lock
from hermes_cli.subcommands.plugins import build_plugins_parser


def _repo(path: Path, *, name: str = "pinned-plugin", valid_manifest: bool = True) -> str:
    path.mkdir()
    if valid_manifest:
        (path / "plugin.yaml").write_text(
            f"name: {name}\nmanifest_version: 1\nhooks: [pre_tool_call]\nprovides_tools: [demo]\n"
        )
    else:
        (path / "plugin.yaml").write_text(": malformed [[")
    (path / "config.yaml.example").write_text("ready: true\n")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one"],
        cwd=path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, capture_output=True
    ).stdout.strip()


def _install(tmp_path: Path, monkeypatch, repo: Path, **kwargs):
    plugins = tmp_path / "plugins"
    plugins.mkdir(exist_ok=True)
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    return pc._install_plugin_core(f"file://{repo}", force=False, **kwargs)


def test_invalid_ref_fails_before_any_git_or_state_change(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    old = plugins / "pinned-plugin"
    old.mkdir()
    (old / "old").write_text("preserved")
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    with patch.object(pc, "_resolve_git_executable") as resolve_git:
        with pytest.raises(pc.PluginOperationError, match="40 lowercase"):
            pc._install_plugin_core("owner/repo", force=True, requested_ref="BAD")
    resolve_git.assert_not_called()
    assert (old / "old").read_text() == "preserved"


def test_exact_ref_checkout_writes_lock_and_capability_report(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    requested = _repo(repo)
    (repo / "plugin.yaml").write_text("name: later\nmanifest_version: 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "two"], cwd=repo, check=True)

    result = _install(tmp_path, monkeypatch, repo, requested_ref=requested)

    assert result.target.name == "pinned-plugin"
    assert result.provenance.resolved_commit == requested
    assert result.provenance.requested_ref == requested
    assert read_provenance_lock(result.target) == result.provenance
    assert result.capabilities.tools == ("demo",)
    assert result.capabilities.warnings == ("CAPABILITY_REPORT_IS_NOT_SECURITY_AUDIT",)
    assert (result.target / "config.yaml").exists()


def test_no_ref_records_resolved_head(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    head = _repo(repo)
    result = _install(tmp_path, monkeypatch, repo)
    assert result.provenance.resolved_commit == head
    assert result.provenance.requested_ref is None


def test_rev_parse_mismatch_leaves_no_target(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    requested = _repo(repo)
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    real_run = pc._run_inspect_git

    def mismatch(command, *, operation, cwd=None):
        if operation == "resolve commit":
            return "0" * 40
        return real_run(command, operation=operation, cwd=cwd)

    monkeypatch.setattr(pc, "_run_inspect_git", mismatch)
    with pytest.raises(pc.PluginOperationError, match="exactly match"):
        pc._install_plugin_core(f"file://{repo}", force=False, requested_ref=requested)
    assert not (plugins / "pinned-plugin").exists()


def test_manifest_failure_leaves_no_target(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _repo(repo, valid_manifest=False)
    with pytest.raises(pc.PluginOperationError, match="manifest"):
        _install(tmp_path, monkeypatch, repo)
    assert not (tmp_path / "plugins" / "pinned-plugin").exists()


def test_lock_write_failure_preserves_force_replacement(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _repo(repo)
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    old = plugins / "pinned-plugin"
    old.mkdir()
    (old / "old").write_text("yes")
    (old / LOCK_FILENAME).write_text("old-lock")
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    monkeypatch.setattr(pc, "write_provenance_lock", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(pc.PluginOperationError, match="provenance lock"):
        pc._install_plugin_core(f"file://{repo}", force=True)
    assert (old / "old").read_text() == "yes"
    assert (old / LOCK_FILENAME).read_text() == "old-lock"


def test_lock_write_failure_leaves_no_new_target(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _repo(repo)
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    monkeypatch.setattr(pc, "write_provenance_lock", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(pc.PluginOperationError, match="provenance lock"):
        pc._install_plugin_core(f"file://{repo}", force=False)
    assert not (plugins / "pinned-plugin").exists()


def test_publication_failure_restores_force_replacement(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _repo(repo)
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    old = plugins / "pinned-plugin"
    old.mkdir()
    (old / "old").write_text("yes")
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    real_replace = pc.os.replace
    calls = 0

    def fail_publish(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("rename failed")
        return real_replace(src, dst)

    monkeypatch.setattr(pc.os, "replace", fail_publish)
    with pytest.raises(pc.PluginOperationError, match="publish"):
        pc._install_plugin_core(f"file://{repo}", force=True)
    assert (old / "old").read_text() == "yes"


@pytest.mark.parametrize("identifier", [
    "https://user:secret@example.com/repo.git",
    "https://example.com/repo.git?token=secret",
])
def test_unsafe_source_rejected_before_git(identifier):
    with patch.object(pc, "_resolve_git_executable") as resolve_git:
        with pytest.raises(pc.PluginOperationError, match="credentials|query"):
            pc._install_plugin_core(identifier, force=False)
    resolve_git.assert_not_called()


def test_dashboard_returns_json_provenance_capabilities_and_enables_after_success(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    sha = _repo(repo)
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    enabled: set[str] = set()
    monkeypatch.setattr(pc, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(pc, "_get_disabled_set", set)
    monkeypatch.setattr(pc, "_save_enabled_set", lambda value: enabled.update(value))
    monkeypatch.setattr(pc, "_save_disabled_set", lambda value: None)
    result = pc.dashboard_install_plugin(f"file://{repo}", force=False, enable=True, requested_ref=sha)
    json.dumps(result)
    assert result["ok"] is True
    assert result["provenance"]["resolved_commit"] == sha
    assert result["capabilities"]["warnings"] == ["CAPABILITY_REPORT_IS_NOT_SECURITY_AUDIT"]
    assert "pinned-plugin" in enabled

    enabled.clear()
    failed = pc.dashboard_install_plugin(f"file://{repo}", force=False, enable=True, requested_ref="bad")
    assert failed["ok"] is False
    assert enabled == set()


def test_cli_install_parser_dispatches_requested_ref():
    parser = ArgumentParser()
    subs = parser.add_subparsers()
    handler = patch.object(pc, "cmd_install").start()
    try:
        build_plugins_parser(subs, cmd_plugins=pc.plugins_command)
        sha = "a" * 40
        args = parser.parse_args(["plugins", "install", "owner/repo", "--ref", sha, "--no-enable"])
        args.func(args)
        handler.assert_called_once_with("owner/repo", force=False, enable=False, requested_ref=sha)
    finally:
        patch.stopall()
