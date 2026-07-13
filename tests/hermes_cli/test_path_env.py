"""Tests for PATH fallback helpers used by CLI diagnostics and tools."""

import os

from hermes_cli.path_env import (
    common_tool_path_dirs,
    ensure_common_tool_paths,
    merge_common_tool_path,
)


def test_common_tool_path_dirs_include_version_manager_and_platform_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    # nvm honours $NVM_DIR ahead of $HOME/.nvm. CI runners commonly set
    # $NVM_DIR to a path outside the test's tmp_path, so clear it to make
    # the version-manager lookup deterministic.
    monkeypatch.delenv("NVM_DIR", raising=False)
    mise_shims = tmp_path / ".local" / "share" / "mise" / "shims"
    asdf_shims = tmp_path / ".asdf" / "shims"
    nvm_node_bin = tmp_path / ".nvm" / "versions" / "node" / "v24.11.0" / "bin"
    for path in (mise_shims, asdf_shims, nvm_node_bin):
        path.mkdir(parents=True)

    dirs = common_tool_path_dirs()

    assert str(mise_shims) in dirs
    assert str(asdf_shims) in dirs
    assert str(nvm_node_bin) in dirs
    assert "/usr/bin" in common_tool_path_dirs(existing_only=False)


def test_common_tool_path_dirs_can_return_non_existing_static_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    dirs = common_tool_path_dirs(existing_only=False)

    assert str(tmp_path / ".local" / "share" / "mise" / "shims") in dirs
    assert "/opt/homebrew/bin" in dirs
    assert "/data/data/com.termux/files/usr/bin" in dirs


def test_merge_common_tool_path_appends_fallbacks_without_duplicates(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    custom_bin = tmp_path / "custom" / "bin"
    mise_shims = tmp_path / ".local" / "share" / "mise" / "shims"
    asdf_shims = tmp_path / ".asdf" / "shims"
    custom_bin.mkdir(parents=True)
    mise_shims.mkdir(parents=True)
    asdf_shims.mkdir(parents=True)

    merged = merge_common_tool_path(os.pathsep.join([str(custom_bin), str(mise_shims)]))
    parts = merged.split(os.pathsep)

    assert parts[0] == str(custom_bin)
    assert parts.count(str(mise_shims)) == 1
    assert str(asdf_shims) in parts


def test_merge_common_tool_path_can_prepend_fallbacks(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    existing_bin = tmp_path / "existing" / "bin"
    mise_shims = tmp_path / ".local" / "share" / "mise" / "shims"
    existing_bin.mkdir(parents=True)
    mise_shims.mkdir(parents=True)

    merged = merge_common_tool_path(str(existing_bin), prepend=True)
    parts = merged.split(os.pathsep)

    assert parts.index(str(mise_shims)) < parts.index(str(existing_bin))


def test_ensure_common_tool_paths_updates_process_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    mise_shims = tmp_path / ".local" / "share" / "mise" / "shims"
    mise_shims.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/usr/bin")

    merged = ensure_common_tool_paths()

    assert os.environ["PATH"] == merged
    assert str(mise_shims) in merged.split(os.pathsep)
