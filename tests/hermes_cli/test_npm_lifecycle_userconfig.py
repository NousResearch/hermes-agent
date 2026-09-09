"""HERMES_HOME npmrc must reach npm during desktop/TUI install (#106373).

Restricted-network Windows updates fail when get-windows cannot fetch its
GitHub Releases prebuilt. Users put ``node_get_windows_binary_host_mirror``
in ``~/.npmrc`` or a project ``.npmrc``, but updater children miss both:
HOME/USERPROFILE is often empty under CreateProcess, and git autostash
strips the tracked project ``.npmrc``. Pinning ``NPM_CONFIG_USERCONFIG``
to a durable ``$HERMES_HOME/npmrc`` (or ``.npmrc``) survives that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from hermes_cli.main_tui_launch import _npm_lifecycle_env


def _write_npmrc(path: Path, body: str = "node_get_windows_binary_host_mirror=https://example.test/mirror/\n") -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_hermes_home_npmrc_sets_npm_config_userconfig(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    npmrc = _write_npmrc(home / "npmrc")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    env = _npm_lifecycle_env({})
    assert env["NPM_CONFIG_USERCONFIG"] == os.path.abspath(str(npmrc))


def test_hermes_home_dot_npmrc_used_when_npmrc_missing(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    npmrc = _write_npmrc(home / ".npmrc")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    env = _npm_lifecycle_env({})
    assert env["NPM_CONFIG_USERCONFIG"] == os.path.abspath(str(npmrc))


def test_bare_npmrc_preferred_over_dot_npmrc(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    preferred = _write_npmrc(home / "npmrc", "registry=https://preferred.test/\n")
    _write_npmrc(home / ".npmrc", "registry=https://dot.test/\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    env = _npm_lifecycle_env({})
    assert env["NPM_CONFIG_USERCONFIG"] == os.path.abspath(str(preferred))


def test_explicit_userconfig_not_overwritten(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _write_npmrc(home / "npmrc")
    override = tmp_path / "explicit.npmrc"
    _write_npmrc(override, "registry=https://explicit.test/\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("NPM_CONFIG_USERCONFIG", str(override))
    env = _npm_lifecycle_env({})
    assert env["NPM_CONFIG_USERCONFIG"] == str(override)


def test_explicit_userconfig_in_env_dict_not_overwritten(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    _write_npmrc(home / "npmrc")
    override = str(tmp_path / "from-env-dict.npmrc")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    env = _npm_lifecycle_env({"NPM_CONFIG_USERCONFIG": override})
    assert env["NPM_CONFIG_USERCONFIG"] == override


def test_missing_hermes_npmrc_leaves_userconfig_unset(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    env = _npm_lifecycle_env({})
    assert not (env.get("NPM_CONFIG_USERCONFIG") or "").strip()


def test_directory_npmrc_is_skipped_in_favor_of_dotfile(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "npmrc").mkdir()
    dot = _write_npmrc(home / ".npmrc")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    env = _npm_lifecycle_env({})
    assert env["NPM_CONFIG_USERCONFIG"] == os.path.abspath(str(dot))


def test_symlink_to_dir_is_not_used_as_userconfig(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    target = tmp_path / "not-a-file"
    target.mkdir()
    (home / "npmrc").symlink_to(target)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("NPM_CONFIG_USERCONFIG", raising=False)
    env = _npm_lifecycle_env({})
    assert not (env.get("NPM_CONFIG_USERCONFIG") or "").strip()


def test_binary_host_mirror_env_not_stripped(monkeypatch):
    monkeypatch.setenv(
        "npm_config_node_get_windows_binary_host_mirror",
        "https://example.test/mirror/",
    )
    env = _npm_lifecycle_env({"SOME_OTHER": "1"})
    assert env["npm_config_node_get_windows_binary_host_mirror"] == "https://example.test/mirror/"


def test_windows_lifecycle_fills_empty_userprofile(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    profile = r"C:\Users\alice"
    monkeypatch.setenv("USERPROFILE", profile)
    env = _npm_lifecycle_env({"USERPROFILE": ""})
    assert env["USERPROFILE"] == profile


def test_windows_lifecycle_fills_empty_home_from_userprofile(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    profile = r"C:\Users\alice"
    monkeypatch.setenv("USERPROFILE", profile)
    env = _npm_lifecycle_env({"HOME": ""})
    assert env["USERPROFILE"] == profile
    assert env["HOME"] == profile


def test_windows_lifecycle_does_not_overwrite_home(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("USERPROFILE", r"C:\Users\alice")
    env = _npm_lifecycle_env({"HOME": r"D:\keep"})
    assert env["HOME"] == r"D:\keep"


def test_windows_lifecycle_fills_userprofile_from_homedrive(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("USERPROFILE", raising=False)
    monkeypatch.setenv("HOMEDRIVE", "C:")
    monkeypatch.setenv("HOMEPATH", r"\Users\alice")
    env = _npm_lifecycle_env({"USERPROFILE": ""})
    assert env["USERPROFILE"] == r"C:\Users\alice"


def test_non_windows_does_not_invent_userprofile(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("USERPROFILE", raising=False)
    env = _npm_lifecycle_env({"USERPROFILE": ""})
    assert not (env.get("USERPROFILE") or "").strip()
