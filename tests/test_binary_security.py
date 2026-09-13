"""Focused tests for credential-bearing CLI discovery and probing."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest import mock

import pytest

from agent.secret_sources import _binary_security as security


def _patch_file_owner(
    monkeypatch, candidate: Path, owner: int, *, safe_parents: bool = False
) -> None:
    original_stat = security.Path.stat
    resolved_candidate = candidate.resolve()

    def fake_stat(self, *args, **kwargs):
        result = original_stat(self, *args, **kwargs)
        if self == resolved_candidate:
            fields = list(result)
            fields[4] = owner  # st_uid
            return os.stat_result(fields)
        if safe_parents and stat.S_ISDIR(result.st_mode):
            fields = list(result)
            fields[4] = 0  # root-owned system directories
            fields[0] &= ~(stat.S_IWGRP | stat.S_IWOTH)
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(security.Path, "stat", fake_stat)


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
def test_root_can_use_root_owned_path_binary(monkeypatch, tmp_path):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o755)
    _patch_file_owner(monkeypatch, candidate, 0, safe_parents=True)
    monkeypatch.setattr(security.os, "geteuid", lambda: 0)

    assert security.resolve_executable(
        candidate,
        check_parent_dirs=True,
        reject_current_owner=True,
    ) == candidate.resolve()


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
def test_non_root_current_owner_remains_rejected(monkeypatch, tmp_path):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o755)
    fake_uid = 424242
    _patch_file_owner(monkeypatch, candidate, fake_uid)
    monkeypatch.setattr(security.os, "geteuid", lambda: fake_uid)

    assert security.resolve_executable(
        candidate, reject_current_owner=True
    ) is None


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
def test_root_rejects_root_owned_binary_in_writable_path(monkeypatch, tmp_path):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o755)
    _patch_file_owner(monkeypatch, candidate, 0)
    monkeypatch.setattr(security.os, "geteuid", lambda: 0)

    assert security.resolve_executable(
        candidate,
        check_parent_dirs=True,
        reject_current_owner=True,
    ) is None


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
def test_explicit_group_world_writable_leaf_is_rejected(tmp_path):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o777)

    assert security.resolve_executable(candidate) is None


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
def test_explicit_user_owned_non_group_world_writable_leaf_is_allowed(tmp_path):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o755)

    assert security.resolve_executable(candidate) == candidate.resolve()


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
def test_explicit_mode_allows_private_current_user_chain(monkeypatch, tmp_path):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o755)
    _patch_file_owner(monkeypatch, candidate, os.geteuid(), safe_parents=True)

    assert security.resolve_executable(
        candidate, check_explicit_parent_dirs=True
    ) == candidate.resolve()


@pytest.mark.parametrize("suffix", [".bat", ".cmd"])
def test_windows_rejects_cmd_interpreted_scripts(monkeypatch, tmp_path, suffix):
    candidate = tmp_path / f"op{suffix}"
    candidate.write_text("echo unsafe\n")
    candidate.chmod(0o755)
    fake_os = mock.Mock(name="windows_os")
    fake_os.name = "nt"
    fake_os.X_OK = os.X_OK
    fake_os.access = os.access
    monkeypatch.setattr(security, "os", fake_os)

    assert security.resolve_executable(candidate) is None


@pytest.mark.parametrize("suffix", [".com", ".exe"])
def test_windows_accepts_native_executables(monkeypatch, tmp_path, suffix):
    candidate = tmp_path / f"op{suffix}"
    candidate.write_text("native placeholder\n")
    candidate.chmod(0o755)
    fake_os = mock.Mock(name="windows_os")
    fake_os.name = "nt"
    fake_os.X_OK = os.X_OK
    fake_os.access = os.access
    monkeypatch.setattr(security, "os", fake_os)

    assert security.resolve_executable(candidate) == candidate.resolve()


def test_windows_explicit_path_is_limited_to_profile_or_machine_root(
    monkeypatch, tmp_path
):
    profile = tmp_path / "profile"
    profile.mkdir()
    candidate = profile / "op.exe"
    candidate.write_text("native placeholder\n")
    candidate.chmod(0o755)
    fake_os = mock.Mock(name="windows_os")
    fake_os.name = "nt"
    fake_os.X_OK = os.X_OK
    fake_os.access = os.access
    fake_os.environ = {"USERPROFILE": str(profile)}
    monkeypatch.setattr(security, "os", fake_os)
    monkeypatch.setattr(security.Path, "home", lambda: profile)

    assert security.resolve_executable(
        candidate, check_explicit_parent_dirs=True
    ) == candidate.resolve()

    outside = tmp_path / "outside" / "op.exe"
    outside.parent.mkdir()
    outside.write_text("native placeholder\n")
    outside.chmod(0o755)
    assert (
        security.resolve_executable(
            outside, check_explicit_parent_dirs=True
        )
        is None
    )


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
def test_path_rejects_unprivileged_leaf_even_for_root(monkeypatch, tmp_path):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o755)
    _patch_file_owner(monkeypatch, candidate, 424242, safe_parents=True)
    monkeypatch.setattr(security.os, "geteuid", lambda: 0)

    assert security.resolve_executable(
        candidate,
        check_parent_dirs=True,
        reject_current_owner=True,
    ) is None


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
def test_path_rejects_parent_owned_by_other_user(monkeypatch, tmp_path):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o755)
    resolved_candidate = candidate.resolve()
    unsafe_parent = resolved_candidate.parent
    original_stat = security.Path.stat

    def fake_stat(self, *args, **kwargs):
        result = original_stat(self, *args, **kwargs)
        fields = list(result)
        if stat.S_ISDIR(result.st_mode):
            fields[4] = 0
            fields[0] &= ~(stat.S_IWGRP | stat.S_IWOTH)
            if self == unsafe_parent:
                fields[4] = 424242
            return os.stat_result(fields)
        if self == resolved_candidate:
            fields[4] = 0
        return os.stat_result(fields)

    effective_uid = 424243
    monkeypatch.setattr(security.Path, "stat", fake_stat)
    monkeypatch.setattr(security.os, "geteuid", lambda: effective_uid)

    assert security.resolve_executable(
        candidate,
        check_parent_dirs=True,
        reject_current_owner=True,
    ) is None


@pytest.mark.skipif(os.name == "nt", reason="ownership policy is POSIX-specific")
@pytest.mark.parametrize("unsafe_mode", [stat.S_IWGRP, stat.S_IWOTH])
def test_path_rejects_group_or_world_writable_parent(
    monkeypatch, tmp_path, unsafe_mode
):
    candidate = tmp_path / "op"
    candidate.write_text("#!/bin/sh\n")
    candidate.chmod(0o755)
    resolved_candidate = candidate.resolve()
    unsafe_parent = resolved_candidate.parent
    original_stat = security.Path.stat

    def fake_stat(self, *args, **kwargs):
        result = original_stat(self, *args, **kwargs)
        fields = list(result)
        if stat.S_ISDIR(result.st_mode):
            fields[4] = 0
            fields[0] &= ~(stat.S_IWGRP | stat.S_IWOTH)
            if self == unsafe_parent:
                fields[0] |= unsafe_mode
            return os.stat_result(fields)
        if self == resolved_candidate:
            fields[4] = 0
        return os.stat_result(fields)

    monkeypatch.setattr(security.Path, "stat", fake_stat)
    monkeypatch.setattr(security.os, "geteuid", lambda: 424243)

    assert security.resolve_executable(
        candidate,
        check_parent_dirs=True,
        reject_current_owner=True,
    ) is None


def test_probe_version_keeps_cli_runtime_keys_without_credentials(monkeypatch):
    required = (
        "SystemDrive",
        "PATHEXT",
        "COMSPEC",
        "ProgramData",
        "APPDATA",
        "LOCALAPPDATA",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
    )
    for key in required:
        monkeypatch.setenv(key, f"probe-{key}")
    monkeypatch.setenv(
        "PATH", f"/tmp/untrusted-probe-path{os.pathsep}{os.environ.get('PATH', '')}"
    )
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "must-not-pass")
    monkeypatch.setenv("OP_SESSION_example", "must-not-pass")
    monkeypatch.setenv("LD_PRELOAD", "must-not-pass")

    captured = {}

    def fake_run(_argv, **kwargs):
        captured.update(kwargs["env"])
        return mock.Mock(returncode=0, stdout="op 2.0.0", stderr="")

    monkeypatch.setattr(security.subprocess, "run", fake_run)

    assert security.probe_version(Path("/unused/op.exe"), r"2\.0\.0")
    assert all(captured[key] == f"probe-{key}" for key in required)
    assert "/tmp/untrusted-probe-path" not in captured["PATH"]
    assert "OP_SERVICE_ACCOUNT_TOKEN" not in captured
    assert "OP_SESSION_example" not in captured
    assert "LD_PRELOAD" not in captured
