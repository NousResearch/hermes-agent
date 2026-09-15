"""Regression coverage for POSIX npm path classification."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.main_install_repair import (
    _is_windows_npm_path,
    _resolve_node_runtime_npm,
)


def test_windows_npm_path_classifies_only_windows_drive_mounts():
    assert _is_windows_npm_path("/mnt/c/nodejs/npm")
    assert not _is_windows_npm_path("/mnt/data/node/bin/npm")


def test_resolve_node_runtime_npm_accepts_native_mnt_data_path():
    native_npm = "/mnt/data/node/bin/npm"
    with (
        patch("hermes_cli.main._is_windows", return_value=False),
        patch("hermes_constants.find_node_executable", return_value=native_npm),
    ):
        assert _resolve_node_runtime_npm() == native_npm


def test_resolve_node_runtime_npm_rejects_windows_drive_mount(monkeypatch):
    monkeypatch.setenv("PATH", "/mnt/c/Program Files/nodejs")
    with (
        patch("hermes_cli.main._is_windows", return_value=False),
        patch(
            "hermes_constants.find_node_executable",
            return_value="/mnt/c/nodejs/npm",
        ),
    ):
        assert _resolve_node_runtime_npm() is None


def test_resolve_node_runtime_npm_rescans_native_mnt_data_path(monkeypatch):
    native_npm = "/mnt/data/node/bin/npm"
    monkeypatch.setenv("PATH", "/mnt/data/node/bin")
    with (
        patch("hermes_cli.main._is_windows", return_value=False),
        patch(
            "hermes_constants.find_node_executable",
            return_value="/mnt/c/nodejs/npm",
        ),
        patch("hermes_cli.main_install_repair.shutil.which", return_value=native_npm),
    ):
        assert _resolve_node_runtime_npm() == native_npm
