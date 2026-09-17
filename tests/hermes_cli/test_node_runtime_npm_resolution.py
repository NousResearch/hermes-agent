"""POSIX npm path classification: only Windows shims are refused (#112041, #30271)."""

from __future__ import annotations

import os
from unittest.mock import patch

from hermes_cli.main_install_repair import _is_windows_npm_path, _resolve_node_runtime_npm


def test_windows_npm_path_refuses_windows_shims_but_not_native_data_mounts():
    """A ``/mnt/<drive>`` WSL interop path or a ``.cmd`` shim is Windows npm; a native Linux
    data mount such as ``/mnt/data`` is not — a folder prefix alone is not a Windows tell."""
    assert _is_windows_npm_path("/mnt/c/Program Files/nodejs/npm")
    assert _is_windows_npm_path("/mnt/d/nodejs/npm")
    assert _is_windows_npm_path("C:\\nodejs\\npm.cmd")
    assert not _is_windows_npm_path("/mnt/data/hermes/home/.local/bin/npm")
    assert not _is_windows_npm_path("/usr/bin/npm")


def test_resolve_node_runtime_npm_rescans_past_windows_drive_to_native_mount(monkeypatch):
    """When PATH interop hands back a Windows npm first, the re-scan skips the Windows drive
    mount but still accepts a native npm living under ``/mnt/data``."""
    native_npm = "/mnt/data/node/bin/npm"
    monkeypatch.setenv("PATH", os.pathsep.join(["/mnt/c/Program Files/nodejs", "/mnt/data/node/bin"]))

    def fake_which(cmd, path=None):
        return native_npm if path == "/mnt/data/node/bin" else None

    with (
        patch("hermes_constants.find_node_executable", return_value="/mnt/c/Program Files/nodejs/npm"),
        patch("hermes_cli.main_install_repair.shutil.which", side_effect=fake_which),
    ):
        assert _resolve_node_runtime_npm() == native_npm
