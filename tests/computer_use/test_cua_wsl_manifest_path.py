from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from tools.computer_use import cua_backend


def test_wsl_windows_manifest_path_translates_to_drvfs():
    with patch("hermes_constants.is_wsl", return_value=True):
        assert cua_backend._wsl_windows_path_to_posix(
            r"C:\Users\Fernando\AppData\Local\cua-driver\cua-driver.exe"
        ) == "/mnt/c/Users/Fernando/AppData/Local/cua-driver/cua-driver.exe"


def test_non_windows_path_is_unchanged_in_wsl():
    with patch("hermes_constants.is_wsl", return_value=True):
        assert cua_backend._wsl_windows_path_to_posix(
            "/usr/local/bin/cua-driver"
        ) == "/usr/local/bin/cua-driver"


def test_windows_manifest_path_is_unchanged_outside_wsl():
    path = r"D:\Tools\cua-driver.exe"
    with patch("hermes_constants.is_wsl", return_value=False):
        assert cua_backend._wsl_windows_path_to_posix(path) == path


def test_resolve_mcp_invocation_normalizes_windows_manifest_command_in_wsl():
    manifest = {
        "mcp_invocation": {
            "command": r"C:\Users\Fernando\AppData\Local\cua-driver\cua-driver.exe",
            "args": ["mcp"],
        }
    }
    proc = SimpleNamespace(returncode=0, stdout=json.dumps(manifest))
    with (
        patch.object(cua_backend.subprocess, "run", return_value=proc),
        patch("hermes_constants.is_wsl", return_value=True),
    ):
        command, args = cua_backend._resolve_mcp_invocation("cua-driver")

    assert command == "/mnt/c/Users/Fernando/AppData/Local/cua-driver/cua-driver.exe"
    assert args == ["mcp"]
