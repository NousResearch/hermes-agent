"""The runtime resolver finds the official per-user Windows Cua install."""

import os
from unittest.mock import patch


def test_official_windows_install_is_resolved_without_a_fresh_path():
    from tools.computer_use import cua_backend

    local_app_data = r"C:\Users\test\AppData\Local"
    expected = (
        local_app_data
        + r"\Programs\Cua\cua-driver\bin\cua-driver.exe"
    )
    with patch.dict(
        os.environ,
        {"LOCALAPPDATA": local_app_data, "PATH": ""},
        clear=True,
    ), patch.object(cua_backend.sys, "platform", "win32"), patch.object(
        cua_backend.shutil,
        "which",
        side_effect=lambda value: value if value == expected else None,
    ):
        assert cua_backend.resolve_cua_driver_cmd() == expected


def test_explicit_override_remains_authoritative():
    from tools.computer_use import cua_backend

    with patch.dict(
        os.environ,
        {"LOCALAPPDATA": r"C:\Users\test\AppData\Local", "PATH": ""},
        clear=True,
    ), patch.object(cua_backend.sys, "platform", "win32"), patch.object(
        cua_backend.shutil, "which", return_value=None
    ) as which:
        assert cua_backend.resolve_cua_driver_cmd(r"D:\missing\cua-driver.exe") is None
        which.assert_called_once_with(r"D:\missing\cua-driver.exe")
