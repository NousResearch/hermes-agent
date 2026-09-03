"""Regression tests for WSL2 clipboard image paste path resolution.

When wsl.conf sets ``appendWindowsPath=false``, WSL processes inherit a
PATH with no Windows directories, so the bare ``powershell.exe`` is
unresolvable and clipboard image paste fails with FileNotFoundError.
``_wsl_powershell_exe()`` resolves powershell.exe via PATH first and
falls back to the standard WSL-interop install path.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import clipboard as clip

_FALLBACK = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def test_wsl_powershell_exe_returns_path_resolved_exe():
    with patch.object(clip.shutil, "which", return_value="/usr/bin/powershell.exe"):
        assert clip._wsl_powershell_exe() == "/usr/bin/powershell.exe"


def test_wsl_powershell_exe_returns_mntc_fallback_when_which_none():
    with patch.object(clip.shutil, "which", return_value=None), \
         patch.object(clip.os.path, "isfile", return_value=True):
        assert clip._wsl_powershell_exe() == _FALLBACK


def test_wsl_powershell_exe_returns_bare_name_when_nothing_resolves():
    with patch.object(clip.shutil, "which", return_value=None), \
         patch.object(clip.os.path, "isfile", return_value=False):
        assert clip._wsl_powershell_exe() == "powershell.exe"


def test_wsl_has_image_passes_resolved_exe():
    with patch.object(clip.shutil, "which", return_value="/usr/bin/powershell.exe"), \
         patch.object(clip, "_powershell_has_image", return_value=True) as has_image:
        assert clip._wsl_has_image() is True
    exe = has_image.call_args[0][0]
    assert exe == "/usr/bin/powershell.exe"
    assert exe != "powershell.exe"


def test_wsl_save_passes_resolved_exe(tmp_path):
    dest = tmp_path / "clip.png"
    with patch.object(clip.shutil, "which", return_value="/usr/bin/powershell.exe"), \
         patch.object(clip, "_powershell_save_image", return_value=True) as save_image:
        assert clip._wsl_save(dest) is True
    exe = save_image.call_args[0][0]
    assert exe == "/usr/bin/powershell.exe"
    assert exe != "powershell.exe"
