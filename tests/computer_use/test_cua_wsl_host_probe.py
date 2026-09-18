"""Issue #114531: under WSL, driver resolution must probe a Windows host install.

Failing-first: on a non-Windows platform with is_wsl() true and WSL interop
(cmd.exe) reachable, _candidate_cua_driver_commands() must append Windows host
candidates (DrvFS /mnt/c/.../cua-driver.exe), keeping Linux guest candidates
first so the host-vs-guest choice is surfaced by order.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

from tools.computer_use import cua_backend_driver


def _wsl_candidates():
    host_exe = "/mnt/c/Users/Test/AppData/Local/Programs/Cua/cua-driver/bin/cua-driver.exe"
    with (
        patch.object(sys, "platform", "linux"),
        patch("hermes_constants.is_wsl", return_value=True),
        patch.object(cua_backend_driver.shutil, "which",
                     side_effect=lambda cmd, *a, **k: cmd if str(cmd).lower().endswith("cmd.exe") else None),
        patch("glob.glob", return_value=[host_exe]),
    ):
        return cua_backend_driver._candidate_cua_driver_commands()


def test_wsl_candidates_include_windows_host_exe():
    assert any(
        str(c).startswith("/mnt/") and str(c).lower().endswith("cua-driver.exe")
        for c in _wsl_candidates()
    ), "no Windows host cua-driver.exe candidate under WSL"


def test_wsl_guest_candidates_stay_first_host_last():
    cands = [str(c) for c in _wsl_candidates()]
    host_idx = next(i for i, c in enumerate(cands)
                    if c.startswith("/mnt/") and c.lower().endswith("cua-driver.exe"))
    assert any("cargo" in c or ".local" in c for c in cands[:host_idx]),         "Linux guest candidates must precede the Windows host candidate"


def test_no_host_candidates_without_interop():
    with (
        patch.object(sys, "platform", "linux"),
        patch("hermes_constants.is_wsl", return_value=True),
        patch.object(cua_backend_driver.shutil, "which", return_value=None),
        patch("glob.glob", return_value=["/mnt/c/Users/Test/AppData/Local/Programs/Cua/cua-driver/bin/cua-driver.exe"]),
    ):
        cands = [str(c) for c in cua_backend_driver._candidate_cua_driver_commands()]
    assert not [c for c in cands if c.startswith("/mnt/")],         "host candidates must be withheld when cmd.exe interop is unreachable"


def test_no_host_candidates_outside_wsl():
    with (
        patch.object(sys, "platform", "linux"),
        patch("hermes_constants.is_wsl", return_value=False),
    ):
        cands = [str(c) for c in cua_backend_driver._candidate_cua_driver_commands()]
    assert not [c for c in cands if c.startswith("/mnt/")]
