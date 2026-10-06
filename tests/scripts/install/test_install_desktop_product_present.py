"""Installer repair/upgrade reruns detect a built desktop app on every arch (#94703).

electron-builder names the unpacked output ``<os>-unpacked`` on x64 but
``<os>-<arch>-unpacked`` elsewhere (``linux-arm64-unpacked``,
``win-arm64-unpacked``; macOS uses ``mac-arm64``). install.sh and install.ps1
only knew the x64 names, so a rerun on an ARM64 desktop install skipped the
desktop rebuild and left a bundle built from the previous code.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = ROOT / "scripts" / "install.sh"
INSTALL_PS1 = ROOT / "scripts" / "install.ps1"

UNPACKED_DIRS = ("linux-unpacked", "linux-arm64-unpacked", "mac", "mac-arm64",
                 "win-unpacked", "win-ia32-unpacked", "win-arm64-unpacked")


def _desktop_product_present(tmp_path: Path) -> int:
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=(tmp_path / "install").as_posix())
    script = f"source {shlex.quote(INSTALL_SH.as_posix())} --manifest >/dev/null\ndesktop_product_present\n"
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30).returncode


@pytest.mark.parametrize("unpacked", UNPACKED_DIRS)
def test_install_sh_detects_desktop_build(tmp_path, unpacked):
    (tmp_path / "install" / "apps" / "desktop" / "release" / unpacked).mkdir(parents=True)
    assert _desktop_product_present(tmp_path) == 0


def test_install_sh_without_desktop_build(tmp_path):
    (tmp_path / "install" / "apps" / "desktop" / "release" / "builder-debug").mkdir(parents=True)
    assert _desktop_product_present(tmp_path) != 0


def test_install_ps1_candidates_match_install_sh():
    body = INSTALL_PS1.read_text(encoding="utf-8")
    fn = body[body.index("function Test-DesktopProductPresent"):]
    listed = re.search(r"foreach \(\$candidate in @\((.*?)\)\)", fn, re.S)
    assert listed, "Test-DesktopProductPresent candidate list not found"
    assert set(re.findall(r'"([^"]+)"', listed.group(1))) == set(UNPACKED_DIRS)
