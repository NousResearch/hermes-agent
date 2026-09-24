"""Regression tests for command discovery during system-package setup."""

import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

pytestmark = [
    pytest.mark.linux_only,
    pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash"),
]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_system_package_probe_finds_commands_in_link_dir(tmp_path):
    bash = shutil.which("bash")
    assert bash is not None

    home = tmp_path / "home"
    link_dir = home / ".local" / "bin"
    path_dir = tmp_path / "path"
    link_dir.mkdir(parents=True)
    path_dir.mkdir()

    _write_executable(link_dir / "rg", "printf 'ripgrep 14.1.0\\n'")
    _write_executable(link_dir / "ffmpeg", "printf 'ffmpeg version 7.1 static\\n'")
    _write_executable(path_dir / "head", "IFS= read -r line || true; printf '%s\\n' \"$line\"")
    _write_executable(path_dir / "awk", "read -r first second third rest; printf '%s\\n' \"$third\"")

    script = (
        f'source "{INSTALL_SH}" --manifest >/dev/null\n'
        'OS=linux\n'
        'DISTRO=unknown\n'
        'install_system_packages\n'
        'printf "ripgrep=%s ffmpeg=%s\\n" "$HAS_RIPGREP" "$HAS_FFMPEG"\n'
    )
    result = subprocess.run(
        [bash, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"HOME": str(home), "PATH": str(path_dir)},
    )

    assert "ripgrep=true ffmpeg=true" in result.stdout
    assert "not installed" not in result.stdout