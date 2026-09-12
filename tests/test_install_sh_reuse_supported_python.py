"""install.sh reuses an already-installed supported Python instead of downloading 3.11 (#10778)."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _exe(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _run_prerequisites(tmp_path: Path, *, uv_find_script: str) -> subprocess.CompletedProcess[str]:
    """Run the real ``--stage prerequisites`` with a stub managed uv whose ``python find`` we script."""
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    (hermes_home / "bin").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _exe(bin_dir / "python3.13", "#!/bin/sh\n[ \"$1\" = --version ] && echo 'Python 3.13.12'\nexit 0\n")
    _exe(hermes_home / "bin" / "uv", "#!/bin/sh\n[ \"$1\" = --version ] && { echo 'uv 0.9.0'; exit 0; }\n"
         "if [ \"$1\" = python ] && [ \"$2\" = find ]; then\n" + uv_find_script + "fi\n"
         "if [ \"$1\" = python ] && [ \"$2\" = install ]; then echo 'DOWNLOAD ATTEMPTED' >&2; exit 1; fi\nexit 0\n")
    for tool in ("git", "node", "npm", "curl", "rg", "g++", "c++"):
        _exe(bin_dir / tool, "#!/bin/sh\ncase \"$1\" in --version|-v) echo 'v22.12.0 2.50.0';; esac\nexit 0\n")
    env = os.environ.copy()
    env.update({"HOME": str(home), "HERMES_HOME": str(hermes_home),
                "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', os.defpath)}"})
    bash = shutil.which("bash") or "/bin/bash"
    return subprocess.run([bash, str(INSTALL_SH), "--stage", "prerequisites", "--non-interactive"],
                          env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


@pytest.mark.linux_only
def test_supported_newer_python_is_reused_instead_of_downloading_311(tmp_path: Path) -> None:
    """3.11 absent, 3.13 present: the installer must take 3.13 and never call ``uv python install``."""
    result = _run_prerequisites(tmp_path, uv_find_script=(
        f"  [ \"$3\" = 3.11 ] && exit 2\n  [ \"$3\" = '>=3.11,<3.14' ] && {{ echo {tmp_path}/bin/python3.13; exit 0; }}\n  exit 2\n"))
    assert "DOWNLOAD ATTEMPTED" not in result.stdout, result.stdout
    assert "Python found: Python 3.13.12" in result.stdout, result.stdout
