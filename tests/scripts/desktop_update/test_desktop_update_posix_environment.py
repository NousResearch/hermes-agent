"""Regression coverage for inherited source-root contamination in the POSIX Desktop updater."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
POSIX_SH = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_handoff_scrubs_inherited_source_environment_before_update(tmp_path: Path) -> None:
    install_root = tmp_path / "hermes-agent"
    bin_dir = install_root / "venv" / "bin"
    bin_dir.mkdir(parents=True)

    (bin_dir / "python").symlink_to(sys.executable)
    (bin_dir / "python3").symlink_to("python")

    captured = tmp_path / "captured-env"
    _write_executable(
        bin_dir / "hermes",
        "#!/bin/bash\n"
        "if [ \"${1:-}\" = update ] && [ \"${2:-}\" = --help ]; then\n"
        "  echo --keep-stash\n"
        "  exit 0\n"
        "fi\n"
        f"printf '%s\\n%s\\n%s\\n%s\\n%s\\n' "
        "\"${HERMES_SOURCE_ROOT-unset}\" \"${PYTHONPATH-unset}\" "
        "\"${PYTHONHOME-unset}\" \"${PYTHONSTARTUP-unset}\" "
        f"\"${{__PYVENV_LAUNCHER__-unset}}\" > {captured!s}\n"
        "exit 0\n",
    )

    env = os.environ.copy()
    env.update(
        HERMES_SOURCE_ROOT=str(tmp_path / "frozen-release"),
        PYTHONPATH=str(tmp_path / "frozen-release"),
        PYTHONHOME=str(tmp_path / "foreign-python"),
        PYTHONSTARTUP=str(tmp_path / "foreign-startup.py"),
        __PYVENV_LAUNCHER__=str(tmp_path / "foreign-python"),
    )
    result = subprocess.run(
        [
            "/bin/bash",
            str(POSIX_SH),
            "--daemonized",
            "--no-ui",
            "--desktop-pid",
            "0",
            "--install-root",
            str(install_root),
            "--branch",
            "main",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert captured.read_text(encoding="utf-8").splitlines() == ["unset"] * 5
