from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALL_SH = Path(__file__).resolve().parents[1] / "scripts" / "install.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _fake_python(path: Path, version: str, supported: bool, probe_log: Path) -> None:
    _write_executable(
        path,
        f"""case "${{1:-}}" in
  -c)
    echo "{path.name} -c" >> {probe_log!s}
    exit {0 if supported else 1}
    ;;
  --version)
    echo "{path.name} --version" >> {probe_log!s}
    echo "Python {version}"
    exit 0
    ;;
esac
exit 0""",
    )


def _run_termux_prerequisites(
    tmp_path: Path,
    *,
    supported_candidate: tuple[str, str] | None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    probe_log = tmp_path / "python-probes.log"
    pkg_log = tmp_path / "pkg.log"

    _fake_python(bin_dir / "python", "3.14.6", False, probe_log)
    if supported_candidate is not None:
        candidate, version = supported_candidate
        _fake_python(bin_dir / candidate, version, True, probe_log)

    _write_executable(
        bin_dir / "pkg",
        f'echo "$*" >> {pkg_log!s}\nexit 0',
    )
    # Keep PATH hermetic so a host-installed python3.11/3.12/3.13 cannot
    # preempt the simulated Termux candidates. Link only non-Python utilities
    # exercised by the public prerequisites stage.
    for tool in ("awk", "head", "mktemp", "rm", "sed", "tr"):
        target = shutil.which(tool)
        assert target is not None
        (bin_dir / tool).symlink_to(target)

    _write_executable(bin_dir / "uname", 'echo "Linux"')
    _write_executable(bin_dir / "git", 'echo "git version 2.50.0"')
    _write_executable(bin_dir / "node", 'echo "v22.22.0"')
    _write_executable(bin_dir / "npm", 'echo "11.9.0"')
    _write_executable(bin_dir / "curl", "exit 0")
    _write_executable(bin_dir / "rg", 'echo "ripgrep 14.1.0"')
    _write_executable(bin_dir / "ffmpeg", 'echo "ffmpeg version 7.1 Copyright"')

    home = tmp_path / "home"
    prefix = tmp_path / "com.termux" / "files" / "usr"
    home.mkdir()
    prefix.mkdir(parents=True)
    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "HERMES_HOME": str(home / ".hermes"),
        "PATH": str(bin_dir),
        "PREFIX": str(prefix),
        "TERMUX_VERSION": "0.118.2",
    })
    result = subprocess.run(
        [
            "/bin/bash",
            str(INSTALL_SH),
            "--stage",
            "prerequisites",
            "--json",
            "--non-interactive",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return result, probe_log, pkg_log


@pytest.mark.parametrize(
    ("candidate", "version"),
    [
        ("python3.11", "3.11.15"),
        ("python3.12", "3.12.11"),
        ("python3.13", "3.13.7"),
    ],
)
def test_termux_prefers_each_supported_explicit_interpreter_over_python314(
    tmp_path: Path,
    candidate: str,
    version: str,
) -> None:
    result, probe_log, pkg_log = _run_termux_prerequisites(
        tmp_path,
        supported_candidate=(candidate, version),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Python found: Python {version}" in result.stdout
    assert probe_log.read_text(encoding="utf-8").splitlines() == [
        f"{candidate} -c",
        f"{candidate} --version",
    ]
    assert "install -y python" not in pkg_log.read_text(encoding="utf-8").splitlines()


def test_termux_rejects_python314_after_package_install_attempt(tmp_path: Path) -> None:
    result, probe_log, pkg_log = _run_termux_prerequisites(
        tmp_path,
        supported_candidate=None,
    )

    assert result.returncode != 0
    assert "Python >=3.11,<3.14" in result.stdout + result.stderr
    assert probe_log.read_text(encoding="utf-8").splitlines() == [
        "python -c",
        "python -c",
    ]
    assert pkg_log.read_text(encoding="utf-8").splitlines() == ["install -y python"]
