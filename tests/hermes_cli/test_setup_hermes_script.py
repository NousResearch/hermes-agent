import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"

_BASH = shutil.which("bash")
# ``posix`` keeps the Linux and macOS lanes running this file; ``windows`` is
# what the Windows lane imports, whose ``-m "platforms and not integration"``
# selects only marked files.
pytestmark = [
    pytest.mark.skipif(_BASH is None, reason="running setup-hermes.sh needs bash"),
    pytest.mark.platforms("posix", "windows"),
]


def test_setup_hermes_script_is_valid_shell():
    # as_posix: on Windows the argument reaches bash through its own command
    # line, where backslashes are escape characters — Git Bash wants forward
    # slashes. _BASH, not a bare "bash": CreateProcess finds System32\bash.exe
    # (WSL) first, which cannot open D:/... and exits 127.
    result = subprocess.run([_BASH, "-n", SETUP_SCRIPT.as_posix()], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    shutil.which("cygpath") is None,
    reason="the drive-relative refuse is gated on cygpath (msys/cygwin)",
)
@pytest.mark.parametrize("bad", ["C:rel-store", "C:"])
def test_setup_hermes_refuses_drive_relative_runtime_dir(tmp_path: Path, bad: str) -> None:
    """setup-hermes.sh binds a native store path: cygpath would swallow the
    colon of a drive-relative override into '<cwd>/C...' and bash has no
    per-drive cwd for pm's resolve() to agree with, so refuse up front."""
    assert _BASH, "running setup-hermes.sh needs bash"
    env = {**os.environ, "HERMES_RUNTIME_DIR": bad}
    result = subprocess.run(
        [_BASH, SETUP_SCRIPT.as_posix()], env=env, cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "drive-relative HERMES_RUNTIME_DIR" in result.stderr, result.stderr
