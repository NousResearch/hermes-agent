"""Regression tests for install.sh Python environment sanitization.

When install.sh is launched from another Python-driven tool session, inherited
PYTHONPATH/PYTHONHOME can shadow the freshly installed checkout. The installer
must sanitize those vars both during installation and at runtime launch.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_unsets_pythonpath_and_pythonhome_early() -> None:
    text = INSTALL_SH.read_text()

    # During install, inherited Python env must be sanitized before pip/venv use.
    assert 'unset PYTHONPATH' in text
    assert 'unset PYTHONHOME' in text


def test_hermes_launcher_wrapper_clears_python_env_before_exec(tmp_path: Path) -> None:
    """Behavioral: run the generated launcher with a poisoned Python env.

    The shim must clear PYTHONPATH/PYTHONHOME before exec'ing the entry point,
    and forward its arguments untouched.
    """
    text = INSTALL_SH.read_text()
    match = re.search(
        r"(?P<block>mkdir -p \"\$command_link_dir\".*?chmod \+x \"\$command_link_dir/hermes\")",
        text,
        re.DOTALL,
    )
    assert match is not None, (
        "Could not locate the setup_path shim-write block in scripts/install.sh"
    )

    home = tmp_path / "home"
    hermes_bin = home / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes"
    hermes_bin.parent.mkdir(parents=True)
    # Stand-in entry point that reports the Python env it was exec'd with.
    hermes_bin.write_text(
        "#!/usr/bin/env bash\n"
        'echo "PYTHONPATH=[${PYTHONPATH-<unset>}]"\n'
        'echo "PYTHONHOME=[${PYTHONHOME-<unset>}]"\n'
        'echo "args=$*"\n'
    )
    hermes_bin.chmod(0o755)

    command_link_dir = home / ".local" / "bin"
    script = (
        "set -e\n"
        f"HOME={home!s}\n"
        f"HERMES_BIN={hermes_bin!s}\n"
        f"command_link_dir={command_link_dir!s}\n"
        f"{match['block']}\n"
    )
    written = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert written.returncode == 0, (
        f"shim-write block failed:\nstdout={written.stdout}\nstderr={written.stderr}"
    )

    result = subprocess.run(
        [str(command_link_dir / "hermes"), "chat"],
        capture_output=True,
        text=True,
        env={
            "HOME": str(home),
            "PATH": "/usr/bin:/bin",
            # Poison the env the way a parent Python tool session would.
            "PYTHONPATH": "/some/other/checkout",
            "PYTHONHOME": "/some/other/python",
        },
    )
    assert result.returncode == 0, f"launcher failed: {result.stderr}"
    assert "PYTHONPATH=[<unset>]" in result.stdout, (
        "launcher must unset an inherited PYTHONPATH before exec"
    )
    assert "PYTHONHOME=[<unset>]" in result.stdout, (
        "launcher must unset an inherited PYTHONHOME before exec"
    )
    assert "args=chat" in result.stdout, "launcher must forward args untouched"
