"""Regression tests for the shell installer."""

import subprocess
from pathlib import Path


INSTALL_SH = Path(__file__).resolve().parents[2] / "scripts" / "install.sh"


def test_uv_installer_does_not_modify_shell_profiles():
    """Hermes owns PATH setup; uv must not add a ~/.local/bin/env source line.

    Astral's uv installer modifies shell startup files by default and can add
    lines such as `. "$HOME/.local/bin/env"`.  If that helper file is later
    missing, every new macOS bash login shell prints:

        -bash: /Users/.../.local/bin/env: No such file or directory

    The Hermes installer already creates the hermes shim and manages PATH, so
    it should disable uv's shell-profile mutation.
    """
    content = INSTALL_SH.read_text()

    assert "INSTALLER_NO_MODIFY_PATH=1" in content
    # The installer is invoked with the mutation guard set as an environment
    # variable.  Other env vars (e.g. UV_UNMANAGED_INSTALL) may sit between the
    # guard and `sh`, so assert it on the same line as the installer call.
    installer_lines = [
        line
        for line in content.splitlines()
        if 'sh "$_uv_installer"' in line
    ]
    assert installer_lines, "could not find the uv installer invocation"
    assert all(
        "INSTALLER_NO_MODIFY_PATH=1" in line for line in installer_lines
    )


def test_cleanup_removes_broken_uv_env_hook(tmp_path):
    """Rerunning the installer repairs stale uv profile hooks when env is gone."""
    bash_profile = tmp_path / ".bash_profile"
    bash_profile.write_text(
        "before\n"
        '. "$HOME/.local/bin/env"\n'
        'source "$HOME/.local/bin/env"\n'
        "after\n"
    )

    script = f"""
set -euo pipefail
export HOME={tmp_path}
source <(sed '/^if .*ENSURE_DEPS/,$d' {INSTALL_SH})
cleanup_broken_uv_shell_env_hooks >/tmp/hermes-cleanup.log
cat "$HOME/.bash_profile"
"""

    result = subprocess.run(
        ["bash", "-lc", script],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout == "before\nafter\n"
