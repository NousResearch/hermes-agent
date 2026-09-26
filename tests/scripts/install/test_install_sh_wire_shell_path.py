"""install.sh PATH wiring must recognize bare `PATH=` assignments.

Regression for #123424: the existing-setup check used
`^[[:space:]]*[^#[:space:]].*PATH=.*\\.local/bin`, whose mandatory pre-`PATH=`
character only matched `export PATH=...`. Bare assignments such as Fedora's
`    PATH="$HOME/.local/bin:$HOME/bin:$PATH"` slipped through, so the installer
appended a duplicate line on every rerun.
"""
import os
import shlex
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")

INSTALL_SH = Path(__file__).resolve().parents[3] / "scripts" / "install.sh"


def _run_wire(tmp_path: Path, rc_name: str, rc_body: str) -> str:
    rc = tmp_path / rc_name
    rc.write_text(rc_body, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("CI", "GITHUB_ACTIONS")}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "home"),
               NO_COLOR="1", TERM="dumb", SHELL="/bin/bash")
    script = (
        f"source {shlex.quote(INSTALL_SH.as_posix())} --manifest\n"
        "wire_shell_path\n"
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=30, env=env)
    assert proc.returncode == 0, proc.stderr
    return rc.read_text(encoding="utf-8")


def test_bare_path_assignment_is_recognized(tmp_path):
    body = '    PATH="$HOME/.local/bin:$HOME/bin:$PATH"\n'
    out = _run_wire(tmp_path, ".bashrc", body)
    assert out == body, "installer appended a duplicate ~/.local/bin line"


def test_export_path_assignment_is_recognized(tmp_path):
    body = 'export PATH="$HOME/.local/bin:$PATH"\n'
    out = _run_wire(tmp_path, ".bashrc", body)
    assert out == body


def test_commented_path_line_does_not_suppress_append(tmp_path):
    body = "# PATH=$HOME/.local/bin:$PATH\n"
    out = _run_wire(tmp_path, ".bashrc", body)
    assert 'export PATH="$HOME/.local/bin:$PATH"' in out


def test_missing_local_bin_appends(tmp_path):
    body = 'export PATH="/usr/local/bin:$PATH"\n'
    out = _run_wire(tmp_path, ".bashrc", body)
    assert 'export PATH="$HOME/.local/bin:$PATH"' in out
