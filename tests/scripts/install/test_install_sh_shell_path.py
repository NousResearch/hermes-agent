"""install.sh wires ~/.local/bin into shell startup files without duplicating it (#123424).

Distro skeletons put ~/.local/bin on PATH with a bare assignment
(`PATH="$HOME/.local/bin:$PATH"`), not `export PATH=...`; that must count as
existing setup. The appended line must also be a no-op when PATH already has the
entry, because Fedora's ~/.bash_profile sources ~/.bashrc in every login shell.
"""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

pytestmark = pytest.mark.platforms("posix")
INSTALL_SH = Path(__file__).resolve().parents[3] / "scripts" / "install.sh"
MARKER = "# Hermes Agent command"

# Fedora /etc/skel (bash-5.2): ~/.bashrc guards and prepends with a bare assignment.
FEDORA_BASHRC = """\
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]; then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH
"""
FEDORA_BASH_PROFILE = """\
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi

# User specific environment and startup programs
"""
# Debian /etc/skel/.profile tail.
DEBIAN_PROFILE = """\
# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
"""


def _wire(home: Path, runs: int = 1) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("CI", "GITHUB_ACTIONS")}
    env.update(HOME=str(home), HERMES_HOME=str(home / "hermes"), SHELL="/bin/bash",
               NO_COLOR="1", TERM="dumb", CI="true")
    script = f"source {shlex.quote(INSTALL_SH.as_posix())} --manifest\n" + "wire_shell_path\n" * runs
    result = subprocess.run(["bash", "-c", script], env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def _login_path(home: Path) -> list[str]:
    result = subprocess.run(["bash", "-lc", 'printf %s "$PATH"'],
                            env={"HOME": str(home), "PATH": "/usr/local/bin:/usr/bin:/bin"},
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout.split(":")


def _home(tmp_path: Path, **files: str) -> Path:
    (tmp_path / ".local" / "bin").mkdir(parents=True)
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("name, text", [(".bashrc", FEDORA_BASHRC), (".profile", DEBIAN_PROFILE)])
def test_bare_assignment_counts_as_existing_setup(tmp_path, name, text):
    home = _home(tmp_path, **{name: text})
    _wire(home)
    assert (home / name).read_text(encoding="utf-8-sig") == text


def test_fedora_login_shell_has_local_bin_once(tmp_path):
    home = _home(tmp_path, **{".bashrc": FEDORA_BASHRC, ".bash_profile": FEDORA_BASH_PROFILE})
    _wire(home, runs=2)
    assert (home / ".bash_profile").read_text(encoding="utf-8-sig").count(MARKER) == 1
    assert _login_path(home).count(str(home / ".local" / "bin")) == 1
