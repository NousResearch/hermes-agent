"""setup-hermes.sh wires ~/.local/bin into a startup file a NEW login shell reads (#149621).

The bootstrap used to touch ``~/.bash_profile`` for bash users without a
``~/.bashrc`` and then skip writing anything when the *installing* shell already
had ``~/.local/bin`` on PATH. That left a 0-byte ``~/.bash_profile`` behind, and
because a login bash reads only the first of ``~/.bash_profile``,
``~/.bash_login``, ``~/.profile``, it hid the ``~/.profile`` line the installer
writes — ``hermes`` disappears from every SSH/console login.

The assertions therefore run the wiring and then read PATH from the file that
login sequence actually executes; a passing run must also leave no file that
shadows ``~/.profile`` and must repair one that already exists.
"""
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("posix")

SETUP = Path(__file__).resolve().parents[3] / "setup-hermes.sh"
MARKER = "# Hermes Agent — ensure ~/.local/bin is on PATH"

# Bash's login half ("looks for ~/.bash_profile, ~/.bash_login, and ~/.profile, in that
# order, and reads and executes commands from the first one that exists"), under
# --noprofile so the host's /etc/profile cannot rewrite HOME before bash looks.
_LOGIN_FILES = ('for f in ~/.bash_profile ~/.bash_login ~/.profile; '
                'do if [ -r "$f" ]; then . "$f"; break; fi; done')

# Debian /etc/skel/.profile tail: the line a fresh install already carries.
DEBIAN_PROFILE = """\
# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
"""


def _home(tmp_path: Path, **files: str) -> Path:
    (tmp_path / ".local" / "bin").mkdir(parents=True)
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


def _wire(home: Path, path: str, shell: str = "/bin/bash", runs: int = 1) -> str:
    """Run the wiring with an explicit inherited PATH (the installing shell's)."""
    env = {"HOME": str(home), "SHELL": shell, "PATH": path, "NO_COLOR": "1", "TERM": "dumb"}
    script = f"source {shlex.quote(SETUP.as_posix())}\n" + "wire_shell_path\n" * runs
    result = subprocess.run([shutil.which("bash") or "/bin/bash", "-c", script], env=env, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _login_path(home: Path) -> list[str]:
    result = subprocess.run(
        [shutil.which("bash") or "/bin/bash", "--noprofile", "--norc",
         "-c", f'{_LOGIN_FILES}; printf %s "$PATH"'],
        env={"HOME": str(home), "PATH": "/usr/local/bin:/usr/bin:/bin"},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.split(":")


def _text(home: Path, name: str) -> str:
    return (home / name).read_text(encoding="utf-8-sig")


def test_bash_without_bashrc_never_creates_bash_profile(tmp_path):
    # The reported estate: ~/.profile already wired, no ~/.bashrc, and the installing
    # shell had ~/.local/bin on PATH — the run that left a 0-byte shadow file behind.
    home = _home(tmp_path, **{".profile": DEBIAN_PROFILE})
    _wire(home, f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin")
    assert not (home / ".bash_profile").exists(), "a login bash would read this instead of ~/.profile"
    assert _text(home, ".profile") == DEBIAN_PROFILE
    assert str(home / ".local" / "bin") in _login_path(home)


def test_an_empty_bash_profile_shadowing_profile_is_repaired(tmp_path):
    # The estate the report describes: a previous run left the 0-byte file behind while
    # ~/.profile already carried the line, so every login shell lost it.
    home = _home(tmp_path, **{".profile": DEBIAN_PROFILE, ".bash_profile": ""})
    _wire(home, f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin")
    assert MARKER in _text(home, ".bash_profile"), "the file a login bash reads must carry the line"
    assert _text(home, ".profile") == DEBIAN_PROFILE
    assert str(home / ".local" / "bin") in _login_path(home)
