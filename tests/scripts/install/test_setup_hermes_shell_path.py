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
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("posix")

SETUP = Path(__file__).resolve().parents[3] / "setup-hermes.sh"
MARKER = "# Hermes Agent — ensure ~/.local/bin is on PATH"

# The fix moved the block into wire_shell_path(); the legacy `if [ "$os" != win32 ]`
# block is still what upstream main runs, so the sabotage run (source reverted, this
# test kept) drives the same contract instead of dying on a missing function.
_FUNCTION = r'/^wire_shell_path\(\) \{/{p=1} p{print} p&&/^\}$/{exit}'
_LEGACY = r'/^if \[ "\$os" != win32 \]; then$/{p=1} p{print} p&&/^fi$/{exit}'

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


def _wiring() -> str:
    """The startup-file wiring setup-hermes.sh runs, as a bash snippet."""
    for pattern, invoker in ((_FUNCTION, "wire_shell_path\n"), (_LEGACY, "")):
        out = subprocess.run(["awk", pattern, str(SETUP)], capture_output=True, text=True, check=False)
        if out.stdout.strip():
            return out.stdout + invoker
    raise AssertionError(f"setup-hermes.sh carries no shell-startup wiring: {SETUP}")


def _home(tmp_path: Path, **files: str) -> Path:
    (tmp_path / ".local" / "bin").mkdir(parents=True)
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


def _wire(home: Path, path: str, shell: str = "/bin/bash", runs: int = 1) -> str:
    """Run the wiring with an explicit inherited PATH (the installing shell's)."""
    env = {"HOME": str(home), "SHELL": shell, "PATH": path,
           "os": "linux", "GREEN": "", "RED": "", "NC": "", "NO_COLOR": "1", "TERM": "dumb"}
    result = subprocess.run(["bash", "-c", _wiring() * runs], env=env, stdin=subprocess.DEVNULL,
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


def test_inherited_path_is_not_evidence_a_new_shell_is_wired(tmp_path):
    # No startup file at all: the inherited PATH says nothing about the next login.
    home = _home(tmp_path)
    _wire(home, "/usr/local/bin:/usr/bin:/bin")
    assert MARKER in _text(home, ".profile")
    assert not (home / ".bash_profile").exists()
    assert str(home / ".local" / "bin") in _login_path(home)


def test_an_empty_bash_profile_shadowing_profile_is_repaired(tmp_path):
    # The estate the report describes: a previous run left the 0-byte file behind while
    # ~/.profile already carried the line, so every login shell lost it.
    home = _home(tmp_path, **{".profile": DEBIAN_PROFILE, ".bash_profile": ""})
    _wire(home, f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin")
    assert MARKER in _text(home, ".bash_profile"), "the file a login bash reads must carry the line"
    assert _text(home, ".profile") == DEBIAN_PROFILE
    assert str(home / ".local" / "bin") in _login_path(home)


def test_an_existing_bash_login_gets_the_line_too(tmp_path):
    # bash reads ~/.bash_login before ~/.profile, so the .profile line alone is hidden.
    home = _home(tmp_path, **{".bash_login": "# mine\n"})
    _wire(home, "/usr/local/bin:/usr/bin:/bin")
    assert str(home / ".local" / "bin") in _login_path(home)


def test_wiring_twice_appends_once(tmp_path):
    home = _home(tmp_path, **{".bashrc": "# mine\n"})
    _wire(home, "/usr/local/bin:/usr/bin:/bin", runs=2)
    assert _text(home, ".bashrc").count(MARKER) == 1
    assert str(home / ".local" / "bin") in _login_path(home)


def test_zsh_users_get_zshrc_only(tmp_path):
    home = _home(tmp_path, **{".zshrc": "# mine\n"})
    _wire(home, "/usr/local/bin:/usr/bin:/bin", shell="/usr/bin/zsh")
    assert MARKER in _text(home, ".zshrc")
    assert not (home / ".profile").exists()
    assert not (home / ".bash_profile").exists()


def test_fish_users_get_their_config_file(tmp_path):
    home = _home(tmp_path)
    _wire(home, "/usr/local/bin:/usr/bin:/bin", shell="/usr/bin/fish")
    config = _text(home, ".config/fish/config.fish")
    assert 'fish_add_path "$HOME/.local/bin"' in config
    assert not (home / ".profile").exists()
