"""install.sh must not provision cua-driver where nobody asked for it (#104413).

The Computer Use pre-install is a convenience for FRESH installs on hosts that
can drive a desktop. A headless VPS (no graphical session) can never use the
toolset, and re-running install.sh over an existing checkout is an *update* —
like ``hermes update``, it may repair a driver that is already present but must
never introduce a new third-party binary (``~/.cua-driver``,
``~/.local/bin/cua-driver``, a ``~/.bashrc`` PATH line) silently. Enabling the
toolset (``hermes tools``, the dashboard, ``hermes computer-use install``) is
the consent that installs it; ``--with-computer-use`` forces the pre-install.

These exercise the real shell functions against a temp HOME rather than
asserting on the text of install.sh (pattern: test_install_sh_bootstrap_marker.py).
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

_FUNCTIONS = ("find_cua_driver", "host_can_use_computer_use",
              "cua_driver_runtime_compatible", "install_computer_use_driver")

# Everything the runtime contract check in install.sh looks for.
_COMPATIBLE_MANIFEST = ('{"mcp_invocation": {"args": ["--socket", "--grant", "--permission-mode", '
                        '"--capability-manifest", "--approve-capability-manifest", "--embedded"]}}')


def _plant_driver(home: Path, version: str) -> None:
    """A fake ~/.local/bin/cua-driver answering --version / manifest (NOT on PATH)."""
    path = home / ".local" / "bin" / "cua-driver"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'[ "$1" = "--version" ] && {{ echo "cua-driver {version}"; exit 0; }}\n'
        f"[ \"$1\" = \"manifest\" ] && {{ echo '{_COMPATIBLE_MANIFEST}'; exit 0; }}\n"
        "exit 1\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def run_install_step(tmp_path, *, display=None, existing_install=False, with_flag=False,
                     skip_flag=False):
    """Source the cua-driver functions from install.sh and run install_computer_use_driver.

    ``run_with_timeout`` is stubbed to record its argv (the upstream installer is never
    reached), so the assertions are about WHETHER and HOW it would have been invoked.
    Returns ``(completed_process, installer_argv_or_None)``.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir(exist_ok=True)
    if existing_install:  # install.sh's own completion stamp (see detect_install_method)
        (install_dir / ".install_method").write_text("git\n", encoding="utf-8")
    invoked = tmp_path / "installer-invoked"
    invoked.unlink(missing_ok=True)  # tests call this more than once per tmp_path
    extract = " ".join(f"-e '/^{fn}()/,/^}}/p'" for fn in _FUNCTIONS)
    script = f"""
set -e
HOME={home!s}
INSTALL_DIR={install_dir!s}
DISTRO=ubuntu
SKIP_COMPUTER_USE={'true' if skip_flag else 'false'}
WITH_COMPUTER_USE={'true' if with_flag else 'false'}
unset DISPLAY WAYLAND_DISPLAY XDG_SESSION_TYPE
{f'export DISPLAY={display}' if display else ''}
eval "$(sed -n {extract} {INSTALL_SH!s})"
log_info() {{ echo "INFO: $*"; }}
log_warn() {{ echo "WARN: $*"; }}
log_success() {{ echo "OK: $*"; }}
run_with_timeout() {{ printf '%s\\n' "$@" > {invoked!s}; return 0; }}
install_computer_use_driver
"""
    env = {**os.environ, "HOME": str(home), "PATH": "/usr/bin:/bin"}  # no cua-driver on PATH
    result = subprocess.run(["bash", "-c", script], capture_output=True, timeout=30, env=env,
                            text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    argv = invoked.read_text(encoding="utf-8").splitlines() if invoked.is_file() else None
    return result, argv


@pytest.mark.linux_only
def test_headless_fresh_install_skips_and_points_at_the_opt_in(tmp_path):
    """The #104413 case: a Linux VPS with no graphical session never gets a silent install.
    ``linux_only``: the display gate is Linux-specific — macOS/Windows always have a desktop."""
    result, argv = run_install_step(tmp_path)
    assert argv is None
    assert "hermes computer-use install" in result.stdout
    assert not (tmp_path / "home" / ".cua-driver").exists()


@pytest.mark.macos_only
def test_macos_is_never_treated_as_headless(tmp_path):
    """No DISPLAY on a Mac is normal (Finder-launched shells); the pre-install still happens."""
    _, argv = run_install_step(tmp_path)
    assert argv is not None


@pytest.mark.linux_only
def test_with_flag_forces_headless_install_and_skip_flag_still_wins(tmp_path):
    _, argv = run_install_step(tmp_path, with_flag=True)
    assert argv is not None
    _, argv = run_install_step(tmp_path, with_flag=True, skip_flag=True)
    assert argv is None


def test_fresh_desktop_install_still_provisions(tmp_path):
    """The 'config flip, not a surprise fetch' UX is kept where the toolset can work."""
    _, argv = run_install_step(tmp_path, display=":0")
    assert argv is not None


def test_update_run_repairs_but_never_adds(tmp_path):
    """Re-running install.sh over an existing checkout is an update: a missing driver stays
    missing (until the toolset is enabled), a present-but-old one is still repaired."""
    _, argv = run_install_step(tmp_path, display=":0", existing_install=True)
    assert argv is None
    _plant_driver(tmp_path / "home", "0.9.0")
    result, argv = run_install_step(tmp_path, display=":0", existing_install=True)
    assert argv is not None
    assert "repairing" in result.stdout


def test_compatible_driver_off_path_is_found_and_left_alone(tmp_path):
    """~/.local/bin/cua-driver counts even when ~/.local/bin is not on this shell's PATH
    (service accounts) — previously every run re-downloaded it."""
    _plant_driver(tmp_path / "home", "0.20.3")
    result, argv = run_install_step(tmp_path, existing_install=True)
    assert argv is None
    assert "already installed and compatible" in result.stdout


def test_installer_never_edits_shell_rc_or_phones_home_and_says_what_it_writes(tmp_path):
    """Upstream appends to ~/.bashrc and fires an install-event by default; Hermes owns the
    PATH line (setup_path) and its telemetry policy is opt-in. The 660s ceiling stays."""
    result, argv = run_install_step(tmp_path, display=":0")
    assert argv[:2] == ["660", "env"]
    assert "CUA_DRIVER_RS_NO_MODIFY_PATH=1" in argv
    assert "CUA_DRIVER_RS_TELEMETRY_ENABLED=0" in argv
    assert "~/.cua-driver" in result.stdout and "~/.local/bin/cua-driver" in result.stdout


def test_with_computer_use_flag_is_accepted_by_the_argument_parser():
    """--help exits 0 only after every preceding flag parsed; an unknown flag exits 1."""
    def _run(flag):
        return subprocess.run(["bash", str(INSTALL_SH), flag, "--help"], capture_output=True,
                              timeout=30, text=True, encoding="utf-8", errors="replace")
    assert _run("--with-computer-use").returncode == 0
    assert _run("--with-computer-use-typo").returncode != 0
