"""Test scripts/mw/mw-setup.sh installs the MW tools and smoke-checks them (MW-P4)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP = REPO_ROOT / "scripts" / "mw" / "mw-setup.sh"

EXPECTED = {
    "mw-work-locks",
    "mw-menu-gate",
    "mw-page-check",
    "mw-doctor",
    "mw-rtm-report",
    "mw-wow-report",
    "mw-backend-check",
    "mw-spec-check",
}


def test_setup_installs_and_smokes_all_tools(tmp_path: Path):
    bindir = tmp_path / "bin"
    env = os.environ.copy()
    env["MW_BIN_DIR"] = str(bindir)
    env["MW_PYTHON"] = sys.executable  # the venv python running the tests
    proc = subprocess.run(
        ["bash", str(SETUP)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    # every expected tool symlinked
    installed = {p.name for p in bindir.iterdir()}
    assert EXPECTED <= installed, (EXPECTED - installed, installed)
    # symlinks resolve to real files under scripts/mw
    for name in EXPECTED:
        link = bindir / name
        assert link.is_symlink()
        assert link.resolve().is_file()
    # smoke summary: 0 failures
    assert "ทดสอบไม่ผ่าน 0" in proc.stdout


def test_setup_fails_closed_when_python_missing(tmp_path: Path):
    env = os.environ.copy()
    env["MW_BIN_DIR"] = str(tmp_path / "bin")
    env["MW_PYTHON"] = "python-does-not-exist-xyz"
    proc = subprocess.run(
        ["bash", str(SETUP)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    assert "ไม่พบ" in (proc.stdout + proc.stderr)


def test_setup_refuses_to_clobber_real_file(tmp_path: Path):
    """GPT-5: ln -sf must NOT silently overwrite a user's real (non-symlink) file."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    victim = bindir / "mw-doctor"
    victim.write_text("MY REAL FILE — do not clobber", encoding="utf-8")
    env = os.environ.copy()
    env["MW_BIN_DIR"] = str(bindir)
    env["MW_PYTHON"] = sys.executable
    proc = subprocess.run(["bash", str(SETUP)], capture_output=True, text=True, env=env)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    # the real file is untouched (not replaced by a symlink)
    assert not victim.is_symlink()
    assert victim.read_text(encoding="utf-8") == "MY REAL FILE — do not clobber"


def test_setup_symlink_to_dir_does_not_clobber_inside(tmp_path: Path):
    """GPT-5 round 2: a pre-existing symlink-to-directory must not let ln follow into it
    and overwrite a real file inside that directory."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    realdir = tmp_path / "elsewhere"
    realdir.mkdir()
    inside = realdir / "mw-doctor"
    inside.write_text("REAL FILE INSIDE DIR", encoding="utf-8")
    # bindir/mw-doctor is a symlink pointing at the directory
    (bindir / "mw-doctor").symlink_to(realdir)
    env = os.environ.copy()
    env["MW_BIN_DIR"] = str(bindir)
    env["MW_PYTHON"] = sys.executable
    proc = subprocess.run(["bash", str(SETUP)], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    # the real file inside the previously-linked dir is intact
    assert inside.read_text(encoding="utf-8") == "REAL FILE INSIDE DIR"
    # and bindir/mw-doctor is now a symlink to the tool file (not the dir)
    link = bindir / "mw-doctor"
    assert link.is_symlink()
    assert link.resolve().name == "mw_doctor.py"
