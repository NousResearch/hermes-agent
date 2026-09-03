"""LIVE Windows E2E for the fail-closed shim quarantine (#87331).

Runs ONLY on a real Windows host (the on-demand ``windows-venv-e2e.yml``
lane). Reproduces the REAL lock shape from the field report: a process
holding ``hermes.exe`` open WITHOUT FILE_SHARE_DELETE, exactly like a
running launcher — then proves the strict quarantine refuses before any
installer runs, and that the non-contended path still installs.

No mocks: real files, a real child process holding a real Windows handle,
the real rename attempt hitting the real sharing violation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="live Windows shim-lock E2E"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Child that opens a file with GENERIC_READ and NO FILE_SHARE_DELETE —
# the exact sharing mode a running .exe image / desktop backend exhibits.
_HOLDER_CODE = r"""
import ctypes, sys, time
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1  # note: NO FILE_SHARE_DELETE
OPEN_EXISTING = 3
h = ctypes.windll.kernel32.CreateFileW(
    sys.argv[1], GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, 0, None
)
if h == -1 or h == 0xFFFFFFFF:
    print("OPEN_FAILED", flush=True)
    sys.exit(1)
print("HOLDING", flush=True)
time.sleep(120)
"""


@pytest.fixture()
def held_shim(tmp_path: Path):
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    shim = scripts / "hermes.exe"
    shim.write_bytes(b"MZ fake shim")
    (scripts / "hermes-gateway.exe").write_bytes(b"MZ fake shim")
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_CODE, str(shim)],
        stdout=subprocess.PIPE,
        text=True,
    )
    line = holder.stdout.readline().strip()
    if line != "HOLDING":
        holder.kill()
        pytest.fail(f"lock-holder child failed: {line!r}")
    yield scripts, shim
    holder.kill()
    holder.wait()


def test_locked_shim_really_cannot_be_renamed(held_shim):
    """Premise check: the no-FILE_SHARE_DELETE handle blocks rename."""
    _scripts, shim = held_shim
    with pytest.raises(OSError):
        os.rename(shim, shim.with_name("hermes.exe.old.premise"))


def test_strict_quarantine_refuses_against_real_lock(held_shim, monkeypatch):
    import hermes_cli.main as cli_main

    scripts, _shim = held_shim
    install_ran: list = []
    monkeypatch.setattr(
        cli_main,
        "_run_install_with_heartbeat",
        lambda cmd, env=None: install_ran.append(cmd),
    )

    with pytest.raises(cli_main.ShimQuarantineError) as exc_info:
        cli_main._run_quarantined_install(
            ["would-be", "uv", "pip", "install"],
            scripts_dir=scripts,
            strict_quarantine=True,
        )

    assert install_ran == [], "installer ran against a contended venv"
    assert "hermes.exe" in exc_info.value.failed_shims
    # The unlocked sibling's rename was rolled back — venv untouched.
    assert (scripts / "hermes-gateway.exe").exists()
    assert not list(scripts.glob("*.old.*"))


def test_recovery_installer_refuses_against_real_lock(held_shim, monkeypatch):
    import hermes_cli._install_repair as ir

    scripts, _shim = held_shim
    monkeypatch.setattr(ir, "_venv_scripts_dir", lambda root: scripts)
    run_calls: list = []
    monkeypatch.setattr(ir.subprocess, "run", lambda *a, **k: run_calls.append(a))

    with pytest.raises(ir.ShimQuarantineError):
        ir._run_install_cmd(["fake"], env=None, root=scripts.parent.parent)
    assert run_calls == []


def test_release_then_strict_quarantine_succeeds(tmp_path, monkeypatch):
    """After the holder exits, the same strict path proceeds normally."""
    import hermes_cli.main as cli_main

    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "hermes.exe").write_bytes(b"MZ fake shim")

    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_CODE, str(scripts / "hermes.exe")],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout.readline().strip() == "HOLDING"
    holder.kill()
    holder.wait()
    time.sleep(0.3)  # handle teardown

    install_ran: list = []
    monkeypatch.setattr(
        cli_main,
        "_run_install_with_heartbeat",
        lambda cmd, env=None: install_ran.append(cmd),
    )
    cli_main._run_quarantined_install(
        ["fake"], scripts_dir=scripts, strict_quarantine=True
    )
    assert install_ran == [["fake"]]
