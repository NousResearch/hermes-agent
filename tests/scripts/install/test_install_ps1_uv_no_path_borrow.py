"""``install.ps1`` must not adopt a uv it found on PATH either (#101269).

Twin of ``test_install_sh_uv_no_path_borrow`` for ``Get-Uv``: a uv on PATH at
least as new as the pin must not become the bootstrap's uv. That would hand
byte authority over the whole install to a user-controlled binary and leave the
store slot empty.

Needs PowerShell (``pwsh`` or Windows PowerShell 5.1), not Windows. The
runnability probe (``Test-UvAtLeastPin``, which executes the staged ``uv.exe``)
is stubbed: a Windows PE cannot be staged as a text fixture, and the assertion
is about which uv ``Get-Uv`` picks, not about the probe. Without the stub a
Windows host would discard the fixture and download the pin from the network.
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "install.ps1"

_POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(
    _POWERSHELL is None, reason="running install.ps1 needs pwsh or powershell"
)


def test_get_uv_stages_the_pin_even_when_a_newer_uv_is_on_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    marker = tmp_path / "path-uv-was-executed"

    user_bin = tmp_path / "user-bin"
    user_bin.mkdir()
    # A uv that claims to be NEWER than the pin and records being executed.
    fake = user_bin / "uv"
    fake.write_text(
        "#!/bin/sh\n"
        f"touch {shlex.quote(str(marker))}\n"
        'echo "uv 99.0.0 (fake 2099-01-01)"\n',
        encoding="utf-8",
    )
    fake.chmod(0o755)

    # Pre-seed the store slot so Get-Uv finds the pin and never reaches the
    # network, so the only variable under test is which uv it chooses.
    # win32-x64 is what Get-WindowsArch reports when the machine arch is
    # anything but ARM64.
    store_entry = home / "tools" / "uv-0.12.3-win32-x64"
    store_entry.mkdir(parents=True)
    staged = store_entry / "uv.exe"
    staged.write_text('#!/bin/sh\necho "uv 0.12.3 (fixture)"\n', encoding="utf-8")
    staged.chmod(0o755)

    env = {**os.environ, "HERMES_HOME": str(home)}
    env["PATH"] = f"{user_bin}{os.pathsep}{env.get('PATH', '')}"
    env.pop("HERMES_RUNTIME_DIR", None)

    # Stub only the runnability probe, never Get-Uv itself: a text fixture cannot
    # run as a Windows PE, and the old borrow path would still return the PATH uv.
    stub_probe = "function Test-UvAtLeastPin([string]$Path) { return $true }; "
    script = f'$ErrorActionPreference = "Stop"; . "{INSTALLER}"; {stub_probe}Get-Uv'
    result = subprocess.run(
        [_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.stdout
    resolved = lines[-1].replace("\\", "/")

    assert resolved == str(staged).replace("\\", "/"), result.stdout
    assert not marker.exists(), (
        "Get-Uv executed the uv on PATH — the pin is the byte authority, "
        "not a user-installed binary"
    )
