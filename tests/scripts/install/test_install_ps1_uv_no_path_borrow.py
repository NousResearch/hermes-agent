"""``install.ps1`` must not adopt a uv it found on PATH (#101269).

Twin of ``test_install_sh_uv_no_path_borrow`` for ``Get-Uv``: whatever uv PATH
offers, the pinned store artifact is the byte authority. Needs PowerShell, not
Windows; the runnability probe is stubbed because a Windows PE cannot be staged
as a text fixture — otherwise a Windows host would download the pin.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "install.ps1"

_POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
# ``any``: the OS lanes import only platforms-marked files, so without it this
# file is never selected by the macOS/Windows lanes even though it is designed to
# run under POSIX pwsh too. The skipif still decides — PowerShell is the real
# prerequisite, not the host OS.
pytestmark = [
    pytest.mark.skipif(
        _POWERSHELL is None, reason="running install.ps1 needs pwsh or powershell"
    ),
    pytest.mark.platforms("any"),
]


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
    # network, so the only variable under test is which uv it chooses. The slot
    # name follows pm/lock.json (a frozen copy would break on the next uv bump)
    # and both arches Get-WindowsArch can report, so an ARM64 host seeds the
    # slot it will actually ask for instead of falling through to a download.
    pin = json.loads((ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))["packages"]["uv"]["version"]
    staged = []
    for arch in ("x64", "arm64"):
        store_entry = home / "tools" / f"uv-{pin}-win32-{arch}"
        store_entry.mkdir(parents=True)
        exe = store_entry / "uv.exe"
        exe.write_text(f'#!/bin/sh\necho "uv {pin} (fixture)"\n', encoding="utf-8")
        exe.chmod(0o755)
        staged.append(exe)

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

    assert resolved in [str(path).replace("\\", "/") for path in staged], result.stdout
    assert not marker.exists(), (
        "Get-Uv executed the uv on PATH — the pin is the byte authority, "
        "not a user-installed binary"
    )
