"""The pinned git .tar.bz2 must unpack even when the inbox bsdtar cannot (#122774).

Stock Windows 10/11 ships ``System32\\tar.exe`` (bsdtar) built without a
bzip2 filter: on a fresh host it cannot decompress the pinned git-for-windows
``.tar.bz2`` ("Can't initialize filter; unable to run program \"bzip2 -d\"")
and the installer failed at the prerequisites stage. install.ps1 now retries
the extraction with the pinned uv's managed Python (stdlib ``tarfile`` reads
bz2 natively), skipping exactly the MSYS /proc links pm skips
(pm/store.py ``extract_tar(git_msys=True)``).

These drive the real installer functions, dot-sourced, under Windows
PowerShell 5.1 -- the shell the one-liner actually delivers the script into.
"""
import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"
GIT_PAYLOAD = b"git.exe fixture bytes\n"


def _powershell() -> str:
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is part of every supported Windows"
    return powershell


def _write_archive(path: Path) -> str:
    """A git-for-windows-shaped tar.bz2: cmd/git.exe plus the five MSYS
    /proc symlinks pm's extractor skips."""
    bash = b"bash fixture bytes\n"
    with tarfile.open(path, "w:bz2") as tf:
        member = tarfile.TarInfo("cmd/git.exe")
        member.size = len(GIT_PAYLOAD)
        tf.addfile(member, io.BytesIO(GIT_PAYLOAD))
        # A second top-level dir keeps the installer's single-wrapper-dir
        # flattening honest, like the real artifact's mingw64/usr/etc.
        other = tarfile.TarInfo("usr/bin/bash.exe")
        other.size = len(bash)
        tf.addfile(other, io.BytesIO(bash))
        for name, target in (("dev/fd", "/proc/self/fd"),
                             ("dev/stdin", "/proc/self/fd/0"),
                             ("dev/stdout", "/proc/self/fd/1"),
                             ("dev/stderr", "/proc/self/fd/2"),
                             ("etc/mtab", "/proc/mounts")):
            link = tarfile.TarInfo(name)
            link.type = tarfile.SYMTYPE
            link.linkname = target
            tf.addfile(link)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_probe(script: str, workdir: Path, timeout: int = 240) -> subprocess.CompletedProcess:
    probe = workdir / "probe.ps1"
    probe.write_text(script, encoding="utf-8-sig")
    return subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(probe)],
        capture_output=True, text=True, timeout=timeout, cwd=str(workdir),
    )


def test_python_fallback_unpacks_archive_and_skips_proc_links(tmp_path):
    """With bsdtar out of the picture, Invoke-PythonTarExtract alone unpacks
    the archive and skips exactly the five /proc symlinks."""
    archive = tmp_path / "git.tar.bz2"
    _write_archive(archive)
    dest = tmp_path / "unpacked"
    dest.mkdir()
    py = os.sys.executable.replace(os.sep, "/")
    script = f"""
$ErrorActionPreference = "Stop"
. "{INSTALLER}"
function Get-BootstrapPython {{ return "{py}" }}
Invoke-PythonTarExtract -Archive "{archive}" -Destination "{dest}"
if ($LASTEXITCODE) {{ "extract exit=$LASTEXITCODE"; exit 1 }}
"ok"
"""
    result = _run_probe(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "cmd" / "git.exe").read_bytes() == GIT_PAYLOAD
    # The /proc symlinks cannot exist on Windows and are skipped; the real
    # directories that share their parents are not.
    assert not (dest / "dev" / "fd").exists()
    assert not (dest / "etc" / "mtab").exists()


def test_bsdtar_failure_falls_back_to_python_extractor(tmp_path):
    """When the inbox tar.exe fails (a filter-less bsdtar on stock Windows),
    Get-PinnedGit still stages git through the Python fallback."""
    archive = tmp_path / "git.tar.bz2"
    sha = _write_archive(archive)
    store = tmp_path / "store"
    py = os.sys.executable.replace(os.sep, "/")
    script = f"""
$ErrorActionPreference = "Stop"
$env:HERMES_RUNTIME_DIR = "{store}"
. "{INSTALLER}"
function Get-BootstrapPython {{ return "{py}" }}
# Stand in for the filter-less inbox bsdtar: fail every tar invocation,
# run everything else (the Python extractor) for real.
function Invoke-Native([scriptblock]$Command) {{
    if ("$Command" -like "*tar.exe*") {{ $global:LASTEXITCODE = 1; return }}
    & $Command
}}
$script:GitPinVersion = "9.9.9"
$script:GitPinFiles["win32-" + (Get-WindowsArch)] = @{{
    Url = "{archive.as_uri()}"; MirrorUrl = ""; Sha256 = "{sha}"
}}
$git = Get-PinnedGit
"staged=$git"
if (-not $git -or $LASTEXITCODE) {{ exit 1 }}
"""
    result = _run_probe(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    staged = sorted(store.glob("git-*/cmd/git.exe"))
    assert len(staged) == 1
    assert staged[0].read_bytes() == GIT_PAYLOAD


def test_bootstrap_python_tolerates_missing_checkout(tmp_path):
    """prerequisites runs before a checkout exists, so Get-BootstrapPython
    must not require pm/lock.json under the install dir."""
    fake_uv = tmp_path / "fake-uv.cmd"
    fake_uv.write_text("@echo C:\\fake\\python.exe\r\n@exit /b 0\r\n", encoding="utf-8")
    install_dir = tmp_path / "no-checkout-yet"
    install_dir.mkdir()
    script = f"""
$ErrorActionPreference = "Stop"
. "{INSTALLER}" -InstallDir "{install_dir}"
function Get-Uv {{ return "{fake_uv}" }}
$py = Get-BootstrapPython
"py=$py"
"""
    result = _run_probe(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "py=C:\\fake\\python.exe" in result.stdout
