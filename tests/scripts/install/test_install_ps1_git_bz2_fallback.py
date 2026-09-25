"""The pinned git .tar.bz2 must unpack even when the inbox bsdtar cannot (#122774).

Older Windows 10 builds ship ``System32\\tar.exe`` (bsdtar/libarchive 3.3.x)
without a bzip2 filter: on a fresh host it cannot decompress the pinned
git-for-windows ``.tar.bz2`` ("Can't initialize filter; unable to run program
\"bzip2 -d\"") and the installer failed at the prerequisites stage. install.ps1
now retries the extraction with the pinned uv's managed Python (stdlib
``tarfile`` reads bz2 natively), skipping the MSYS /proc links pm skips
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
                             ("etc/mtab", "/proc/mounts"),
                             # The pm predicate skips ANY dev/* -> /proc/* symlink,
                             # not just the five git-for-windows ships today.
                             ("dev/newlink", "/proc/newthing")):
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
    the archive and skips exactly the MSYS /proc symlinks."""
    archive = tmp_path / "git.tar.bz2"
    _write_archive(archive)
    dest = tmp_path / "unpacked"
    dest.mkdir()
    py = os.sys.executable.replace(os.sep, "/")
    script = f"""
$ErrorActionPreference = "Stop"
. "{INSTALLER}"
function Resolve-ManagedPython {{ return "{py}" }}
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
    # A future git build adding another dev/* -> /proc/* link (the pm
    # predicate's general form) is skipped too, not just the five literals.
    assert not (dest / "dev" / "newlink").exists()


def test_bsdtar_failure_falls_back_to_python_extractor(tmp_path):
    """When the inbox tar.exe fails (a filter-less bsdtar on older Windows 10
    builds), Get-PinnedGit still stages git through the Python fallback.

    The Invoke-Native stand-in must fail the exact scriptblock the bsdtar
    call site builds (`& $inboxTar @excludes -xf $tarPath -C $extractDir`,
    resolved from the $inboxTar variable) -- a literal ``tar.exe`` never
    appears in that text, so a stand-in keyed on it would let the runner's
    real bsdtar succeed and never exercise the fallback.
    """
    archive = tmp_path / "git.tar.bz2"
    sha = _write_archive(archive)
    store = tmp_path / "store"
    py = os.sys.executable.replace(os.sep, "/")
    script = f"""
$ErrorActionPreference = "Stop"
$env:HERMES_RUNTIME_DIR = "{store}"
. "{INSTALLER}"
function Resolve-ManagedPython {{ return "{py}" }}
# Stand in for the filter-less inbox bsdtar: fail exactly the archive
# extraction call, run everything else (the Python extractor) for real.
function Invoke-Native([scriptblock]$Command) {{
    if ("$Command" -like "*-xf*") {{ $global:LASTEXITCODE = 1; return }}
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
