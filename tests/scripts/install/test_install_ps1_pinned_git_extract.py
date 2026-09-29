"""Pinned git staging must not need a bzip2-capable tar (#122512, #122774).

Stock Windows 10 ships a System32 tar.exe without a bzip2 filter, so the
old .tar.bz2 pin died at stage=prerequisites ("tar.exe: Error opening
archive: Can't initialize filter; unable to run program \"bzip2 -d\"").
Get-PinnedGit now stages the PortableGit self-extractor, which needs
neither tar nor bzip2. The driver below runs the real Get-PinnedGit
against the real pinned archive with a PATH that carries no bzip2 (or
anything else to fall back on) and demands a working git.exe plus the
bundled bash contract pm/shell.py relies on.
"""
import hashlib
import io
import subprocess
import textwrap
import zipfile
from pathlib import Path

import pytest

from pm.store import tree_digest
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "install.ps1"


@pytest.mark.platforms("windows")
def test_pinned_git_extracts_without_a_bzip2_capable_tar(tmp_path):
    driver = tmp_path / "driver.ps1"
    driver.write_text(
        textwrap.dedent(f"""
            . '{INSTALLER}' -HermesHome '{tmp_path / "home"}'
            $env:HERMES_RUNTIME_DIR = '{tmp_path / "tools"}'
            # This machine has no bzip2 (and nothing else to fall back on).
            $env:PATH = "$env:SystemRoot\\System32;$env:SystemRoot"
            $git = Get-PinnedGit
            if (-not $git) {{ exit 9 }}
            & $git --version 2>$null | Out-Null
            if ($LASTEXITCODE) {{ exit 9 }}
            $entry = Split-Path (Split-Path $git)
            if (-not (Test-Path (Join-Path $entry 'usr\\bin\\bash.exe'))) {{ exit 9 }}
        """),
        encoding="ascii",
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver)],
        capture_output=True, timeout=900,
    )
    out = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert result.returncode == 0, out


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(("corrupt", "wrong_tree"), [(False, False), (True, False), (False, True)])
def test_prepared_git_uses_stock_zip_without_running_raw_installer(tmp_path, dl_server, corrupt, wrong_tree):
    entry = tmp_path / "expected"
    (entry / "cmd").mkdir(parents=True)
    (entry / "usr/bin").mkdir(parents=True)
    (entry / "cmd/git.exe").write_bytes(b"prepared git")
    (entry / "usr/bin/bash.exe").write_bytes(b"prepared bash")
    expected_tree = tree_digest(entry)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cmd/git.exe", b"prepared git")
        zf.writestr("usr/bin/bash.exe", b"prepared bash")
    good = archive.getvalue()
    sha = hashlib.sha256(good).hexdigest()
    RangeHandler.payloads = {"/prepared": b"bad bytes" if corrupt else good}
    prepared_url = url(dl_server, "/prepared")
    raw_url = url(dl_server, "/raw-never-requested")
    pinned_tree = "f" * 64 if wrong_tree else expected_tree
    driver = tmp_path / "prepared-git.ps1"
    driver.write_text(textwrap.dedent(f"""
        . '{INSTALLER}' -HermesHome '{tmp_path / "home"}'
        $env:HERMES_RUNTIME_DIR = '{tmp_path / "tools"}'
        $env:PATH = "$env:SystemRoot\\System32;$env:SystemRoot"
        $target = "win32-$(Get-WindowsArch)"
        $script:GitPinFiles[$target] = @{{
            PreparedUrl = '{prepared_url}'
            PreparedMirrorUrl = '{url(dl_server, "/backup")}'
            PreparedSha256 = '{sha}'
            PreparedDigest = '{pinned_tree}'
            GitHubUrl = '{raw_url}'
            Url = '{raw_url}'
            MirrorUrl = '{raw_url}'
            Sha256 = '{"a" * 64}'
        }}
        try {{
            $git = Get-PinnedGit
            if ({'$true' if corrupt or wrong_tree else '$false'}) {{ exit 9 }}
            if (-not (Test-Path -LiteralPath $git)) {{ exit 9 }}
            if ([IO.File]::ReadAllBytes($git).Length -ne 12) {{ exit 9 }}
        }} catch {{
            if ({'$false' if corrupt or wrong_tree else '$true'} -or $_.Exception.Message -notlike '*digest mismatch*') {{ exit 9 }}
        }}
    """), encoding="ascii")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver)], capture_output=True, timeout=180,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    assert "/raw-never-requested" not in RangeHandler.requests_seen


@pytest.mark.platforms("windows")
def test_prepared_git_http_client_error_never_runs_raw_installer(tmp_path, dl_server):
    RangeHandler.status_codes = {"/prepared": 400}
    RangeHandler.payloads = {"/raw": b"raw must not run"}
    driver = tmp_path / "rejected-git.ps1"
    driver.write_text(textwrap.dedent(f"""
        . '{INSTALLER}' -HermesHome '{tmp_path / "home"}'
        $env:HERMES_RUNTIME_DIR = '{tmp_path / "tools"}'
        $target = "win32-$(Get-WindowsArch)"
        $script:GitPinFiles[$target] = @{{
            PreparedUrl = '{url(dl_server, "/prepared")}'
            PreparedMirrorUrl = '{url(dl_server, "/missing")}'
            PreparedSha256 = '{"a" * 64}'
            GitHubUrl = '{url(dl_server, "/raw")}'
            Url = '{url(dl_server, "/raw")}'
            MirrorUrl = '{url(dl_server, "/raw")}'
            Sha256 = '{"b" * 64}'
        }}
        try {{ Get-PinnedGit | Out-Null; exit 9 }} catch {{
            if ($_.Exception.Message -notlike '*prepared download rejected*') {{ exit 9 }}
        }}
    """), encoding="ascii")
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(driver)], capture_output=True, timeout=180,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
    assert "/prepared" in RangeHandler.requests_seen
    assert "/raw" not in RangeHandler.requests_seen
