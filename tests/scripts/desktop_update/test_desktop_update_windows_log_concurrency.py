"""Regression: concurrent writers must not lose hand-off log lines.

``scripts/desktop-update/windows.ps1`` used to append to
``desktop-update-handoff.log`` with ``Add-Content``. That cmdlet is unsafe
once a second writer is live: PowerShell caches a content writer per path, so
the second writer receives a handle the first one already consumed. Measured
on the shape this fixture reproduces (8 writers x 100 lines, #126152):

* Windows PowerShell 5.1 -- a ``GetContentWriterArgumentError`` ("stream is
  not readable") error flood;
* PowerShell 7 -- no errors and **58% of lines silently missing**.

The hand-off log is the only record of why a Desktop update failed, and a
second writer is a real event, not a hypothesis: two hand-off processes can
overlap (the marker claim is a plain write with no mutual exclusion, and its
failure path logs a warning and continues), and every future runspace writer
joins the same race. ``Write-HandoffLog`` now serializes appends behind a
named, per-log-path mutex and one ``FileStream`` per write.

The fixture runs the REAL ``Write-HandoffLog`` (fetched as function text and
re-homed into each runspace with its own lazily created mutex instance, the
same way an overlapping process joins the named mutex) from a runspace pool,
then holds the file to the exact line count, per-writer count, and intact
line shape. It is ``platforms("windows")`` because Linux CI cannot execute
the PowerShell hand-off.

It runs under both interpreters the hand-off can be started by: Windows
PowerShell 5.1 and PowerShell 7 (the desktop shim launches ``pwsh``). The log
writer once used a ``FileStream`` constructor that exists only on .NET
Framework; under pwsh it threw, the writer's ``catch`` swallowed it, and
``desktop-update-handoff.log`` was never created (#156217).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _run_log_concurrency_self_test(powershell: str, tmp_path: Path) -> None:
    """Drive 8 concurrent runspaces through the real Write-HandoffLog.

    A pass means all 800 lines landed: the total matches, each writer's 100
    lines match, and no line lost its timestamp-prefix + payload shape (the
    signature of a torn interleave). Under the pre-fix ``Add-Content`` the
    same run drops lines on PowerShell 7 and errors on 5.1.
    """
    env = dict(os.environ)
    # The fixture writes its target log under TEMP; point that at tmp_path
    # so the test leaves nothing behind.
    env["TEMP"] = os.fspath(tmp_path)
    env["TMP"] = os.fspath(tmp_path)
    # The self-test touches no update state: no marker claim (which its exit would never release).
    home = tmp_path / "home"
    home.mkdir()
    env["HERMES_HOME"] = os.fspath(home)

    argv = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        os.fspath(WINDOWS_PS1),
        "-SelfTestLogConcurrency",
    ]
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env=env,
        cwd=os.fspath(REPO_ROOT),
    )

    (tmp_path / "log-concurrency.stdout.log").write_text(
        result.stdout, encoding="utf-8"
    )
    (tmp_path / "log-concurrency.stderr.log").write_text(
        result.stderr, encoding="utf-8"
    )
    diagnosis = result.stdout[-6000:] + result.stderr[-6000:]
    if "LOG-CONCURRENCY SELF-TEST: PASS" not in result.stdout or result.returncode != 0:
        pytest.fail(diagnosis, pytrace=False)
    left = sorted(p.name for p in home.iterdir() if p.name != "logs")
    assert not left, f"the self-test left update state in HERMES_HOME: {left}"


@pytest.mark.platforms("windows")
def test_concurrent_writers_lose_no_handoff_log_lines(tmp_path: Path) -> None:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = (
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if not powershell.is_file():
        pytest.skip(f"Windows PowerShell not found at {powershell}")
    _run_log_concurrency_self_test(os.fspath(powershell), tmp_path)


@pytest.mark.platforms("windows")
def test_handoff_log_is_written_under_powershell_7(tmp_path: Path) -> None:
    """The same self-test under pwsh, which the desktop shim launches the hand-off with."""
    pwsh = shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell 7 (pwsh) not installed")
    _run_log_concurrency_self_test(pwsh, tmp_path)

_MUTEX_TIMEOUT_PROBE = r"""
param([string]$Source, [string]$LogPath)
$ErrorActionPreference = 'Stop'
$script:HandoffLogMutexName = ('Global\HermesLogMutexTest-{0}' -f $PID)
$script:HandoffLogMutex = $null
$script:PendingLogLines = New-Object System.Collections.Generic.List[string]
$script:HandoffLogError = $null

# Execute the real script's function without launching an update.
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$null, [ref]$null)
$definition = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq 'Sync-HandoffLog'
}, $true))[0]
if (-not $definition) { throw 'Sync-HandoffLog is missing' }
. ([scriptblock]::Create($definition.Extent.Text))

$ready = New-Object System.Threading.ManualResetEventSlim($false)
$release = New-Object System.Threading.ManualResetEventSlim($false)
$owner = [System.Management.Automation.PowerShell]::Create()
$holderCode = @'
param($Name, $Ready, $Release)
$mutex = New-Object System.Threading.Mutex($false, $Name)
$held = $false
try {
    $held = $mutex.WaitOne(10000)
    if (-not $held) { throw 'holder could not acquire the mutex' }
    $Ready.Set()
    [void]$Release.Wait(30000)
} finally {
    if ($held) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
'@
[void]$owner.AddScript($holderCode).AddArgument($script:HandoffLogMutexName).AddArgument($ready).AddArgument($release)
$handle = $owner.BeginInvoke()
try {
    if (-not $ready.Wait(15000)) { throw 'mutex holder did not acquire' }
    $script:PendingLogLines.Add('must stay queued')
    # Uses the production WaitOne(5000) timeout; no shortened limits.
    $blocked = Sync-HandoffLog
    $pending = $script:PendingLogLines.Count
    $written = Test-Path -LiteralPath $LogPath
} finally {
    $release.Set()
    [void]$owner.EndInvoke($handle)
    $owner.Dispose()
    $ready.Dispose()
    $release.Dispose()
}
if ($blocked -ne $false -or $pending -ne 1 -or $written) {
    throw "mutex timeout bypassed ownership: returned=$blocked pending=$pending file=$written"
}
if (-not (Sync-HandoffLog)) { throw 'queued append was not retried after release' }
if ([IO.File]::ReadAllText($LogPath).Trim() -ne 'must stay queued') {
    throw 'queued line was not flushed exactly once'
}
Write-Output 'LOG-MUTEX-TIMEOUT SELF-TEST: PASS'
"""


@pytest.mark.platforms("windows")
def test_handoff_log_mutex_timeout_keeps_lines_queued(tmp_path: Path) -> None:
    """#156217: a timed-out writer must not bypass the cross-process append mutex."""
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    powershell = system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if not powershell.is_file():
        pytest.skip(f"Windows PowerShell not found at {powershell}")
    probe = tmp_path / "mutex-timeout.ps1"
    probe.write_text(_MUTEX_TIMEOUT_PROBE, encoding="utf-8")
    result = subprocess.run(
        [os.fspath(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", os.fspath(probe), os.fspath(WINDOWS_PS1), os.fspath(tmp_path / "handoff.log")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=55, check=False,
    )
    assert result.returncode == 0 and "LOG-MUTEX-TIMEOUT SELF-TEST: PASS" in result.stdout, (
        result.stdout + result.stderr
    )
