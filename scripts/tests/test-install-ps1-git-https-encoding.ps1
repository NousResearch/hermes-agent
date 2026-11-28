# Regression: a pinned git whose --exec-path contains non-ASCII characters
# must survive Windows PowerShell's console-codepage decoding.
#
# Run from a PowerShell prompt:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-git-https-encoding.ps1
#
# Background: Windows PowerShell 5.1 decodes a native command's captured
# stdout with [Console]::OutputEncoding (the console code page, e.g. CP936
# on zh-CN), while git for Windows always emits UTF-8. On a machine whose
# user profile carries non-ASCII characters, the pinned git's exec path is
# such a path: decoded under CP936 it turns into different code points, so
# Test-PinnedGitHttpsReady's Test-Path missed git-remote-https and
# Get-PinnedGit deleted the freshly extracted tree. Only the git call site
# missed Invoke-Native's -Utf8Output: the `uv python find` call sites pass
# it, which is exactly how a UTF-8 producer is opted in.
#
# HOW THIS RUNS THE CODE: install.ps1 is dot-sourced, which loads every
# function definition and stops before the entry dispatch (its dot-source
# guard), so the real Test-PinnedGitHttpsReady runs against a real native
# child. That child is a fixture compiled with the inbox .NET Framework
# compiler (no git download needed) that reports its own tree's exec path
# the way MSYS2 git does: raw UTF-8 bytes on a pipe. A fixture that let
# .NET encode to the console code page would silently match the buggy
# decoder and never discriminate. The positive tree carries
# git-remote-https under the non-ASCII path; the negative tree omits it,
# so the check cannot pass for the wrong reason. Every code page asserts
# the helper's presence/absence verdict and that the caller's encoding
# survives the call.
param([switch]$HiddenChild)
$ErrorActionPreference = "Stop"
$suitePath = $PSCommandPath
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installer = Join-Path $repoRoot "scripts/install.ps1"
$testRoot = $null
$savedEncoding = [Console]::OutputEncoding

if (-not (Test-Path -LiteralPath $installer)) {
    throw "Could not locate install.ps1 at $installer"
}

function Assert-Equal($Expected, $Actual, [string]$Label) {
    if ($Expected -cne $Actual) { throw "FAIL: ${Label}: expected '$Expected', got '$Actual'" }
}

try {
    if (-not $HiddenChild) {
        # Match the real failure's shell shape: a console-less child with
        # piped streams (the same harness idea as test-install-ps1-utf8.ps1).
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = (Get-Process -Id $PID).Path
        $psi.Arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$suitePath`" -HiddenChild"
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $child = New-Object System.Diagnostics.Process
        $child.StartInfo = $psi
        try {
            if (-not $child.Start()) { throw "could not start hidden PowerShell host" }
            $child.StandardInput.Close()
            $stdoutTask = $child.StandardOutput.ReadToEndAsync()
            $stderrTask = $child.StandardError.ReadToEndAsync()
            if (-not $child.WaitForExit(120000)) {
                $child.Kill()
                throw "hidden PowerShell regression timed out"
            }
            $stdout = $stdoutTask.Result
            $stderr = $stderrTask.Result
            if ($child.ExitCode -ne 0) { throw "hidden PowerShell regression failed: $stdout $stderr" }
            Assert-Equal $true $stdout.Contains("git-https-encoding regression passed.") "hidden suite completed"
        } finally {
            $child.Dispose()
        }
        Write-Host "git-https-encoding regression passed under 5.1 and pwsh."
        return
    }

    $testRoot = Join-Path ([IO.Path]::GetTempPath()) ("hermes-git-enc-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
    # The exec path the fixture reports, shared by both trees. Non-ASCII on
    # purpose: CJK, an accent, a quote and a Polish letter, as real profiles
    # carry.
    $profileDir = Join-Path $testRoot (([string][char]0x5F20) + [char]0x4E09 + " Ren" + [char]0xE9 + " O'Brien Pawe" + [char]0x142)
    $goodTree = Join-Path $profileDir "git-good-win32-x64"
    $badTree = Join-Path $profileDir "git-bad-win32-x64"

    # MSYS2-style git.exe: answers --exec-path with this tree's exec dir as
    # raw UTF-8 bytes on stdout, exactly where git for Windows writes them.
    $stubSource = Join-Path $testRoot "stub-git.cs"
    [IO.File]::WriteAllText($stubSource, @'
using System;
using System.IO;
using System.Text;

class StubGit
{
    static int Main(string[] args)
    {
        if (args.Length != 1 || args[0] != "--exec-path") { return 64; }
        string cmdDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
        string execPath = Path.GetFullPath(Path.Combine(cmdDir, "..", "ucrt64", "libexec", "git-core"));
        byte[] bytes = new UTF8Encoding(false).GetBytes(execPath + "\n");
        Stream stdout = Console.OpenStandardOutput();
        stdout.Write(bytes, 0, bytes.Length);
        return 0;
    }
}
'@)
    $compiler = Join-Path $env:SystemRoot "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    New-Item -ItemType Directory -Path (Join-Path $goodTree "cmd") -Force | Out-Null
    & $compiler /nologo /target:exe "/out:$goodTree\cmd\git.exe" $stubSource
    if ($LASTEXITCODE) { throw "native fixture compilation failed" }
    # The helper's presence is what Test-Path judges; content is irrelevant.
    $goodCore = Join-Path $goodTree "ucrt64\libexec\git-core"
    New-Item -ItemType Directory -Path $goodCore -Force | Out-Null
    [IO.File]::WriteAllText((Join-Path $goodCore "git-remote-https.exe"), "")
    New-Item -ItemType Directory -Path (Join-Path $badTree "cmd") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $goodTree "cmd\git.exe") -Destination (Join-Path $badTree "cmd\git.exe")

    # Dot-sourced: definitions only, the installer's entry never runs.
    . $installer

    # CodePages 936/437 ship with Windows; 65001 is the control the buggy
    # decoder already passed. The pytest lane marks this suite windows-only.
    foreach ($codePage in @(936, 437, 65001)) {
        [Console]::OutputEncoding = [Text.Encoding]::GetEncoding($codePage)
        Assert-Equal $codePage ([Console]::OutputEncoding.CodePage) "code page $codePage applied"

        Assert-Equal $true (Test-PinnedGitHttpsReady (Join-Path $goodTree "cmd\git.exe")) "helper found under CP$codePage despite non-ASCII exec path"
        Assert-Equal $false (Test-PinnedGitHttpsReady (Join-Path $badTree "cmd\git.exe")) "missing helper rejected under CP$codePage"
        Assert-Equal $false (Test-PinnedGitHttpsReady (Join-Path $testRoot "absent\git.exe")) "absent git.exe rejected under CP$codePage"
        Assert-Equal $codePage ([Console]::OutputEncoding.CodePage) "caller code page restored under CP$codePage"
    }

    Write-Host "git-https-encoding regression passed."
} finally {
    [Console]::OutputEncoding = $savedEncoding
    if ($testRoot) { Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue }
}
