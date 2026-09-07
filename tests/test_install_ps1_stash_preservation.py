"""Installer updates cannot cross an unverified local-work preservation boundary."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.windows_only


def git(root, *args, input=None, binary=False):
    result = subprocess.run(
        [shutil.which("git"), "-C", str(root), *args],
        input=input,
        capture_output=True,
        text=not binary,
        encoding="utf-8" if not binary else None,
        errors="replace" if not binary else None,
        check=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.stdout.strip() if not binary else result.stdout


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@pytest.mark.parametrize("host", ["powershell.exe", "pwsh.exe"])
@pytest.mark.parametrize("mode", ["fail", "created-error", "foreign-stash", "restore-conflict", "unmerged"])
def test_stash_boundary_owns_exact_recovery_object(tmp_path, host, mode):
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-b", "main")
    git(remote, "config", "user.name", "Fixture")
    git(remote, "config", "user.email", "fixture@example.invalid")
    (remote / "tracked.txt").write_text("original\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "original")
    live = tmp_path / "owned install # one"
    git(tmp_path, "clone", str(remote), str(live))
    git(live, "config", "user.name", "Fixture")
    git(live, "config", "user.email", "fixture@example.invalid")
    (live / "tracked.txt").write_text("personal edits\n")
    (live / "personal.txt").write_text("untracked work\n")
    head = git(live, "rev-parse", "HEAD")
    if mode == "restore-conflict":
        (remote / "tracked.txt").write_text("upstream change\n")
        git(remote, "commit", "-am", "upstream")
    if mode == "unmerged":
        original = git(live, "rev-parse", "HEAD:tracked.txt")
        ours = git(live, "hash-object", "-w", "tracked.txt")
        git(live, "update-index", "--index-info", input=(
            f"0 {'0' * 40}\ttracked.txt\n"
            f"100644 {original} 1\ttracked.txt\n"
            f"100644 {original} 2\ttracked.txt\n"
            f"100644 {ours} 3\ttracked.txt\n"
        ).encode("ascii"), binary=True)
        unmerged = git(live, "ls-files", "--unmerged")
    driver = tmp_path / "driver.ps1"
    events = tmp_path / "events.txt"
    driver.write_text(
        f". {literal(REPO / 'scripts/install.ps1')} -InstallDir {literal(live)} -HermesHome {literal(tmp_path / 'home')} -Branch main\n"
        f"$realGit = {literal(shutil.which('git'))}\n"
        f"$eventLog = {literal(events)}\n"
        f"$fixtureMode = {literal(mode)}\n"
        # Explicit pipes avoid the nested native stdout loss of a hidden,
        # redirected PowerShell host; the real Git repository operations stay real.
        "function Invoke-FixtureGit {\n"
        "  $start = [Diagnostics.ProcessStartInfo]::new()\n"
        "  $start.FileName = $realGit\n"
        "  $start.Arguments = ($args | ForEach-Object { [char]34 + [string]$_ + [char]34 }) -join ' '\n"
        "  $start.WorkingDirectory = (Get-Location).Path\n"
        "  $start.UseShellExecute = $false; $start.CreateNoWindow = $true\n"
        "  $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true\n"
        "  $child = [Diagnostics.Process]::Start($start)\n"
        "  try {\n"
        "    $outTask = $child.StandardOutput.ReadToEndAsync(); $errTask = $child.StandardError.ReadToEndAsync()\n"
        "    $child.WaitForExit(); $global:LASTEXITCODE = $child.ExitCode\n"
        "    if ($outTask.Result) { $outTask.Result.TrimEnd() -split \"`r?`n\" }\n"
        "    if ($errTask.Result) { [Console]::Error.Write($errTask.Result) }\n"
        "  } finally { $child.Dispose() }\n"
        "}\n"
        "function git {\n"
        "  $call = $args -join ' '\n"
        "  [IO.File]::AppendAllText($eventLog, $call + \"`n\")\n"
        "  if ($call -match 'stash push') {\n"
        "    if ($fixtureMode -eq 'fail') { $global:LASTEXITCODE = 17; return }\n"
        "    Invoke-FixtureGit @args\n"
        "    if ($fixtureMode -eq 'created-error') { $global:LASTEXITCODE = 17; return }\n"
        "    if ($fixtureMode -eq 'foreign-stash') {\n"
        "      [IO.File]::WriteAllText((Join-Path $InstallDir 'foreign.txt'), 'foreign work')\n"
        "      Invoke-FixtureGit stash push --include-untracked -m foreign-owned\n"
        "    }\n"
        "    return\n"
        "  }\n"
        "  Invoke-FixtureGit @args\n"
        "}\n"
        "try { Install-Repository; exit 0 } catch { [Console]::Error.WriteLine([string]$_); exit 1 }\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run(
        [shutil.which(host), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(driver)],
        env={**os.environ, "TEMP": str(tmp_path), "TMP": str(tmp_path)},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    output = result.stdout + result.stderr
    calls = events.read_text()
    if mode == "unmerged":
        assert result.returncode != 0, output
        assert git(live, "ls-files", "--unmerged") == unmerged
        assert "stash push" not in calls
        assert " fetch " not in calls
        assert (live / "tracked.txt").read_text() == "personal edits\n"
        return
    assert "stash push" in calls, output + "\n" + calls
    if mode == "restore-conflict":
        assert result.returncode != 0, output
        assert git(live, "ls-files", "--unmerged")
        assert "reset --hard" not in calls
        oid = git(live, "rev-parse", "refs/stash")
        assert git(live, "show", f"{oid}:tracked.txt") == "personal edits"
        assert oid in output
        return
    assert git(live, "rev-parse", "HEAD") == head
    if mode == "foreign-stash":
        assert result.returncode == 0, output
        assert (live / "tracked.txt").read_text() == "personal edits\n"
        assert (live / "personal.txt").read_text() == "untracked work\n"
        assert not (live / "foreign.txt").exists()
        stashes = git(live, "stash", "list", "--format=%gs")
        assert "foreign-owned" in stashes
        assert "hermes-install-autostash-" in stashes
        assert "stash drop" not in calls
    else:
        assert result.returncode != 0, output
        assert "stash" in output.lower()
        assert " fetch " not in calls
        assert "reset --hard" not in calls
        if mode == "fail":
            assert (live / "tracked.txt").read_text() == "personal edits\n"
            assert (live / "personal.txt").read_text() == "untracked work\n"
        else:
            oid = git(live, "rev-parse", "refs/stash")
            assert git(live, "show", f"{oid}:tracked.txt") == "personal edits"
            assert git(live, "show", f"{oid}^3:personal.txt") == "untracked work"
            assert oid in output
