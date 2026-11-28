# Dot-source only. Prefer the published command; never rescue a broken one
# through PATH, the user's shared bin, or an obsolete checkout venv.
function Get-SourceHermes([string]$Root) {
    foreach ($name in @('hermes.exe', 'hermes.cmd')) {
        $command = Join-Path $Root ".hermes/bin/$name"
        if (Test-Path -LiteralPath $command) {
            if (-not (Test-Path -LiteralPath $command -PathType Leaf)) {
                throw "Invalid published launcher: $command"
            }
            return $command
        }
    }
    if (Test-Path -LiteralPath (Join-Path $Root 'pm/lock.json')) {
        throw "Missing published launcher under $Root/.hermes/bin"
    }
    $legacy = Join-Path $Root 'venv/Scripts/hermes.exe'
    if (Test-Path -LiteralPath $legacy -PathType Leaf) { return $legacy }
    throw "No installed Hermes command under $Root"
}

# Hand out a command to DRIVE the next ordinary startup, even when the
# published launcher is not there yet.
#
# A pre-handoff release cannot flip during `hermes update` -- there is no
# retired-hook seam on its update path to reach, so the update ends with the
# tree at HEAD and no `.hermes/bin/*`. The NEXT ordinary startup is what
# completes it: hermes_bootstrap calls prepare_launch() before importing
# anything, which syncs PM, publishes the launchers and re-execs.
#
# Deliberately NOT used for `--version` probes: those stay under
# HERMES_DISABLE_LAZY_INSTALLS so a probe can never complete an unfinished
# update. Only a real startup may heal.
function Get-SourceHermesForStartup([string]$Root) {
    foreach ($name in @('hermes.exe', 'hermes.cmd')) {
        $published = Join-Path $Root ".hermes/bin/$name"
        if (Test-Path -LiteralPath $published -PathType Leaf) { return $published }
    }
    $legacy = Join-Path $Root 'venv/Scripts/hermes.exe'
    if (Test-Path -LiteralPath $legacy -PathType Leaf) { return $legacy }
    throw "No installed Hermes command to start under $Root"
}

# Official source checkouts default to the stable channel (the latest published
# vX.Y.Z), and these fixtures publish commits on main, not releases: a fresh
# install at HEAD is newer than the latest release and correctly waits, so the
# HEAD -> NEXT legs would find no update. Record `main` for the install BEFORE
# the user-state snapshot (config.yaml is part of the baseline). Releases that
# predate --set-channel follow main already.
function Set-SourceMainChannel([string]$Root) {
    $exe = Get-SourceHermes $Root
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        $help = & $exe update --help 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { throw "update --help failed: $help" }
        if ($help -notmatch '--set-channel') { return }
        $out = & $exe update --set-channel main 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) { throw "could not record the main update channel: $out" }
        Write-Host "  $($out.Trim())"
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Invoke-SourceDriverGit([string]$GitExe, [string[]]$GitArgs) {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $GitExe @GitArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE): $output"
        }
        return ($output | Out-String).Trim()
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Clear-SourceInstallerChurn {
    param(
        [Parameter(Mandatory = $true)][string]$InstallDir,
        [Parameter(Mandatory = $true)][string]$WorkRoot,
        [Parameter(Mandatory = $true)][string]$RealGitExe,
        [Parameter(Mandatory = $true)][string]$DriverNode,
        [Parameter(Mandatory = $true)][string]$AssetsDir,
        [string[]]$LegacyLaunchers = @()
    )
    $lines = @((Invoke-SourceDriverGit $RealGitExe @("-C", $InstallDir, "-c", "core.quotepath=false", "status", "--porcelain=v2", "--untracked-files=all")) -split "\r?\n" |
        Where-Object { $_ })
    if ($lines.Count -eq 0) { return }
    Write-Host "  source status before GUI update:"
    $lines | ForEach-Object { Write-Host "    $_" }

    $collisionScript = Join-Path $AssetsDir "case-collision.cjs"
    $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        $collisionJson = (& $DriverNode $collisionScript "--list" $InstallDir "--git" $RealGitExe | Out-String).Trim()
        $collisionExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
    if ($collisionExit -ne 0) {
        throw "E2E ASSERTION FAILED: case-collision check failed (exit $collisionExit)"
    }
    $collisions = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
    $parsedCollisions = $collisionJson | ConvertFrom-Json
    foreach ($item in @($parsedCollisions.collisions)) {
        if ($item) { $collisions.Add([string]$item) | Out-Null }
    }

    # --numstat honours --ignore-cr-at-eol (--name-only does not): a modified
    # path with no record and an unchanged mode differs from its index blob
    # only by CRs.
    $content = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
    (Invoke-SourceDriverGit $RealGitExe @("-C", $InstallDir, "-c", "core.quotepath=false", "diff", "--numstat", "--ignore-cr-at-eol")) -split "\r?\n" |
        Where-Object { $_ } | ForEach-Object { $content.Add([string]($_ -split "`t", 3)[2]) | Out-Null }

    $locks = @(); $other = @()
    foreach ($line in $lines) {
        $fields = $line -split " ", 9
        if ($fields[0] -eq "1" -and $fields[1] -eq ".M" -and $fields.Count -eq 9 -and
            ($fields[8] -eq "package-lock.json" -or $fields[8] -like "*/package-lock.json" -or
             ($fields[4] -eq $fields[5] -and -not $content.Contains($fields[8])))) {
            $locks += $fields[8]
        } elseif ($line -eq "? .install_method") {
            Add-Content -LiteralPath (Join-Path $InstallDir ".git\info\exclude") -Value "/.install_method"
        } elseif ($line -cin $LegacyLaunchers) {
            Add-Content -LiteralPath (Join-Path $InstallDir ".git\info\exclude") -Value ("/" + $line.Substring(2))
        } elseif ($fields[0] -eq "1" -and $fields[1] -eq ".M" -and $fields.Count -eq 9 -and $collisions.Contains($fields[8])) {
            Write-Host "  tolerating HEAD case-collision dirty path: $($fields[8])"
        } else {
            $other += $line
        }
    }
    if ($other.Count -ne 0) {
        throw "E2E ASSERTION FAILED: installed source has only installer-generated changes (other: $($other -join '; '))"
    }
    Write-Host "  [ok] installed source has only installer-generated changes (other: )"
    if ($locks.Count) {
        # From a file: a v2026.7.1 clone churns hundreds of paths, and passing
        # them as arguments overflows the Windows command line ("The filename
        # or extension is too long"). NUL-separated, literal: no quoting or
        # glob magic applies to a path.
        $pathspecs = Join-Path $WorkRoot "installer-churn.pathspec"
        [System.IO.File]::WriteAllText($pathspecs, (($locks -join "`0") + "`0"), (New-Object System.Text.UTF8Encoding $false))
        Invoke-SourceDriverGit $RealGitExe @("--literal-pathspecs", "-C", $InstallDir, "checkout", "--pathspec-from-file=$pathspecs", "--pathspec-file-nul") | Out-Null
    }
    $left = @((Invoke-SourceDriverGit $RealGitExe @("-C", $InstallDir, "status", "--porcelain", "--untracked-files=all")) -split "\r?\n" |
        Where-Object { $_ })
    if ($left.Count -gt 0) {
        $prevEap = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try {
            $collisionOut = & $DriverNode $collisionScript "--only-collisions" $InstallDir "--git" $RealGitExe
            $onlyCollisionsExit = $LASTEXITCODE
            $collisionOut | ForEach-Object { Write-Host "  $_" }
        } finally {
            $ErrorActionPreference = $prevEap
        }
        if ($onlyCollisionsExit -ne 0) {
            throw "E2E ASSERTION FAILED: undid only installer-generated source churn before the GUI update (remaining: $($left -join '; '))"
        }
    }
    Write-Host "  [ok] undid only installer-generated source churn before the GUI update"
}
