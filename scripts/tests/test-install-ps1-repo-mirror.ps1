# Tests for install.ps1's opt-in repo-mirror clone ladder (#122888).
#
# Run from a PowerShell prompt:
#
#   pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-repo-mirror.ps1
#
# On restricted networks the official GitHub URL stalls while mirrors of the
# same repository stay reachable. install.ps1 retries the official URL hard
# (3 direct clones + a deferred-history clone) but never tries a second
# source: every retry re-hits the same dead route. The mirror ladder added
# for #122888 is opt-in via HERMES_REPO_MIRROR_URL -- no mirror is baked in
# -- so the contracts to pin here are:
#
#   1. Without the env var, the ladder never leaves the official URL and a
#      fully blocked clone fails with the same error as before.
#   2. With the env var, the official URL gets its whole ladder (4 attempts)
#      before the first mirror is contacted, and an answering mirror
#      publishes the checkout.
#   3. Semicolon-separated lists are honored: blank entries dropped,
#      duplicates of the official URL skipped, entries tried in order, and
#      the first mirror that answers wins.
#   4. A mirror env var is inert when the official clone succeeds on the
#      first try.
#   5. A mirror rescue can also arrive through the deferred-checkout path.
#
# HOW THIS RUNS THE CODE: install.ps1 is dot-sourced (its dot-source guard
# loads only the function definitions), then Stage-Repository runs against a
# fake `git` that records every clone's URL and exits non-zero for URLs the
# case marks as blocked. Nothing here parses install.ps1's source; the real
# routing code runs and only the network edge is faked. PowerShell resolves
# functions before native commands, so shadowing `git` with a function is
# enough. Start-Sleep is shadowed too so a blocked ladder retries instantly
# instead of backing off 5-10 seconds per attempt.

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot 'scripts/install.ps1'

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$official = 'https://github.com/NousResearch/hermes-agent.git'
$mirrorA = 'https://mirror-a.example/hermes-agent.git'
$mirrorB = 'https://mirror-b.example/hermes-agent.git'

$script:Failures = 0
function Assert-True($Condition, [string]$Label) {
    if ($Condition) { Write-Host "PASS: $Label" }
    else { Write-Host "FAIL: $Label"; $script:Failures++ }
}

function New-CaseRoot {
    param([System.Collections.Generic.List[string]]$Roots)
    $root = Join-Path ([IO.Path]::GetTempPath()) ("hermes-mirror-test-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    $Roots.Add($root)
    return $root
}

# Tripwires exist before dot-sourcing, so a broken guard cannot run an
# install. The recording `git` shadow below replaces the tripwire AFTER the
# dot-source (install.ps1 does not define a git function, so the order is
# safe), and Ensure-Git is re-shadowed after it because install.ps1 DOES
# define that one and Stage-Repository must not provision PortableGit here.
function Invoke-WebRequest { throw 'unexpected download' }
function Invoke-RestMethod { throw 'unexpected download' }
function git { throw 'unexpected git command' }

$script:Roots = New-Object System.Collections.Generic.List[string]
$caseRoot = New-CaseRoot $script:Roots
. $installScript -HermesHome (Join-Path $caseRoot 'home') -InstallDir (Join-Path $caseRoot 'agent')

function Start-Sleep { param($Seconds, $Milliseconds) }

$script:CloneUrls = New-Object System.Collections.Generic.List[string]
$script:BlockedUrls = @()
$script:DeferredOkUrls = @()
function git {
    $argv = @($args | ForEach-Object { "$_" })
    if ($argv -and $argv[0] -eq 'clone') {
        $url = @($argv | Where-Object { $_ -match '^https?://' })[0]
        $script:CloneUrls.Add($url)
        if ($script:BlockedUrls -contains $url) {
            # A URL whose graph downloads but whose tree materialization
            # stalls still answers the deferred --no-checkout clone.
            if (($argv -contains '--no-checkout') -and ($script:DeferredOkUrls -contains $url)) {
                New-Item -ItemType Directory -Force -Path $argv[-1] | Out-Null
                $global:LASTEXITCODE = 0
                return
            }
            $global:LASTEXITCODE = 128
            return
        }
        New-Item -ItemType Directory -Force -Path $argv[-1] | Out-Null
        $global:LASTEXITCODE = 0
        return
    }
    $global:LASTEXITCODE = 0
}
function Ensure-Git { $true }

function Invoke-RepositoryStage {
    $script:CloneUrls.Clear()
    $failed = $null
    try { Stage-Repository } catch { $failed = "$_" }
    return $failed
}

try {
    # --- Case 1: no env var -- the ladder never leaves the official URL ---
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    Remove-Item Env:\HERMES_REPO_MIRROR_URL -ErrorAction SilentlyContinue
    $script:BlockedUrls = @($official)
    $script:DeferredOkUrls = @()
    $failed = Invoke-RepositoryStage
    Assert-True ($failed -match 'git clone failed') "no-env blocked clone fails with the stock error (got: $failed)"
    Assert-True (@($script:CloneUrls | Where-Object { $_ -ne $official }).Count -eq 0) 'no-env ladder never contacts a non-official URL'
    Assert-True (@($script:CloneUrls).Count -eq 4) "no-env ladder exhausts 3 direct + 1 deferred attempt (got $(@($script:CloneUrls).Count))"
    Assert-True (-not (Test-Path -LiteralPath $InstallDir)) 'no-env blocked clone publishes nothing'

    # --- Case 2: mirror env -- official ladder first, then the mirror ----
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = $mirrorA
    $script:BlockedUrls = @($official)
    $failed = Invoke-RepositoryStage
    Assert-True (-not $failed) "answering mirror completes the stage (got: $failed)"
    Assert-True (@($script:CloneUrls).Count -eq 5) "official ladder (4 attempts) precedes the first mirror attempt (got $(@($script:CloneUrls).Count))"
    Assert-True ($script:CloneUrls[3] -eq $official -and $script:CloneUrls[4] -eq $mirrorA) 'mirror contact starts only after the official ladder is exhausted'
    Assert-True (Test-Path -LiteralPath $InstallDir) 'mirror-rescued clone publishes the checkout'

    # --- Case 3: semicolon list -- blanks dropped, order kept, first win --
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = " ; $mirrorA ; $mirrorB "
    $script:BlockedUrls = @($official, $mirrorA)
    $failed = Invoke-RepositoryStage
    Assert-True (-not $failed) "second list entry rescues the install (got: $failed)"
    Assert-True (@($script:CloneUrls).Count -eq 9) "each list entry gets the full 4-attempt ladder in order (got $(@($script:CloneUrls).Count))"
    Assert-True ($script:CloneUrls[-1] -eq $mirrorB) 'the answering mirror is the last URL contacted'

    # --- Case 4: official answers -- the mirror env stays inert ----------
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = $mirrorA
    $script:BlockedUrls = @()
    $failed = Invoke-RepositoryStage
    Assert-True (-not $failed) "unblocked official clone completes the stage (got: $failed)"
    Assert-True (@($script:CloneUrls).Count -eq 1 -and $script:CloneUrls[0] -eq $official) 'mirror env is inert when the official clone answers first try'

    # --- Case 5: a mirror entry equal to the official URL is deduplicated -
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = $official
    $script:BlockedUrls = @($official)
    $failed = Invoke-RepositoryStage
    Assert-True ($failed -match 'git clone failed') 'mirror equal to the official URL is deduplicated, not retried'
    Assert-True (@($script:CloneUrls).Count -eq 4) "deduplicated ladder still stops at 4 official attempts (got $(@($script:CloneUrls).Count))"

    # --- Case 6: rescue via the deferred-checkout path of a mirror -------
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = $mirrorA
    $script:BlockedUrls = @($official, $mirrorA)
    $script:DeferredOkUrls = @($mirrorA)
    $failed = Invoke-RepositoryStage
    Assert-True (-not $failed) "deferred-checkout rescue via mirror completes the stage (got: $failed)"
    Assert-True (@($script:CloneUrls).Count -eq 8) "mirror runs 3 direct + 1 deferred attempt before the rescue (got $(@($script:CloneUrls).Count))"
    Assert-True ($script:CloneUrls[-1] -eq $mirrorA -and (Test-Path -LiteralPath $InstallDir)) 'deferred mirror rescue publishes the checkout'
} finally {
    Remove-Item Env:\HERMES_REPO_MIRROR_URL -ErrorAction SilentlyContinue
    foreach ($root in $script:Roots) {
        Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($script:Failures -gt 0) {
    Write-Host "$($script:Failures) assertion(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host 'all repo-mirror ladder assertions passed' -ForegroundColor Green
exit 0
