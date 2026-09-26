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
#   6. A candidate abandoned after a half-materialized deferred clone is
#      cleaned up: the next candidate's first direct attempt is not burned
#      refusing the leftover directory.
#   7. Non-https mirror entries fail closed before any clone happens.
#   8. URL variants of the official source (.git-less, slash-suffixed)
#      deduplicate to it instead of re-running the official ladder.
#   9. A mirror rescue names the URL that answered and states how to point
#      origin back at the official repository.
#  10. A rerun whose origin is not the managed URL warns about it on every
#      run, not just the one that contacted the mirror; an official origin
#      reruns with no origin warnings.
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
# `remote get-url origin` reports an explicit override when set, else the URL
# of the last clone this fake performed (the real clone would have set
# origin to it).
$script:OriginOverride = $null
$script:LastClonedUrl = $null
# While > 0, that many `reset --hard` invocations fail: a mirror whose
# deferred clone lands but whose tree materialization stalls.
$script:FailResets = 0
function git {
    $argv = @($args | ForEach-Object { "$_" })
    if ($argv -and $argv[0] -eq 'clone') {
        $url = @($argv | Where-Object { $_ -match '^https?://' })[0]
        $script:CloneUrls.Add($url)
        if ($script:BlockedUrls -contains $url) {
            # A URL whose graph downloads but whose tree materialization
            # stalls still answers the deferred --no-checkout clone.
            if (($argv -contains '--no-checkout') -and ($script:DeferredOkUrls -contains $url)) {
                # The real deferred clone leaves a populated .git; a later
                # clone into the same directory must be refused like real
                # git refuses a non-empty destination.
                New-Item -ItemType Directory -Force -Path $argv[-1] | Out-Null
                Set-Content -Path (Join-Path $argv[-1] '.git') -Value 'staged git dir'
                $script:LastClonedUrl = $url
                $global:LASTEXITCODE = 0
                return
            }
            $global:LASTEXITCODE = 128
            return
        }
        if ((Test-Path -LiteralPath $argv[-1]) -and @(Get-ChildItem -Force -LiteralPath $argv[-1]).Count -gt 0) {
            # Real git refuses a non-empty destination before any network I/O.
            $global:LASTEXITCODE = 128
            return
        }
        New-Item -ItemType Directory -Force -Path $argv[-1] | Out-Null
        $script:LastClonedUrl = $url
        $global:LASTEXITCODE = 0
        return
    }
    if ($argv -contains 'get-url') {
        $global:LASTEXITCODE = 0
        if ($null -ne $script:OriginOverride) { return $script:OriginOverride }
        return $script:LastClonedUrl
    }
    if (($argv -and $argv[0] -eq '-C') -and ($argv -contains 'reset')) {
        if ($script:FailResets -gt 0) {
            $script:FailResets--
            $global:LASTEXITCODE = 1
            return
        }
        $global:LASTEXITCODE = 0
        return
    }
    $global:LASTEXITCODE = 0
}
function Ensure-Git { $true }
# install.ps1 defines Write-Warn; shadowing it after the dot-source collects
# the warnings so origin/mirror contracts can assert on them.
$script:Warnings = New-Object System.Collections.Generic.List[string]
function Write-Warn { param([string]$msg) $script:Warnings.Add($msg) }

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

    # --- Case 7: an abandoned deferred clone must not poison the next -----
    # candidate's first direct attempt
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = "$mirrorA;$mirrorB"
    $script:BlockedUrls = @($official, $mirrorA)
    $script:DeferredOkUrls = @($mirrorA)
    $script:FailResets = 2   # mirrorA's deferred clone lands, both resets stall
    $script:OriginOverride = $null
    $failed = Invoke-RepositoryStage
    Assert-True (-not $failed) "a healthy second mirror rescues the install (got: $failed)"
    # official 4 + mirrorA 4 (3 direct + a deferred clone whose resets all
    # failed). The abandoned mirrorA leaves a populated tree behind, so
    # mirrorB's FIRST direct clone must still succeed: 9 contacts, mirrorB
    # contacted exactly once.
    Assert-True (@($script:CloneUrls).Count -eq 9) "abandoned deferred clone is cleaned, not retried around (got $(@($script:CloneUrls).Count))"
    Assert-True (@($script:CloneUrls | Where-Object { $_ -eq $mirrorB }).Count -eq 1 -and $script:CloneUrls[-1] -eq $mirrorB) 'the next mirror answers on its first direct attempt'

    # --- Case 8: non-https mirror entries fail closed ----------------------
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = 'ext::sh -c pwd'
    $script:BlockedUrls = @()
    $script:DeferredOkUrls = @()
    $script:FailResets = 0
    $failed = Invoke-RepositoryStage
    Assert-True ($failed -match 'https://') "helper-protocol mirror entry fails closed (got: $failed)"
    Assert-True (@($script:CloneUrls).Count -eq 0) 'no clone happens before the entry is validated'

    # --- Case 9: a URL variant of the official source deduplicates ---------
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = "$official/;$($official.Replace('.git', ''))"
    $script:BlockedUrls = @($official)
    $failed = Invoke-RepositoryStage
    Assert-True ($failed -match 'git clone failed') 'URL variants of the official source dedup to it, not re-clone'
    Assert-True (@($script:CloneUrls).Count -eq 4) "variant duplicates do not re-run the official ladder (got $(@($script:CloneUrls).Count))"

    # --- Case 10: a mirror rescue names its source and the way back --------
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    $env:HERMES_REPO_MIRROR_URL = $mirrorA
    $script:BlockedUrls = @($official)
    $script:DeferredOkUrls = @()
    $script:OriginOverride = $null
    $script:Warnings.Clear()
    $failed = Invoke-RepositoryStage
    Assert-True (-not $failed) "mirror rescue completes the stage (got: $failed)"
    Assert-True (@($script:Warnings | Where-Object { $_ -match [regex]::Escape($mirrorA) }).Count -ge 1) 'the mirror that answered is named'
    Assert-True (@($script:Warnings | Where-Object { $_ -match 'HERMES_REPO_URL' }).Count -ge 1) 'the way back to the official URL is stated'

    # --- Case 11: a rerun against a mirror-stuck origin warns every time ---
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir '.git') | Out-Null
    Remove-Item Env:\HERMES_REPO_MIRROR_URL -ErrorAction SilentlyContinue
    Remove-Item Env:\HERMES_REPO_URL -ErrorAction SilentlyContinue
    $script:BlockedUrls = @()
    $script:OriginOverride = $mirrorA
    $script:Warnings.Clear()
    $failed = Invoke-RepositoryStage
    Assert-True (-not $failed) "update against an existing clone completes (got: $failed)"
    Assert-True (@($script:CloneUrls).Count -eq 0) 'an update never clones'
    Assert-True (@($script:Warnings | Where-Object { $_ -match [regex]::Escape($mirrorA) }).Count -ge 1) 'a non-official origin is named on every rerun'
    Assert-True (@($script:Warnings | Where-Object { $_ -match 'HERMES_REPO_URL' }).Count -ge 1) 'the way back is stated on every rerun'

    # --- Case 12: an official origin reruns without origin warnings --------
    $InstallDir = Join-Path (New-CaseRoot $script:Roots) 'agent'
    New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir '.git') | Out-Null
    $script:OriginOverride = $official
    $script:Warnings.Clear()
    $failed = Invoke-RepositoryStage
    Assert-True (-not $failed) "update against the official origin completes (got: $failed)"
    Assert-True (@($script:Warnings | Where-Object { $_ -match 'origin is' }).Count -eq 0) 'an official origin produces no origin warnings'
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
