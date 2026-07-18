# Smoke tests for the install.ps1 stage protocol.
#
# Run from a PowerShell prompt:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/tests/test-install-ps1-stage-protocol.ps1
#
# These tests only exercise the metadata surface (-ProtocolVersion, -Manifest,
# unknown -Stage handling).  They DO NOT actually run any install stages --
# those have heavy side effects (winget, git clone, pip install, PATH writes)
# and are out of scope for a unit smoke test.  All three metadata commands
# below return without invoking Main / Invoke-AllStages.
#
# To exercise real install stages, drive the script from a clean VM.

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$installScript = Join-Path $repoRoot "scripts\install.ps1"

if (-not (Test-Path $installScript)) {
    throw "Could not locate install.ps1 at $installScript"
}

$failures = 0
function Assert-Equal {
    param([Parameter(Mandatory=$true)] $Expected,
          [Parameter(Mandatory=$true)] $Actual,
          [Parameter(Mandatory=$true)] [string]$Label)
    if ($Expected -ne $Actual) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        Write-Host "  expected: $Expected"
        Write-Host "  actual:   $Actual"
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}
function Assert-True {
    param([Parameter(Mandatory=$true)] $Condition,
          [Parameter(Mandatory=$true)] [string]$Label)
    if (-not $Condition) {
        Write-Host "FAIL: $Label" -ForegroundColor Red
        $script:failures++
    } else {
        Write-Host "OK: $Label" -ForegroundColor Green
    }
}

# -----------------------------------------------------------------------------
# Test: -ProtocolVersion emits a single integer
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "-- -ProtocolVersion --"
$output = & powershell -NoProfile -ExecutionPolicy Bypass -File $installScript -ProtocolVersion
Assert-Equal -Expected 0 -Actual $LASTEXITCODE -Label "-ProtocolVersion exits 0"
Assert-True ($output -match '^\d+$') -Label "-ProtocolVersion emits an integer (got: $output)"

# -----------------------------------------------------------------------------
# Test: -Manifest emits valid JSON with expected shape
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "-- -Manifest --"
$manifestJson = & powershell -NoProfile -ExecutionPolicy Bypass -File $installScript -Manifest
Assert-Equal -Expected 0 -Actual $LASTEXITCODE -Label "-Manifest exits 0"

$manifest = $null
try {
    $manifest = $manifestJson | ConvertFrom-Json
    Assert-True $true -Label "-Manifest output parses as JSON"
} catch {
    Assert-True $false -Label "-Manifest output parses as JSON (parse error: $_)"
}

if ($manifest) {
    Assert-True ($manifest.protocol_version -is [int] -or $manifest.protocol_version -is [long]) `
        -Label "manifest.protocol_version is an integer"
    Assert-True ($manifest.stages.Count -gt 0) -Label "manifest.stages is non-empty"

    # Every stage has the four required fields
    $allValid = $true
    foreach ($stage in $manifest.stages) {
        foreach ($field in @("name", "title", "category", "needs_user_input")) {
            if (-not ($stage.PSObject.Properties.Name -contains $field)) {
                Write-Host "  stage missing field '$field': $($stage | ConvertTo-Json -Compress)" -ForegroundColor Red
                $allValid = $false
            }
        }
    }
    Assert-True $allValid -Label "every stage has name/title/category/needs_user_input"

    # Specific stage names that the GUI driver will rely on
    $names = $manifest.stages | ForEach-Object { $_.name }
    foreach ($expected in @("uv", "python", "git", "venv", "dependencies", "configure", "gateway")) {
        Assert-True ($names -contains $expected) -Label "manifest contains stage '$expected'"
    }

    # The two known-interactive stages must declare needs_user_input
    $interactive = $manifest.stages | Where-Object { $_.needs_user_input } | ForEach-Object { $_.name }
    Assert-True ($interactive -contains "configure") -Label "'configure' stage flagged needs_user_input"
    Assert-True ($interactive -contains "gateway") -Label "'gateway' stage flagged needs_user_input"
}

# -----------------------------------------------------------------------------
# Test: unknown stage name -> exit 2, structured JSON error
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "-- -Stage with unknown name --"
$errOutput = & powershell -NoProfile -ExecutionPolicy Bypass -File $installScript -Stage "does-not-exist"
Assert-Equal -Expected 2 -Actual $LASTEXITCODE -Label "unknown -Stage exits 2"

$errFrame = $null
try {
    $errFrame = $errOutput | ConvertFrom-Json
    Assert-True $true -Label "unknown-stage output parses as JSON"
} catch {
    Assert-True $false -Label "unknown-stage output parses as JSON (parse error: $_)"
}

if ($errFrame) {
    Assert-Equal -Expected $false -Actual $errFrame.ok -Label "unknown-stage frame has ok=false"
    Assert-Equal -Expected "does-not-exist" -Actual $errFrame.stage -Label "unknown-stage frame echoes stage name"
    Assert-True ($errFrame.reason -match "unknown stage") -Label "unknown-stage frame explains why"
}

# -----------------------------------------------------------------------------
# Test: install recovery marker + stash-ref-verify logic (#46791)
#
# These tests don't drive the install script end-to-end (that has heavy side
# effects). Instead they exercise the git primitives the recovery flow relies
# on, in a throwaway temp repo. Regression: if these assumptions break on a
# future git version, the installer's recovery path becomes unsafe.
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "-- install recovery primitives (#46791) --"

function New-TempGitRepo {
    $dir = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-recov-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $dir | Out-Null
    Push-Location $dir
    git init -q
    git config user.email "test@local"
    git config user.name "test"
    # Disable autocrlf so the stashed/dirtied files round-trip cleanly on
    # Windows test hosts.
    git config core.autocrlf false
    Pop-Location
    return $dir
}

function Remove-TempGitRepo {
    param([string]$Dir)
    if ($Dir -and (Test-Path $Dir)) {
        Remove-Item -Recurse -Force $Dir -ErrorAction SilentlyContinue
    }
}

# Case 1: stash@{0} resolves to a real commit after a successful push, and
# `git rev-parse --verify` returns exit 0.
$tmp1 = New-TempGitRepo
Push-Location $tmp1
try {
    "a" | Out-File -FilePath "tracked.txt" -Encoding ASCII -NoNewline
    git add tracked.txt | Out-Null
    git commit -q -m "init"
    "dirty" | Out-File -FilePath "tracked.txt" -Encoding ASCII -NoNewline
    git stash push --include-untracked -m "hermes-install-autostash-test" 2>$null
    Assert-Equal -Expected 0 -Actual $LASTEXITCODE -Label "stash push succeeds on dirty tracked file"
    git rev-parse --verify "stash@{0}" 2>$null
    Assert-Equal -Expected 0 -Actual $LASTEXITCODE -Label "stash@{0} resolves after push"
} finally {
    Pop-Location
    Remove-TempGitRepo -Dir $tmp1
}

# Case 2: after `git stash drop stash@{0}`, `git rev-parse --verify stash@{0}`
# must FAIL with non-zero exit -- this is the exact guard the installer uses
# before calling `git stash apply` (#46791 bug class).
$tmp2 = New-TempGitRepo
Push-Location $tmp2
try {
    "a" | Out-File -FilePath "tracked.txt" -Encoding ASCII -NoNewline
    git add tracked.txt | Out-Null
    git commit -q -m "init"
    "dirty" | Out-File -FilePath "tracked.txt" -Encoding ASCII -NoNewline
    git stash push --include-untracked -m "hermes-install-autostash-test" 2>$null
    git stash drop "stash@{0}" 2>$null
    Assert-Equal -Expected 0 -Actual $LASTEXITCODE -Label "stash drop succeeds"
    git rev-parse --verify "stash@{0}" 2>$null
    Assert-True ($LASTEXITCODE -ne 0) -Label "stash@{0} no longer resolves after drop (regression guard for #46791)"
} finally {
    Pop-Location
    Remove-TempGitRepo -Dir $tmp2
}

# Case 3: a second stash push shifts refs -- stash@{0} becomes the new stash
# and the original moves to stash@{1}. The installer's post-pull-dirt block
# relies on this exact shift to track the user's autostash ref.
$tmp3 = New-TempGitRepo
Push-Location $tmp3
try {
    "a" | Out-File -FilePath "tracked.txt" -Encoding ASCII -NoNewline
    git add tracked.txt | Out-Null
    git commit -q -m "init"
    "dirty" | Out-File -FilePath "tracked.txt" -Encoding ASCII -NoNewline
    git stash push --include-untracked -m "user-autostash" 2>$null
    $userStash = (git rev-parse "stash@{0}" 2>$null).Trim()
    "dirty-2" | Out-File -FilePath "tracked.txt" -Encoding ASCII -NoNewline
    git stash push --include-untracked -m "hermes-install-post-pull-dirt" 2>$null
    $dirtStash = (git rev-parse "stash@{0}" 2>$null).Trim()
    $userStashAfterShift = (git rev-parse "stash@{1}" 2>$null).Trim()
    Assert-True ($userStash -eq $userStashAfterShift) -Label "user autostash ref shifts from stash@{0} to stash@{1} after a second push (#46791)"
    Assert-True ($dirtStash -ne $userStash) -Label "new post-pull-dirt stash is distinct from user stash"
} finally {
    Pop-Location
    Remove-TempGitRepo -Dir $tmp3
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
Write-Host ""
if ($failures -gt 0) {
    Write-Host "FAILED: $failures assertion(s) failed" -ForegroundColor Red
    exit 1
} else {
    Write-Host "All smoke tests passed." -ForegroundColor Green
    exit 0
}
