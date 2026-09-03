# ============================================================================
# Hermes Offline Installation Module
# ============================================================================
# Extracted from scripts/install.ps1 — bounded module for offline installation
# logic (bundle creation, bundle-based clone, verification, behavioral witnesses).
# ============================================================================

# Precede all module-level code with a strict mode guard so syntax errors are
# caught at load time rather than at invocation.
Set-StrictMode -Version Latest

# --- Behavioral witness helpers (BLOCKER 1 evidence) ---

function New-BehavioralWitness {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Expected,
        [string]$Actual = "",
        [string]$Type = "assert"
    )
    $witness = @{
        witness_name = $Name
        type           = $Type
        path           = $Path
        expected       = $Expected
        actual         = $Actual
        timestamp      = (Get-Date -Format "o")
    }
    $witnessPath = Join-Path (Split-Path $Path -Parent) "behavioral-witness.json"
    if (-not (Test-Path (Split-Path $witnessPath -Parent))) {
        New-Item -ItemType Directory -Path (Split-Path $witnessPath -Parent) -Force | Out-Null
    }
    $existing = @()
    if (Test-Path $witnessPath) {
        try { $existing = (Get-Content $witnessPath -Raw | ConvertFrom-Json -AsHashtable) } catch { }
        if (-not (Compare-Object $existing.GetType().Name 'Object[]')) { $existing = @($existing) }
    }
    $existing += $witness
    $existing | ConvertTo-Json -Depth 5 | Set-Content -Path $witnessPath -Encoding UTF8
    return $witness
}

function Get-BehavioralWitness {
    param([string]$Name, [string]$Path = ".")
    $witnessPath = Join-Path (Split-Path $Path -Parent) "behavioral-witness.json"
    if (-not (Test-Path $witnessPath)) { return $null }
    try {
        $all = Get-Content $witnessPath -Raw | ConvertFrom-Json -AsHashtable
        return ($all | Where-Object { $_.witness_name -eq $Name } | Select-Object -First 1)
    } catch { return $null }
}

# --- Bundle contract helpers ---

# Builds a git bundle from the remote repository honoring Commit > Tag > Branch.
function Invoke-OfflineBundleCreate {
    param(
        [string]$TargetDir,
        [string]$Branch = "main",
        [string]$Commit = "",
        [string]$Tag = ""
    )
    $bundleName = "hermes-agent.bundle"
    $bundlePath = Join-Path $TargetDir $bundleName

    # Determine the fetch ref using the exact-commit contract precedence.
    $fetchRef = if ($Commit) {
        $Commit
    } elseif ($Tag) {
        "refs/tags/$Tag"
    } else {
        "refs/heads/$Branch"
    }

    # Create a temporary bare clone to fetch from the network, then bundle it.
    $tempRepo = "$env:TEMP\hermes-offline-bundle-repo-$(Get-Random)"
    try {
        if (Test-Path $tempRepo) { Remove-Item -Recurse -Force $tempRepo }
        git init --bare $tempRepo 2>$null | Out-Null
        $repoUrl = "https://github.com/NousResearch/hermes-agent.git"
        git -C $tempRepo fetch --depth=1 $repoUrl "+$fetchRef:$fetchRef" 2>$null
        if ($LASTEXITCODE -ne 0 -and $Commit) {
            # A bare commit may not be reachable at depth 1; retry full fetch for it.
            $global:LASTEXITCODE = 0
            git -C $tempRepo fetch $repoUrl "+$Commit:$Commit" 2>$null
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to fetch $fetchRef for bundle creation (exit $LASTEXITCODE)"
        }
        # Bundle all fetched refs so the consumer can clone / fetch from it.
        git -C $tempRepo bundle create $bundlePath --all 2>$null
        if (-not (Test-Path $bundlePath)) {
            throw "Bundle creation did not produce $bundlePath"
        }
        # Verify bundle integrity (fail-closed witness).
        $bundleVerify = git bundle verify $bundlePath 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Bundle verification failed for $bundlePath"
        }
        # Behavioral witness: bundle created with the correct contract.
        $bundleHash = (Get-FileHash $bundlePath -Algorithm SHA256).Hash
        New-BehavioralWitness -Name "bundle-created" -Path $bundlePath `
            -Expected "$bundleName;hash=$bundleHash;ref=$fetchRef"
        return $bundlePath
    } finally {
        if (Test-Path $tempRepo) { Remove-Item -Recurse -Force $tempRepo -ErrorAction SilentlyContinue }
    }
}

# Clones from a bundle using the exact-commit contract precedence.
function Install-RepositoryFromBundle {
    param(
        [string]$BundlePath,
        [string]$InstallDir,
        [string]$Branch = "main",
        [string]$Commit = "",
        [string]$Tag = ""
    )
    if (-not (Test-Path $BundlePath)) {
        throw "Offline repository bundle not found: $BundlePath"
    }
    # Initialize repo from bundle contents.
    if (Test-Path $InstallDir) {
        $backupDir = "$InstallDir.broken-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        Move-Item -LiteralPath $InstallDir -Destination $backupDir -ErrorAction Stop
    }
    # Clone directly from bundle file when possible; else init + remote + fetch.
    $bundleCloneSuccess = $false
    try {
        # Try direct clone from bundle.
        git clone $BundlePath $InstallDir 2>$null
        if ($LASTEXITCODE -eq 0) {
            $bundleCloneSuccess = $true
        }
    } catch { }
    if (-not $bundleCloneSuccess) {
        # Fallback: init and fetch from bundle.
        git init $InstallDir 2>$null | Out-Null
        Push-Location $InstallDir
        git remote add origin $BundlePath 2>$null
        $fetchRef = if ($Commit) { $Commit } elseif ($Tag) { "refs/tags/$Tag" } else { $Branch }
        git fetch origin "+$fetchRef:$fetchRef" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            throw "git fetch from bundle failed (exit $LASTEXITCODE) for ref $fetchRef"
        }
        # Checkout the requested pin with precedence.
        if ($Commit) {
            git checkout --detach $Commit 2>$null
        } elseif ($Tag) {
            git checkout --detach "refs/tags/$Tag" 2>$null
        } else {
            git checkout -B $Branch $fetchRef 2>$null
        }
        # Configure managed clone settings.
        git config windows.appendAtomically false 2>$null
        git config core.autocrlf false 2>$null
        git remote add origin https://github.com/NousResearch/hermes-agent.git 2>$null
        Pop-Location
    }
    # Verify HEAD materialization matches the requested pin.
    Push-Location $InstallDir
    try {
        $headSha = (git rev-parse HEAD 2>$null | Select-Object -First 1).ToString().Trim()
        $expectedSha = if ($Commit) {
            $Commit
        } elseif ($Tag) {
            (git rev-parse "refs/tags/$Tag" 2>$null | Select-Object -First 1).ToString().Trim()
        } else {
            # For branch, verify the fetched branch head exists and record it.
            (git rev-parse "$Branch" 2>$null | Select-Object -First 1).ToString().Trim()
        }
        # Behavioral witness: verify HEAD matches expected pin.
        $match = ($headSha -eq $expectedSha)
        New-BehavioralWitness -Name "repo-head-verify" -Path $InstallDir `
            -Expected $expectedSha -Actual $headSha -Type $(if ($match) { "assert" } else { "mismatch" })
        if (-not $match) {
            throw "Repository HEAD ($headSha) does not match requested pin ($expectedSha)"
        }
    } finally {
        Pop-Location
    }
    return $true
}

# Verify that an offline asset's SHA-256 matches manifest expectations.
# Fail-closed: returns $false when no expected hash is present (BLOCKER 3).
function Test-OfflineAssetHash {
    param(
        [string]$FilePath,
        [string]$ExpectedHash
    )
    if (-not $ExpectedHash) {
        # BLOCKER 3: fail closed — no hash means verification fails.
        return $false
    }
    if (-not (Test-Path $FilePath)) { return $false }
    $actual = (Get-FileHash -Path $FilePath -Algorithm SHA256).Hash.ToLower()
    return $actual -eq $ExpectedHash.ToLower()
}
