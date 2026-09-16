# Run by tests/scripts/install/test_install_ps1_script_suites.py under both PS 5.1 and 7.
$ErrorActionPreference = 'Stop'
$installer = Join-Path (Split-Path $PSScriptRoot -Parent) 'install.ps1'
$testRoot = Join-Path $PSScriptRoot ("hermes-marker-" + [guid]::NewGuid().ToString('N'))
$checkout = Join-Path $testRoot 'checkout'
try {
    New-Item -ItemType Directory -Path $checkout -Force | Out-Null
    & git -C $checkout init -q
    if ($LASTEXITCODE -ne 0) { throw 'fixture git init failed' }
    & git -C $checkout -c user.name=Test -c user.email=test@example.invalid commit --allow-empty -qm installed
    if ($LASTEXITCODE -ne 0) { throw 'fixture git commit failed' }
    $installedHead = (& git -C $checkout rev-parse HEAD).Trim()
    $requested = 'a' * 40

    . $installer -InstallDir $checkout -HermesHome (Join-Path $testRoot 'home') -Commit $requested
    $script:Commit = $requested
    # The upstream staged-Git test covers actual provisioning. This focused
    # marker test avoids downloading dependencies from the network.
    function Ensure-Git { return $true }
    Stage-Complete
    $marker = Get-Content (Join-Path $checkout '.hermes-bootstrap-complete') -Raw | ConvertFrom-Json
    if ($marker.pinnedCommit -ne $installedHead) { throw 'marker did not record installed HEAD' }
    if ($marker.pinnedCommit -eq $requested) { throw 'marker recorded the stale requested pin' }

    if ($Commit -ne $requested) { throw "dot-source lost request pin: '$Commit'" }
    function Ensure-Git { return $false }
    Stage-Complete
    $fallback = Get-Content (Join-Path $checkout '.hermes-bootstrap-complete') -Raw | ConvertFrom-Json
    if ($fallback.pinnedCommit -ne $requested) { throw 'archive fallback did not keep requested pin' }

    # Stub the child runner, not the installer boundary under test: it must
    # clear inherited CI identity while spawning and restore it afterward.
    function Get-BootstrapPython { return 'unused-bootstrap-python' }
    function Invoke-Logged {
        param([string]$StatusLabel, [scriptblock]$NativeBlock)
        foreach ($name in @('GITHUB_SHA', 'GITHUB_REF_NAME', 'GITHUB_HEAD_REF')) {
            if ([Environment]::GetEnvironmentVariable($name, 'Process')) {
                throw "CI variable $name leaked into local build"
            }
        }
        $global:LASTEXITCODE = 0
    }
    $oldSha = [Environment]::GetEnvironmentVariable('GITHUB_SHA', 'Process')
    $oldRef = [Environment]::GetEnvironmentVariable('GITHUB_REF_NAME', 'Process')
    $oldHead = [Environment]::GetEnvironmentVariable('GITHUB_HEAD_REF', 'Process')
    try {
        $env:GITHUB_SHA = $requested
        $env:GITHUB_REF_NAME = 'stale'
        $env:GITHUB_HEAD_REF = 'older'
        Invoke-SourceCompletion $true
        if ($env:GITHUB_SHA -ne $requested -or $env:GITHUB_REF_NAME -ne 'stale' -or $env:GITHUB_HEAD_REF -ne 'older') {
            throw 'caller CI variables were not restored'
        }
        function Invoke-Logged { throw 'simulated build failure' }
        $failed = $false
        try { Invoke-SourceCompletion $true } catch { $failed = $true }
        if (-not $failed) { throw 'simulated build failure was not propagated' }
        if ($env:GITHUB_SHA -ne $requested -or $env:GITHUB_REF_NAME -ne 'stale' -or $env:GITHUB_HEAD_REF -ne 'older') {
            throw 'caller CI variables were not restored after failure'
        }
    } finally {
        [Environment]::SetEnvironmentVariable('GITHUB_SHA', $oldSha, 'Process')
        [Environment]::SetEnvironmentVariable('GITHUB_REF_NAME', $oldRef, 'Process')
        [Environment]::SetEnvironmentVariable('GITHUB_HEAD_REF', $oldHead, 'Process')
    }
} finally {
    if (Test-Path $testRoot) { Remove-Item $testRoot -Recurse -Force }
}
