# Behavioral test for install.ps1's persisted-User-PATH handling of the
# managed Node directory.
#
# Run:  pwsh -NoProfile -File scripts/ci/test_install_ps1_path_migration.ps1
#
# Not wired into the default CI lane -- the Linux runners have no PowerShell
# host. It runs on any machine with pwsh (including via nixpkgs#powershell),
# and on a Windows runner if one is ever added.
#
# This is NOT a source-regex test. Like its predecessor (which covered
# Set-ManagedNodeFirstOnUserPath), it parses install.ps1, lifts real function
# bodies out of the AST, and rewrites *only* the registry calls into an
# in-memory store so the actual shipped logic -- split, filter, join,
# change-detection -- executes for real. Rewriting from the AST rather than
# hand-copying the bodies means the tests cannot silently drift away from the
# functions they claim to cover.  Install-ManagedNodeShims defers registry
# access to Add-DirToUserPathIfMissing, so it is lifted verbatim (0 direct
# reads/writes) and exercises its real guards + real .cmd file writes.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$installPs1 = Join-Path $PSScriptRoot '..' 'install.ps1' | Resolve-Path

function Find-InstallFunction {
    param([string]$Name)
    $parsed = [System.Management.Automation.Language.Parser]::ParseFile(
        $installPs1, [ref]$null, [ref]$null)
    return $parsed.Find({
        param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $n.Name -eq $Name
    }, $true)
}

function Get-RewrittenDefinition {
    # Lift the function definition and swap its User-PATH registry calls for
    # the in-memory store. Both call shapes must match the expected counts, or
    # the function has changed shape and this harness is no longer exercising
    # it. Install-ManagedNodeShims has no direct registry access (it delegates
    # to Add-DirToUserPathIfMissing), so it is lifted with 0/0 -- a future
    # inline registry call then fails the count and forces a harness update.
    param([string]$Name, [int]$ExpectedReads = 1, [int]$ExpectedWrites = 1)
    $fn = Find-InstallFunction -Name $Name
    if (-not $fn) {
        throw "$Name not found in $installPs1"
    }
    $definition = $fn.Extent.Text
    $reads  = ([regex]'\[Environment\]::GetEnvironmentVariable\("Path", "User"\)').Matches($definition).Count
    $writes = ([regex]'\[Environment\]::SetEnvironmentVariable\("Path", ([^,]+), "User"\)').Matches($definition).Count
    if ($reads -ne $ExpectedReads -or $writes -ne $ExpectedWrites) {
        throw "expected $ExpectedReads read(s) and $ExpectedWrites write(s) in ${Name}; found $reads read(s), $writes write(s). Update this harness."
    }
    $definition = $definition -replace `
        '\[Environment\]::GetEnvironmentVariable\("Path", "User"\)', '$script:FakeUserPath'
    $definition = $definition -replace `
        '\[Environment\]::SetEnvironmentVariable\("Path", ([^,]+), "User"\)', '$script:FakeUserPath = $1; $script:FakeWrites++'
    return $definition
}

Invoke-Expression (Get-RewrittenDefinition -Name 'Remove-ManagedNodeFromUserPath')
Invoke-Expression (Get-RewrittenDefinition -Name 'Add-DirToUserPathIfMissing')
Invoke-Expression (Get-RewrittenDefinition -Name 'Install-ManagedNodeShims' -ExpectedReads 0 -ExpectedWrites 0)

# Install-ManagedNodeShims calls install.ps1's write helpers on failure paths
# and writes real .cmd files -- stub the helpers and point it at temp dirs so
# the shipped logic (guards, shim content, delegation) runs for real.
function Write-Info    { Write-Host "[info]  $args" }
function Write-Success { Write-Host "[ok]    $args" }
function Write-Warn    { Write-Host "[warn]  $args" }

$script:FakeTree = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-node-shim-tree-" + [guid]::NewGuid().ToString("N"))
$script:FakeBin  = Join-Path ([System.IO.Path]::GetTempPath()) ("hermes-node-shim-bin-"  + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $script:FakeTree | Out-Null
New-Item -ItemType Directory -Force -Path $script:FakeBin  | Out-Null

$NODE = 'C:\Users\me\AppData\Local\hermes\node'
$BIN  = 'C:\Users\me\AppData\Local\hermes\bin'
$script:Failures = 0

function Invoke-Migration {
    param([string]$Start, [string]$NodeDir = $NODE)
    $script:FakeUserPath = $Start
    $script:FakeWrites = 0
    Remove-ManagedNodeFromUserPath $NodeDir
}

function Invoke-AddIfMissing {
    param([string]$Start, [string]$Dir)
    $script:FakeUserPath = $Start
    $script:FakeWrites = 0
    Add-DirToUserPathIfMissing -Dir $Dir
}

function Assert-Equal {
    param($Expected, $Actual, [string]$Name)
    if ($Expected -ceq $Actual) {
        Write-Host "  PASS  $Name"
    } else {
        Write-Host "  FAIL  $Name"
        Write-Host "        expected: [$Expected]"
        Write-Host "        actual:   [$Actual]"
        $script:Failures++
    }
}

Write-Host "install.ps1 Remove-ManagedNodeFromUserPath"

# The migration this function exists for: an install made by an installer
# that registered the managed Node dir itself in User PATH (and moved it to
# the front). Removing the entry restores the user's own node/npm selection;
# unrelated entries must survive untouched.
Invoke-Migration "$NODE;C:\Program Files\nodejs;C:\Users\me\bin"
Assert-Equal "C:\Program Files\nodejs;C:\Users\me\bin" $script:FakeUserPath `
    'upgrade from PATH-registration installer: managed dir removed'

Invoke-Migration "C:\Program Files\nodejs;$NODE;C:\Users\me\bin"
Assert-Equal "C:\Program Files\nodejs;C:\Users\me\bin" $script:FakeUserPath `
    'entry at tail position is also removed'

Invoke-Migration "C:\Program Files\nodejs"
Assert-Equal "C:\Program Files\nodejs" $script:FakeUserPath `
    'clean User PATH: unchanged'
Assert-Equal 0 $script:FakeWrites 'clean User PATH: no registry write'

# Empty segments are legal in a real User PATH (a trailing ';' is common) and
# the installer's other PATH code preserves them. Migration must not quietly
# rewrite parts of PATH it was not asked to touch.
Invoke-Migration "C:\Program Files\nodejs;;C:\Users\me\bin;"
Assert-Equal "C:\Program Files\nodejs;;C:\Users\me\bin;" $script:FakeUserPath `
    'empty segments are preserved when the entry is absent'
Invoke-Migration "$NODE;;C:\Users\me\bin;"
Assert-Equal ";C:\Users\me\bin;" $script:FakeUserPath `
    'removal drops only the entry itself, keeping empty segments around it'

# Windows paths are case-insensitive, and -ne on strings is too.
Invoke-Migration "c:\users\me\appdata\local\HERMES\Node;C:\Program Files\nodejs"
Assert-Equal "C:\Program Files\nodejs" $script:FakeUserPath `
    'existing entry in different case is removed, not partially matched'

# Users edit User PATH in the GUI, where a trailing backslash is common; the
# legacy writer never emitted one, but a hand-edited variant must still go.
Invoke-Migration "$NODE\;C:\Program Files\nodejs"
Assert-Equal "C:\Program Files\nodejs" $script:FakeUserPath `
    'trailing-backslash variant of the managed dir is also removed'

# Duplicate occurrences (older installers could accumulate them) all go.
Invoke-Migration "$NODE;C:\Program Files\nodejs;$NODE"
Assert-Equal "C:\Program Files\nodejs" $script:FakeUserPath 'duplicates collapse away'

# Exactly one write on a run that changes anything.
Invoke-Migration "$NODE"
Assert-Equal "" $script:FakeUserPath 'sole entry removal empties PATH'
Assert-Equal 1 $script:FakeWrites 'migration persists exactly once'

Invoke-Migration "C:\Program Files\nodejs" ""
Assert-Equal "C:\Program Files\nodejs" $script:FakeUserPath 'empty NodeDir is a no-op'
Assert-Equal 0 $script:FakeWrites 'empty NodeDir does not write'

Write-Host ""
Write-Host "install.ps1 Add-DirToUserPathIfMissing (shim dir registration)"

# Fresh install: bin dir not yet on PATH -> prepended, exactly one write.
Invoke-AddIfMissing -Start "C:\Program Files\nodejs" -Dir $BIN
Assert-Equal "$BIN;C:\Program Files\nodejs" $script:FakeUserPath `
    'fresh: shim dir prepended'
Assert-Equal 1 $script:FakeWrites 'fresh: persists exactly once'

# Already present (the normal install/update case): no write at all.
Invoke-AddIfMissing -Start "$BIN;C:\Program Files\nodejs" -Dir $BIN
Assert-Equal "$BIN;C:\Program Files\nodejs" $script:FakeUserPath `
    'already present: unchanged'
Assert-Equal 0 $script:FakeWrites 'already present: no registry write'

# Present with a trailing backslash or different case still counts as present.
Invoke-AddIfMissing -Start "c:\users\me\appdata\local\hermes\bin\;C:\Program Files\nodejs" -Dir $BIN
Assert-Equal "c:\users\me\appdata\local\hermes\bin\;C:\Program Files\nodejs" $script:FakeUserPath `
    'trailing-backslash / case variant counts as present (no duplicate)'
Assert-Equal 0 $script:FakeWrites 'variant match: no registry write'

# Unrelated entries and empty segments are never touched.
Invoke-AddIfMissing -Start "C:\Program Files\nodejs;;C:\Users\me\bin;" -Dir $BIN
Assert-Equal "$BIN;C:\Program Files\nodejs;;C:\Users\me\bin;" $script:FakeUserPath `
    'empty segments preserved when prepending'

# Empty Dir is a guard-clause no-op.
Invoke-AddIfMissing -Start "C:\Program Files\nodejs" -Dir ""
Assert-Equal "C:\Program Files\nodejs" $script:FakeUserPath 'empty Dir is a no-op'
Assert-Equal 0 $script:FakeWrites 'empty Dir does not write'

# Empty/unset User PATH: prepend the dir with no trailing empty segment.
Invoke-AddIfMissing -Start "" -Dir $BIN
Assert-Equal "$BIN" $script:FakeUserPath 'empty User PATH: no trailing empty segment'
Assert-Equal 1 $script:FakeWrites 'empty User PATH: persists exactly once'

Write-Host ""
Write-Host "install.ps1 Install-ManagedNodeShims (node/npm/npx delegators)"

function Invoke-Shims {
    param([string]$TreeDir, [string]$Destination = $script:FakeBin)
    Install-ManagedNodeShims -NodeDir $TreeDir -Destination $Destination
}

function Reset-ShimEnv {
    # Fresh User-PATH state, no skip-links flag, no stale shims.
    $script:FakeUserPath = ""
    $script:FakeWrites = 0
    $script:NoVenv = $false
    Remove-Item Env:HERMES_NODE_SKIP_LINKS -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $script:FakeBin "node.cmd"), `
                (Join-Path $script:FakeBin "npm.cmd"), `
                (Join-Path $script:FakeBin "npx.cmd") -ErrorAction SilentlyContinue
}

# --- managed tree present: three shims written, bin dir registered once ---
Reset-ShimEnv
New-Item -ItemType File -Force -Path (Join-Path $script:FakeTree "node.exe") | Out-Null
Invoke-Shims -TreeDir $script:FakeTree
Assert-Equal $true (Test-Path (Join-Path $script:FakeBin "node.cmd")) 'node.cmd written'
Assert-Equal $true (Test-Path (Join-Path $script:FakeBin "npm.cmd"))  'npm.cmd written'
Assert-Equal $true (Test-Path (Join-Path $script:FakeBin "npx.cmd"))  'npx.cmd written'
$nodeRaw = [System.IO.File]::ReadAllText((Join-Path $script:FakeBin "node.cmd")).TrimEnd("`r`n")
Assert-Equal "@echo off`r`n`"$script:FakeTree\node.exe`" %*" $nodeRaw `
    'node shim delegates to the managed tree by absolute path'
$npmRaw = [System.IO.File]::ReadAllText((Join-Path $script:FakeBin "npm.cmd")).TrimEnd("`r`n")
Assert-Equal $true ($npmRaw.StartsWith("@echo off`r`ncall `"")) 'npm shim uses call to preserve exit codes'
Assert-Equal $true ($npmRaw.Contains("$script:FakeTree\npm.cmd")) 'npm shim targets the managed npm.cmd'
Assert-Equal $true ($script:FakeUserPath.StartsWith($script:FakeBin)) 'bin dir registered on User PATH'
Assert-Equal 1 $script:FakeWrites 'bin dir registered exactly once'
Remove-Item (Join-Path $script:FakeTree "node.exe") -Force

# --- the regression this guard exists for: no managed tree -> no shims ---
Reset-ShimEnv
Invoke-Shims -TreeDir $script:FakeTree
Assert-Equal $false (Test-Path (Join-Path $script:FakeBin "node.cmd")) 'absent tree: no node.cmd'
Assert-Equal $false (Test-Path (Join-Path $script:FakeBin "npm.cmd"))  'absent tree: no npm.cmd'
Assert-Equal $false (Test-Path (Join-Path $script:FakeBin "npx.cmd"))  'absent tree: no npx.cmd'
Assert-Equal 0 $script:FakeWrites 'absent tree: no registry write'
Assert-Equal "" $script:FakeUserPath 'absent tree: User PATH untouched'

# --- idempotent re-run refreshes the shims, no second PATH write ---
New-Item -ItemType File -Force -Path (Join-Path $script:FakeTree "node.exe") | Out-Null
Reset-ShimEnv
Invoke-Shims -TreeDir $script:FakeTree
Invoke-Shims -TreeDir $script:FakeTree
Assert-Equal $true (Test-Path (Join-Path $script:FakeBin "node.cmd")) 're-run: shims still present'
Assert-Equal 1 $script:FakeWrites 're-run: bin already registered, no second write'
Remove-Item (Join-Path $script:FakeTree "node.exe") -Force

# --- empty NodeDir is a guard-clause no-op ---
Reset-ShimEnv
Invoke-Shims -TreeDir ""
Assert-Equal 0 $script:FakeWrites 'empty NodeDir: no registry write'

# --- HERMES_NODE_SKIP_LINKS=1: private-only, and removes prior shims ---
New-Item -ItemType File -Force -Path (Join-Path $script:FakeTree "node.exe") | Out-Null
Reset-ShimEnv
Invoke-Shims -TreeDir $script:FakeTree            # non-private install first
$env:HERMES_NODE_SKIP_LINKS = "1"
Invoke-Shims -TreeDir $script:FakeTree            # then the user opts out
Assert-Equal $false (Test-Path (Join-Path $script:FakeBin "node.cmd")) 'skip-links: stale node.cmd removed'
Assert-Equal $false (Test-Path (Join-Path $script:FakeBin "npm.cmd"))  'skip-links: stale npm.cmd removed'
Assert-Equal $false (Test-Path (Join-Path $script:FakeBin "npx.cmd"))  'skip-links: stale npx.cmd removed'
Assert-Equal 1 $script:FakeWrites 'skip-links: no additional registry write'
Remove-Item (Join-Path $script:FakeTree "node.exe") -Force

# --- NoVenv: managed tree stays private, checkout stays clean ---
New-Item -ItemType File -Force -Path (Join-Path $script:FakeTree "node.exe") | Out-Null
Reset-ShimEnv
$script:NoVenv = $true
Invoke-Shims -TreeDir $script:FakeTree
Assert-Equal $false (Test-Path (Join-Path $script:FakeBin "node.cmd")) 'NoVenv: no shims written'
Assert-Equal 0 $script:FakeWrites 'NoVenv: no registry write'
Remove-Item (Join-Path $script:FakeTree "node.exe") -Force

Remove-Item -Recurse -Force $script:FakeTree, $script:FakeBin -ErrorAction SilentlyContinue

if ($script:Failures -gt 0) {
    Write-Host ""
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ""
Write-Host "all assertions passed"
