# Behavioral test for install.ps1's HERMES_HOME handling (issue #118988).
#
# Run:  powershell.exe -NoProfile -File scripts/ci/test_install_ps1_hermes_home_scope.ps1
#
# Not wired into the default CI lane - the Linux runners have no PowerShell
# host. It runs on any machine with pwsh, and on a Windows runner if one is
# ever added.
#
# This IS a behavioral test, not a source-regex test. It parses install.ps1,
# lifts the real Set-PathVariable body out of the AST, and rewrites *only* the
# User-scope registry calls into an in-memory store, exactly as
# scripts/ci/test_install_ps1_path_migration.ps1 does for PATH. The shipped
# logic - the read-compare-write decision and the process-scope assignment -
# therefore executes for real, while no registry write happens anywhere in the
# run and the test stays independent of whatever HERMES_HOME the developer's
# machine happens to have. Rewriting from the AST rather than hand-copying the
# body means the test cannot silently drift away from the function it claims to
# cover.
#
# Contract under test (#118988):
#   install.ps1 must NOT persist HERMES_HOME at User scope. A user-level
#   HERMES_HOME leaks into every Desktop SSH remote probe: the Windows remote
#   probe reads $env:HERMES_HOME, which resolves through the client's inherited
#   environment, and a local Windows path like
#   C:\Users\<u>\AppData\Local\hermes\hermes-agent then fails the remote-home
#   safety check as a Windows-style absolute path. `hermes update` also
#   re-runs install.ps1's path stage on repair, so a leaked value re-poisons
#   the environment after the user removes it by hand.
#
#   The process-scope assignment must stay: the rest of the install run and
#   every child it spawns resolve config/data through $env:HERMES_HOME.
#
#   A value the user set deliberately (setx HERMES_HOME ...) must survive
#   verbatim - never read, rewritten, or deleted by an install.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$installPs1 = Join-Path $PSScriptRoot '../install.ps1' | Resolve-Path
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installPs1, [ref]$null, [ref]$null)

$fn = $ast.Find({
    param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq 'Set-PathVariable'
}, $true)

if (-not $fn) {
    throw "Set-PathVariable not found in $installPs1"
}

# Rewrite the whole definition extent (which already carries `function <name>
# { param(...) ... }`) so the shipped param block and body run verbatim.
$definition = $fn.Extent.Text

# Shape guard: the harness must know what it is rewriting. The body either
# performs ZERO HERMES_HOME registry calls (the #118988 contract) or one Get
# plus one Set (the pre-fix behaviour this test must catch). Anything else means
# the function changed shape and the harness is no longer exercising it.
$hhReads = ([regex]'\[Environment\]::GetEnvironmentVariable\(\s*"HERMES_HOME"').Matches($definition).Count
$hhWrites = ([regex]'\[Environment\]::SetEnvironmentVariable\(\s*"HERMES_HOME"').Matches($definition).Count
if (-not (($hhReads -eq 0 -and $hhWrites -eq 0) -or ($hhReads -eq 1 -and $hhWrites -eq 1))) {
    throw ("expected 0 or 1 HERMES_HOME User-scope reads and matching writes in Set-PathVariable; " +
           "found $hhReads read(s), $hhWrites write(s). Update this harness.")
}

# Swap the User-scope registry calls for the in-memory store. Get becomes a
# store lookup; Set becomes a store assignment that also counts HERMES_HOME
# writes separately from the (legitimate, unrelated) PATH writes.
$definition = $definition -replace `
    '\[Environment\]::GetEnvironmentVariable\(\s*"(?<name>[A-Za-z_]+)"\s*,\s*"User"\s*\)',
    '$script:Store["${name}"]'
$definition = $definition -replace `
    '\[Environment\]::SetEnvironmentVariable\(\s*"(?<name>[A-Za-z_]+)"\s*,\s*(?<value>[^,]+?)\s*,\s*"User"\s*\)',
    '$script:Store["${name}"] = ${value}; $script:Writes++; if ("${name}" -eq "HERMES_HOME") { $script:HHWrites++ }'

Invoke-Expression $definition

$script:Failures = 0

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

function Invoke-SetPathVariable {
    param([hashtable]$Store)
    # $script:InstallDir / $HermesHome / $NoVenv / Write-* emulate the real
    # script scope the body reads, so no external state leaks in.
    $script:Store = $Store
    $script:Writes = 0
    $script:HHWrites = 0
    $script:InstallDir = 'C:\Users\me\AppData\Local\hermes\hermes-agent'
    $script:HermesHome = 'C:\Users\me\AppData\Local\hermes'
    $script:NoVenv = $true
    # Install-HermesCommandLaunchers would stage real launchers; the venv-less
    # path never calls it, so the stub is belt-and-braces only.
    function global:Install-HermesCommandLaunchers { param($Root, $Destination) $Destination }
    function global:Write-Info { param($m) }
    function global:Write-Success { param($m) }
    # Start from a clean process env so the assertion observes what the shipped
    # body actually assigns, never a value inherited from the test host.
    $env:HERMES_HOME = $null
    Set-PathVariable
}

Write-Host "install.ps1 Set-PathVariable HERMES_HOME scope (#118988)"

# 1. The regression: a fresh install must not persist HERMES_HOME at User
#    scope. The launcher/venv resolution reads it from the process env, and the
#    CLI sets it explicitly per-spawn; a persisted value only exists to leak
#    into remote SSH probes. Seeded with the empty string a machine that has
#    never had the variable would present.
$seed = @{ HERMES_HOME = ''; Path = 'C:\Program Files\nodejs;C:\Users\me\bin' }
Invoke-SetPathVariable $seed
Assert-Equal '' $script:Store['HERMES_HOME'] `
    'fresh install leaves the User-scope HERMES_HOME untouched'
Assert-Equal 0 $script:HHWrites `
    'fresh install performs zero HERMES_HOME User-scope writes'

# 2. The process environment must still carry HERMES_HOME for the rest of the
#    install run and every child it spawns. This is the whole reason the
#    assignment is kept: only its persistence was the bug.
Assert-Equal 'C:\Users\me\AppData\Local\hermes' $env:HERMES_HOME `
    'process HERMES_HOME is set for this install run'

# 3. The user's own deliberate choice must survive untouched. Someone who set
#    HERMES_HOME by hand (or via setx) to relocate their home is a legitimate
#    configuration and must not be cleared or rewritten by an install.
$deliberate = 'D:\Hermes\data'
Invoke-SetPathVariable @{ HERMES_HOME = $deliberate; Path = 'C:\Program Files\nodejs' }
Assert-Equal $deliberate $script:Store['HERMES_HOME'] `
    'user-set HERMES_HOME value is preserved verbatim'
Assert-Equal 0 $script:HHWrites `
    'user-set HERMES_HOME is never rewritten'

# 4. Pinning the process env is unconditional, so it holds even when the user
#    relocated their home: the installer must work against the tree it is
#    actually installing, not against the user's data directory.
Assert-Equal 'C:\Users\me\AppData\Local\hermes' $env:HERMES_HOME `
    'process HERMES_HOME is pinned to the install home even when the user set one'

# 5. The rewrite must not have disabled the unrelated PATH work the same
#    function performs; otherwise a green result here would prove nothing about
#    the shipped body.
$seed = @{ HERMES_HOME = ''; Path = 'C:\Program Files\nodejs' }
Invoke-SetPathVariable $seed
Assert-Equal 'C:\Users\me\AppData\Local\hermes\hermes-agent;C:\Program Files\nodejs' `
    $script:Store['Path'] 'the PATH mutation the same function owns still runs'

if ($script:Failures -gt 0) {
    Write-Host "install.ps1 HERMES_HOME scope: FAIL ($script:Failures)"
    exit 1
}
Write-Host "install.ps1 HERMES_HOME scope: PASS"
exit 0
