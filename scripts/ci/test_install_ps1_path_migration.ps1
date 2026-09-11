# Behavioral test for install.ps1's persisted-User-PATH migration.
#
# Run:  pwsh -NoProfile -File scripts/ci/test_install_ps1_path_migration.ps1
#
# Not wired into the default CI lane — the Linux runners have no PowerShell
# host. It runs on any machine with pwsh (including via nixpkgs#powershell),
# and on a Windows runner if one is ever added.
#
# This is NOT a source-regex test. It parses install.ps1, lifts the real
# Set-ManagedNodeFirstOnUserPath body out of the AST, and rewrites *only* the
# two registry calls into an in-memory store so the actual shipped logic —
# split, dedupe, prepend, change-detection — executes for real. Rewriting from
# the AST rather than hand-copying the body means the test cannot silently
# drift away from the function it claims to cover.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$installPs1 = Join-Path $PSScriptRoot '..' 'install.ps1' | Resolve-Path
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $installPs1, [ref]$null, [ref]$null)

$fn = $ast.Find({
    param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq 'Set-ManagedNodeFirstOnUserPath'
}, $true)

if (-not $fn) {
    throw "Set-ManagedNodeFirstOnUserPath not found in $installPs1"
}

# Swap the two registry calls for the in-memory store. Both must match, or the
# function has changed shape and this harness is no longer exercising it.
# Rewrite the whole definition extent (which already carries `function <name>
# { param(...) ... }`) so the shipped param block and body run verbatim.
$definition = $fn.Extent.Text
$reads = ([regex]'\[Environment\]::GetEnvironmentVariable\("Path", "User"\)').Matches($definition).Count
$writes = ([regex]'\[Environment\]::SetEnvironmentVariable\("Path", ([^,]+), "User"\)').Matches($definition).Count
if ($reads -ne 1 -or $writes -ne 1) {
    throw "expected exactly one User PATH read and one write in the function body; found $reads read(s), $writes write(s). Update this harness."
}

$definition = $definition -replace `
    '\[Environment\]::GetEnvironmentVariable\("Path", "User"\)', '$script:FakeUserPath'
$definition = $definition -replace `
    '\[Environment\]::SetEnvironmentVariable\("Path", ([^,]+), "User"\)', '$script:FakeUserPath = $1; $script:FakeWrites++'

Invoke-Expression $definition

$NODE = 'C:\Users\me\AppData\Local\hermes\node'
$script:Failures = 0

function Invoke-Migration {
    param([string]$Start, [string]$NodeDir = $NODE)
    $script:FakeUserPath = $Start
    $script:FakeWrites = 0
    Set-ManagedNodeFirstOnUserPath $NodeDir
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

Write-Host "install.ps1 Set-ManagedNodeFirstOnUserPath"

# The regression this function exists for: an install made by an older
# install.ps1, which *appended*. A system Node leads and the managed dir is
# stranded at the tail, so every new shell resolves the wrong node.exe. An
# add-if-missing check would see the entry present and leave it there forever.
Invoke-Migration "C:\Program Files\nodejs;C:\Users\me\bin;$NODE"
Assert-Equal "$NODE;C:\Program Files\nodejs;C:\Users\me\bin" $script:FakeUserPath `
    'upgrade from appending installer: managed dir becomes first entry'
Assert-Equal 1 (@($script:FakeUserPath -split ';' | Where-Object { $_ -eq $NODE }).Count) `
    'upgrade: managed dir is not duplicated'
Assert-Equal "C:\Program Files\nodejs;C:\Users\me\bin" `
    (($script:FakeUserPath -split ';' | Where-Object { $_ -ne $NODE }) -join ';') `
    'upgrade: unrelated entries keep their relative order'
Assert-Equal 1 $script:FakeWrites 'upgrade: persists exactly once'

Invoke-Migration "$NODE;C:\Program Files\nodejs"
Assert-Equal "$NODE;C:\Program Files\nodejs" $script:FakeUserPath 'already correct: unchanged'
Assert-Equal 0 $script:FakeWrites 'already correct: no registry write'

Invoke-Migration "C:\Program Files\nodejs"
Assert-Equal "$NODE;C:\Program Files\nodejs" $script:FakeUserPath 'fresh install: prepended'

# Empty segments are legal in a real User PATH (a trailing ';' is common) and
# the installer's other PATH code preserves them. Migration must not quietly
# rewrite parts of PATH it was not asked to touch.
Invoke-Migration "C:\Program Files\nodejs;;C:\Users\me\bin;"
Assert-Equal "$NODE;C:\Program Files\nodejs;;C:\Users\me\bin;" $script:FakeUserPath `
    'empty segments are preserved'

# Windows paths are case-insensitive, and -ne on strings is too.
Invoke-Migration "C:\Program Files\nodejs;c:\users\me\appdata\local\HERMES\Node"
Assert-Equal "$NODE;C:\Program Files\nodejs" $script:FakeUserPath `
    'existing entry in different case is replaced, not duplicated'

Invoke-Migration "$NODE;C:\Program Files\nodejs;$NODE"
Assert-Equal "$NODE;C:\Program Files\nodejs" $script:FakeUserPath 'duplicates collapse'

Invoke-Migration ""
Assert-Equal $NODE $script:FakeUserPath 'empty User PATH'

Invoke-Migration "C:\Program Files\nodejs" ""
Assert-Equal "C:\Program Files\nodejs" $script:FakeUserPath 'empty NodeDir is a no-op'
Assert-Equal 0 $script:FakeWrites 'empty NodeDir does not write'

if ($script:Failures -gt 0) {
    Write-Host ""
    Write-Host "$script:Failures assertion(s) failed"
    exit 1
}

Write-Host ""
Write-Host "all assertions passed"
