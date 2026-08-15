# bootstrap.ps1 -- dependency-free escape hatch for Windows update deadlocks.
#
# This script is intentionally independent from hermes_cli.  It is run by the
# OS-level Desktop handoff after an update child exits with the recoverable
# self-lock code (2).  Its job is to advance a pre-fix checkout to code that
# contains the honest, post-fetch self-lock handling before starting Python
# again.  It must remain compatible with Windows PowerShell 5.1.

param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot,
    [string]$Branch = "main",
    [switch]$NoRelaunch
)

$ErrorActionPreference = "Stop"
$root = [System.IO.Path]::GetFullPath($InstallRoot)
$git = "git"
$python = Join-Path $root "venv\Scripts\python.exe"

if ([string]::IsNullOrWhiteSpace($Branch) -or
    $Branch -notmatch '^[A-Za-z0-9._/-]+$' -or $Branch.Contains("..")) {
    throw "Invalid update branch: $Branch"
}

function Invoke-Native([string]$FilePath, [string[]]$Arguments) {
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath $($Arguments -join ' ') exited with code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $root ".git"))) {
    throw "Not a git checkout: $root"
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing venv interpreter: $python"
}

# Do not silently destroy user work.  The normal updater owns stash policy;
# this escape hatch only runs when that updater could not reach its new code.
$status = (& $git -C $root status --porcelain 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect working tree: $root"
}
if ($status) {
    throw "Working tree is not clean; preserve local changes, then rerun the update bootstrap"
}

# This is deliberately a fresh, non-Python process.  The old updater may have
# mapped cryptography._rust; git does not import the venv and can advance the
# checkout without touching the locked extension.
Invoke-Native $git @("-C", $root, "fetch", "origin", $Branch)
Invoke-Native $git @("-C", $root, "merge", "--ff-only", "origin/$Branch")

# Start a new interpreter only after the checkout has advanced.  The new
# hermes_cli.main contains the post-fetch self-lock placement and early
# recovery path.  --force is safe here because this script already verified
# that the working tree is clean; the Python holder guard remains active.
$pythonArgs = @("-m", "hermes_cli.main", "update", "--yes", "--gateway", "--force", "--branch", $Branch)
& $python @pythonArgs
$code = $LASTEXITCODE
if ($code -ne 0) {
    throw "Fresh Hermes updater exited with code $code"
}

exit 0
