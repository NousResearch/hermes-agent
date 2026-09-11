#Requires -Version 5.1
<#
.SYNOPSIS
  Install the ast-grep binary on Windows.

.DESCRIPTION
  Tries package managers in priority order, then falls back to downloading a
  pinned release zip from GitHub into <skill_root>/bin/sg.exe.

  Order:
    1. Already installed?  -> nothing to do
    2. Scoop (most common Windows dev tool installer)
    3. Winget (Microsoft built-in)
    4. Chocolatey (choco)
    5. npm (@ast-grep/cli)
    6. cargo binstall / cargo install
    7. pip (ast-grep-cli)
    8. GitHub release zip -> <skill_root>/bin/sg.exe

.PARAMETER Method
  Force one method: scoop | winget | choco | npm | cargo | pip | github

.PARAMETER Version
  Pin a specific version when downloading from GitHub. Default: 0.45.0

.PARAMETER NoFallback
  Don't fall back to GitHub zip; fail if all package managers miss

.PARAMETER Quiet
  Suppress non-error output

.EXAMPLE
  .\install.ps1
  .\install.ps1 -Method scoop
  .\install.ps1 -Version 0.42.0 -Method github
#>

param(
    [string]$Method = "",
    [string]$Version = "0.45.0",
    [switch]$NoFallback,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CacheBinDir = if ($env:OMO_AST_GREP_BIN_DIR) { $env:OMO_AST_GREP_BIN_DIR } else { Join-Path $ScriptDir "bin" }

function Log([string]$msg) {
    if (-not $Quiet) {
        [Console]::Error.WriteLine("[install.ps1] $msg")
    }
}

function Err([string]$msg) {
    [Console]::Error.WriteLine("[install.ps1] error: $msg")
}

function Has-Cmd([string]$name) {
    $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Test-AstGrep {
    if (Has-Cmd 'ast-grep') { return $true }
    if (Has-Cmd 'sg') { return $true }
    if (Test-Path (Join-Path $CacheBinDir 'sg.exe')) { return $true }
    if (Test-Path (Join-Path $CacheBinDir 'ast-grep.exe')) { return $true }
    return $false
}

if (-not $Method -and (Test-AstGrep)) {
    Log "ast-grep already installed"
    exit 0
}

function Detect-Arch {
    $a = $env:PROCESSOR_ARCHITECTURE
    switch -Wildcard ($a) {
        'AMD64' { return 'x86_64' }
        'ARM64' { return 'aarch64' }
        default { return 'unknown' }
    }
}

$Arch = Detect-Arch

function Try-Scoop {
    if (-not (Has-Cmd 'scoop')) { return $false }
    Log "trying: scoop install main/ast-grep"
    try { scoop install main/ast-grep; return $LASTEXITCODE -eq 0 }
    catch { return $false }
}

function Try-Winget {
    if (-not (Has-Cmd 'winget')) { return $false }
    Log "trying: winget install --id ast-grep.ast-grep"
    try { winget install --id ast-grep.ast-grep --silent --accept-package-agreements --accept-source-agreements; return $LASTEXITCODE -eq 0 }
    catch { return $false }
}

function Try-Choco {
    if (-not (Has-Cmd 'choco')) { return $false }
    Log "trying: choco install ast-grep -y"
    try { choco install ast-grep -y; return $LASTEXITCODE -eq 0 }
    catch { return $false }
}

function Try-Npm {
    if (-not (Has-Cmd 'npm')) { return $false }
    Log "trying: npm install -g @ast-grep/cli"
    try { npm install -g '@ast-grep/cli'; return $LASTEXITCODE -eq 0 }
    catch { return $false }
}

function Try-Cargo {
    if (Has-Cmd 'cargo-binstall') {
        Log "trying: cargo binstall -y ast-grep"
        try { cargo binstall -y ast-grep; if ($LASTEXITCODE -eq 0) { return $true } } catch {}
    }
    if (-not (Has-Cmd 'cargo')) { return $false }
    Log "trying: cargo install ast-grep --locked"
    try { cargo install ast-grep --locked; return $LASTEXITCODE -eq 0 }
    catch { return $false }
}

function Try-Pip {
    $pip = $null
    foreach ($p in 'pip3','pip','py') {
        if (Has-Cmd $p) { $pip = $p; break }
    }
    if (-not $pip) { return $false }
    Log "trying: $pip install --user ast-grep-cli"
    try {
        if ($pip -eq 'py') { py -m pip install --user ast-grep-cli }
        else { & $pip install --user ast-grep-cli }
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Triple-For-Windows {
    switch ($Arch) {
        'x86_64' { return 'x86_64-pc-windows-msvc' }
        'aarch64' { return 'aarch64-pc-windows-msvc' }
        default { return '' }
    }
}

function Try-Github {
    $triple = Triple-For-Windows
    if (-not $triple) {
        Err "no GitHub release asset for arch $Arch"
        return $false
    }

    $asset = "app-$triple.zip"
    $url = "https://github.com/ast-grep/ast-grep/releases/download/$Version/$asset"
    $tmp = Join-Path $env:TEMP ("ast-grep-install-" + [guid]::NewGuid().ToString('N').Substring(0,8))
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    try {
        Log "downloading $url"
        Invoke-WebRequest -Uri $url -OutFile (Join-Path $tmp $asset) -UseBasicParsing
        Expand-Archive -Path (Join-Path $tmp $asset) -DestinationPath (Join-Path $tmp 'extract') -Force

        New-Item -ItemType Directory -Path $CacheBinDir -Force | Out-Null

        $candidates = @(
            (Join-Path $tmp 'extract/ast-grep.exe'),
            (Join-Path $tmp 'extract/sg.exe')
        )
        $src = $null
        foreach ($c in $candidates) {
            if (Test-Path $c) { $src = $c; break }
        }
        if (-not $src) {
            Err "no ast-grep.exe or sg.exe found inside $asset"
            return $false
        }

        $dest = Join-Path $CacheBinDir 'sg.exe'
        Copy-Item -Path $src -Destination $dest -Force
        Log "installed cached binary: $dest"
        Log "verify: & '$dest' --version"
        Log ""
        Log "Add to PATH for direct sg use:"
        Log "  `$env:Path = '$CacheBinDir;' + `$env:Path"
        return $true
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

function Run-Method([string]$m) {
    switch ($m) {
        'scoop'  { return Try-Scoop }
        'winget' { return Try-Winget }
        'choco'  { return Try-Choco }
        'npm'    { return Try-Npm }
        'cargo'  { return Try-Cargo }
        'pip'    { return Try-Pip }
        'github' { return Try-Github }
        default  { Err "unknown method: $m"; return $false }
    }
}

if ($Method) {
    if (Run-Method $Method) { exit 0 }
    Err "method '$Method' failed"
    exit 2
}

$methods = @('scoop', 'winget', 'choco', 'npm', 'cargo', 'pip')
foreach ($m in $methods) {
    if (Run-Method $m) {
        Log "installed via $m"
        exit 0
    }
    Log "$m unavailable or failed; trying next"
}

if (-not $NoFallback) {
    Log "all package managers failed; falling back to GitHub release"
    if (Try-Github) { exit 0 }
}

Err "all install methods failed."
Err ""
Err "Manual options:"
Err "  scoop install main/ast-grep                            # Scoop"
Err "  winget install --id ast-grep.ast-grep                  # Winget"
Err "  choco install ast-grep                                 # Chocolatey"
Err "  npm install -g @ast-grep/cli                           # any OS with Node"
Err "  cargo install ast-grep --locked                        # any OS with Rust"
Err "  pip install ast-grep-cli                               # any OS with Python"
Err "  https://github.com/ast-grep/ast-grep/releases          # manual binary"
exit 2
