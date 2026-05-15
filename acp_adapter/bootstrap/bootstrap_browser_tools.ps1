# bootstrap_browser_tools.ps1 — install agent-browser + Playwright Chromium
# into ~/.hermes/node/ for use by Hermes Agent's browser tools on Windows.
#
# Targets the registry-install path: users who got Hermes via
# `uvx --from 'hermes-agent[acp]==X' hermes-acp` don't have a repo clone,
# so the install.ps1 `npm install`-in-repo flow doesn't apply. This script
# is a self-contained, idempotent slice of install.ps1's browser block.
#
# Usage:
#   .\bootstrap_browser_tools.ps1                # use defaults
#   .\bootstrap_browser_tools.ps1 -Yes           # accept Chromium download
#   .\bootstrap_browser_tools.ps1 -SkipChromium  # Node + agent-browser only
#
# Idempotent: re-running this is safe and fast.

[CmdletBinding()]
param(
    [switch]$Yes,
    [switch]$SkipChromium
)

$ErrorActionPreference = "Stop"
$NodeVersion = "22"

# ─────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────

function Write-Info    { param([string]$msg) Write-Host "[*] $msg" -ForegroundColor Cyan    }
function Write-Success { param([string]$msg) Write-Host "[+] $msg" -ForegroundColor Green   }
function Write-Warn    { param([string]$msg) Write-Host "[!] $msg" -ForegroundColor Yellow  }
function Write-Err     { param([string]$msg) Write-Host "[x] $msg" -ForegroundColor Red     }

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────

$HermesHome = $env:HERMES_HOME
if (-not $HermesHome) {
    $HermesHome = Join-Path $env:USERPROFILE ".hermes"
}
$NodePrefix = Join-Path $HermesHome "node"

# ─────────────────────────────────────────────────────────────────────────
# Step 1: Node.js
# ─────────────────────────────────────────────────────────────────────────

function Resolve-NpmExe {
    # Same gotcha as install.ps1: prefer npm.cmd over npm.ps1 so the
    # PowerShell execution policy doesn't block us.
    $cmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $cmd) { return $null }
    $npmExe = $cmd.Source
    if ($npmExe -like "*.ps1") {
        $sibling = Join-Path (Split-Path $npmExe -Parent) "npm.cmd"
        if (Test-Path $sibling) { return $sibling }
    }
    return $npmExe
}

function Resolve-NpxExe {
    $cmd = Get-Command npx -ErrorAction SilentlyContinue
    if (-not $cmd) { return $null }
    $npxExe = $cmd.Source
    if ($npxExe -like "*.ps1") {
        $sibling = Join-Path (Split-Path $npxExe -Parent) "npx.cmd"
        if (Test-Path $sibling) { return $sibling }
    }
    return $npxExe
}

function Ensure-Node {
    # System Node on PATH?
    $sysNode = Get-Command node -ErrorAction SilentlyContinue
    if ($sysNode) {
        try {
            $v = & $sysNode.Source --version
            $major = [int]($v -replace '^v(\d+).*', '$1')
            if ($major -ge 20) {
                Write-Success "Node.js $v found on PATH"
                return
            }
            Write-Warn "Node.js $v is older than v20 — installing managed Node."
        } catch {
            Write-Warn "Failed to query Node version: $_"
        }
    }

    # Hermes-managed Node?
    $managedNode = Join-Path $NodePrefix "node.exe"
    if (Test-Path $managedNode) {
        $v = & $managedNode --version
        Write-Success "Node.js $v found (Hermes-managed at $NodePrefix)"
        # Prepend to current-process PATH so subsequent npm/npx calls find it.
        $env:PATH = "$NodePrefix;$env:PATH"
        return
    }

    Write-Info "Installing Node.js $NodeVersion LTS into $NodePrefix ..."

    $arch = if ([Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
    $indexUrl = "https://nodejs.org/dist/latest-v${NodeVersion}.x/"

    try {
        $indexPage = Invoke-WebRequest -Uri $indexUrl -UseBasicParsing
        $matches = [regex]::Matches($indexPage.Content, "node-v${NodeVersion}\.\d+\.\d+-win-${arch}\.zip")
        if ($matches.Count -eq 0) {
            Write-Err "Could not locate Node.js $NodeVersion zip for win-$arch"
            throw "no tarball"
        }
        $zipName = $matches[0].Value
        $zipUrl = "$indexUrl$zipName"

        $tmpDir = Join-Path $env:TEMP "hermes-node-$([guid]::NewGuid().ToString('N'))"
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        $zipPath = Join-Path $tmpDir $zipName

        Write-Info "Downloading $zipName ..."
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing

        Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force
        $extracted = Get-ChildItem -Path $tmpDir -Directory | Where-Object { $_.Name -like "node-v*" } | Select-Object -First 1

        if (-not $extracted) { Write-Err "Node.js extraction failed"; throw "extract" }

        if (Test-Path $NodePrefix) { Remove-Item -Recurse -Force $NodePrefix }
        New-Item -ItemType Directory -Force -Path $HermesHome | Out-Null
        Move-Item -Path $extracted.FullName -Destination $NodePrefix

        Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue

        $env:PATH = "$NodePrefix;$env:PATH"
        $v = & "$NodePrefix\node.exe" --version
        Write-Success "Node.js $v installed to $NodePrefix"
    } catch {
        Write-Err "Node.js install failed: $_"
        Write-Info "Install Node 20+ manually from https://nodejs.org/en/download/ and re-run."
        throw
    }
}

# ─────────────────────────────────────────────────────────────────────────
# Step 2: agent-browser
# ─────────────────────────────────────────────────────────────────────────

function Ensure-AgentBrowser {
    $npmExe = Resolve-NpmExe
    if (-not $npmExe) {
        Write-Err "npm not on PATH after Node install — aborting"
        throw "npm missing"
    }

    # Already installed?
    $existing = Get-Command agent-browser -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Success "agent-browser already installed at $($existing.Source)"
        return
    }

    # When the user has system Node (winget / installer-based), `npm install
    # -g` writes to a directory that may require admin rights. Force the
    # prefix to the user-writable Hermes-managed Node directory so we never
    # need elevation and the agent can always find the result. Mirrors the
    # bash bootstrap's `--prefix $NODE_PREFIX` strategy.
    New-Item -ItemType Directory -Force -Path $NodePrefix | Out-Null

    Write-Info "Installing agent-browser (npm, prefix=$NodePrefix)..."
    & $npmExe install -g --prefix $NodePrefix --silent `
        "agent-browser@^0.26.0" "@askjo/camofox-browser@^1.5.2"
    if ($LASTEXITCODE -ne 0) {
        Write-Err "npm install -g agent-browser failed (exit $LASTEXITCODE)"
        throw "npm install"
    }

    # Windows npm global installs drop shims at $NodePrefix\ root (not bin/).
    # Prepend to PATH so any subsequent npx call resolves them.
    $env:PATH = "$NodePrefix;$env:PATH"

    Write-Success "agent-browser installed to $NodePrefix"
}

# ─────────────────────────────────────────────────────────────────────────
# Step 3: Playwright Chromium
# ─────────────────────────────────────────────────────────────────────────

function Find-SystemBrowser {
    $candidates = @(
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "C:\Program Files\Chromium\Application\chromium.exe",
        "${env:LOCALAPPDATA}\Google\Chrome\Application\chrome.exe",
        "${env:LOCALAPPDATA}\Chromium\Application\chromium.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    # Edge — Chromium-based, agent-browser can use it
    foreach ($p in @(
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    )) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Write-BrowserEnv {
    param([string]$BrowserPath)
    $envFile = Join-Path $HermesHome ".env"
    New-Item -ItemType Directory -Force -Path $HermesHome | Out-Null
    if (Test-Path $envFile) {
        $existing = Get-Content $envFile -Raw -ErrorAction SilentlyContinue
        if ($existing -and ($existing -match "(?m)^AGENT_BROWSER_EXECUTABLE_PATH=")) {
            return
        }
    }
    Add-Content -Path $envFile -Value ""
    Add-Content -Path $envFile -Value "# Hermes Agent browser tools — use the system Chrome/Chromium/Edge binary."
    Add-Content -Path $envFile -Value "AGENT_BROWSER_EXECUTABLE_PATH=$BrowserPath"
    Write-Success "Configured browser tools to use $BrowserPath"
}

function Confirm-ChromiumDownload {
    if ($Yes) { return $true }
    if (-not [Environment]::UserInteractive) {
        Write-Warn "Non-interactive shell — skipping Chromium prompt."
        Write-Info "Re-run with -Yes to install Chromium (~400 MB download)."
        return $false
    }
    $reply = Read-Host "Install Playwright Chromium (~400 MB download)? [y/N]"
    return ($reply -match "^(y|yes)$")
}

function Ensure-Chromium {
    if ($SkipChromium) {
        Write-Info "Skipping Chromium install (-SkipChromium)"
        return
    }

    # agent-browser on Windows expects a Playwright-managed Chromium under
    # %LOCALAPPDATA%\ms-playwright. The system-browser shortcut from the
    # Linux/macOS path doesn't apply the same way on Windows — Playwright's
    # default launch path won't pick up a stock Chrome install without an
    # explicit AGENT_BROWSER_EXECUTABLE_PATH. We still offer it as a
    # fallback when the user doesn't want the download.

    if (-not (Confirm-ChromiumDownload)) {
        $sys = Find-SystemBrowser
        if ($sys) {
            Write-Info "Using system browser at $sys (Chromium download skipped)."
            Write-BrowserEnv -BrowserPath $sys
        } else {
            Write-Info "Chromium install skipped. Browser tools won't launch until"
            Write-Info "Chromium is installed or AGENT_BROWSER_EXECUTABLE_PATH is set."
        }
        return
    }

    $npxExe = Resolve-NpxExe
    if (-not $npxExe) {
        Write-Err "npx not on PATH — cannot install Playwright Chromium"
        throw "npx missing"
    }

    Write-Info "Installing Playwright Chromium (~400 MB) ..."
    & $npxExe --yes playwright install chromium
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Playwright Chromium install failed (exit $LASTEXITCODE)"
        Write-Info "Try again later: npx --yes playwright install chromium"
        throw "playwright"
    }
    Write-Success "Playwright Chromium installed"
}

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

Write-Info "Hermes Agent: bootstrapping browser tools"
Write-Info "  HERMES_HOME = $HermesHome"
Write-Info "  OS          = Windows"

Ensure-Node
Ensure-AgentBrowser
Ensure-Chromium

Write-Success "Browser tools setup complete."
Write-Info "Hermes Agent will pick up agent-browser from $NodePrefix on next launch."
