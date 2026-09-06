function Find-SystemBrowser {
    # Honor ONLY an explicit, user-set AGENT_BROWSER_EXECUTABLE_PATH override.
    #
    # We no longer scan well-known install locations for a system browser.
    # Auto-detection silently bound the install to an arbitrary binary instead
    # of the bundled Playwright Chromium, which made the browser tool behave
    # differently across hosts (and, on Linux, picked up a sandboxed Snap
    # Chromium that hangs every browser_navigate). Every install now uses the
    # bundled Chromium unless the user explicitly points elsewhere.
    $override = $env:AGENT_BROWSER_EXECUTABLE_PATH
    if ([string]::IsNullOrWhiteSpace($override)) { return $null }
    if (Test-Path $override) { return $override }
    return $null
}

function Write-BrowserEnv {
    param([string]$BrowserPath)
    if (-not (Test-Path $HermesHome)) {
        New-Item -ItemType Directory -Force -Path $HermesHome | Out-Null
    }
    $envFile = Join-Path $HermesHome ".env"
    if (-not (Test-Path $envFile)) {
        Set-Content -Path $envFile -Value "AGENT_BROWSER_EXECUTABLE_PATH=$BrowserPath" -Encoding UTF8
        return
    }
    $content = Get-Content $envFile -Raw -ErrorAction SilentlyContinue
    if ($content -and $content -match "AGENT_BROWSER_EXECUTABLE_PATH=") { return }
    Add-Content -Path $envFile -Value "AGENT_BROWSER_EXECUTABLE_PATH=$BrowserPath" -Encoding UTF8
}

function Install-AgentBrowser {
    $npm = Resolve-NpmCmd
    if (-not $npm) {
        Write-Err "npm not found -- install Node.js first"
        throw "npm not found"
    }

    # agent-browser itself is intentionally NOT installed here (#43564 /
    # PR #44772 review): it resolves lazily via `npx agent-browser` instead,
    # which every consumer (tools/browser_tool.py, `hermes update`'s npx
    # cache warm) already goes through. Eagerly npm-installing a second,
    # separately version-pinned copy here -- only reachable via this
    # explicit -Ensure browser fallback in the first place -- was redundant
    # complexity and an extra credential/supply-chain surface for a path
    # npx already covers.
    Write-Info "Installing camofox browser server..."
    $prefixDir = Join-Path $HermesHome "node"
    if (-not (Test-Path $prefixDir)) {
        New-Item -ItemType Directory -Path $prefixDir -Force | Out-Null
    }
    $npmLog = [System.IO.Path]::GetTempFileName()
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $npm install -g --prefix $prefixDir --silent --ignore-scripts "@askjo/camofox-browser@^1.5.2" 2>&1 | Tee-Object -FilePath $npmLog | Out-Null
    $npmExit = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($npmExit -ne 0) {
        $npmDetail = Get-Content $npmLog -Raw -ErrorAction SilentlyContinue
        Remove-Item $npmLog -Force -ErrorAction SilentlyContinue
        Write-Err "npm install -g failed (exit $npmExit): $npmDetail"
        Show-NpmCertHint $npmDetail | Out-Null
        # This install runs with --silent, so $npmDetail is often near-empty;
        # npm's debug log is the only place the real error survives.
        Write-NpmDebugLogTail -NpmOutput $npmDetail
        throw "npm install failed"
    }
    Remove-Item $npmLog -Force -ErrorAction SilentlyContinue

    $sysBrowser = Find-SystemBrowser
    if ($sysBrowser) {
        Write-BrowserEnv -BrowserPath $sysBrowser
        Write-Info "Explicit browser override set -- Chromium download will be skipped when agent-browser installs on demand"
    }
    Write-Success "Agent-browser ready"
}

