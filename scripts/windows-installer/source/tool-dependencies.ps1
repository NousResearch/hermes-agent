function Install-NodeDeps {
    if (-not $HasNode) {
        # Cross-process driver mode (Hermes-Setup.exe runs each -Stage NAME
        # in a fresh powershell.exe) means $script:HasNode set by Stage-Node
        # in the previous process isn't visible here. Re-probe rather than
        # trust the stale global -- Stage-Node already ran successfully or
        # the bootstrap would've aborted, so npm is reachable.
        if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
            Write-Info "Skipping Node.js dependencies (Node not installed)"
            return
        }
    }

    # npm lifecycle scripts need node.exe on the PATH visible to child
    # cmd.exe processes.  Stage-Node may have run in a prior process, so
    # re-apply here before any npm install (regression #48130).
    Ensure-NodeExeOnPath | Out-Null

    # Resolve npm explicitly to npm.cmd, NOT npm.ps1.  Node.js on Windows
    # ships BOTH npm.cmd (a batch shim) and npm.ps1 (a PowerShell shim).
    # Get-Command's default ordering picks whichever comes first in PATHEXT,
    # and on many systems that's .ps1 -- but .ps1 requires scripts to be
    # enabled in PowerShell's execution policy, which most Windows users
    # don't have (the Restricted / RemoteSigned default blocks unsigned
    # .ps1 files).  .cmd has no such restriction and works on every box.
    #
    # Strategy: look next to the npm shim we found and prefer npm.cmd if
    # it exists in the same directory.  Fall back to whatever Get-Command
    # returned if we can't find a .cmd sibling.
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        Write-Warn "npm not found on PATH -- skipping Node.js dependencies."
        Write-Info "Open a new PowerShell window and re-run 'hermes setup tools' later."
        return
    }
    $npmExe = $npmCmd.Source
    if ($npmExe -like "*.ps1") {
        $npmCmdSibling = Join-Path (Split-Path $npmExe -Parent) "npm.cmd"
        if (Test-Path $npmCmdSibling) {
            Write-Info "Using npm.cmd (PowerShell execution policy blocks npm.ps1)"
            $npmExe = $npmCmdSibling
        } else {
            Write-Warn "Only npm.ps1 available -- install may fail if script execution is disabled."
            Write-Info "  If it fails, either enable PS script execution or install Node via winget."
        }
    }

    # Wall-clock ceiling for each npm / Playwright invocation in this stage.
    # scripts/install.sh has time-boxed the same work with
    # ``run_with_timeout "$NODE_DEPS_TIMEOUT"`` (600s default) since #39219;
    # Windows never got the guard, so a stalled registry fetch or a wedged
    # Chromium extraction (#76222, #84614) froze the installer forever -- one
    # user left it running 12+ hours overnight.  Same env override as bash
    # for very slow links.
    $nodeDepsTimeoutSec = 600
    if ($env:NODE_DEPS_TIMEOUT -match '^\d+$') {
        $nodeDepsTimeoutSec = [int]$env:NODE_DEPS_TIMEOUT
    }

    # Helper: run a native command with a hard wall-clock timeout while
    # still streaming its output live.  Returns the exit code, or 124 on
    # timeout (the same convention as coreutils ``timeout`` and bash's
    # run_with_timeout).
    #
    # Launcher notes: ``Start-Process -FilePath npm.cmd`` fails with
    # ``%1 is not a valid Win32 application`` on some PowerShell versions
    # because Start-Process bypasses cmd.exe / PATHEXT and expects a real
    # PE file -- so route through cmd.exe, which IS a real PE, honours .cmd
    # batch shims, and performs the stdout+stderr merge into the log file
    # natively.  The parent then tails the log into the console each poll
    # tick, preserving the live progress that makes a 3-minute download
    # distinguishable from a hang (the whole reason _Run-NpmInstall streams
    # output in the first place).  ``Wait-Job -Timeout`` was rejected: jobs
    # swallow live output, and Stop-Job leaves the npm child running.
    # taskkill /T kills the real process tree.  Works on Windows PowerShell
    # 5.1 -- no pwsh-only primitives.
    function _Invoke-NativeWithTimeout(
        [string]$exePath, [string]$argLine, [string]$workDir,
        [string]$logPath, [int]$timeoutSec
    ) {
        $cmdLine = "/d /s /c "" ""$exePath"" $argLine > ""$logPath"" 2>&1 """
        $proc = Start-Process -FilePath $env:ComSpec -ArgumentList $cmdLine `
            -WorkingDirectory $workDir -NoNewWindow -PassThru
        $deadline = [DateTime]::UtcNow.AddSeconds($timeoutSec)
        $shown = 0
        function _Drain-NewLines([string]$path, [ref]$count) {
            $lines = @(Get-Content $path -ErrorAction SilentlyContinue)
            if ($lines.Count -gt $count.Value) {
                $lines[$count.Value..($lines.Count - 1)] | ForEach-Object {
                    Write-Host "    $_" -ForegroundColor DarkGray
                }
                $count.Value = $lines.Count
            }
        }
        while (-not $proc.HasExited) {
            if ([DateTime]::UtcNow -gt $deadline) {
                & taskkill /T /F /PID $proc.Id 2>&1 | Out-Null
                return 124
            }
            Start-Sleep -Milliseconds 750
            _Drain-NewLines $logPath ([ref]$shown)
        }
        _Drain-NewLines $logPath ([ref]$shown)
        return $proc.ExitCode
    }

    # Helper: run "npm install" in a given directory and surface the real
    # error when it fails.  Returns $true on success.
    function _Run-NpmInstall([string]$label, [string]$installDir, [string]$logPath, [string]$npmPath) {
        Push-Location $installDir
        # Capture EAP outside the try block so the catch's restore call always
        # has a meaningful value (see Install-Uv for the full rationale).
        $prevEAP = $ErrorActionPreference
        try {
            # The helper streams npm's output to BOTH the console and the log
            # file, so the user watches real progress instead of a frozen
            # "Installing..." line (on a fresh VM the install is 1-3 minutes;
            # total silence is indistinguishable from a hang) -- and the
            # wall-clock ceiling turns a genuinely stalled install (#76222
            # class) into a diagnosable failure instead of an overnight freeze.
            #
            # Relax EAP around the invocation: with EAP=Stop (set at the top
            # of this script), PowerShell can wrap stray stderr from the
            # launcher plumbing as ErrorRecord objects and throw even though
            # npm exited 0.  This is the same issue Test-Python and Install-Uv
            # work around for uv's stderr-emitting installer.  Check success
            # via the returned exit code, which is reliable regardless of
            # stderr noise.
            $ErrorActionPreference = "Continue"
            $code = _Invoke-NativeWithTimeout $npmPath "install --silent" `
                $installDir $logPath $nodeDepsTimeoutSec
            $ErrorActionPreference = $prevEAP
            if ($code -eq 0) {
                Write-Success "$label dependencies installed"
                Remove-Item -Force $logPath -ErrorAction SilentlyContinue
                return $true
            }
            if ($code -eq 124) {
                Write-Warn "$label npm install timed out after $([math]::Round($nodeDepsTimeoutSec / 60)) minutes -- a stalled download, wedged extraction, or file lock is the usual cause."
                Write-Info "  Re-run the installer to retry (completed stages are skipped)."
                Write-Info "  Slow connection? Raise the ceiling: set NODE_DEPS_TIMEOUT to seconds (default 600)."
            } else {
                Write-Warn "$label npm install failed -- exit code $code"
            }
            if (Test-Path $logPath) {
                $errText = (Get-Content $logPath -Raw -ErrorAction SilentlyContinue)
                if ($errText) {
                    $snippet = if ($errText.Length -gt 1200) { $errText.Substring(0, 1200) + "..." } else { $errText }
                    Write-Info "  npm output:"
                    foreach ($line in $snippet -split "`n") {
                        Write-Host "    $line" -ForegroundColor DarkGray
                    }
                    Write-Info "  Full log: $logPath"
                    Show-NpmCertHint $errText | Out-Null
                    Write-NpmDebugLogTail -NpmOutput $errText
                }
            }
            Write-Info "Run manually later: cd `"$installDir`"; npm install"
            return $false
        } catch {
            if ($prevEAP) { $ErrorActionPreference = $prevEAP }
            Write-Warn "$label npm install could not be launched: $_"
            return $false
        } finally {
            Pop-Location
        }
    }

    # Browser tools
    if (Test-Path "$InstallDir\package.json") {
        Write-Info "Installing Node.js dependencies (browser tools)..."
        $browserLog = "$env:TEMP\hermes-npm-browser-$(Get-Random).log"
        $browserNpmOk = _Run-NpmInstall "Browser tools" $InstallDir $browserLog $npmExe

        # Install Playwright Chromium (mirrors scripts/install.sh behaviour for
        # Linux).  Without this, tools/browser_tool.py::check_browser_requirements
        # returns False (no Chromium under %LOCALAPPDATA%\ms-playwright), and the
        # browser_* tools are silently filtered out of the agent's tool schema.
        # System Chrome at "C:\Program Files\Google\Chrome\..." is NOT used by
        # agent-browser -- it expects a Playwright-managed Chromium.
        if ($browserNpmOk) {
            Write-Info "Installing browser engine (Playwright Chromium)..."
            # npx lives next to npm in the same bin dir.  Prefer .cmd to dodge
            # the same execution-policy gotcha that affects npm.ps1 (see above).
            $npmDir = Split-Path $npmExe -Parent
            $npxExe = $null
            foreach ($cand in @("npx.cmd", "npx.exe", "npx")) {
                $try = Join-Path $npmDir $cand
                if (Test-Path $try) { $npxExe = $try; break }
            }
            if (-not $npxExe) {
                $npxCmd = Get-Command npx -ErrorAction SilentlyContinue
                if ($npxCmd) { $npxExe = $npxCmd.Source }
            }
            if (-not $npxExe) {
                Write-Warn "npx not found -- cannot install Playwright Chromium."
                Write-Info "Run manually later: cd `"$InstallDir`"; npx playwright install chromium"
            } else {
                $pwLog = "$env:TEMP\hermes-playwright-install-$(Get-Random).log"
                Push-Location $InstallDir
                # Capture EAP outside the try block so the catch's restore call
                # always has a meaningful value (see Install-Uv for the full
                # rationale).
                $prevEAP = $ErrorActionPreference
                try {
                    # Playwright Chromium is ~170MB compressed and the
                    # download regularly takes 3-10 minutes on a fresh
                    # VM.  Tee the output to console + log so the user
                    # sees download progress in real time instead of
                    # staring at a silent prompt that looks hung.  See
                    # _Run-NpmInstall above for the same pattern and
                    # the rationale behind 2>&1 before the pipe.
                    Write-Info "(this can take several minutes -- streaming progress below)"
                    # --yes auto-accepts npx's "Need to install playwright@X.Y.Z"
                    # confirmation prompt.  Without it, npx 7+ blocks on stdin
                    # waiting for a y/N answer that never comes when this is
                    # invoked through a pipeline (Tee-Object disconnects stdin
                    # from the user's TTY), and the install hangs indefinitely
                    # after printing "Need to install the following packages:
                    # playwright@X.Y.Z".
                    #
                    # Relax EAP around the playwright invocation: playwright
                    # emits a "Chromium downloaded to ..." success banner to
                    # stderr after a successful install.  The launcher merges
                    # stderr into the log natively, but keep EAP relaxed so
                    # stray plumbing stderr can't fire the catch block with a
                    # mangled banner even though the install succeeded.  Check
                    # the returned exit code instead, which is the reliable
                    # signal.
                    #
                    # The wall-clock ceiling is the #76222 / #84614 fix: the
                    # Chromium download reaches 100% and the extraction wedges
                    # (or the registry fetch stalls), and without a bound the
                    # installer sits on this line forever.  bash has carried
                    # the same 600s guard via run_playwright_install since
                    # #39219.
                    $ErrorActionPreference = "Continue"
                    $pwCode = _Invoke-NativeWithTimeout $npxExe "--yes playwright install chromium" `
                        $InstallDir $pwLog $nodeDepsTimeoutSec
                    $ErrorActionPreference = $prevEAP
                    if ($pwCode -eq 0) {
                        Write-Success "Playwright Chromium installed (browser tools ready)"
                        Remove-Item -Force $pwLog -ErrorAction SilentlyContinue
                    } elseif ($pwCode -eq 124) {
                        Write-Warn "Playwright Chromium install timed out after $([math]::Round($nodeDepsTimeoutSec / 60)) minutes."
                        Write-Warn "This usually means a stalled download or a wedged archive extraction (a locked previous browser version can also cause it)."
                        Write-Warn "Browser tools will not work until Chromium is installed."
                        if (Test-Path $pwLog) { Write-Info "  Partial log: $pwLog" }
                        Write-Info "Run manually later: cd `"$InstallDir`"; npx playwright install chromium"
                    } else {
                        Write-Warn "Playwright Chromium install failed -- exit code $pwCode"
                        Write-Warn "Browser tools will not work until Chromium is installed."
                        if (Test-Path $pwLog) {
                            $pwErr = Get-Content $pwLog -Raw -ErrorAction SilentlyContinue
                            if ($pwErr) {
                                $snippet = if ($pwErr.Length -gt 1200) { $pwErr.Substring(0, 1200) + "..." } else { $pwErr }
                                Write-Info "  playwright output:"
                                foreach ($line in $snippet -split "`n") {
                                    Write-Host "    $line" -ForegroundColor DarkGray
                                }
                                Write-Info "  Full log: $pwLog"
                            }
                        }
                        Write-Info "Run manually later: cd `"$InstallDir`"; npx playwright install chromium"
                    }
                } catch {
                    if ($prevEAP) { $ErrorActionPreference = $prevEAP }
                    Write-Warn "Playwright Chromium install could not be launched: $_"
                    Write-Info "Run manually later: cd `"$InstallDir`"; npx playwright install chromium"
                } finally {
                    Pop-Location
                }
            }
        }
    }

    # TUI
    $tuiDir = "$InstallDir\ui-tui"
    if (Test-Path "$tuiDir\package.json") {
        Write-Info "Installing TUI dependencies..."
        $tuiLog = "$env:TEMP\hermes-npm-tui-$(Get-Random).log"
        [void](_Run-NpmInstall "TUI" $tuiDir $tuiLog $npmExe)
    }

    Install-BrowserUseCli
    Install-CuaDriver
}

# The Browser Use CLI is the default browser backend when it is runnable
# (tools/browser_use_cli.py). Provision it at install time so fresh installs
# don't silently fall back to the built-in browser tools. Best-effort: any
# failure is non-fatal (browser_exec can still run via uvx, and `hermes tools`
# can install it later).
function Install-BrowserUseCli {
    if (-not $script:UvCmd) { Resolve-UvCmd }
    if (-not $script:UvCmd) {
        Write-Info "Skipping Browser Use CLI install (uv unavailable)"
        return
    }
    $managedBin = Join-Path $HermesHome "bin"
    $managedBu = Join-Path $managedBin "browser-use.exe"
    # MANAGED-FIRST: only Hermes' managed copy short-circuits. A browser-use
    # on the user's PATH is a side install -- resolution prefers the managed
    # copy, so it must be provisioned regardless.
    if (Test-Path $managedBu) {
        Write-Success "Browser Use CLI already installed"
        return
    }

    Write-Info "Installing Browser Use CLI (default browser backend)..."
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # UV_TOOL_BIN_DIR keeps the binary inside Hermes' managed bin dir,
        # where the browser tool resolves it -- no reliance on the user PATH.
        $env:UV_TOOL_BIN_DIR = $managedBin
        $env:UV_NO_CONFIG = "1"
        & $script:UvCmd tool install browser-use 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Browser Use CLI installed"
        } else {
            Write-Warn "Browser Use CLI install failed (exit $LASTEXITCODE) -- browser automation falls back to built-in tools."
            Write-Info "Install later with: uv tool install browser-use  (or via 'hermes tools')"
        }
    } catch {
        Write-Warn "Browser Use CLI install failed: $_"
    } finally {
        $ErrorActionPreference = $prevEAP
        Remove-Item Env:\UV_TOOL_BIN_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:\UV_NO_CONFIG -ErrorAction SilentlyContinue
    }
}

function Test-CuaDriverRuntimeContract {
    param([Parameter(Mandatory = $true)][string]$DriverPath)

    try {
        $versionOutput = (& $DriverPath --version 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        $versionMatch = [regex]::Match($versionOutput, '(\d+\.\d+\.\d+)')
        if (-not $versionMatch.Success) {
            return $false
        }
        if ([version]($versionMatch.Groups[1].Value) -lt [version]'0.20.0') {
            return $false
        }

        $manifestOutput = (& $DriverPath manifest 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $manifestOutput) {
            return $false
        }
        $manifest = $manifestOutput | ConvertFrom-Json
        if (-not $manifest.mcp_invocation.args) {
            return $false
        }

        $required = @{
            mcp = @('--socket', '--grant')
            serve = @(
                '--socket', '--permission-mode', '--capability-manifest',
                '--approve-capability-manifest', '--embedded'
            )
            stop = @('--socket')
        }
        foreach ($commandName in $required.Keys) {
            $command = $manifest.subcommands | Where-Object { $_.name -eq $commandName }
            if (-not $command) {
                return $false
            }
            $argNames = @($command.args | ForEach-Object { $_.name })
            foreach ($requiredArg in $required[$commandName]) {
                if ($requiredArg -notin $argNames) {
                    return $false
                }
            }
        }
        return $true
    } catch {
        return $false
    }
}

# cua-driver powers the computer_use toolset (background desktop control).
# Provision it at install time so enabling the tool later -- via `hermes
# tools`, the dashboard, or the desktop app -- is a config flip, not a
# surprise multi-minute binary fetch. Best-effort and non-fatal: the enable
# paths still lazy-install via install_cua_driver() (hermes_cli/tools_config)
# when this step was skipped or failed.
function Install-CuaDriver {
    if ($SkipComputerUse) {
        Write-Info "Skipping Computer Use (cua-driver) install (-SkipComputerUse)"
        return
    }
    $existingCuaDriver = Get-Command cua-driver -ErrorAction SilentlyContinue
    if ($existingCuaDriver) {
        if (Test-CuaDriverRuntimeContract -DriverPath $existingCuaDriver.Source) {
            Write-Success "Computer Use driver (cua-driver) already installed and compatible"
            return
        }
        Write-Warn "Existing cua-driver is old or incomplete; repairing it"
    }

    Write-Info "Installing Computer Use driver (cua-driver)..."
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        # Same upstream installer `hermes computer-use install` runs. Bounded
        # via a background job: the upstream installer serializes with its own
        # lock (600s stale window), so the ceiling sits above that -- matching
        # Hermes' _CUA_INSTALLER_TIMEOUT (660s).
        $job = Start-Job -ScriptBlock {
            Invoke-RestMethod -UseBasicParsing "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.ps1" | Invoke-Expression
        }
        if (Wait-Job $job -Timeout 660) {
            Receive-Job $job -ErrorAction SilentlyContinue | Out-Null
            Remove-Job $job -Force -ErrorAction SilentlyContinue
            $installedCuaDriver = Get-Command cua-driver -ErrorAction SilentlyContinue
            if ($installedCuaDriver -and (Test-CuaDriverRuntimeContract -DriverPath $installedCuaDriver.Source)) {
                Write-Success "Computer Use driver installed (enable via 'hermes tools' -> Computer Use)"
            } else {
                Write-Warn "Computer Use driver install did not produce a compatible runtime -- repair it before enabling the tool."
                Write-Info "Install later with: hermes computer-use install"
            }
        } else {
            Stop-Job $job -ErrorAction SilentlyContinue
            Remove-Job $job -Force -ErrorAction SilentlyContinue
            Write-Warn "Computer Use driver install timed out -- it will install on demand when you enable the tool."
            Write-Info "Install later with: hermes computer-use install"
        }
    } catch {
        Write-Warn "Computer Use driver install failed: $_"
        Write-Info "Install later with: hermes computer-use install"
    } finally {
        $ErrorActionPreference = $prevEAP
    }
}

