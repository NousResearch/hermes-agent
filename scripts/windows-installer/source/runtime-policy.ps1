# ============================================================================
# Dependency checks
# ============================================================================

# Resolve the PowerShell host executable used to spawn child PowerShell
# processes (the astral uv installer below).  We must NOT hardcode the bare
# name `powershell`: it names *Windows PowerShell* and only resolves when its
# System32 directory is on PATH.  When install.ps1 is run under PowerShell 7+
# (`pwsh`) -- or any session where `powershell` isn't on PATH -- a bare
# `powershell` spawn dies with "The term 'powershell' is not recognized",
# aborting uv installation (field report: Windows install stuck, uv install
# failed with exactly that message).  Prefer the absolute path of the host we
# are already running in (PATH-independent), then fall back to whichever of
# powershell/pwsh is resolvable, and only then to the bare name.
function Get-PowerShellHostExe {
    try {
        $hostExe = (Get-Process -Id $PID).Path
        if ($hostExe -and (Test-Path $hostExe)) {
            $leaf = Split-Path $hostExe -Leaf
            # Only trust the current host when it is a real PowerShell CLI
            # (not e.g. powershell_ise.exe or an embedded host that can't take
            # `-ExecutionPolicy`/`-Command`).
            if ($leaf -match '^(?i:powershell|pwsh)\.exe$') { return $hostExe }
        }
    } catch { }
    foreach ($candidate in @("powershell", "pwsh")) {
        $cmd = Get-Command $candidate -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($cmd -and $cmd.Source) { return $cmd.Source }
    }
    # Last-ditch: hand back the bare name so the spawn surfaces its own error.
    return "powershell"
}

function Install-Uv {
    # Hermes owns its own uv at $HermesHome\bin\uv.exe.  Always install there --
    # no PATH probing, no conda guards, no multi-location resolution chains.
    # The runtime update path (hermes_cli/managed_uv.py) looks in the same
    # place, so install.ps1 and `hermes update` stay in sync.
    $managedUv = Join-Path $HermesHome "bin\uv.exe"

    if (Test-Path $managedUv) {
        $script:UvCmd = $managedUv
        $version = & $managedUv --version
        Write-Success "Managed uv found ($version)"
        return $true
    }

    Write-Info "Installing managed uv into $HermesHome\bin ..."
    New-Item -ItemType Directory -Path (Join-Path $HermesHome "bin") -Force | Out-Null

    # UV_INSTALL_DIR tells the astral installer to place the binary
    # directly into $HermesHome\bin instead of ~/.local/bin.
    $prevEAP = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $env:UV_INSTALL_DIR = Join-Path $HermesHome "bin"
        # Spawn via the resolved host exe (see Get-PowerShellHostExe) rather
        # than a bare `powershell`, which isn't guaranteed to be on PATH under
        # PowerShell 7 / pwsh-only setups.
        $psHostExe = Get-PowerShellHostExe

        # Rungs 1 + 2: run the uv installer -- astral.sh first, then the
        # byte-identical copy published on GitHub releases.  Corporate
        # proxies and AV products frequently block astral.sh while
        # github.com is reachable (issue #69216), so a second source turns
        # a hard failure into a working install.  Capture the installer
        # output (Tee-Object) instead of discarding it: when every source
        # fails, the real error (download blocked, AV quarantine,
        # permissions) must reach the user instead of only the generic
        # "installed but not found" message.
        $installerOutput = @()
        $astralOut = @()
        & $psHostExe -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" 2>&1 | Tee-Object -Variable astralOut | Out-Null
        $installerOutput += "--- uv installer source: astral.sh ---"
        $installerOutput += @($astralOut | ForEach-Object { "$_" })
        if (Test-Path $managedUv) {
            Write-Info "uv installer succeeded via astral.sh"
        } else {
            Write-Info "astral.sh uv installer did not produce $managedUv; trying GitHub releases mirror ..."
            $ghOut = @()
            & $psHostExe -ExecutionPolicy ByPass -c "irm https://github.com/astral-sh/uv/releases/latest/download/uv-installer.ps1 | iex" 2>&1 | Tee-Object -Variable ghOut | Out-Null
            $installerOutput += "--- uv installer source: GitHub releases ---"
            $installerOutput += @($ghOut | ForEach-Object { "$_" })
            if (Test-Path $managedUv) {
                Write-Info "uv installer succeeded via GitHub releases"
            }
        }

        # Rung 3: salvage an existing uv.exe.  When the installer cannot run
        # at all (network fully blocked) but a working uv already exists --
        # on PATH, or at ~/.local/bin (the astral default location when
        # UV_INSTALL_DIR was ignored by an older installer) -- copy it into
        # the managed location so the managed-first invariant holds
        # (hermes_cli/managed_uv.py looks only at $HermesHome\bin\uv.exe).
        if (-not (Test-Path $managedUv)) {
            $existingUv = $null
            $uvOnPath = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue |
                Select-Object -First 1
            if ($uvOnPath -and $uvOnPath.Source -and (Test-Path $uvOnPath.Source)) {
                $existingUv = $uvOnPath.Source
            }
            if (-not $existingUv) {
                $defaultUv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
                if (Test-Path $defaultUv) { $existingUv = $defaultUv }
            }
            if ($existingUv) {
                Write-Info "Salvaging existing uv from $existingUv"
                try {
                    Copy-Item $existingUv $managedUv -Force
                    # Verify the salvaged binary actually runs before
                    # trusting it as the managed uv.
                    $null = & $managedUv --version
                } catch {
                    Write-Info "Existing uv at $existingUv could not be salvaged: $_"
                    Remove-Item $managedUv -Force -ErrorAction SilentlyContinue
                }
            }
        }

        $ErrorActionPreference = $prevEAP

        if (Test-Path $managedUv) {
            $script:UvCmd = $managedUv
            $version = & $managedUv --version
            Write-Success "Managed uv installed ($version)"
            return $true
        }

        Write-Err "uv installed but not found at $managedUv"
        if ($installerOutput.Count -gt 0) {
            Write-Info "uv installer output (last 15 lines):"
            $installerOutput | Select-Object -Last 15 | ForEach-Object { Write-Info "  $_" }
        }
        Write-Info "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        return $false
    } catch {
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Write-Err "Failed to install uv: $_"
        Write-Info "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        return $false
    }
}

# Refresh $env:Path from the User + Machine registry hives.  Stage drivers
# invoke each stage in a fresh powershell process, but those processes
# inherit env from the parent driver shell, NOT from the registry.  When
# an earlier stage (Stage-Git, Stage-Node, ...) installs a binary and
# pushes its directory into User PATH, the next child process's $env:Path
# is stale and the binary appears missing.  This helper re-reads PATH
# from the registry so every Invoke-Stage starts from a fresh, up-to-date
# PATH view.  Cheap (registry reads, no I/O elsewhere) and idempotent.
function Sync-EnvPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
}

# npm lifecycle scripts on Windows spawn ``cmd.exe /d /s /c node <script>``.
# PowerShell can resolve ``node`` via Get-Command while the child cmd process
# still sees a PATH without node.exe's directory (nvm4w shims, App Paths
# aliases, stale cross-process PATH).  Prepend the resolved node.exe parent
# directory so postinstall hooks (electron-winstaller, native modules, etc.)
# can find ``node``.  Regression for #48130.
function Ensure-NodeExeOnPath {
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    if (-not $nodeCmd) { return $false }

    $nodeExeDir = Split-Path $nodeCmd.Source -Parent
    if (-not $nodeExeDir) { return $false }

    $pathParts = $env:Path -split ";"
    if ($pathParts -notcontains $nodeExeDir) {
        $env:Path = "$nodeExeDir;$env:Path"
    }
    return $true
}

# Put the Hermes-managed Node dir at the FRONT of the persisted User PATH.
#
# Appending is not enough: it leaves a pre-existing system Node ahead of the
# bundled one in every new shell, so anything launched without a curated
# environment (a standalone hermes-setup.exe run, a user typing `npm`) silently
# resolves the wrong Node.  Bundled must win.
#
# Move-to-front rather than add-if-missing, because installs made by an older
# install.ps1 already have this dir in User PATH -- at the tail.  An
# add-if-missing check sees it present and leaves the broken ordering in place
# forever, so the very users the ordering bug hurt would never be repaired.
#
# Unrelated entries keep their relative order, including empty segments (a
# trailing ';' is legal and common in a real User PATH; Install-Git's splitting
# preserves them too, so this must not quietly rewrite them).  Duplicate
# occurrences of the managed dir collapse into the single leading entry.
# PowerShell's -ne is case-insensitive for strings, which is the right
# comparison on Windows.  Persists only when the resulting string differs, so
# an already-correct PATH costs one registry read and no write.
function Set-ManagedNodeFirstOnUserPath {
    param([string]$NodeDir)

    if (-not $NodeDir) { return }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $items = if ($userPath) { @($userPath -split ";") } else { @() }

    $rest = @($items | Where-Object { $_ -ne $NodeDir })
    $updated = (@($NodeDir) + $rest) -join ";"

    if ($updated -ne $userPath) {
        [Environment]::SetEnvironmentVariable("Path", $updated, "User")
    }
}

# The npm range to install into the managed Node tree.  Prefers the checkout's
# root package.json so the installer and the manifest cannot drift; falls back
# to the $NpmRange constant, which is the common case here because Test-Node
# runs before the repo is cloned.
function Get-NpmRange {
    $manifest = Join-Path $InstallDir "package.json"
    if (Test-Path $manifest) {
        try {
            $engines = (Get-Content $manifest -Raw | ConvertFrom-Json).engines
            if ($engines -and $engines.npm) { return [string]$engines.npm }
        } catch { }
    }
    return $NpmRange
}

# Convert the numeric core of an npm version or range operand into a stable
# three-component System.Version. npm reports semantic versions, but the
# installer only needs the numeric core for the comparator ranges authored in
# package.json (for example, <11.10.0 || >=11.17.0).
function ConvertTo-NpmVersion {
    param([string]$Version)

    if (-not $Version) { return $null }

    $core = ($Version.Trim() -replace '^v', '' -replace '-.*$', '')
    $parts = @($core -split '\.')
    if ($parts.Count -lt 1 -or $parts.Count -gt 3) { return $null }
    foreach ($part in $parts) {
        if ($part -notmatch '^\d+$') { return $null }
    }
    while ($parts.Count -lt 3) { $parts += '0' }

    try {
        return [version]($parts -join '.')
    } catch {
        return $null
    }
}

# Evaluate the comparator-only npm ranges used by the root manifest and the
# pre-clone fallback. Alternatives are separated with || and each alternative
# may contain one or more whitespace-separated <, <=, >, or >= comparators.
# Unknown range syntax fails closed so an incompatible system npm cannot reach
# npm ci and fail later with EBADENGINE.
function Test-NpmVersionOk {
    param(
        [string]$Version,
        [string]$Range = (Get-NpmRange)
    )

    $actual = ConvertTo-NpmVersion $Version
    if (-not $actual -or -not $Range) { return $false }

    foreach ($alternative in @($Range -split '\s*\|\|\s*')) {
        $clause = $alternative.Trim()
        if (-not $clause) { continue }

        $comparators = [regex]::Matches(
            $clause,
            '(?:^|\s)(<=|>=|<|>)\s*(\d+(?:\.\d+){0,2})(?=\s|$)'
        )
        if ($comparators.Count -eq 0) { continue }

        $remainder = [regex]::Replace(
            $clause,
            '(?:^|\s)(?:<=|>=|<|>)\s*\d+(?:\.\d+){0,2}(?=\s|$)',
            ''
        ).Trim()
        if ($remainder) { continue }

        $matchesClause = $true
        foreach ($comparator in $comparators) {
            $target = ConvertTo-NpmVersion $comparator.Groups[2].Value
            if (-not $target) {
                $matchesClause = $false
                break
            }

            $matchesComparator = switch ($comparator.Groups[1].Value) {
                '<'  { $actual -lt $target }
                '<=' { $actual -le $target }
                '>'  { $actual -gt $target }
                '>=' { $actual -ge $target }
                default { $false }
            }
            if (-not $matchesComparator) {
                $matchesClause = $false
                break
            }
        }

        if ($matchesClause) { return $true }
    }

    return $false
}

# Upgrade the Hermes-managed Node tree's bundled npm into $NpmRange when
# needed. Managed Node trees survive updates, so their bundled npm can drift
# outside a newer root package.json engine range. The repo .npmrc sets
# `engine-strict=true`, making that mismatch fatal at the first `npm ci`.
# Provision the right npm here instead of reacting to EBADENGINE later.
#
# Three details are load-bearing, mirroring _nb_ensure_bundled_npm_range in
# scripts/lib/node-bootstrap.sh and upgrade_managed_npm in
# hermes_cli/npm_engine.py:
#   - a temp cwd, so the checkout's own .npmrc (engine-strict,
#     min-release-age) does not gate the very upgrade meant to satisfy it;
#   - npm_config_min_release_age=0, which also neutralises a user ~/.npmrc;
#   - an explicit --prefix at the managed tree, so the upgrade rewrites the
#     tree's own npm rather than installing a second copy elsewhere.
#
# Best-effort: a failure leaves a working Node with an old npm, which beats no
# Node at all, and npm_engine.py still covers the EBADENGINE that follows.
function Update-ManagedNpm {
    param([string]$NodeDir)

    $npmCmd = Join-Path $NodeDir "npm.cmd"
    if (-not (Test-Path $npmCmd)) { return $false }

    $range = Get-NpmRange

    # Skip the network round-trip when the bundled npm already satisfies the
    # same range used by the system-Node acceptance gate.
    try {
        $have = (& $npmCmd --version 2>$null | Select-Object -First 1)
        if ($have -and (Test-NpmVersionOk $have $range)) { return $true }
    } catch { }

    # In-app updates run while the desktop app's Node processes are alive.
    # The managed npm lives inside the very tree they execute from, so an
    # in-place upgrade would hit WinError 5 (Access denied) on npm.cmd
    # (#80926).  Defer; the next update with the app closed retries.
    if (Test-ManagedNodeInUse $NodeDir) {
        Write-Warn "Hermes-managed Node.js is in use by a running app; skipping the bundled npm upgrade (applies on a later update with the app closed)."
        return $false
    }

    Write-Info "Upgrading bundled npm to satisfy $range ..."

    $tmpCwd = Join-Path $env:TEMP ("hermes-npm-upgrade-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmpCwd | Out-Null
    $prevAge = $env:npm_config_min_release_age
    $prevCI = $env:CI
    $prevEAP = $ErrorActionPreference
    Push-Location $tmpCwd
    try {
        $env:npm_config_min_release_age = "0"
        $env:CI = "1"
        # Relax EAP=Stop so npm's stderr lines don't get wrapped as
        # ErrorRecords and short-circuit before $LASTEXITCODE is checked.
        # Same pattern as Install-Uv.
        $ErrorActionPreference = "Continue"
        & $npmCmd install --global --prefix $NodeDir "npm@$range" `
            --no-fund --no-audit --progress=false 2>&1 | Out-Null
        $exit = $LASTEXITCODE
    } catch {
        $exit = 1
    } finally {
        $ErrorActionPreference = $prevEAP
        Pop-Location
        $env:npm_config_min_release_age = $prevAge
        $env:CI = $prevCI
        Remove-Item -Recurse -Force $tmpCwd -ErrorAction SilentlyContinue
    }

    if ($exit -ne 0) {
        Write-Warn "Could not upgrade bundled npm to $range -- ``npm ci`` may fail with EBADENGINE."
        Write-Info  "Fix manually: npm install -g --prefix `"$NodeDir`" npm@`"$range`""
        return $false
    }

    Write-Success "npm $(& $npmCmd --version 2>$null) installed"
    return $true
}

function Test-ManagedNodeInUse {
    param([string]$NodeDir)
    # Windows locks files that running processes execute from.  During an
    # in-app update the desktop app's Node processes may hold the managed
    # tree open, and rewriting it then fails with WinError 5 (Access denied)
    # on npm.cmd (#80926).  Cheap pre-check used to skip destructive steps;
    # the rename/move itself remains the authoritative guard.
    #
    # Check the executable path AND the command line: a cmd.exe wrapper
    # running npm.cmd from the tree reports its own exe (cmd.exe lives in
    # System32) while the tree path appears only in the command line.
    # Win32_Process.CommandLine is available on Windows PowerShell 5.1 and
    # 7+ (the Get-Process .CommandLine ETS property is 7.4+ only), and a
    # single CIM query beats a per-process property access loop.
    return @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                ($_.ExecutablePath -like "$NodeDir\*") -or
                ($_.CommandLine -like "*$NodeDir*")
            }
    ).Count -gt 0
}

# Re-discover uv without re-installing it.  Cross-process stage drivers
# (the desktop GUI's onboarding wizard, CI step-runners) invoke each stage
# in a fresh powershell process, so $script:UvCmd set by Install-Uv in a
# prior process is not visible here.  Later stages (Test-Python,
# Install-Venv, Install-Dependencies, Install-PlatformSdks) call this
# at the top to populate $script:UvCmd from the managed location.
# Throws if uv is not findable -- the caller's stage then surfaces a
# clean error via the stage-driver's try/catch.
function Resolve-UvCmd {
    # Already resolved (default invocation path: Install-Uv ran earlier
    # in the same process and set $script:UvCmd).
    if ($script:UvCmd) {
        if ($script:UvCmd -eq "uv") {
            # "uv" on PATH -- verify it's still resolvable (PATH could have
            # changed mid-session; cheap to recheck).
            if (Get-Command uv -ErrorAction SilentlyContinue) { return }
        } elseif (Test-Path $script:UvCmd) {
            return
        }
        # Stale; fall through to re-discover.
    }

    # Check the managed location first -- this is where Install-Uv puts it.
    $managedUv = Join-Path $HermesHome "bin\uv.exe"
    if (Test-Path $managedUv) {
        $script:UvCmd = $managedUv
        return
    }

    # Fall back to PATH (covers edge cases where the installer ran in a
    # sibling process and HERMES_HOME wasn't propagated).
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $script:UvCmd = "uv"
        return
    }

    # Refresh PATH from registry in case the current process started before
    # Install-Uv updated User PATH.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        $script:UvCmd = "uv"
        return
    }

    throw "uv is not installed. Run install.ps1 -Stage uv first."
}

function Initialize-ManagedPythonEnvironment {
    # Python used by Hermes belongs to the checkout, never to another
    # application or a user-level uv configuration. Keep this aligned with
    # hermes_cli.managed_uv.managed_python_env(), which owns the update path.
    foreach ($name in @(
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT",
        "UV_NO_MANAGED_PYTHON", "UV_PYTHON", "UV_PYTHON_DOWNLOADS",
        "UV_SYSTEM_PYTHON", "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"
    )) {
        Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue
    }

    $managedRoot = Join-Path $InstallDir ".hermes-runtime\python"
    New-Item -ItemType Directory -Force -Path $managedRoot | Out-Null
    $env:UV_MANAGED_PYTHON = "1"
    $env:UV_NO_CONFIG = "1"
    $env:UV_PYTHON_INSTALL_BIN = "0"
    $env:UV_PYTHON_INSTALL_DIR = $managedRoot
    $env:UV_PYTHON_INSTALL_REGISTRY = "0"
    return [System.IO.Path]::GetFullPath($managedRoot)
}

function Resolve-AvailablePythonVersion {
    # Return the path and minor version of the first Hermes-managed interpreter
    # uv can find, preferring the requested version and then fallback minors.
    # System and application-owned interpreters are deliberately ineligible.
    #
    # Under Hermes-Setup.exe each stage runs in a fresh powershell.exe. The
    # venv stage therefore re-resolves both version and provenance rather than
    # relying on state selected by the earlier Python stage (#50769).
    [string]$managedRoot = Initialize-ManagedPythonEnvironment
    $managedPrefix = $managedRoot.TrimEnd('\') + '\'
    $candidates = @($PythonVersion) + $PythonFallbackVersions
    $seen = @{}
    foreach ($ver in $candidates) {
        if (-not $ver -or $seen.ContainsKey($ver)) { continue }
        $seen[$ver] = $true
        $process = $null
        try {
            # PowerShell 5.1 can lose a nested native command's stdout when
            # this installer itself is redirected by the desktop bootstrapper.
            # Capture uv directly through ProcessStartInfo instead of relying
            # on the native-command pipeline for the interpreter path.
            $process = New-Object System.Diagnostics.Process
            $startInfo = New-Object System.Diagnostics.ProcessStartInfo
            $startInfo.FileName = $UvCmd
            $startInfo.Arguments = "python find $ver --managed-python --no-config"
            $startInfo.UseShellExecute = $false
            $startInfo.CreateNoWindow = $true
            $startInfo.RedirectStandardOutput = $true
            $startInfo.RedirectStandardError = $true
            $process.StartInfo = $startInfo
            if (-not $process.Start()) { continue }
            $stdoutTask = $process.StandardOutput.ReadToEndAsync()
            $stderrTask = $process.StandardError.ReadToEndAsync()
            if (-not $process.WaitForExit($PythonFindTimeoutMs)) {
                try { $process.Kill() } catch { }
                $process.WaitForExit()
                throw "uv python find $ver timed out after $PythonFindTimeoutMs ms"
            }
            $stdout = $stdoutTask.Result
            $stderrTask.Result | Out-Null
            if ($process.ExitCode -ne 0) { continue }
            [string]$foundPath = ($stdout.Trim() -split "`r?`n") | Select-Object -Last 1
            if ($foundPath) {
                $absolute = [System.IO.Path]::GetFullPath($foundPath)
                if ($absolute.StartsWith($managedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                    return [PSCustomObject]@{
                        Path = $absolute
                        Version = $ver
                    }
                }
            }
        } catch {
            throw "Failed to resolve Hermes-managed Python $ver`: $_"
        } finally {
            if ($process) { $process.Dispose() }
        }
    }
    return $null
}

function Test-Python {
    Initialize-ManagedPythonEnvironment | Out-Null
    Write-Info "Checking Python $PythonVersion..."

    # Only a checkout-private uv-managed interpreter satisfies this stage.
    try {
        $resolvedPython = Resolve-AvailablePythonVersion
        if ($resolvedPython) {
            $ver = & $resolvedPython.Path --version 2>$null
            Write-Success "Python found: $ver"
            return $true
        }
    } catch { }
    
    # Python not found -- use uv to install it (no admin needed!)
    Write-Info "Python $PythonVersion not found, installing via uv..."
    # Capture EAP outside the try block so the catch's restore call always
    # has a meaningful value (see Install-Uv for the full rationale).
    $prevEAP = $ErrorActionPreference
    try {
        # Temporarily relax ErrorActionPreference: uv writes download progress
        # ("Downloading cpython-3.11.15-windows-x86_64-none (24.5MiB)") to
        # stderr.  With $ErrorActionPreference = "Stop" (set at the top of this
        # script) PowerShell wraps stderr lines from native commands as
        # ErrorRecord objects when captured via 2>&1, then throws a terminating
        # exception on the first one -- even though uv exits 0 and Python was
        # installed successfully.  Verify success via `uv python find`
        # afterwards, which is the reliable signal regardless of exit-code
        # semantics or stderr noise.  This fix was previously landed as
        # commit ec1714e71 and then lost in a release squash; reapplied here.
        $ErrorActionPreference = "Continue"
        $uvOutput = & $UvCmd python install $PythonVersion --no-bin --no-registry --no-config 2>&1
        $uvExitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP

        # Check if Python is now available (more reliable than exit code
        # since uv may return non-zero due to "already installed" etc.)
        $resolvedPython = Resolve-AvailablePythonVersion
        if ($resolvedPython) {
            $ver = & $resolvedPython.Path --version 2>$null
            Write-Success "Python installed: $ver"
            return $true
        }

        # uv ran but Python still not findable -- show what happened
        if ($uvExitCode -ne 0) {
            Write-Warn "uv python install output:"
            Write-Host $uvOutput -ForegroundColor DarkGray
        }
    } catch {
        # Restore EAP in case the try block threw before the assignment
        if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        Write-Warn "uv python install error: $_"
    }

    # Preserve the established minor-version fallback contract, but provision
    # every fallback into the same private store instead of borrowing a system
    # interpreter. This path is reached only when the preferred install failed.
    foreach ($fallbackVer in $PythonFallbackVersions) {
        try {
            Write-Info "Trying managed Python fallback $fallbackVer..."
            $previousFallbackEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            & $UvCmd python install $fallbackVer --no-bin --no-registry --no-config 2>&1 | Out-Null
            $ErrorActionPreference = $previousFallbackEAP
            $resolvedPython = Resolve-AvailablePythonVersion
            if ($resolvedPython) {
                $ver = & $resolvedPython.Path --version 2>$null
                Write-Success "Python fallback installed: $ver"
                return $true
            }
        } catch {
            if ($previousFallbackEAP) { $ErrorActionPreference = $previousFallbackEAP }
        }
    }

    Write-Err "Failed to install Python $PythonVersion"
    Write-Info "Check network access to uv's managed Python downloads, then retry."
    return $false
}

