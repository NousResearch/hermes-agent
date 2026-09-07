function Install-Venv {
    if ($NoVenv) {
        Write-Info "Skipping virtual environment (-NoVenv)"
        return
    }

    # Re-resolve the interpreter before creating the venv.  Under Hermes-Setup.exe
    # each stage runs in its own powershell.exe, so the fallback the `python`
    # stage picked (e.g. 3.12 when 3.11 is absent) did NOT propagate into this
    # fresh process -- $PythonVersion is back at its "3.11" default.  Trusting it
    # here made `uv venv venv --python 3.11` fail with exit 2 on machines without
    # 3.11 even though the `python` stage reported success (issue #50769).
    $resolvedPython = Resolve-AvailablePythonVersion
    if (-not $resolvedPython) {
        throw "Hermes-managed Python is unavailable. Run install.ps1 -Stage python first."
    }

    Write-Info "Creating virtual environment with Python $($resolvedPython.Version)..."
    
    Push-Location $InstallDir

    # Tasks we disabled below and must re-enable no matter how this stage
    # exits. Populated only with tasks that were ENABLED before we touched
    # them, so a task the user deliberately disabled is never re-armed.
    $gatewayTasksDisabled = @()
    $venvHadExistingVenv = $false
    $venvBackupName = $null
    $venvParked = $false
    try {
    # A previous venv stage may have finished (or been interrupted after
    # parking the old tree) without reaching dependency validation. Restore
    # that transaction's original generation before beginning another recreate
    # so a retry cannot promote the unverified replacement to rollback source.
    $pendingVenvBackup = Get-PendingVenvBackup
    if ($pendingVenvBackup) {
        Write-Warn "Reconciling unfinished venv transaction before retrying"
        Restore-VenvBackup
        if (Get-PendingVenvBackup) {
            throw "Could not restore the original venv from $pendingVenvBackup; retry aborted during transaction recovery."
        }
    }

    if (Test-Path -LiteralPath "venv") {
        $venvHadExistingVenv = $true
        Write-Info "Virtual environment already exists, recreating..."
        # On Windows, native Python extensions (e.g. _bcrypt.pyd, tornado's
        # speedups.pyd) are loaded as DLLs by any running hermes process.
        # Windows denies deletion of loaded DLLs, so every process running out
        # of this venv must be stopped before retiring it. This keeps cleanup
        # from accumulating locked stale trees and avoids carrying a live
        # gateway into the replacement venv.
        if ($env:OS -eq "Windows_NT") {
            $myPid = $PID
            Write-Info "Stopping any running hermes processes before recreating venv..."
            # Disarm the respawner FIRST: the gateway autostart Scheduled Task
            # relaunches a killed gateway within seconds, and losing that race
            # re-locks the venv's .pyd files between our kill sweep and
            # venv parking/cleanup (the July 2026 _brotlicffi.pyd incident). schtasks
            # /End stops a running task instance; /Change /DISABLE stops it
            # from re-firing mid-install. (The Startup-folder .vbs fallback is
            # NOT touched: it only fires at logon, so it cannot respawn a
            # gateway mid-install.) Re-enabled in the finally below -- including
            # on failure -- but only for tasks that were enabled to begin with.
            # Best-effort: a missing task just errors quietly.
            try {
                schtasks /Query /FO CSV 2>$null | ConvertFrom-Csv | Where-Object { $_.TaskName -like '*Hermes_Gateway*' } | ForEach-Object {
                    $tn = $_.TaskName
                    if ($_.Status -eq 'Disabled') {
                        Write-Info "  gateway autostart task $tn is already disabled; leaving it that way"
                        return
                    }
                    schtasks /End /TN $tn 2>$null | Out-Null
                    schtasks /Change /TN $tn /DISABLE 2>$null | Out-Null
                    $gatewayTasksDisabled += $tn
                    Write-Info "  disabled gateway autostart task $tn for the duration of the install"
                }
            } catch {
                Write-Warn "Could not enumerate gateway scheduled tasks: $($_.Exception.Message)"
            }
            # The launcher CLI (hermes.exe) plus its child tree.
            & taskkill /F /T /IM hermes.exe /FI "PID ne $myPid" 2>$null | Out-Null
            # taskkill /IM hermes.exe is NOT enough: the gateway/agent that a
            # scheduled task or watchdog autostarts runs as
            # `pythonw.exe -m hermes_cli.main gateway run` straight out of
            # venv\Scripts\, so its image name is python/pythonw, not hermes.exe.
            # That process holds the venv's .pyd files open and re-triggers the
            # access-denied failure. Select only roots whose executable lives
            # under this venv, then stop each root's whole process tree. Some
            # Hermes children re-exec through .hermes-runtime, so killing only
            # the selected venv process can leave its child holding the install
            # open. The path-prefix check still keeps unrelated Python processes
            # outside this venv untouched.
            #
            # The gateway autostart task registers with /RL LIMITED as the current
            # user (see hermes_cli/gateway_windows.py), so the installer always
            # runs at equal-or-higher integrity and can read its executable path.
            # Get-CimInstance is used over Get-Process because it returns a null
            # ExecutablePath for a process it cannot inspect (a different session)
            # instead of throwing, so an unreadable process is skipped rather than
            # aborting the whole sweep.
            #
            # The sweep is a bounded LOOP, not single-shot: supervised processes
            # (the Desktop app's backend, a watchdog-managed gateway) respawn in
            # the window between one kill pass and venv parking. Each pass re-
            # enumerates; three consecutive clean passes (or the attempt cap)
            # ends the loop.
            $venvPrefix = [System.IO.Path]::GetFullPath((Join-Path $InstallDir "venv")).TrimEnd('\') + '\'
            $cleanPasses = 0
            for ($sweep = 0; $sweep -lt 10 -and $cleanPasses -lt 3; $sweep++) {
                $found = 0
                try {
                    Get-CimInstance Win32_Process -ErrorAction Stop |
                        Where-Object { $_.ProcessId -ne $myPid -and $_.ExecutablePath -and $_.ExecutablePath.StartsWith($venvPrefix, [System.StringComparison]::OrdinalIgnoreCase) } |
                        ForEach-Object {
                            $found++
                            $treePid = [string]$_.ProcessId
                            Write-Info "  stopping process tree at PID $treePid ($($_.Name)) running from venv"
                            & taskkill /F /T /PID $treePid 2>$null | Out-Null
                        }
                } catch {
                    Write-Warn "Could not enumerate venv processes: $($_.Exception.Message)"
                    break
                }
                if ($found -eq 0) { $cleanPasses++ } else { $cleanPasses = 0 }
                Start-Sleep -Milliseconds 400
            }
        }
        # Move the old venv aside before creating its replacement. A directory
        # rename is atomic on the same volume and does not require deleting
        # files mapped as DLLs. NEVER fall back to deleting the live venv
        # (#83149): Remove-Item -Recurse can delete most of site-packages and
        # then fail on one locked .pyd, leaving a gutted venv with no usable
        # interpreter and no rollback source. Abort with the previous install
        # intact so the user can close holders and retry.
        $venvBackupName = "venv.stale.{0}-{1}" -f (Get-Date -Format "yyyyMMddHHmmss"), ([Guid]::NewGuid().ToString("N"))
        # Publish intent before the atomic rename. If this process is killed
        # after publication but before the rename, Get-PendingVenvBackup drops
        # the marker because its target does not exist. If it is killed after
        # the rename, the next attempt can restore this original generation.
        Set-Content -LiteralPath (Join-Path $InstallDir "venv.pending-backup") -Value $venvBackupName -Encoding ascii
        try {
            Rename-Item -LiteralPath "venv" -NewName $venvBackupName -ErrorAction Stop
            $venvParked = $true
        } catch {
            $renameErr = $_.Exception.Message
            Remove-Item -LiteralPath (Join-Path $InstallDir "venv.pending-backup") -Force -ErrorAction SilentlyContinue
            throw (
                "Could not move the existing venv aside ($renameErr). " +
                "A process still has the install directory open (often a non-Hermes " +
                "python.exe that resolved into this venv via PATH). Close those " +
                "processes and retry - the previous install was left intact."
            )
        }
    }
    
    # Pass the already-validated private interpreter path and prohibit uv from
    # resolving or downloading a different Python during venv creation. Use
    # ProcessStartInfo because the desktop bootstrapper redirects this script;
    # Windows PowerShell 5.1 can otherwise lose nested native output/exit state.
    $venvProcess = New-Object System.Diagnostics.Process
    try {
        $venvStartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $venvStartInfo.FileName = $UvCmd
        $venvStartInfo.Arguments = "venv venv --python `"$($resolvedPython.Path)`" --managed-python --no-python-downloads --no-config"
        $venvStartInfo.WorkingDirectory = $InstallDir
        $venvStartInfo.UseShellExecute = $false
        $venvStartInfo.CreateNoWindow = $true
        $venvStartInfo.RedirectStandardOutput = $true
        $venvStartInfo.RedirectStandardError = $true
        $venvProcess.StartInfo = $venvStartInfo
        if (-not $venvProcess.Start()) {
            throw "Failed to start uv while creating the virtual environment"
        }
        $venvStdoutTask = $venvProcess.StandardOutput.ReadToEndAsync()
        $venvStderrTask = $venvProcess.StandardError.ReadToEndAsync()
        $venvProcess.WaitForExit()
        $venvStdout = $venvStdoutTask.Result
        $venvStderr = $venvStderrTask.Result
        $venvExitCode = $venvProcess.ExitCode
        if ($venvStdout) { Write-Host $venvStdout.TrimEnd() }
        if ($venvStderr) { Write-Host $venvStderr.TrimEnd() }
    } finally {
        $venvProcess.Dispose()
    }
    # Fail fast so the stage cannot report ok=true when uv failed.
    if ($venvExitCode -ne 0) {
        throw "Failed to create virtual environment (uv venv exited with $venvExitCode)"
    }

    # uv can return success without leaving the interpreter expected by the
    # installer (for example after an interrupted filesystem operation). Treat
    # that as a failed transaction so the previous venv can be restored.
    $venvPythonExe = Join-Path $InstallDir "venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPythonExe -PathType Leaf)) {
        throw "uv reported success but venv interpreter is missing at $venvPythonExe"
    }

    # The replacement has a working interpreter, but the transaction is only
    # committed after Install-Dependencies' baseline-import gate passes -- the
    # bootstrap runs the stages as separate processes, and every dependency
    # tier (or the import validation) can still fail after this stage
    # succeeds. The marker was published before parking the original so an
    # interruption cannot strand the only rollback source without a pointer.
    if ($venvParked) {
        Write-Info "Previous venv parked at $venvBackupName until the dependency install is verified"
    }

    # Clean up parked venvs from previous installs whose handles have since
    # been released. Best-effort -- a still-held tree just stays for next time.
    # The backup parked THIS run is excluded: it is the rollback source until
    # Install-Dependencies commits the transaction.
    Get-ChildItem -Directory -Filter "venv.stale.*" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne $venvBackupName } | ForEach-Object {
            Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue
        }

    # Neutralize any inherited UV_PYTHON (e.g. $env:UV_PYTHON = "3.14" left in
    # the user's shell). uv honours UV_PYTHON over an existing venv for the
    # later `uv sync` / `uv pip install` tiers, so without this it would
    # silently delete this 3.11 venv and recreate it at the inherited version
    # -- building Rust transitives that have no wheel for that version from
    # source via maturin, which fails. Pinning UV_PYTHON to the interpreter we
    # just created forces every subsequent uv command onto it.
    $env:UV_PYTHON = $venvPythonExe
    } catch {
        $originalError = $_
        $rollbackError = $null

        if ($venvParked -and $venvBackupName -and (Test-Path -LiteralPath $venvBackupName)) {
            try {
                if (Test-Path -LiteralPath "venv") {
                    $failedVenvName = "venv.failed.{0}-{1}" -f (Get-Date -Format "yyyyMMddHHmmss"), ([Guid]::NewGuid().ToString("N"))
                    Rename-Item -LiteralPath "venv" -NewName $failedVenvName -ErrorAction Stop
                    Write-Warn "Failed replacement parked at $failedVenvName"
                }
                Rename-Item -LiteralPath $venvBackupName -NewName "venv" -ErrorAction Stop
                Remove-Item -LiteralPath (Join-Path $InstallDir "venv.pending-backup") -Force -ErrorAction SilentlyContinue
                Write-Warn "Restored previous virtual environment after failed recreate"
            } catch {
                $rollbackError = $_.Exception.Message
            }

            if ($rollbackError) {
                throw "Virtual environment recreate failed: $($originalError.Exception.Message). Rollback failed: $rollbackError. Previous venv remains at $venvBackupName."
            }
        } elseif (-not $venvHadExistingVenv -and (Test-Path -LiteralPath "venv")) {
            # Preserve a partial first install too. This branch must not touch a
            # pre-existing venv whose move-aside failed above.
            try {
                $failedVenvName = "venv.failed.{0}-{1}" -f (Get-Date -Format "yyyyMMddHHmmss"), ([Guid]::NewGuid().ToString("N"))
                Rename-Item -LiteralPath "venv" -NewName $failedVenvName -ErrorAction Stop
                Write-Warn "Partial virtual environment parked at $failedVenvName"
            } catch {
                $rollbackError = $_.Exception.Message
            }
            if ($rollbackError) {
                throw "Virtual environment creation failed: $($originalError.Exception.Message). Could not park partial venv: $rollbackError"
            }
        }

        throw $originalError
    } finally {
        Pop-Location
        # Re-arm the gateway autostart tasks disabled during the venv teardown
        # -- in a finally so a failed teardown/creation can never strand the
        # user's gateway autostart in the disabled state. Same function scope,
        # so the list survives even under the stage-per-process bootstrap.
        # Deliberately NOT started here -- dependencies aren't installed yet;
        # the task fires normally on next logon and `hermes update` / the
        # gateway resume path handles the immediate restart.
        if ($gatewayTasksDisabled -and $gatewayTasksDisabled.Count -gt 0) {
            foreach ($tn in $gatewayTasksDisabled) {
                schtasks /Change /TN $tn /ENABLE 2>$null | Out-Null
            }
            Write-Info "Re-enabled gateway autostart task(s): $($gatewayTasksDisabled -join ', ')"
        }
    }

    Write-Success "Virtual environment ready (Python $($resolvedPython.Version))"
}

function Get-PendingVenvBackup {
    # Rollback source recorded by Install-Venv (#83149). Returns the parked
    # directory name, or $null when there is nothing to roll back to. A marker
    # pointing at a directory that no longer exists is stale -- drop it.
    $markerPath = Join-Path $InstallDir "venv.pending-backup"
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) { return $null }
    $name = (Get-Content -LiteralPath $markerPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($name) { $name = $name.Trim() }
    if (-not $name -or -not (Test-Path -LiteralPath (Join-Path $InstallDir $name))) {
        Remove-Item -LiteralPath $markerPath -Force -ErrorAction SilentlyContinue
        return $null
    }
    return $name
}

function Complete-VenvTransaction {
    # Commit: dependency install + baseline imports passed, so the previous
    # venv is no longer needed as a rollback source. Best-effort delete; a
    # tree still held open just stays parked for the next install's sweep.
    $backupName = Get-PendingVenvBackup
    if (-not $backupName) { return }
    $backupPath = Join-Path $InstallDir $backupName
    Remove-Item -LiteralPath $backupPath -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $backupPath) {
        Write-Warn "Old venv parked at $backupName (a process still holds files in it); it will be cleaned up on the next install"
    }
    Remove-Item -LiteralPath (Join-Path $InstallDir "venv.pending-backup") -Force -ErrorAction SilentlyContinue
}

function Restore-VenvBackup {
    # Rollback: the dependency stage failed after Install-Venv replaced the
    # venv. Park the unusable replacement and restore the previous working
    # venv so Hermes (and the venv-blocker probe) stay usable (#83149).
    $backupName = Get-PendingVenvBackup
    if (-not $backupName) { return }
    try {
        if (Test-Path -LiteralPath (Join-Path $InstallDir "venv")) {
            $failedVenvName = "venv.failed.{0}-{1}" -f (Get-Date -Format "yyyyMMddHHmmss"), ([Guid]::NewGuid().ToString("N"))
            Rename-Item -LiteralPath (Join-Path $InstallDir "venv") -NewName $failedVenvName -ErrorAction Stop
            Write-Warn "Failed replacement parked at $failedVenvName"
        }
        Rename-Item -LiteralPath (Join-Path $InstallDir $backupName) -NewName "venv" -ErrorAction Stop
        Remove-Item -LiteralPath (Join-Path $InstallDir "venv.pending-backup") -Force -ErrorAction SilentlyContinue
        Write-Warn "Restored previous virtual environment after failed dependency install"
    } catch {
        Write-Warn "Could not restore previous venv (still parked at $backupName): $($_.Exception.Message)"
    }
}

function Install-Dependencies {
    Write-Info "Installing dependencies..."
    
    Push-Location $InstallDir
    
    if (-not $NoVenv) {
        # Tell uv to install into our venv (no activation needed)
        $env:VIRTUAL_ENV = "$InstallDir\venv"
    }

    # Re-pin UV_PYTHON to the venv interpreter. Install-Venv already does this,
    # but the bootstrap runs install stages (venv, python-deps) as separate
    # processes, so the env var set in Install-Venv does NOT survive into a
    # separate python-deps invocation. Re-deriving it here covers that path.
    # Without it, an inherited $env:UV_PYTHON = "3.14" makes the uv sync/pip
    # tiers below recreate the venv at 3.14 and fail the maturin source build
    # (no cp314 wheels yet).
    if (-not $NoVenv) {
        $venvPythonExe = Join-Path $InstallDir "venv\Scripts\python.exe"
        if (Test-Path $venvPythonExe) {
            $env:UV_PYTHON = $venvPythonExe
        }
    }

    # Hash-verified install (Tier 0) -- when uv.lock is present, prefer
    # `uv sync --locked`. The lockfile records SHA256 hashes for every
    # transitive dependency, so a compromised transitive (different hash
    # than what we shipped) is REJECTED by the resolver. This is the
    # *only* path that protects against the "direct dep is fine, but the
    # dep's dep got worm-poisoned overnight" failure mode. The
    # `uv pip install` tiers below re-resolve transitives fresh from PyPI
    # without any hash verification -- they exist to keep installs working
    # when the lockfile is stale, missing, or out-of-sync with the
    # current extras spec, NOT because they're equivalent in posture.
    #
    # Everything through the baseline-import gate runs inside the venv
    # transaction opened by Install-Venv (#83149): on any failure the parked
    # previous venv is restored before the error propagates, and the parked
    # tree is deleted only after the imports prove the replacement usable.
    try {
    if (Test-Path "uv.lock") {
        Write-Info "Trying tier: hash-verified (uv.lock) ..."
        # Critical flag choice: `--extra all`, NOT `--all-extras`.
        #   --all-extras = every [project.optional-dependencies] key,
        #                  bypassing the curated [all] extra. On Windows
        #                  that means [matrix] -> python-olm (no wheel,
        #                  needs `make` to build from sdist) and the
        #                  install fails.
        #   --extra all  = just the [all] extra's contents (curated).
        #
        # UV_PROJECT_ENVIRONMENT pins the sync target to our venv\.
        # Without it, modern uv (>=0.5) ignores VIRTUAL_ENV for `sync`
        # and creates a sibling .venv\ inside the repo -- leaving venv\
        # empty and producing the broken state where `hermes.exe` exists
        # in the wrong directory and imports fail with ModuleNotFoundError.
        # (Mirrors the same flag in scripts/install.sh::install_deps.)
        $env:UV_PROJECT_ENVIRONMENT = "$InstallDir\venv"
        Invoke-NativeWithRelaxedErrorAction { & $UvCmd sync --extra all --locked }
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Main package installed (hash-verified via uv.lock)"
            $script:InstalledTier = "hash-verified (uv.lock)"
            # Skip the rest of the tiered cascade -- we already have a
            # complete, hash-verified install.
            $skipPipFallback = $true
        } else {
            Write-Warn "uv.lock sync failed (lockfile may be stale), falling back to PyPI resolve..."
            $skipPipFallback = $false
        }
    } else {
        Write-Info "uv.lock not found -- falling back to PyPI resolve (no hash verification)"
        $skipPipFallback = $false
    }

    # Install main package.  Tiered fallback so a single flaky transitive
    # doesn't silently drop everything.  Each tier's stdout/stderr is
    # preserved -- no Out-Null swallowing -- so the user can see what failed.
    #
    # Tier 1: [all] -- the curated extra in pyproject.toml.
    # Tier 2: [all] minus the currently-broken extras list ($brokenExtras).
    #         Edit $brokenExtras below when something on PyPI breaks; this
    #         lets users keep the rest of [all] when one transitive is
    #         unavailable. The list of [all]'s contents is parsed from
    #         pyproject.toml at runtime -- there is NO hand-mirrored copy
    #         to drift out of sync.
    # Tier 3: bare `.` -- last-resort so at least the core CLI launches.

    # Currently-broken extras. Edit this list when an upstream package
    # gets quarantined / yanked / breaks resolution. Empty means everything
    # in [all] should be installable; populate with the names of extras
    # whose deps are temporarily unavailable.
    $brokenExtras = @()

    # Parse [project.optional-dependencies].all from pyproject.toml.
    # tomllib is stdlib on Python 3.11+ which the bootstrap guarantees.
    $pythonExeForParse = if (-not $NoVenv) { "$InstallDir\venv\Scripts\python.exe" } else { (& $UvCmd python find $PythonVersion) }
    $allExtras = @()
    if (Test-Path $pythonExeForParse) {
        $parsed = & $pythonExeForParse -c @"
import re, sys, tomllib
try:
    with open('pyproject.toml', 'rb') as fh:
        data = tomllib.load(fh)
    specs = data['project']['optional-dependencies']['all']
    out = []
    for s in specs:
        m = re.search(r'hermes-agent\[([\w-]+)\]', s)
        if m: out.append(m.group(1))
    print(','.join(out))
except Exception:
    sys.exit(1)
"@ 2>$null
        if ($LASTEXITCODE -eq 0 -and $parsed) {
            $allExtras = $parsed.Trim().Split(',')
        }
    }
    if (-not $allExtras -or $allExtras.Count -eq 0) {
        Write-Warn "Could not parse [all] from pyproject.toml; Tier 2 will be a no-op."
        $safeAll = "all"
    } else {
        $safeAll = ($allExtras | Where-Object { $brokenExtras -notcontains $_ }) -join ","
    }
    $brokenLabel = if ($brokenExtras) { ($brokenExtras -join ", ") } else { "none" }

    $installTiers = @(
        @{ Name = "all"; Spec = ".[all]" },
        @{ Name = "all minus known-broken ($brokenLabel)"; Spec = ".[$safeAll]" },
        @{ Name = "core only (no extras)"; Spec = "." }
    )
    $installed = $skipPipFallback
    if (-not $skipPipFallback) {
        foreach ($tier in $installTiers) {
        Write-Info "Trying tier: $($tier.Name) ..."
        Invoke-NativeWithRelaxedErrorAction { & $UvCmd pip install -e $tier.Spec }
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Main package installed ($($tier.Name))"
            $script:InstalledTier = $tier.Name
            $installed = $true
            break
        }
        Write-Warn "Tier '$($tier.Name)' failed (exit $LASTEXITCODE). Trying next tier..."
        }
    }
    if (-not $installed) {
        throw "Failed to install hermes-agent package even with no extras. Inspect the uv pip install output above."
    }

    # Baseline-import gate. Even if a tier reported success above, the
    # actual deps may have landed somewhere other than $InstallDir\venv\
    # (e.g. uv 0.5+ syncing into a sibling .venv\ when UV_PROJECT_ENVIRONMENT
    # isn't set, leaving venv\ empty and hermes.exe broken with
    # `ModuleNotFoundError: No module named 'dotenv'` on first run).
    # We probe via the venv's own python so a misdirected sync is caught
    # here, not 30 seconds later when the user runs `hermes`.
    if (-not $NoVenv) {
        $venvPython = "$InstallDir\venv\Scripts\python.exe"
        if (-not (Test-Path $venvPython)) {
            throw "Install reported success but $venvPython does not exist. The dependency sync likely landed in a sibling .venv\ directory. Re-run the installer; if it persists, close Hermes processes and preserve existing venv directories before retrying. Do not delete venv in place."
        }
        # Relax EAP=Stop while running the import probe.  Python writes
        # deprecation warnings and import-system info to stderr; under
        # EAP=Stop the 2>&1 merge wraps those as ErrorRecord objects and
        # throws even when the imports succeed.  $LASTEXITCODE is the
        # reliable signal (it's 0 iff the python invocation exited 0,
        # regardless of what was written to stderr).
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $venvPython -c "import dotenv, openai, rich, prompt_toolkit" 2>&1 | Out-Null
        $importExitCode = $LASTEXITCODE
        $ErrorActionPreference = $prevEAP
        if ($importExitCode -ne 0) {
            $sibling = "$InstallDir\.venv"
            $hint = if (Test-Path $sibling) {
                "Detected sibling .venv\ at $sibling -- uv synced there instead of venv\. Close Hermes processes, preserve the existing venv, and rerun the installer so the transactional recovery path can move directories safely."
            } else {
                "Recover with: cd '$InstallDir'; `$env:UV_PROJECT_ENVIRONMENT='$InstallDir\venv'; uv sync --extra all --locked"
            }
            throw "Baseline imports failed in $InstallDir\venv (dotenv/openai/rich/prompt_toolkit). The install completed but dependencies are not in the venv. $hint"
        }
        Write-Success "Baseline imports verified in venv"
    }

    # Commit the venv transaction: the dependency install completed and the
    # baseline imports passed, so the previous venv is no longer needed as a
    # rollback source (#83149).
    Complete-VenvTransaction
    } catch {
        # Dependency install or import validation failed: restore the previous
        # working venv (parked by Install-Venv) before surfacing the error, so
        # a failed update leaves Hermes and its blocker probe usable.
        Restore-VenvBackup
        Pop-Location
        throw
    }

    if (-not $NoVenv) {
        # uv on Windows can register hermes.exe in dist-info/RECORD but fail to
        # materialise the .exe (file lock during self-update, distlib edge case).
        # Catch it here so a fresh install/update does not finish with a broken
        # `hermes` command while hermes-agent.exe / hermes-acp.exe exist
        $scriptsDir = Join-Path $InstallDir "venv\Scripts"
        $pythonExe = Join-Path $scriptsDir "python.exe"
        if ((Test-Path $scriptsDir) -and (Test-Path $pythonExe)) {
            $scriptNames = & $pythonExe -c @"
import tomllib
with open('pyproject.toml', 'rb') as fh:
    scripts = tomllib.load(fh).get('project', {}).get('scripts', {}) or {}
print(','.join(scripts))
"@ 2>$null
            if ($LASTEXITCODE -eq 0 -and $scriptNames) {
                $expected = @($scriptNames.Trim().Split(',') | Where-Object { $_ })
                $missing = @()
                foreach ($name in $expected) {
                    $exe = Join-Path $scriptsDir "$name.exe"
                    if (-not (Test-Path $exe)) { $missing += "$name.exe" }
                }
                if ($missing.Count -gt 0) {
                    Write-Warn "Console entry point(s) missing: $($missing -join ', ')"
                    Write-Info "Reinstalling entry points..."
                    $env:UV_PROJECT_ENVIRONMENT = "$InstallDir\venv"
                    Invoke-NativeWithRelaxedErrorAction { & $UvCmd pip install --reinstall -e . }
                    $stillMissing = @()
                    foreach ($name in $expected) {
                        $exe = Join-Path $scriptsDir "$name.exe"
                        if (-not (Test-Path $exe)) { $stillMissing += "$name.exe" }
                    }
                    if ($stillMissing.Count -gt 0) {
                        Write-Warn "Entry points still missing after repair: $($stillMissing -join ', ')"
                        Write-Info "Workaround: `"$pythonExe`" -m hermes_cli.main <command>"
                    } else {
                        Write-Success "Console entry points restored"
                    }
                }
            }
        }
    }

    # Verify the dashboard deps specifically -- they're the most common thing
    # users hit and lazy-import errors from `hermes dashboard` are confusing.
    # If tier 1 failed (the common case), [web] was still picked up by tiers
    # 2-3; only tier 4 leaves you without it.
    $pythonExe = if (-not $NoVenv) { "$InstallDir\venv\Scripts\python.exe" } else { (& $UvCmd python find $PythonVersion) }
    if (Test-Path $pythonExe) {
        $webOk = $false
        $webServerSyntaxOk = $false
        # Relax EAP=Stop while running the import probe; see the matching
        # comment on the baseline-imports check above.  Python writes
        # deprecation warnings to stderr and we don't want those wrapped
        # as ErrorRecords that silently force the "not importable" path
        # even when fastapi/uvicorn are actually installed.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $pythonExe -c "import fastapi, uvicorn" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { $webOk = $true }
        } catch { }
        try {
            & $pythonExe -m py_compile "$InstallDir\hermes_cli\web_server.py" 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { $webServerSyntaxOk = $true }
        } catch { }
        $ErrorActionPreference = $prevEAP
        if (-not $webOk) {
            Write-Warn "fastapi/uvicorn not importable -- `hermes dashboard` will not work."
            Write-Info "Attempting targeted install of [web] extra as last resort..."
            & $UvCmd pip install -e ".[web]"
            if ($LASTEXITCODE -eq 0) {
                Write-Success "[web] extra installed; `hermes dashboard` should now work."
            } else {
                Write-Warn "Could not install [web] extra. Run manually: uv pip install --python `"$pythonExe`" `"fastapi>=0.104,<1`" `"uvicorn[standard]>=0.24,<1`""
            }
        }
        if (-not $webServerSyntaxOk) {
            throw "dashboard backend source failed syntax check: hermes_cli/web_server.py"
        }
    }
    
    Pop-Location
    
    Write-Success "All dependencies installed"
}

