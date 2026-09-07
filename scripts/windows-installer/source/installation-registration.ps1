function Install-HermesCommandLaunchers {
    param(
        [Parameter(Mandatory=$true)] [string]$Root,
        [Parameter(Mandatory=$true)] [string]$Destination
    )

    # Expose ONLY the hermes launchers on PATH -- never the whole
    # venv\Scripts directory, which contains python.exe / pip.exe and
    # silently hijacks the `python` command in every terminal (#83797).
    # Requiring hermes.exe before creating the destination keeps the PATH
    # stage from reporting success with an unusable command (PR #92092).
    $scriptsDir = Join-Path $Root "venv\Scripts"
    $requiredSource = Join-Path $scriptsDir "hermes.exe"
    if (-not (Test-Path -LiteralPath $requiredSource -PathType Leaf)) {
        throw "Cannot set up the hermes command: required launcher not found: $requiredSource"
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    # Launcher form depends on the venv (keep in lockstep with
    # hermes_cli/_install_repair.py): a normal venv's exe trampoline
    # embeds an absolute interpreter path and survives copying; a
    # relocatable venv's trampoline (managed_uv rebuilds use
    # --relocatable) resolves relative to its own location, and a copy
    # dies with 'uv trampoline failed to canonicalize script path' --
    # those get a .cmd delegator invoking the in-venv exe instead.
    $pyvenvCfg = Join-Path $Root "venv\pyvenv.cfg"
    $venvRelocatable = $false
    if (Test-Path -LiteralPath $pyvenvCfg) {
        $venvRelocatable = [bool](Select-String -Path $pyvenvCfg -Pattern '^\s*relocatable\s*=\s*true\s*$' -Quiet)
    }
    foreach ($launcher in @("hermes", "hermes-acp")) {
        $src = Join-Path $scriptsDir "$launcher.exe"
        if (-not (Test-Path -LiteralPath $src -PathType Leaf)) { continue }
        if ($venvRelocatable) {
            Remove-Item (Join-Path $Destination "$launcher.exe") -Force -ErrorAction SilentlyContinue
            Set-Content -Path (Join-Path $Destination "$launcher.cmd") -Value "@echo off`r`n`"$src`" %*" -Encoding Ascii
        } else {
            Remove-Item (Join-Path $Destination "$launcher.cmd") -Force -ErrorAction SilentlyContinue
            Copy-Item -Force -LiteralPath $src -Destination (Join-Path $Destination "$launcher.exe")
        }
    }

    # Verify either staged form before the caller mutates PATH.
    $requiredExe = Join-Path $Destination "hermes.exe"
    $requiredCmd = Join-Path $Destination "hermes.cmd"
    if (-not ((Test-Path -LiteralPath $requiredExe -PathType Leaf) -or
              (Test-Path -LiteralPath $requiredCmd -PathType Leaf))) {
        throw "Cannot set up the hermes command: launcher was not installed: $requiredExe"
    }
    return $Destination
}

function Set-PathVariable {
    Write-Info "Setting up hermes command..."
    
    if ($NoVenv) {
        $hermesBin = "$InstallDir"
    } else {
        # $HermesHome\bin is the managed binary dir (shared with the managed
        # uv), OUTSIDE the git checkout: `hermes update`'s autostash
        # (git stash push --include-untracked) deletes untracked files from
        # the working tree, which silently removed the launchers an earlier
        # installer staged under hermes-agent\bin. No git operation can ever
        # touch this dir. Staging and verification live in
        # Install-HermesCommandLaunchers, which throws BEFORE any PATH
        # mutation when the launchers cannot be staged.
        $hermesBin = "$HermesHome\bin"
        Install-HermesCommandLaunchers -Root $InstallDir -Destination $hermesBin | Out-Null
    }
    
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

    # Migrate older layouts off the user PATH:
    #   venv\Scripts     -- shadowed the user's python (#83797)
    #   hermes-agent\bin -- lived inside the git checkout, where the update
    #                       autostash could sweep the launchers off disk
    # The hermes-agent\bin FILES are left in place on purpose: editor/ACP
    # configs that captured absolute launcher paths keep working, and the
    # dir is git-ignored so it cannot dirty the checkout.
    if (-not $NoVenv) {
        $legacyEntries = @("$InstallDir\venv\Scripts", "$InstallDir\bin")
        $items = @(($currentPath -split ';') | Where-Object { $_ })
        $cleaned = @($items | Where-Object { $legacyEntries -notcontains $_ })
        if ($cleaned.Count -ne $items.Count) {
            $currentPath = $cleaned -join ";"
            [Environment]::SetEnvironmentVariable("Path", $currentPath, "User")
            Write-Info "Removed legacy launcher entries from user PATH (kept hermes via $hermesBin)"
        }
    }
    
    if ($currentPath -notlike "*$hermesBin*") {
        [Environment]::SetEnvironmentVariable(
            "Path",
            "$hermesBin;$currentPath",
            "User"
        )
        Write-Success "Added to user PATH: $hermesBin"
    } else {
        Write-Info "PATH already configured"
    }
    
    # Set HERMES_HOME so the Python code finds config/data in the right place.
    # Only needed on Windows where we install to %LOCALAPPDATA%\hermes instead
    # of the Unix default ~/.hermes
    $currentHermesHome = [Environment]::GetEnvironmentVariable("HERMES_HOME", "User")
    if (-not $currentHermesHome -or $currentHermesHome -ne $HermesHome) {
        [Environment]::SetEnvironmentVariable("HERMES_HOME", $HermesHome, "User")
        Write-Success "Set HERMES_HOME=$HermesHome"
    }
    $env:HERMES_HOME = $HermesHome
    
    # Update current session
    $env:Path = "$hermesBin;$env:Path"
    
    Write-Success "hermes command ready"
}

function Write-BootstrapMarker {
    # Writes $InstallDir\.hermes-bootstrap-complete which tells the Hermes
    # desktop app (apps/desktop/electron/main.ts) "install.ps1 ran
    # successfully -- DON'T trigger the legacy first-launch bootstrap
    # runner."
    #
    # Schema mirrors what main.ts's writeBootstrapMarker() / isBootstrap
    # Complete() expect. Keep this in lockstep when either side changes:
    #   apps/desktop/electron/main.ts lines 1199-1222
    #   BOOTSTRAP_MARKER_SCHEMA_VERSION = 1 (line 187)
    #
    # Pinned commit/branch come from -Commit + -Branch flags (passed by
    # Hermes-Setup.exe) or fall back to whatever git resolves in the
    # checkout. The desktop validates schemaVersion + pinnedCommit
    # length but doesn't enforce that HEAD matches the pin (users
    # update via `hermes update` which moves HEAD legitimately).
    if (-not (Test-Path $InstallDir)) {
        Write-Warn "Skipping bootstrap marker: $InstallDir doesn't exist"
        return
    }

    # Resolve the pinned commit: explicit -Commit wins, otherwise read
    # the checkout's HEAD via git. If git can't run, leave commit empty
    # and the marker will fail desktop validation (pinnedCommit.length
    # >= 7) -- better to be invalid than wrong.
    $pinnedCommit = $Commit
    if (-not $pinnedCommit) {
        # PS 5.1 doesn't support the ?. null-conditional operator, so
        # check Get-Command's result explicitly before reading .Source.
        $gitCmd = Get-Command git -ErrorAction SilentlyContinue
        $gitExe = if ($gitCmd) { $gitCmd.Source } else { $null }
        if ($gitExe) {
            Push-Location $InstallDir
            try {
                $resolved = & $gitExe rev-parse HEAD 2>$null
                if ($LASTEXITCODE -eq 0 -and $resolved) {
                    $pinnedCommit = $resolved.Trim()
                }
            } catch {
                # Ignore -- pinnedCommit stays empty, marker stays invalid,
                # desktop falls through to its legacy bootstrap path.
            } finally {
                Pop-Location
            }
        }
    }

    $pinnedBranch = $Branch
    if (-not $pinnedBranch) {
        $pinnedBranch = "main"  # install.ps1's own default for -Branch
    }

    $markerPath = Join-Path $InstallDir ".hermes-bootstrap-complete"
    $marker = [ordered]@{
        schemaVersion = 1
        pinnedCommit  = $pinnedCommit
        pinnedBranch  = $pinnedBranch
        completedAt   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
        # desktopVersion field intentionally omitted -- only the desktop
        # app knows its own version, and the marker validator doesn't
        # require it. The desktop fills it in if/when it writes its
        # own marker (e.g. after a future in-app upgrade).
    }
    $json = $marker | ConvertTo-Json -Compress:$false

    # Write WITHOUT a UTF-8 BOM. PowerShell 5.1's `Set-Content -Encoding UTF8`
    # always emits a BOM, and Node's plain JSON.parse rejects the BOM as an
    # unexpected character -- so a BOM'd marker would silently fail the
    # desktop's readJson(), make isBootstrapComplete() return null, and the
    # desktop would re-run the legacy bootstrap runner anyway. Defeats the
    # whole point. Use the .NET API directly for BOM-less UTF-8.
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($markerPath, $json, $utf8NoBom)

    Write-Success "Bootstrap marker written: $markerPath"
}

function Copy-ConfigTemplates {
    Write-Info "Setting up configuration files..."
    
    # Create the HERMES_HOME directory structure ($HermesHome, default %LOCALAPPDATA%\hermes)
    New-Item -ItemType Directory -Force -Path "$HermesHome\cron" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\sessions" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\logs" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\pairing" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\hooks" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\image_cache" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\audio_cache" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\memories" | Out-Null
    New-Item -ItemType Directory -Force -Path "$HermesHome\skills" | Out-Null

    
    # Create .env
    $envPath = "$HermesHome\.env"
    if (-not (Test-Path $envPath)) {
        $examplePath = "$InstallDir\.env.example"
        if (Test-Path $examplePath) {
            Copy-Item $examplePath $envPath
            Write-Success "Created $envPath from template"
        } else {
            New-Item -ItemType File -Force -Path $envPath | Out-Null
            Write-Success "Created $envPath"
        }
    } else {
        Write-Info "$envPath already exists, keeping it"
    }
    
    # Create config.yaml
    $configPath = "$HermesHome\config.yaml"
    if (-not (Test-Path $configPath)) {
        $examplePath = "$InstallDir\cli-config.yaml.example"
        if (Test-Path $examplePath) {
            Copy-Item $examplePath $configPath
            Write-Success "Created $configPath from template"
        }
    } else {
        Write-Info "$configPath already exists, keeping it"
    }
    
    # Create SOUL.md if it doesn't exist (global persona file).
    # IMPORTANT: write without a BOM.  Windows PowerShell 5.1's
    # ``Set-Content -Encoding UTF8`` writes UTF-8 WITH a byte-order-mark
    # (the default PS5 behaviour), and Hermes's prompt-injection scanner
    # flags the BOM as an invisible unicode character and refuses to
    # load the file.  PS7's ``-Encoding utf8NoBOM`` fixes that but we
    # don't control which PowerShell version the user has.  Go direct
    # to .NET with an explicit UTF8Encoding($false) -- BOM-free on every
    # PowerShell version.
    $soulPath = "$HermesHome\SOUL.md"
    if (-not (Test-Path $soulPath)) {
        # MUST match DEFAULT_SOUL_MD in hermes_cli/default_soul.py. The runtime
        # upgrades the old comment-only scaffold to this text on next run, so
        # drift is self-healing, but keep them in sync to avoid first-run churn.
        $soulContent = @"
You are Hermes Agent, built by Nous Research. Be direct: match the length of your reply to the weight of the ask -- a one-line question gets a one-line answer, and finished work gets a short report of what changed, what's verified, and what's left, never a replay of the process. No filler ("Great question," "I'd be happy to"), no restating the request back, no re-summarizing what you already said, no narrating tool calls the user can see. Plain claims over adjectives; when unsure, say so plainly. Agree because it's right, not because the user said it. Depth is earned -- give it when the user asks for detail, teaches, or the stakes demand it, not by default.
"@
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($soulPath, $soulContent, $utf8NoBom)
        Write-Success "Created $soulPath (edit to customize personality)"
    }
    
    Write-Success "Configuration directory ready: $HermesHome"
    
    # Seed bundled skills into $HermesHome\skills (manifest-based, one-time per skill)
    Write-Info "Syncing bundled skills to $HermesHome\skills ..."
    $pythonExe = "$InstallDir\venv\Scripts\python.exe"
    if (Test-Path $pythonExe) {
        try {
            # Force the child python.exe to emit UTF-8 on its stdout/stderr.
            # On non-UTF-8 Windows locales (CP936/GBK zh-CN) Python defaults
            # its stream encoding to the active codepage and crashes on glyphs
            # like the checkmark (U+2713) that the codepage can't encode; the
            # resulting non-UTF-8 bytes break this script's JSON result frame on
            # stdout and abort the config-templates stage. Scope to this call
            # only. (Comment kept ASCII per this file's PS 5.1 contract above.)
            $prevPythonioencoding = $env:PYTHONIOENCODING
            $prevPythonutf8 = $env:PYTHONUTF8
            $env:PYTHONIOENCODING = "utf-8"
            $env:PYTHONUTF8 = "1"
            try {
                & $pythonExe "$InstallDir\tools\skills_sync.py" 2>$null
            } finally {
                $env:PYTHONIOENCODING = $prevPythonioencoding
                $env:PYTHONUTF8 = $prevPythonutf8
            }
            Write-Success "Skills synced to $HermesHome\skills"
        } catch {
            # Fallback: simple directory copy
            $bundledSkills = "$InstallDir\skills"
            $userSkills = "$HermesHome\skills"
            if ((Test-Path $bundledSkills) -and -not (Get-ChildItem $userSkills -Exclude '.bundled_manifest' -ErrorAction SilentlyContinue)) {
                Copy-Item -Path "$bundledSkills\*" -Destination $userSkills -Recurse -Force -ErrorAction SilentlyContinue
                Write-Success "Skills copied to $HermesHome\skills"
            }
        }
    }
}

