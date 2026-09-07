$script:GitInstallFailureReason = $null
$script:GitBashPath = $null
$script:GitBashProbeOutput = $null

function Test-GitBashCompatibility {
    <#
    .SYNOPSIS
    Verify that Git Bash can launch external MSYS programs, not just evaluate
    shell builtins. Mandatory ASLR can allow bash.exe itself to start while
    every child linked to msys-2.0.dll fails during fork/spawn.
    #>
    param([Parameter(Mandatory = $true)][string]$BashPath)

    $script:GitBashProbeOutput = $null
    if (-not (Test-Path -LiteralPath $BashPath)) {
        $script:GitBashProbeOutput = "bash.exe was not found at $BashPath"
        return $false
    }

    $process = New-Object System.Diagnostics.Process
    try {
        $startInfo = New-Object System.Diagnostics.ProcessStartInfo
        $startInfo.FileName = $BashPath
        $startInfo.Arguments = '--noprofile --norc -c "/usr/bin/true; /usr/bin/cat --version >/dev/null"'
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $process.StartInfo = $startInfo

        if (-not $process.Start()) {
            $script:GitBashProbeOutput = "bash.exe did not start"
            return $false
        }
        if (-not $process.WaitForExit(15000)) {
            try { $process.Kill() } catch { }
            $script:GitBashProbeOutput = "Git Bash compatibility probe timed out"
            return $false
        }

        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $script:GitBashProbeOutput = ("$stdout`n$stderr").Trim()
        return ($process.ExitCode -eq 0)
    } catch {
        $script:GitBashProbeOutput = $_.Exception.Message
        return $false
    } finally {
        $process.Dispose()
    }
}

function Test-MandatoryAslrEnabled {
    <# Return true only when Windows reports system-wide ForceRelocateImages=ON. #>
    try {
        $cmd = Get-Command Get-ProcessMitigation -ErrorAction SilentlyContinue
        if (-not $cmd) { return $false }
        $mitigations = & $cmd -System
        $value = $mitigations.Aslr.ForceRelocateImages
        return ($null -ne $value -and $value.ToString().ToUpperInvariant() -eq "ON")
    } catch {
        return $false
    }
}

function Get-GitRootFromBashPath {
    param([Parameter(Mandatory = $true)][string]$BashPath)

    $binDir = Split-Path -Path $BashPath -Parent
    if ((Split-Path -Path $binDir -Leaf) -ine "bin") {
        return (Split-Path -Path $binDir -Parent)
    }

    $parent = Split-Path -Path $binDir -Parent
    if ((Split-Path -Path $parent -Leaf) -ieq "usr") {
        return (Split-Path -Path $parent -Parent)
    }
    return $parent
}

function New-GitBashAslrFailureReason {
    param([Parameter(Mandatory = $true)][string]$BashPath)

    $gitRoot = Get-GitRootFromBashPath -BashPath $BashPath
    $escapedRoot = $gitRoot -replace "'", "''"
    return @(
        "Git Bash at $BashPath cannot launch required MSYS child processes because Windows Mandatory ASLR (ForceRelocateImages) is enabled system-wide. Reinstalling Git will not change this policy."
        "Open PowerShell as Administrator and run:"
        "`$gitRoot = '$escapedRoot'"
        'Get-Item "$gitRoot\bin\bash.exe", "$gitRoot\usr\bin\*.exe" -ErrorAction SilentlyContinue | ForEach-Object { Set-ProcessMitigation -Name $_.FullName -Disable ForceRelocateImages }'
        "Then rerun Hermes setup. If the override is blocked or later re-applied, ask your Windows administrator to allow this per-program exception."
    ) -join [Environment]::NewLine
}

function Install-Git {
    <#
    .SYNOPSIS
    Ensure Git (and Git Bash) are installed.  Git for Windows bundles bash.exe
    which Hermes uses to run shell commands.

    Priority order (deliberately simple -- no winget, no registry, no system
    package manager):
      1. Existing ``git`` on PATH -- use it as-is (the common fast path).
      2. Download **PortableGit** from the official git-for-windows GitHub
         release (self-extracting 7z.exe) and unpack it to
         ``%LOCALAPPDATA%\hermes\git`` -- never touches system Git, never
         requires admin, works even on locked-down machines and machines
         with a broken system Git install.

    **Why PortableGit, not MinGit:**  MinGit is the minimal-automation
    distribution and ships ONLY ``git.exe`` -- no bash, no POSIX utilities.
    Hermes needs ``bash.exe`` to run shell commands.  PortableGit is the
    full Git for Windows distribution without the installer UI; it ships
    ``git.exe`` + ``bash.exe`` + ``sh``, ``awk``, ``sed``, ``grep``, ``curl``,
    ``ssh``, etc. in ``usr\bin\``.

    We deliberately skip winget because it fails badly when the system Git
    install is in a half-installed state (partially registered, or uninstall-
    blocked).  Owning the Hermes copy of Git ourselves is predictable and
    recoverable: if it ever breaks, ``Remove-Item %LOCALAPPDATA%\hermes\git``
    and re-running this installer fully recovers.

    After install we locate ``bash.exe`` and persist the path in
    ``HERMES_GIT_BASH_PATH`` (User scope) so Hermes can find it in a fresh
    shell without a second PATH refresh.
    #>
    $script:GitInstallFailureReason = $null
    Write-Info "Checking Git..."

    if (Get-Command git -ErrorAction SilentlyContinue) {
        $version = git --version
        Write-Success "Git found ($version)"
        Set-GitBashEnvVar
        if ($script:GitBashPath -and (Test-GitBashCompatibility -BashPath $script:GitBashPath)) {
            Write-Success "Git Bash can launch MSYS programs"
            return $true
        }

        if ($script:GitBashPath -and (Test-MandatoryAslrEnabled)) {
            $script:GitInstallFailureReason = New-GitBashAslrFailureReason -BashPath $script:GitBashPath
            Write-Err $script:GitInstallFailureReason
            return $false
        }

        if ($script:GitBashPath) {
            $probeDetail = if ($script:GitBashProbeOutput) { ": $script:GitBashProbeOutput" } else { "" }
            Write-Warn "System Git Bash could not launch required MSYS programs$probeDetail"
        } else {
            Write-Warn "Git is on PATH, but its Git Bash installation could not be located."
        }
        Write-Info "Trying a Hermes-managed PortableGit install instead..."
    }

    # Download PortableGit into $HermesHome\git.  Always works as long as
    # we can reach github.com -- no admin, no winget, no reliance on the
    # user's possibly-broken system Git install.
    Write-Info "Git not found -- downloading PortableGit to $HermesHome\git\ ..."
    Write-Info "(no admin rights required; isolated from any system Git install)"

    try {
        $arch = Get-WindowsArch
        if ($arch -eq 'arm64') {
            $assetTag = 'arm64'
            $downloadIsZip = $false
        } elseif ($arch -eq 'x64') {
            $assetTag = '64-bit'
            $downloadIsZip = $false
        } else {
            # PortableGit does not ship 32-bit / arm builds -- fall back to MinGit
            # 32-bit with a warning that bash-based features will be unavailable.
            $assetTag = '32-bit-mingit'
            $downloadIsZip = $true
        }

        # Pinned git-for-windows release. We deliberately do NOT hit
        # api.github.com/repos/.../releases/latest here: that endpoint
        # is rate-limited to 60 requests/hour/IP for unauthenticated
        # callers, and users behind CGNAT / corporate NAT / dorm WiFi
        # routinely hit the limit, breaking the installer.
        # Static github.com/.../releases/download/<tag>/<asset> URLs
        # are not subject to the API rate limit.
        $gitTag    = "v2.54.0.windows.1"
        $gitVer    = "2.54.0"
        $gitVerTag = "$gitVer.windows.1"

        if ($arch -eq "32-bit-mingit") {
            Write-Warn "32-bit Windows detected -- PortableGit is 64-bit only.  Installing MinGit 32-bit as a last resort; bash-dependent Hermes features (terminal tool, agent-browser) will not work on this machine."
            $assetName    = "MinGit-$gitVer-32-bit.zip"
            $downloadIsZip = $true
        } elseif ($arch -eq "arm64") {
            $assetName    = "PortableGit-$gitVer-arm64.7z.exe"
            $downloadIsZip = $false
        } else {
            $assetName    = "PortableGit-$gitVer-64-bit.7z.exe"
            $downloadIsZip = $false
        }

        $downloadUrl = "https://github.com/git-for-windows/git/releases/download/$gitTag/$assetName"
        $downloadExt = if ($downloadIsZip) { "zip" } else { "7z.exe" }
        $tmpFile = "$env:TEMP\$assetName"
        $gitDir = "$HermesHome\git"

        Write-Info "Downloading $assetName (Git for Windows $gitVerTag)..."
        Invoke-WebRequest -Uri $downloadUrl -OutFile $tmpFile -UseBasicParsing

        if (Test-Path $gitDir) {
            Write-Info "Removing previous Git install at $gitDir ..."
            Remove-Item -Recurse -Force $gitDir
        }
        New-Item -ItemType Directory -Path $gitDir -Force | Out-Null

        if ($downloadIsZip) {
            Expand-Archive -Path $tmpFile -DestinationPath $gitDir -Force
        } else {
            # PortableGit is a self-extracting 7z archive.  Invoke it with
            # `-o<target> -y` (silent) to extract to $gitDir.  No 7z install
            # required; it's fully self-contained.
            Write-Info "Extracting PortableGit to $gitDir ..."
            $extractProc = Start-Process -FilePath $tmpFile `
                -ArgumentList "-o`"$gitDir`"", "-y" `
                -NoNewWindow -Wait -PassThru
            if ($extractProc.ExitCode -ne 0) {
                throw "PortableGit extraction failed (exit code $($extractProc.ExitCode))"
            }
        }
        Remove-Item -Force $tmpFile -ErrorAction SilentlyContinue

        # PortableGit layout: cmd\git.exe + bin\bash.exe + usr\bin\ (coreutils)
        # MinGit layout:      cmd\git.exe + usr\bin\bash.exe (if present)
        $gitExe = "$gitDir\cmd\git.exe"
        if (-not (Test-Path $gitExe)) {
            throw "Git extraction did not produce git.exe at $gitExe"
        }

        # Add to session PATH so the rest of this install run can use git.
        $env:Path = "$gitDir\cmd;$env:Path"

        # Persist to User PATH so fresh shells see it.  PortableGit needs
        # cmd\ (for git.exe), bin\ (for bash.exe + core tools), and
        # usr\bin\ (for perl, ssh, curl, and other POSIX coreutils).
        $newPathEntries = @(
            "$gitDir\cmd",
            "$gitDir\bin",
            "$gitDir\usr\bin"
        )
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        $userPathItems = if ($userPath) { $userPath -split ";" } else { @() }
        $changed = $false
        foreach ($entry in $newPathEntries) {
            if ($userPathItems -notcontains $entry) {
                $userPathItems += $entry
                $changed = $true
            }
        }
        if ($changed) {
            [Environment]::SetEnvironmentVariable("Path", ($userPathItems -join ";"), "User")
        }

        $version = & $gitExe --version
        Write-Success "Git $version installed to $gitDir (portable, user-scoped)"
        Set-GitBashEnvVar
        if (-not $script:GitBashPath) {
            throw "PortableGit extraction did not produce a usable bash.exe"
        }
        if (-not (Test-GitBashCompatibility -BashPath $script:GitBashPath)) {
            if (Test-MandatoryAslrEnabled) {
                $script:GitInstallFailureReason = New-GitBashAslrFailureReason -BashPath $script:GitBashPath
            } else {
                $probeDetail = if ($script:GitBashProbeOutput) { " Probe output: $script:GitBashProbeOutput" } else { "" }
                $script:GitInstallFailureReason = "Git Bash at $script:GitBashPath exists but cannot launch required MSYS programs.$probeDetail"
            }
            throw $script:GitInstallFailureReason
        }
        Write-Success "Git Bash can launch MSYS programs"
        return $true
    } catch {
        if ($script:GitInstallFailureReason) {
            Write-Err $script:GitInstallFailureReason
            return $false
        }
        Write-Err "Could not install portable Git: $_"
        Write-Info ""
        Write-Info "Fallback: install Git manually from https://git-scm.com/download/win"
        Write-Info "then re-run this installer.  Hermes needs Git Bash on Windows to run"
        Write-Info "shell commands (same as Claude Code and other coding agents)."
        return $false
    }
}

function Set-GitBashEnvVar {
    <#
    .SYNOPSIS
    Locate ``bash.exe`` from an already-installed Git and persist the path in
    ``HERMES_GIT_BASH_PATH`` (User env scope) so Hermes can find it even before
    PATH propagation completes in a newly-spawned shell.
    #>
    $script:GitBashPath = $null
    $candidates = @()

    # Our own portable Git install is ALWAYS checked first, so a broken
    # system Git doesn't hijack us.  If the user had a working system Git
    # we'd have returned early from Install-Git's fast path and never called
    # this with a system-Git-only installation anyway.
    #
    # Layouts:
    #   PortableGit (our default): $HermesHome\git\bin\bash.exe
    #   MinGit (32-bit fallback):  $HermesHome\git\usr\bin\bash.exe
    $candidates += "$HermesHome\git\bin\bash.exe"       # PortableGit layout (primary)
    $candidates += "$HermesHome\git\usr\bin\bash.exe"   # MinGit / PortableGit usr\bin fallback

    # git.exe on PATH can tell us where the install root is
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCmd) {
        $gitExe = $gitCmd.Source
        # Git for Windows (full installer): <root>\cmd\git.exe + <root>\bin\bash.exe
        # MinGit:                           <root>\cmd\git.exe + <root>\usr\bin\bash.exe
        $gitRoot = Split-Path (Split-Path $gitExe -Parent) -Parent
        $candidates += "$gitRoot\bin\bash.exe"
        $candidates += "$gitRoot\usr\bin\bash.exe"
    }

    # Standard system install locations as a final fallback.  Note:
    # ProgramFiles(x86) can't be referenced via ${env:...} string interpolation
    # because of the parens -- use [Environment]::GetEnvironmentVariable().
    $candidates += "${env:ProgramFiles}\Git\bin\bash.exe"
    $pf86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    if ($pf86) { $candidates += "$pf86\Git\bin\bash.exe" }
    $candidates += "${env:LocalAppData}\Programs\Git\bin\bash.exe"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            [Environment]::SetEnvironmentVariable("HERMES_GIT_BASH_PATH", $candidate, "User")
            $env:HERMES_GIT_BASH_PATH = $candidate
            $script:GitBashPath = $candidate
            Write-Info "Set HERMES_GIT_BASH_PATH=$candidate"
            return
        }
    }

    Write-Warn "Could not locate bash.exe -- Hermes may not find Git Bash."
    Write-Info "If needed, set HERMES_GIT_BASH_PATH manually to your bash.exe path."
}

# The dependency tree supports Node 22.22+, 24.11+, and 26+. nanoid 6 excludes
# Node 23 and 25 while its >=26 arm accepts later releases, and @babel/* 8.x
# requires ^22.18.0 || >=24.11.0 -- so accepting 23/25 or an early Node 24
# only defers the failure to `npm ci` under engine-strict. Keep this in sync
# with the root package.json.
function Test-NodeVersionOk {
    param([string]$Version)
    if ($Version -match '-') { return $false }
    try {
        $v = [version]($Version -replace '^v', '')
    } catch {
        return $false
    }
    if ($v.Major -eq 22) { return ($v.Minor -ge 22) }
    if ($v.Major -eq 24) { return ($v.Minor -ge 11) }
    return ($v.Major -ge 26)
}

# Accept a system Node only when its companion npm also satisfies the same
# range used to provision the Hermes-managed tree. Keeping this probe separate
# lets the initial PATH check and the post-winget check share one authority.
function Test-SystemNodeReady {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { return $false }

    $version = node --version
    if (Test-NodeVersionOk $version) {
        Ensure-NodeExeOnPath | Out-Null
    } else {
        Write-Warn "Node.js $version is unsupported (Hermes requires Node 22.22+, 24.11+, or 26+)"
        return $false
    }

    $npmRange = Get-NpmRange
    $npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    }

    $npmVersion = $null
    if ($npmCmd) {
        try {
            $npmVersion = (& $npmCmd --version 2>$null | Select-Object -First 1)
        } catch { }
    }

    if ($npmVersion -and (Test-NpmVersionOk $npmVersion $npmRange)) {
        Write-Success "Node.js $version with npm $npmVersion found"
        return $true
    }

    if ($npmVersion) {
        Write-Warn "Node.js $version uses npm $npmVersion, which does not satisfy Hermes requirement $npmRange"
    } else {
        Write-Warn "Node.js $version was found, but npm is missing or could not report its version"
    }
    return $false
}

function Test-Node {
    Write-Info "Checking Node.js (for browser tools)..."

    if (Test-SystemNodeReady) {
        $script:HasNode = $true
        return $true
    }

    Write-Info "Using a Hermes-managed Node.js installation instead..."

    # Prefer a Hermes-managed Node from a previous run over a too-old system one.
    $managedNode = "$HermesHome\node\node.exe"
    if ((Test-Path $managedNode) -and (Test-NodeVersionOk (& $managedNode --version))) {
        $version = & $managedNode --version
        $env:Path = "$HermesHome\node;$env:Path"
        Set-ManagedNodeFirstOnUserPath "$HermesHome\node"
        Write-Success "Node.js $version found (Hermes-managed)"
        # A tree from an older install still has that Node major's bundled
        # npm, which is below the current engines.npm floor. No-ops when the
        # npm is already in range, so reruns cost one --version probe.
        Update-ManagedNpm "$HermesHome\node" | Out-Null
        $script:HasNode = $true
        return $true
    }

    Write-Info "Installing Hermes-managed Node.js $NodeVersion LTS..."

    # Try the portable-zip path FIRST -- no UAC, no admin, no winget MSI.
    # winget install OpenJS.NodeJS.LTS triggers a system-wide MSI install
    # which prompts UAC (the dialog often appears minimized in the taskbar
    # and the install silently waits for consent, looking like a hang).
    # The portable zip path drops node.exe + npm into $HermesHome\node\
    # which is user-scoped and identical to how Install-Git handles
    # PortableGit.  Same UX guarantee: works on locked-down enterprise
    # machines with no admin rights.
    Write-Info "Downloading portable Node.js $NodeVersion to $HermesHome\node\ ..."
    Write-Info "(no admin rights required; isolated from any system Node install)"
    try {
        $arch = Get-WindowsArch
        $indexUrl = "https://nodejs.org/dist/latest-v${NodeVersion}.x/"
        $indexPage = Invoke-WebRequest -Uri $indexUrl -UseBasicParsing
        $zipName = ($indexPage.Content | Select-String -Pattern "node-v${NodeVersion}\.\d+\.\d+-win-${arch}\.zip" -AllMatches).Matches[0].Value

        if ($zipName) {
            $downloadUrl = "${indexUrl}${zipName}"
            $tmpZip = "$env:TEMP\$zipName"
            $tmpDir = "$env:TEMP\hermes-node-extract"

            Invoke-WebRequest -Uri $downloadUrl -OutFile $tmpZip -UseBasicParsing
            if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
            Expand-Archive -Path $tmpZip -DestinationPath $tmpDir -Force

            $extractedDir = Get-ChildItem $tmpDir -Directory | Select-Object -First 1
            if ($extractedDir) {
                # Rename-swap instead of delete-then-move: the live tree is
                # never removed before its replacement is fully extracted.
                # Windows permits renaming a tree with running executables,
                # but if a process holds it without FILE_SHARE_DELETE the
                # rename fails with WinError 5 -- that refusal means the tree
                # is in use, so defer instead of forcing the write (#80926).
                # Best-effort sweep of staging/backup litter from interrupted
                # runs; locked files simply stay for the next attempt.  Only
                # dirs older than 10 minutes are removed so a concurrent
                # heal's in-flight swap is never disturbed.
                Get-ChildItem "$HermesHome" -Directory -Filter "node.old-*" -ErrorAction SilentlyContinue |
                    Where-Object { $_.LastWriteTime -lt (Get-Date).AddMinutes(-10) } |
                    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
                Get-ChildItem "$HermesHome" -Directory -Filter "node.new-*" -ErrorAction SilentlyContinue |
                    Where-Object { $_.LastWriteTime -lt (Get-Date).AddMinutes(-10) } |
                    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
                $stamp = [Guid]::NewGuid().ToString("N")
                $staged = "$HermesHome\node.new-$stamp"
                $backup = "$HermesHome\node.old-$stamp"
                # Stage to a sibling directory so the final swap is a
                # same-volume rename (atomic), not a cross-volume Move-Item
                # (copy+delete, non-atomic -- a partial copy would leave a
                # broken tree).  Move from $env:TEMP here, rename below.
                try {
                    Move-Item $extractedDir.FullName $staged -ErrorAction Stop
                } catch {
                    Write-Warn "Failed to stage the new Node.js tree; aborting the Node upgrade."
                    Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
                    Remove-Item -Force $tmpZip -ErrorAction SilentlyContinue
                    return $false
                }
                if (Test-Path "$HermesHome\node") {
                    try {
                        Rename-Item "$HermesHome\node" $backup -ErrorAction Stop
                    } catch {
                        Write-Warn "Hermes-managed Node.js is in use by a running app; deferring its upgrade. Close the app and re-run the update."
                        Remove-Item -Recurse -Force $staged -ErrorAction SilentlyContinue
                        Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
                        Remove-Item -Force $tmpZip -ErrorAction SilentlyContinue
                        return $false
                    }
                    # A rename preserves LastWriteTime, so a backup renamed
                    # from a long-lived tree would instantly look older than
                    # the litter-sweep cutoff to a concurrent heal.  Touch it
                    # (best-effort) so the in-flight backup is never swept.
                    try {
                        (Get-Item $backup).LastWriteTime = Get-Date
                    } catch { }
                    try {
                        Rename-Item $staged "$HermesHome\node" -ErrorAction Stop
                    } catch {
                        # Restore the live tree before bailing.  The swap is a
                        # same-volume rename, so a failure leaves no partial
                        # target to clear.
                        Rename-Item $backup "$HermesHome\node" -ErrorAction SilentlyContinue
                        Remove-Item -Recurse -Force $staged -ErrorAction SilentlyContinue
                        Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
                        Remove-Item -Force $tmpZip -ErrorAction SilentlyContinue
                        return $false
                    }
                    Remove-Item -Recurse -Force $backup -ErrorAction SilentlyContinue
                } else {
                    try {
                        Rename-Item $staged "$HermesHome\node" -ErrorAction Stop
                    } catch {
                        Remove-Item -Recurse -Force $staged -ErrorAction SilentlyContinue
                        Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
                        Remove-Item -Force $tmpZip -ErrorAction SilentlyContinue
                        return $false
                    }
                }

                # Session PATH so the rest of this run sees node/npm.
                $env:Path = "$HermesHome\node;$env:Path"

                # Persist to User PATH so fresh shells (and future stages
                # in cross-process driver mode) see it.  Matches the
                # pattern Install-Git uses for PortableGit.  See
                # Set-ManagedNodeFirstOnUserPath for why this is a
                # move-to-front and not an add-if-missing.
                Set-ManagedNodeFirstOnUserPath "$HermesHome\node"

                $version = & "$HermesHome\node\node.exe" --version
                Write-Success "Node.js $version installed to $HermesHome\node\ (portable, user-scoped)"
                # The zip's bundled npm is below the repo's engines.npm floor.
                Update-ManagedNpm "$HermesHome\node" | Out-Null
                $script:HasNode = $true

                Remove-Item -Force $tmpZip -ErrorAction SilentlyContinue
                Remove-Item -Recurse -Force $tmpDir -ErrorAction SilentlyContinue
                return $true
            }
        }
    } catch {
        Write-Warn "Portable Node.js download failed: $_"
    }

    # Fallback: try winget (used to be primary, demoted because the MSI
    # install triggers a UAC prompt that frequently appears minimized in
    # the taskbar -- looks like a hang to users on stock Windows).
    # Kept for environments where the portable download fails (proxy,
    # locked firewall, etc.) but the user is willing to consent to UAC.
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Info "Falling back to winget (may prompt UAC -- check your taskbar for a flashing icon)..."
        # Capture EAP outside the try block so the catch's restore call always
        # has a meaningful value (see Install-Uv for the full rationale).
        $prevEAP = $ErrorActionPreference
        try {
            # Relax EAP=Stop so stderr lines from winget don't get wrapped
            # as ErrorRecords and short-circuit the 2>&1 pipe before we can
            # check the post-condition.  See the long comment in Install-Uv
            # for the same pattern.
            $ErrorActionPreference = "Continue"
            # On ARM64, force winget to fetch the ARM64 installer.  Without
            # the explicit override, winget on WoW64 sometimes still resolves
            # to x64 manifests, leaving us with an emulated Node toolchain
            # even after a "successful" install.  The OpenJS manifest does
            # publish an arm64 installer, so this is safe.
            $wingetArgs = @(
                'install','OpenJS.NodeJS','--silent',
                '--accept-package-agreements','--accept-source-agreements'
            )
            if ((Get-WindowsArch) -eq 'arm64') {
                $wingetArgs += @('--architecture','arm64')
            }
            winget @wingetArgs 2>&1 | Out-Null
            $ErrorActionPreference = $prevEAP
            # Refresh PATH
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + [Environment]::GetEnvironmentVariable("Path", "Machine")
            if (Test-SystemNodeReady) {
                $script:HasNode = $true
                return $true
            }
        } catch {
            if ($prevEAP) { $ErrorActionPreference = $prevEAP }
        }
    }


    Write-Info "Install manually: https://nodejs.org/en/download/"
    $script:HasNode = $false
    return $true
}

function Update-ProcessPathForPackages {
    # Make freshly-installed shims (rg.exe, ffmpeg.exe) visible to Get-Command in
    # THIS process without spawning a new shell, by folding the persisted
    # User+Machine hives plus winget's alias-shim directory into $env:Path.
    # Called after every package-manager attempt (winget/choco/scoop): previously
    # PATH was only refreshed inside the winget branch, so a successful
    # choco/scoop fallback -- or any install on a box without winget -- could be
    # misreported as "not installed".
    #
    # MERGE rather than overwrite: start from the existing process PATH so any
    # process-only entries added earlier in this installer run survive, then
    # APPEND hive/winget-Links entries not already present (case-insensitive,
    # order-preserving dedupe). A wholesale replace would silently drop those
    # process-only entries.
    $candidates = @()
    $candidates += $env:Path
    $candidates += [Environment]::GetEnvironmentVariable("Path", "User")
    $candidates += [Environment]::GetEnvironmentVariable("Path", "Machine")
    $wingetLinks = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
    if (Test-Path $wingetLinks) {
        $candidates += $wingetLinks
    }
    $seen = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
    $ordered = New-Object System.Collections.Generic.List[string]
    foreach ($chunk in $candidates) {
        if ([string]::IsNullOrEmpty($chunk)) { continue }
        foreach ($entry in $chunk.Split(';')) {
            $trimmed = $entry.Trim()
            if ($trimmed -and $seen.Add($trimmed)) {
                $ordered.Add($trimmed)
            }
        }
    }
    $env:Path = [string]::Join(';', $ordered)
}

function Install-SystemPackages {
    $script:HasRipgrep = $false
    $script:HasFfmpeg = $false
    $needRipgrep = $false
    $needFfmpeg = $false

    Write-Info "Checking ripgrep (fast file search)..."
    if (Get-Command rg -ErrorAction SilentlyContinue) {
        $version = rg --version | Select-Object -First 1
        Write-Success "$version found"
        $script:HasRipgrep = $true
    } else {
        $needRipgrep = $true
    }

    Write-Info "Checking ffmpeg (TTS voice messages)..."
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-Success "ffmpeg found"
        $script:HasFfmpeg = $true
    } else {
        $needFfmpeg = $true
    }

    if (-not $needRipgrep -and -not $needFfmpeg) { return }

    # Build description and package lists for each package manager
    $descParts = @()
    $wingetPkgs = @()
    $chocoPkgs = @()
    $scoopPkgs = @()

    if ($needRipgrep) {
        $descParts += "ripgrep for faster file search"
        $wingetPkgs += "BurntSushi.ripgrep.MSVC"
        $chocoPkgs += "ripgrep"
        $scoopPkgs += "ripgrep"
    }
    if ($needFfmpeg) {
        $descParts += "ffmpeg for TTS voice messages"
        $wingetPkgs += "Gyan.FFmpeg"
        $chocoPkgs += "ffmpeg"
        $scoopPkgs += "ffmpeg"
    }

    $description = $descParts -join " and "
    $hasWinget = Get-Command winget -ErrorAction SilentlyContinue
    $hasChoco = Get-Command choco -ErrorAction SilentlyContinue
    $hasScoop = Get-Command scoop -ErrorAction SilentlyContinue

    # Try winget first (most common on modern Windows)
    if ($hasWinget) {
        Write-Info "Installing $description via winget..."
        # Per-package log paths -- key the lookup by package id so we can
        # decide AFTER the post-install Get-Command check whether to keep
        # the log (still missing -> keep as breadcrumb) or delete it (now
        # present -> happy path, no clutter).
        $pkgLogs = @{}
        foreach ($pkg in $wingetPkgs) {
            $log = "$env:TEMP\hermes-winget-$($pkg -replace '[^A-Za-z0-9]','_')-$(Get-Random).log"
            $pkgLogs[$pkg] = $log
            # --source winget pins us to the github-backed source.  Without this,
            # a broken msstore source (cert validation failures like 0x8a15005e
            # are common on Windows-on-ARM and some corporate networks) makes
            # winget bail with "please specify --source" *before* attempting any
            # install -- and it exits 0, so the surrounding try/catch never fires.
            # We don't ship anything from msstore, so pinning is safe.
            try {
                $output = winget install --exact --id $pkg --source winget --silent `
                    --accept-package-agreements --accept-source-agreements 2>&1
                $code = $LASTEXITCODE
                $output | Out-File -FilePath $log -Encoding utf8
                "winget exit: $code" | Out-File -FilePath $log -Encoding utf8 -Append
                # 0x8A15002B (-1978335189) = APPINSTALLER_CLI_ERROR_UPDATE_NOT_APPLICABLE.
                # winget treats `install` on a package it already has registered as
                # an *upgrade*, finds no newer version, and bails with this code --
                # even when the binary is gone from disk/PATH (stale registration,
                # files removed outside winget, or a missing alias shim). We KNOW the
                # command was missing (that's why we're here), so a plain install
                # dead-ends forever. Force a reinstall to repair the registration so
                # the shim reappears.
                if ($code -eq -1978335189) {
                    "-> already-installed/no-upgrade; retrying with --force" | Out-File -FilePath $log -Encoding utf8 -Append
                    $output = winget install --exact --id $pkg --source winget --silent --force `
                        --accept-package-agreements --accept-source-agreements 2>&1
                    $output | Out-File -FilePath $log -Encoding utf8 -Append
                    "winget exit (force): $LASTEXITCODE" | Out-File -FilePath $log -Encoding utf8 -Append
                }
            } catch {
                $_ | Out-File -FilePath $log -Encoding utf8 -Append
                "winget exit: <exception>" | Out-File -FilePath $log -Encoding utf8 -Append
            }
        }
        # Refresh PATH so packages winget exposed via "command line aliases" in
        # %LOCALAPPDATA%\Microsoft\WinGet\Links (added to PATH only in
        # newly-spawned shells, not this process) are visible to Get-Command below.
        Update-ProcessPathForPackages
        if ($needRipgrep -and (Get-Command rg -ErrorAction SilentlyContinue)) {
            Write-Success "ripgrep installed"
            $script:HasRipgrep = $true
            $needRipgrep = $false
            Remove-Item -Path $pkgLogs["BurntSushi.ripgrep.MSVC"] -ErrorAction SilentlyContinue
        } elseif ($pkgLogs.ContainsKey("BurntSushi.ripgrep.MSVC")) {
            Write-Warn "winget could not install ripgrep; details: $($pkgLogs['BurntSushi.ripgrep.MSVC'])"
        }
        if ($needFfmpeg -and (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Success "ffmpeg installed"
            $script:HasFfmpeg = $true
            $needFfmpeg = $false
            Remove-Item -Path $pkgLogs["Gyan.FFmpeg"] -ErrorAction SilentlyContinue
        } elseif ($pkgLogs.ContainsKey("Gyan.FFmpeg")) {
            Write-Warn "winget could not install ffmpeg; details: $($pkgLogs['Gyan.FFmpeg'])"
        }
        if (-not $needRipgrep -and -not $needFfmpeg) { return }
    }

    # Fallback: choco
    if ($hasChoco -and ($needRipgrep -or $needFfmpeg)) {
        Write-Info "Trying Chocolatey..."
        foreach ($pkg in $chocoPkgs) {
            try { choco install $pkg -y 2>&1 | Out-Null } catch { }
        }
        Update-ProcessPathForPackages
        if ($needRipgrep -and (Get-Command rg -ErrorAction SilentlyContinue)) {
            Write-Success "ripgrep installed via chocolatey"
            $script:HasRipgrep = $true
            $needRipgrep = $false
        }
        if ($needFfmpeg -and (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Success "ffmpeg installed via chocolatey"
            $script:HasFfmpeg = $true
            $needFfmpeg = $false
        }
    }

    # Fallback: scoop
    if ($hasScoop -and ($needRipgrep -or $needFfmpeg)) {
        Write-Info "Trying Scoop..."
        foreach ($pkg in $scoopPkgs) {
            try { scoop install $pkg 2>&1 | Out-Null } catch { }
        }
        Update-ProcessPathForPackages
        if ($needRipgrep -and (Get-Command rg -ErrorAction SilentlyContinue)) {
            Write-Success "ripgrep installed via scoop"
            $script:HasRipgrep = $true
            $needRipgrep = $false
        }
        if ($needFfmpeg -and (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
            Write-Success "ffmpeg installed via scoop"
            $script:HasFfmpeg = $true
            $needFfmpeg = $false
        }
    }

    # Show manual instructions for anything still missing
    if ($needRipgrep) {
        Write-Warn "ripgrep not installed (file search will use findstr fallback)"
        Write-Info "  winget install BurntSushi.ripgrep.MSVC"
    }
    if ($needFfmpeg) {
        Write-Warn "ffmpeg not installed (TTS voice messages will be limited)"
        Write-Info "  winget install Gyan.FFmpeg"
    }
}

