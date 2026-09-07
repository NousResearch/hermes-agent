# ============================================================================
# Installation
# ============================================================================

function Install-Repository {
    Write-Info "Installing to $InstallDir..."

    $didUpdate = $false

    if (Test-Path $InstallDir) {
        # Test-Path "$InstallDir\.git" returns True when .git is a file OR a
        # directory OR a symlink OR a submodule-style gitfile -- and also when
        # it's a broken stub left over from a failed previous install (e.g.
        # a partial Remove-Item that couldn't delete a locked index.lock).
        # Validate the repo properly by asking git itself.  Three checks
        # belt-and-braces: rev-parse (work tree), git status, and a resolvable
        # HEAD (an initial commit).  If any fails the repo is broken and we
        # fall through to a fresh clone.
        $repoValid = $false
        if (Test-Path "$InstallDir\.git") {
            Push-Location $InstallDir
            try {
                # Reset $LASTEXITCODE before the probe so we don't pick up
                # a stale 0 from an earlier git call in this session.
                $global:LASTEXITCODE = 0
                $revParseOut = & git -c windows.appendAtomically=false rev-parse --is-inside-work-tree 2>&1
                $revParseOk = ($LASTEXITCODE -eq 0) -and ($revParseOut -match "true")

                $global:LASTEXITCODE = 0
                $null = & git -c windows.appendAtomically=false status --short 2>&1
                $statusOk = ($LASTEXITCODE -eq 0)

                # An interrupted previous clone leaves a repo with NO initial
                # commit. rev-parse/status still succeed there, but the update
                # path's `git stash` (and later `git checkout`) abort with
                # "You do not have the initial commit yet" and fail the install
                # (#40998). Require a resolvable HEAD so such partial checkouts
                # are treated as broken and re-cloned fresh below.
                $global:LASTEXITCODE = 0
                $null = & git -c windows.appendAtomically=false rev-parse --verify HEAD 2>&1
                $hasCommit = ($LASTEXITCODE -eq 0)

                if ($revParseOk -and $statusOk -and $hasCommit) {
                    $repoValid = $true
                }
            } catch {}
            Pop-Location
        }

        if ($repoValid) {
            Write-Info "Existing installation found, updating..."
            Push-Location $InstallDir
            # Wrap the entire fetch+checkout block in EAP=Continue so git's
            # routine stderr output (e.g. 'From <url>' info lines emitted by
            # `git fetch`) doesn't terminate the script under the global
            # EAP=Stop.  We rely on $LASTEXITCODE for actual failures.
            $prevEAP = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $autostashRef = ""
            try {
                # This is a MANAGED checkout, not a repo the user edits. Git for
                # Windows defaults to core.autocrlf=true, which renormalizes the
                # repo's LF-only text files to CRLF in the working tree -- so
                # tracked files (.envrc, AGENTS.md, agent/*.py, workflows, ...)
                # show as locally modified even though nobody touched them. A
                # bare `git checkout` then aborts with "Your local changes would
                # be overwritten by checkout", which is exactly the failure GUI
                # users hit on update. Pin autocrlf=false so the dirt is never
                # created in the first place.
                git -c windows.appendAtomically=false config core.autocrlf false 2>$null
                Discard-LockfileChurn $InstallDir
                # Preserve any real local changes before the checkout instead of
                # discarding them with `reset --hard HEAD`. The old hard reset
                # silently destroyed agent-edited source on managed clones (the
                # #38542 data-loss class). Stash + restore mirrors install.sh:
                # nothing is lost, and a failed restore leaves the work in a
                # git stash for manual recovery. Untracked files are included so
                # agent-created dirs (e.g. tinker-atropos/) survive too.
                $statusOut = git -c windows.appendAtomically=false status --porcelain 2>$null
                if (-not [string]::IsNullOrWhiteSpace(($statusOut -join "`n"))) {
                    # A previously interrupted update can leave the index with
                    # unmerged entries. In that state `git stash` aborts with
                    # "could not write index" and the following `git checkout`
                    # aborts with "you need to resolve your current index first"
                    # -- the GUI "git checkout main failed (exit 1)" install
                    # failure. Clear the conflict markers with `git reset` first:
                    # working-tree changes are kept (and stashed just below); only
                    # the index conflict state is dropped. Mirrors the `hermes
                    # update` path (#4735).
                    $unmergedOut = git -c windows.appendAtomically=false ls-files --unmerged 2>$null
                    if (-not [string]::IsNullOrWhiteSpace(($unmergedOut -join "`n"))) {
                        Write-Info "Clearing unmerged index entries from a previous conflict..."
                        git -c windows.appendAtomically=false reset -q 2>$null
                    }
                    $stashName = "hermes-install-autostash-" + (Get-Date -Format "yyyyMMdd-HHmmss")
                    Write-Info "Local changes detected, stashing before update..."
                    git -c windows.appendAtomically=false stash push --include-untracked -m "$stashName"
                    if ($LASTEXITCODE -eq 0) { $autostashRef = "stash@{0}" }
                }
                git -c windows.appendAtomically=false fetch origin $Branch
                if ($LASTEXITCODE -ne 0) { throw "git fetch failed (exit $LASTEXITCODE)" }
                # Precedence: Commit > Tag > Branch.  Commit and Tag check
                # out as detached HEAD intentionally -- they're meant to be
                # reproducible pins, not branches the user pulls into.
                if ($Commit) {
                    # Make sure we have the commit locally (a tag-less commit
                    # SHA isn't always reachable from any one branch fetch).
                    git -c windows.appendAtomically=false fetch origin $Commit
                    # A commit pin must never move an existing install
                    # BACKWARDS. hermes-setup.exe bakes its build-time commit
                    # into the binary (BUILD_PIN_COMMIT) and passes it as
                    # -Commit on every install-mode run -- including the retry
                    # the desktop's "Update didn't finish" screen kicks off. An
                    # installer built months ago would otherwise rewind a
                    # current checkout to its build commit, leaving ancient
                    # code against a current venv (npm workspaces and Python
                    # deps that no longer match: the #74xxx report). Skip the
                    # pin when the target is already an ancestor of HEAD; a
                    # fresh clone has no such ancestry and pins normally.
                    $skipRollback = $false
                    if (-not $ForceCommit) {
                        git -c windows.appendAtomically=false merge-base --is-ancestor $Commit HEAD 2>$null
                        $isAncestor = ($LASTEXITCODE -eq 0)
                        $pinnedSha = (& git -c windows.appendAtomically=false rev-parse "$Commit^{commit}" 2>$null)
                        $headSha = (& git -c windows.appendAtomically=false rev-parse HEAD 2>$null)
                        $skipRollback = $isAncestor -and ($pinnedSha -ne $headSha)
                    }
                    if ($skipRollback) {
                        Write-Warn "Ignoring -Commit $Commit`: the checkout is already newer."
                        Write-Warn "Pinning to it would roll this install back. Pass -ForceCommit to override."
                    } else {
                        git -c windows.appendAtomically=false checkout --detach $Commit
                        if ($LASTEXITCODE -ne 0) { throw "git checkout $Commit failed (exit $LASTEXITCODE)" }
                    }
                } elseif ($Tag) {
                    git -c windows.appendAtomically=false fetch origin "refs/tags/${Tag}:refs/tags/${Tag}"
                    git -c windows.appendAtomically=false checkout --detach "refs/tags/$Tag"
                    if ($LASTEXITCODE -ne 0) { throw "git checkout tag $Tag failed (exit $LASTEXITCODE)" }
                } else {
                    git -c windows.appendAtomically=false checkout $Branch
                    if ($LASTEXITCODE -ne 0) { throw "git checkout $Branch failed (exit $LASTEXITCODE)" }
                    # Managed installs should follow origin/$Branch exactly. If
                    # the checkout has diverged (or has local-only commits),
                    # ff-only pull cannot succeed -- mirror ``hermes update`` and
                    # reset to the fetched remote so bootstrap/install can recover.
                    git -c windows.appendAtomically=false pull --ff-only origin $Branch
                    if ($LASTEXITCODE -ne 0) {
                        Write-Warn "Fast-forward not possible; resetting managed install to origin/$Branch..."
                        git -c windows.appendAtomically=false reset --hard "origin/$Branch"
                        if ($LASTEXITCODE -ne 0) { throw "git reset --hard origin/$Branch failed (exit $LASTEXITCODE)" }
                    }
                }

                if ($autostashRef) {
                    # Default to restoring so work is never silently dropped.
                    # Only prompt when we're certain a human can answer: an
                    # interactive session AND a real, non-redirected console on
                    # both stdin and stdout. The desktop "Update" button and
                    # bootstrap run the installer without a usable console -- in
                    # those cases Read-Host would hang or return empty, so we
                    # skip the prompt and just restore (the safe default).
                    $restoreNow = $true
                    $hasConsole = $false
                    try {
                        $hasConsole = (
                            [Environment]::UserInteractive `
                            -and (-not [Console]::IsInputRedirected) `
                            -and (-not [Console]::IsOutputRedirected) `
                            -and ($Host.Name -eq "ConsoleHost")
                        )
                    } catch { $hasConsole = $false }
                    if ($hasConsole) {
                        Write-Warn "Local changes were stashed before updating."
                        Write-Warn "Restoring them may reapply local customizations onto the updated codebase."
                        $restoreAnswer = Read-Host "Restore local changes now? [Y/n]"
                        if ($restoreAnswer -match '^(n|no)$') { $restoreNow = $false }
                    }

                    if ($restoreNow) {
                        Write-Info "Restoring local changes..."
                        $restoreOutput = @(git -c windows.appendAtomically=false stash apply $autostashRef 2>&1)
                        $restoreExit = $LASTEXITCODE
                        $conflictedFiles = @(
                            git -c windows.appendAtomically=false diff --name-only --diff-filter=U 2>$null
                        ) | Where-Object { $_ -and $_.ToString().Trim() }
                        if (($restoreExit -eq 0) -and ($conflictedFiles.Count -eq 0)) {
                            git -c windows.appendAtomically=false stash drop $autostashRef 2>$null
                            Write-Warn "Local changes were restored on top of the updated codebase."
                            Write-Warn "Review git diff / git status if Hermes behaves unexpectedly."
                        } else {
                            Write-Err "Update pulled new code, but restoring local changes hit conflicts."
                            foreach ($line in $restoreOutput) {
                                if ($line -and $line.ToString().Trim()) {
                                    Write-Host $line
                                }
                            }
                            if ($conflictedFiles.Count -gt 0) {
                                Write-Host ""
                                Write-Host "Conflicted files:"
                                foreach ($file in $conflictedFiles) {
                                    Write-Host "  - $file"
                                }
                            }
                            Write-Host ""
                            Write-Info "Your stashed changes are preserved -- nothing is lost."
                            Write-Info "  Stash ref: $autostashRef"
                            git -c windows.appendAtomically=false reset --hard HEAD 2>$null | Out-Null
                            Write-Info "Working tree reset to clean state."
                            Write-Info "Restore your changes later with: git stash apply $autostashRef"
                        }
                    } else {
                        Write-Info "Skipped restoring local changes."
                        Write-Info "Your changes are still preserved in git stash."
                        Write-Info "Restore manually with: git stash apply $autostashRef"
                    }
                    $autostashRef = ""
                }
            } finally {
                if ($autostashRef) {
                    # We stashed but never reached the restore block (a fetch/
                    # checkout/pull failure threw). Leave the stash in place and
                    # tell the user how to recover it -- never silently drop it.
                    Write-Warn "Update did not complete. Your local changes are preserved in git stash."
                    Write-Info "Restore manually with: git stash apply $autostashRef"
                }
                $ErrorActionPreference = $prevEAP
                Pop-Location
            }
            $didUpdate = $true
        } else {
            # Directory exists but isn't a usable git repo -- e.g. an
            # interrupted clone with no initial commit (#40998), or a leftover
            # ``.git`` stub from a partial uninstall that used to lock the
            # installer into the "update" branch forever. Move it aside rather
            # than deleting it -- never destroy a directory the user might still
            # want -- and fall through to a fresh clone.
            $backupDir = "$InstallDir.broken-" + (Get-Date -Format "yyyyMMdd-HHmmss")
            Write-Warn "Existing directory at $InstallDir is not a valid git repo."
            Write-Warn "Moving it aside to $backupDir before re-cloning."
            try {
                Move-Item -LiteralPath $InstallDir -Destination $backupDir -ErrorAction Stop
            } catch {
                Write-Err "Could not move $InstallDir aside : $_"
                Write-Info "Close any programs that might be using files in $InstallDir (editors,"
                Write-Info "terminals, running hermes processes) and try again."
                throw
            }
        }
    }

    if (-not $didUpdate) {
        $cloneSuccess = $false

        # Fix Windows git "copy-fd: write returned: Invalid argument" error.
        # Git for Windows can fail on atomic file operations (hook templates,
        # config lock files) due to antivirus, OneDrive, or NTFS filter drivers.
        # The -c flag injects config before any file I/O occurs.
        Write-Info "Configuring git for Windows compatibility..."
        $env:GIT_CONFIG_COUNT = "1"
        $env:GIT_CONFIG_KEY_0 = "windows.appendAtomically"
        $env:GIT_CONFIG_VALUE_0 = "false"
        git config --global windows.appendAtomically false 2>$null

        # Try SSH first, then HTTPS, with -c flag for atomic write fix
        Write-Info "Trying SSH clone..."
        $env:GIT_SSH_COMMAND = "ssh -o BatchMode=yes -o ConnectTimeout=5"
        try {
            Invoke-NativeWithRelaxedErrorAction { git -c windows.appendAtomically=false clone --depth 1 --branch $Branch $RepoUrlSsh $InstallDir }
            if ($LASTEXITCODE -eq 0) { $cloneSuccess = $true }
        } catch { }
        $env:GIT_SSH_COMMAND = $null

        if (-not $cloneSuccess) {
            if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
            Write-Info "SSH failed, trying HTTPS..."
            try {
                Invoke-NativeWithRelaxedErrorAction { git -c windows.appendAtomically=false clone --depth 1 --branch $Branch $RepoUrlHttps $InstallDir }
                if ($LASTEXITCODE -eq 0) { $cloneSuccess = $true }
            } catch { }
        }

        # Fallback: download ZIP archive (bypasses git file I/O issues entirely)
        if (-not $cloneSuccess) {
            if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir -ErrorAction SilentlyContinue }
            Write-Warn "Git clone failed -- downloading ZIP archive instead..."
            try {
                # Pick the ZIP URL for the most-specific ref the caller asked
                # for.  GitHub supports archive URLs for commits, tags, and
                # branches; we honour Commit > Tag > Branch.
                if ($Commit) {
                    $zipUrl = "https://github.com/NousResearch/hermes-agent/archive/$Commit.zip"
                    $zipLabel = $Commit
                } elseif ($Tag) {
                    $zipUrl = "https://github.com/NousResearch/hermes-agent/archive/refs/tags/$Tag.zip"
                    $zipLabel = $Tag
                } else {
                    $zipUrl = "https://github.com/NousResearch/hermes-agent/archive/refs/heads/$Branch.zip"
                    $zipLabel = $Branch
                }
                $zipPath = "$env:TEMP\hermes-agent-$zipLabel.zip"
                $extractPath = "$env:TEMP\hermes-agent-extract"

                Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
                if (Test-Path $extractPath) { Remove-Item -Recurse -Force $extractPath }
                Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

                # GitHub ZIPs extract to repo-branch/ subdirectory
                $extractedDir = Get-ChildItem $extractPath -Directory | Select-Object -First 1
                if ($extractedDir) {
                    New-Item -ItemType Directory -Force -Path (Split-Path $InstallDir) -ErrorAction SilentlyContinue | Out-Null
                    Move-Item $extractedDir.FullName $InstallDir -Force
                    Write-Success "Downloaded and extracted"

                    # Initialize git repo so updates work later. A bare
                    # `git init` leaves NO HEAD -- desktop's write-build-stamp
                    # then hard-fails with "could not determine git commit"
                    # (#50823 / #61657). Fetch the requested ref and force-check
                    # it out (-f) so untracked ZIP files cannot block checkout.
                    Push-Location $InstallDir
                    git -c windows.appendAtomically=false init 2>$null
                    git -c windows.appendAtomically=false config windows.appendAtomically false 2>$null
                    # Pin autocrlf=false BEFORE the checkout below. Git for Windows
                    # defaults to core.autocrlf=true, which would renormalize the
                    # repo's LF text files to CRLF in the working tree during
                    # `checkout -f FETCH_HEAD` -- leaving this freshly-created
                    # managed checkout dirty vs HEAD and aborting the next
                    # `hermes update` (see the notes at the shared clone-path
                    # config below and install.ps1:1461-1469). The later pin on
                    # the shared path is idempotent and still covers git clones.
                    git -c windows.appendAtomically=false config core.autocrlf false 2>$null
                    git remote add origin $RepoUrlHttps 2>$null
                    $fetchRef = if ($Commit) { $Commit } elseif ($Tag) { "refs/tags/$Tag" } else { $Branch }
                    Write-Info "Fetching $fetchRef so the ZIP checkout has a resolvable HEAD..."
                    $prevZipEAP = $ErrorActionPreference
                    $ErrorActionPreference = "Continue"
                    try {
                        git -c windows.appendAtomically=false fetch --depth 1 origin $fetchRef 2>&1 | Out-Null
                        if ($LASTEXITCODE -eq 0) {
                            if ($Commit -or $Tag) {
                                git -c windows.appendAtomically=false checkout -f --detach FETCH_HEAD 2>&1 | Out-Null
                            } else {
                                git -c windows.appendAtomically=false checkout -f -B $Branch FETCH_HEAD 2>&1 | Out-Null
                            }
                            if ($LASTEXITCODE -eq 0) {
                                Write-Success "ZIP checkout pinned to $fetchRef"
                            } else {
                                # Checkout blocked, but FETCH_HEAD still has a SHA we can stamp with.
                                $fetchSha = & git -c windows.appendAtomically=false rev-parse FETCH_HEAD 2>$null
                                if ($LASTEXITCODE -eq 0 -and $fetchSha) {
                                    if (-not $env:GITHUB_SHA) { $env:GITHUB_SHA = ("$fetchSha").Trim() }
                                    Write-Warn "ZIP checkout failed; seeded GITHUB_SHA from FETCH_HEAD for desktop stamp"
                                } else {
                                    Write-Warn "ZIP extract succeeded but git checkout failed -- desktop build may need `$env:GITHUB_SHA"
                                }
                            }
                        } else {
                            Write-Warn "ZIP extract succeeded but git fetch of $fetchRef failed -- desktop build may need `$env:GITHUB_SHA"
                        }
                    } finally {
                        $ErrorActionPreference = $prevZipEAP
                    }
                    Pop-Location
                    Write-Success "Git repo initialized for future updates"

                    $cloneSuccess = $true
                }

                # Cleanup temp files
                Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
                Remove-Item -Recurse -Force $extractPath -ErrorAction SilentlyContinue
            } catch {
                Write-Err "ZIP download also failed: $_"
            }
        }

        if (-not $cloneSuccess) {
            throw "Failed to download repository (tried git clone SSH, HTTPS, and ZIP)"
        }
    }

    # Set per-repo config (harmless if it fails)
    Push-Location $InstallDir
    git -c windows.appendAtomically=false config windows.appendAtomically false 2>$null
    # Pin autocrlf=false on the managed clone so git never renormalizes the
    # repo's LF text files to CRLF in the working tree. Without this, the very
    # next `hermes update` checkout aborts on a "dirty" tree the user never
    # touched (see the update path above).
    git -c windows.appendAtomically=false config core.autocrlf false 2>$null

    # Post-clone pin: when a clone (or ZIP-fallback init) just landed us on
    # $Branch's tip, honour the higher-precedence $Commit / $Tag by checking
    # the exact ref out as a detached HEAD.  Skipped for the in-place update
    # path (above) since that already routed via the same precedence.
    if (-not $didUpdate) {
        # Same EAP=Continue wrap as the update path -- git fetch's 'From <url>'
        # info line goes to stderr and would terminate the script under the
        # global EAP=Stop otherwise.  We check $LASTEXITCODE for real errors.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            if ($Commit) {
                Write-Info "Pinning to commit $Commit..."
                git -c windows.appendAtomically=false fetch origin $Commit
                git -c windows.appendAtomically=false checkout --detach $Commit
                if ($LASTEXITCODE -ne 0) {
                    throw "git checkout $Commit failed (exit $LASTEXITCODE)"
                }
            } elseif ($Tag) {
                Write-Info "Pinning to tag $Tag..."
                git -c windows.appendAtomically=false fetch origin "refs/tags/${Tag}:refs/tags/${Tag}"
                git -c windows.appendAtomically=false checkout --detach "refs/tags/$Tag"
                if ($LASTEXITCODE -ne 0) {
                    throw "git checkout tag $Tag failed (exit $LASTEXITCODE)"
                }
            }
        } finally {
            $ErrorActionPreference = $prevEAP
        }
    }

    Write-Success "Repository ready"
}

