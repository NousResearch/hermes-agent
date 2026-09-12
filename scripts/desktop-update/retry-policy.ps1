function Test-HermesUpdateShouldRetry {
    param(
        [int]$ExitCode,
        [string]$InstallRoot
    )

    if ($ExitCode -eq 0) { return $false }
    if ($ExitCode -ne 2) { return $true }

    # Exit 2 is shared by non-retryable safety refusals and the self-lock
    # deferral. Only the latter writes this marker. The handoff treats it as a
    # retry signal for one fresh-process attempt, whose early-recovery pass
    # completes core dependencies before native modules load.
    $deferredInstallMarker = Join-Path $InstallRoot ".update-incomplete"
    return Test-Path -LiteralPath $deferredInstallMarker
}

function Test-HermesDesktopBuildFailed {
    param([string]$Output)

    # `hermes update` treats a Desktop build failure as non-fatal and prints it in
    # two shapes: "Desktop build failed" from the dependency stage and
    # "Desktop GUI build failed" from the packaged-app stage. The hand-off matched
    # only the first, so the packaged-app class -- the one a missing desktop
    # workspace produces -- never triggered the single rebuild retry (#107685).
    if ([string]::IsNullOrEmpty($Output)) { return $false }
    return $Output -match "Desktop( GUI)? build failed"
}
