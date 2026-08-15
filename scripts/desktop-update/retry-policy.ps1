function Test-HermesUpdateShouldRetry {
    param(
        [int]$ExitCode,
        [string]$InstallRoot
    )

    if ($ExitCode -eq 0) { return $false }
    if ($ExitCode -ne 2) { return $true }

    # Exit 2 is shared by non-retryable safety refusals and the self-lock
    # deferral. Only the latter writes this marker, which a fresh Python
    # process consumes before importing native modules.
    $deferredInstallMarker = Join-Path $InstallRoot ".update-incomplete"
    return Test-Path -LiteralPath $deferredInstallMarker
}
