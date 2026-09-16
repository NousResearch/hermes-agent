function Resolve-HermesVenvDir {
    param([Parameter(Mandatory = $true)][string]$InstallRoot)

    foreach ($name in @("venv", ".venv")) {
        $candidate = Join-Path $InstallRoot $name
        if (Test-Path -LiteralPath $candidate -PathType Container) {
            return $candidate
        }
    }

    # Preserve the installer's historical target in diagnostics and repair
    # paths when neither supported environment has a runnable interpreter.
    return (Join-Path $InstallRoot "venv")
}