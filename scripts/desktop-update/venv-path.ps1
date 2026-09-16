function Resolve-HermesVenvDir {
    param([Parameter(Mandatory = $true)][string]$InstallRoot)

    foreach ($name in @("venv", ".venv")) {
        $candidate = Join-Path $InstallRoot $name
        $python = Join-Path $candidate "Scripts\python.exe"
        if (Test-Path -LiteralPath $python -PathType Leaf) {
            return $candidate
        }
    }

    # Preserve the installer's historical target in diagnostics and repair
    # paths when neither supported environment has a runnable interpreter.
    return (Join-Path $InstallRoot "venv")
}