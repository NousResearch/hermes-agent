function Get-HermesRuntimeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$InstallRoot,
        [string]$Module = 'hermes_cli.main'
    )

    # The launcher owns interpreter/ABI and generation selection. Never infer
    # PM's store layout or borrow a different installation's PATH command.
    foreach ($name in @('hermes.exe', 'hermes.cmd')) {
        $launcher = Join-Path $InstallRoot ".hermes\bin\$name"
        if (Test-Path -LiteralPath $launcher -PathType Leaf) {
            $json = & $launcher --print-runtime-command --module $Module
            if ($LASTEXITCODE) { throw "Installation launcher failed (exit $LASTEXITCODE): $launcher" }
            $command = @((($json -join "`n") | ConvertFrom-Json))
            if ($command.Count -lt 2 -or @($command | Where-Object { $_ -isnot [string] -or -not $_ }).Count) {
                throw "Installation launcher returned invalid command: $launcher"
            }
            return $command
        }
    }

    # Earlier PM installers published only to user-bin. The established
    # --version surface reports the bound source root; never trust PATH alone.
    if (Test-Path -LiteralPath (Join-Path $InstallRoot 'hermes_cli/_launchers.py') -PathType Leaf) {
        $directories = @(
            (Join-Path $env:HERMES_HOME 'bin'),
            (Join-Path (Split-Path -Parent $InstallRoot) 'bin')
        )
        if ($env:LOCALAPPDATA) { $directories += Join-Path $env:LOCALAPPDATA 'hermes/bin' }
        foreach ($directory in ($directories | Select-Object -Unique)) {
            foreach ($name in @('hermes.exe', 'hermes.cmd')) {
                $legacy = Join-Path $directory $name
                if (-not (Test-Path -LiteralPath $legacy -PathType Leaf)) { continue }
                $version = (& $legacy --version 2>$null) -join "`n"
                if ($LASTEXITCODE -or $version -notmatch '(?m)^Install directory: (.+)\r?$') { continue }
                $reported = [IO.Path]::GetFullPath($Matches[1].Trim()).TrimEnd('\', '/')
                $expected = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\', '/')
                if (-not [string]::Equals($reported, $expected, [StringComparison]::OrdinalIgnoreCase)) { continue }
                if ($Module -ne 'hermes_cli.main') {
                    throw "This older installation needs its launcher refreshed before running $Module. Run the update through $legacy."
                }
                return @($legacy)
            }
        }
    }

    # Only an older, pre-PM checkout may use the historical interpreter.
    if (-not (Test-Path -LiteralPath (Join-Path $InstallRoot 'pm') -PathType Container)) {
        $python = Join-Path $InstallRoot 'venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $python -PathType Leaf) { return @($python, '-m', $Module) }
    }
    throw "Installation launcher is missing under $InstallRoot\.hermes\bin. Repair this installation."
}