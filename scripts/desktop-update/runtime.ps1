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
            # A machine boundary that names executables under the profile.
            # The JSON is ASCII-escaped today (json.dumps default), but every
            # other consumer of this boundary (hermes_cli.windows_ssh_runtime,
            # the Electron updater, the Rust bootstrap) decodes it as UTF-8;
            # scope the capture to the same contract so the boundary stays
            # byte-exact if the emitter ever stops escaping, and so a
            # non-ASCII path cannot arrive OEM-decoded (#124526).
            $previousNativeOutputEncoding = [Console]::OutputEncoding
            try {
                [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
                $json = & $launcher --print-runtime-command --module $Module
            } finally {
                [Console]::OutputEncoding = $previousNativeOutputEncoding
            }
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
                # The "Install directory:" line is UTF-8 (hermes_bootstrap
                # reconfigures the CLI's stdio on import); PS 5.1 decodes
                # captured native stdout with the console OEM code page, so a
                # profile path like C:\Users\Balázs arrives mojibaked and the
                # GetFullPath comparison below never matches (#124526).
                $previousNativeOutputEncoding = [Console]::OutputEncoding
                try {
                    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
                    $version = (& $legacy --version 2>$null) -join "`n"
                } finally {
                    [Console]::OutputEncoding = $previousNativeOutputEncoding
                }
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
        foreach ($directory in @('venv', '.venv')) {
            $python = Join-Path $InstallRoot "$directory\Scripts\python.exe"
            if (Test-Path -LiteralPath $python -PathType Leaf) { return @($python, '-m', $Module) }
        }
    }
    throw "Installation launcher is missing under $InstallRoot\.hermes\bin. Repair this installation."
}