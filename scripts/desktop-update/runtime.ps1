function Invoke-HermesProbe {
    param([Parameter(Mandatory = $true)][string]$Exe, [string[]]$Arguments = @())
    # Every launcher probe is bounded (60 s): a launcher that hangs (a stuck
    # import, an AV scan) fails the probe instead of parking the hand-off
    # forever after the Desktop already closed. Timeout kills the probe tree.
    # windows.ps1 -ProbeTimeoutSeconds sets the bound for tests.
    $timeout = if ($HermesProbeTimeoutSeconds -gt 0) { $HermesProbeTimeoutSeconds } else { 60 }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = (@($Arguments) | ForEach-Object { '"{0}"' -f ($_ -replace '"', '\"') }) -join ' '
    # A command-file launcher runs through cmd.exe explicitly, exactly as the
    # update step does: never hand CreateProcess a .cmd and let it pick an
    # interpreter and re-parse the arguments (BatBadBut). cmd metacharacters
    # that this quoting cannot neutralise are refused.
    if ([IO.Path]::GetExtension($Exe) -in @('.cmd', '.bat')) {
        if ($Exe -match '[%!"\x0D\x0A]' -or @($Arguments | Where-Object { $_ -match '[%!"\x0D\x0A]' }).Count) {
            throw 'The legacy command launcher cannot safely quote this probe; refresh the installation launcher first.'
        }
        $psi.Arguments = '/d /s /c ""' + $Exe + '" ' + $psi.Arguments + '"'
        $psi.FileName = if ($env:ComSpec) { $env:ComSpec } else { Join-Path $env:SystemRoot 'System32\cmd.exe' }
    }
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $process = [System.Diagnostics.Process]::Start($psi)
    try { $process.StandardInput.Close() } catch {}
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit($timeout * 1000)) {
        & taskkill.exe /T /F /PID $process.Id 2>&1 | Out-Null
        return @{ Code = 124; Output = ''; TimedOut = $true }
    }
    # A descendant may hold the pipes open; never wait on it.
    $text = if ($stdout.Wait(5000)) { $stdout.Result } else { '' }
    [void]$stderr
    return @{ Code = $process.ExitCode; Output = $text; TimedOut = $false }
}

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
            $probe = Invoke-HermesProbe $launcher @('--print-runtime-command', '--module', $Module)
            if ($probe.TimedOut) { throw "Installation launcher did not answer within the probe timeout: $launcher" }
            if ($probe.Code) { throw "Installation launcher failed (exit $($probe.Code)): $launcher" }
            $command = @(($probe.Output | ConvertFrom-Json))
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
                $probe = Invoke-HermesProbe $legacy @('--version')
                $version = $probe.Output
                if ($probe.TimedOut -or $probe.Code -or $version -notmatch '(?m)^Install directory: (.+)\r?$') { continue }
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