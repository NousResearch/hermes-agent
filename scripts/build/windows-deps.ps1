# Build-only prerequisites; setup-hermes.ps1 uses the same installer functions.
param(
    [Parameter(Mandatory = $true)][string]$StateRoot,
    [string]$OpenSSLRoot,
    [string]$EnvironmentFile,
    [string]$GithubEnv,
    [string]$GithubPath
)
$ErrorActionPreference = 'Stop'

$before = @{}
Get-ChildItem Env: | ForEach-Object { $before[$_.Name] = $_.Value }
. (Join-Path $PSScriptRoot '..\windows-build-deps.ps1')
Initialize-HermesArm64BuildTools -StateRoot $StateRoot -OpenSSLRoot $OpenSSLRoot

$after = @{}
Get-ChildItem Env: | ForEach-Object { $after[$_.Name] = $_.Value }
$utf8 = New-Object System.Text.UTF8Encoding($false)
if ($EnvironmentFile) {
    # The caller owns this temporary file: it includes the inherited environment.
    [IO.File]::WriteAllText($EnvironmentFile, ($after | ConvertTo-Json -Compress), $utf8)
}
if ($GithubEnv) {
    foreach ($name in ($after.Keys | Sort-Object)) {
        # GitHub rejects writes to its own variables (and NODE_OPTIONS).
        if ($name -match '^(GITHUB_|RUNNER_)' -or $name -eq 'NODE_OPTIONS') { continue }
        if ($name -eq 'PATH' -and $GithubPath) { continue }
        # Rust homes must stay explicit when a later child isolates HOME.
        if ($name -notin @('CARGO_HOME', 'RUSTUP_HOME') -and
            $before.ContainsKey($name) -and $before[$name] -ceq $after[$name]) { continue }
        $delimiter = 'hermes_' + [Guid]::NewGuid().ToString('N')
        [IO.File]::AppendAllText($GithubEnv, "$name<<$delimiter`n$($after[$name])`n$delimiter`n", $utf8)
    }
}
if ($GithubPath) {
    # The runner prepends each line. Reverse only new entries so their compiler
    # search order survives, without replacing the caller's inherited PATH.
    $inherited = @($before['PATH'] -split ';')
    $added = @($after['PATH'] -split ';' | Where-Object { $_ -and $_ -notin $inherited } | Select-Object -Unique)
    [array]::Reverse($added)
    foreach ($entry in $added) {
        [IO.File]::AppendAllText($GithubPath, "$entry`n", $utf8)
    }
}
