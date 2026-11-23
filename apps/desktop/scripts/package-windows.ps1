param([string]$Destination = "")
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    if (!$Destination) { $Destination = Join-Path $PWD "dist" }
    $stage = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
    $root = Join-Path $stage "Hermes"
    $null = New-Item -ItemType Directory -Path "$root/Notices" -Force
    $null = New-Item -ItemType Directory -Path $Destination -Force
    cargo build --locked --release -p hermes-desktop -p hermes-gateway
    if ($LASTEXITCODE -ne 0) { throw "Cargo build failed ($LASTEXITCODE)" }
    $meta = cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "Cargo metadata failed" }
    Copy-Item "$($meta.target_directory)/release/hermes-desktop.exe" "$root/Hermes.exe"
    Copy-Item "$($meta.target_directory)/release/hermes-gateway.exe" $root
    Copy-Item "assets/icon.ico" "$root/Hermes.ico"
    Copy-Item "../../LICENSE" "$root/Notices/Hermes-LICENSE"
    Get-ChildItem "crates/desktop/assets/fonts" -File | Where-Object { $_.Name -match '(NOTICE|LICENSE).*\.txt$' } | Copy-Item -Destination "$root/Notices"
    Compress-Archive -Path $root -DestinationPath (Join-Path $Destination "Hermes-windows-$env:PROCESSOR_ARCHITECTURE.zip") -Force
} finally {
    if ($stage -and (Test-Path $stage)) { Remove-Item -LiteralPath $stage -Recurse -Force }
    Pop-Location
}
