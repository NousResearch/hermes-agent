# Native package replacement acceptance. Never run on a developer desktop.
param(
    [Parameter(Mandatory=$true)][string]$ManifestUrl,
    [ValidateSet('x64','arm64')][string]$Arch = 'x64'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:GITHUB_ACTIONS -ne 'true' -or $env:OS -ne 'Windows_NT') { throw 'Disposable Windows Actions runner required' }
$Repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$Assets = Join-Path $PSScriptRoot 'e2e-assets'
$Work = Join-Path $env:RUNNER_TEMP 'hermes-bundled-update'
if (Test-Path $Work) { throw "Refusing to reuse $Work" }
$Proof = Join-Path $Work 'proof'
New-Item -ItemType Directory -Path $Proof -Force | Out-Null
$env:HERMES_HOME = Join-Path $Work 'home'
New-Item -ItemType Directory -Path $env:HERMES_HOME | Out-Null
$env:HERMES_DESKTOP_USER_DATA = Join-Path $Work 'electron-user-data'
$env:HERMES_DESKTOP_FEED_BASE_URL = ''
$Node = (Get-Command node.exe).Source
function Run-Node([string[]]$Argv) {
    & $Node @Argv
    if ($LASTEXITCODE -ne 0) { throw "Node failed: $($Argv[0])" }
}
Run-Node @((Join-Path $Assets 'bundle-inputs.mjs'), '--manifest-url', $ManifestUrl, '--platform', 'windows', '--arch', $Arch, '--out', $Work)
$ManifestPath = Join-Path $Work 'bundle-inputs.json'
$m = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
Run-Node @((Join-Path $Assets 'windows-bundled-helpers.mjs'), 'validate-manifest', '--manifest', $ManifestPath, '--arch', $Arch)
if (Get-AppxPackage -Name $m.old.identity) { throw 'Package identity is already installed; refusing to modify it' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
function Assert-Bundle($Side) {
    $artifact = $Side.artifact.path
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $artifact).Hash.ToLowerInvariant() -cne $Side.artifact.sha256) { throw 'Bundle hash mismatch' }
    $signature = Get-AuthenticodeSignature -LiteralPath $artifact
    if ($signature.Status -ne 'Valid') { throw "Bundle signature invalid: $($signature.Status)" }
    $zip = [IO.Compression.ZipFile]::OpenRead($artifact)
    try {
        $entry = $zip.GetEntry('AppxMetadata/AppxBundleManifest.xml')
        if (-not $entry) { throw 'Not an MSIX bundle' }
        $reader = New-Object IO.StreamReader($entry.Open())
        try { [xml]$xml = $reader.ReadToEnd() } finally { $reader.Dispose() }
        if ($xml.Bundle.Identity.Name -cne $Side.identity -or $xml.Bundle.Identity.Publisher -cne $Side.publisher -or $xml.Bundle.Identity.Version -cne $Side.version) { throw 'Bundle identity disagrees with pinned manifest' }
        if (-not @($xml.Bundle.Packages.Package | Where-Object { $_.Architecture -eq $Arch }).Count) { throw 'Requested architecture absent from bundle' }
    } finally { $zip.Dispose() }
}
Assert-Bundle $m.old
Assert-Bundle $m.new

function Installed($Side) {
    $packages = @(Get-AppxPackage -Name $Side.identity)
    if ($packages.Count -ne 1) { throw 'Expected exactly one installed package' }
    $pkg = $packages[0]
    if ($pkg.Version.ToString() -cne $Side.version -or $pkg.Publisher -cne $Side.publisher -or $pkg.Architecture.ToString() -ine $Arch) { throw 'Installed package identity/version/architecture mismatch' }
    $stampPath = Join-Path $pkg.InstallLocation 'app\resources\install-stamp.json'
    $stamp = Get-Content -Raw -LiteralPath $stampPath | ConvertFrom-Json
    if ($stamp.commit -cne $Side.commit -or $stamp.tag -cne $Side.tag -or $stamp.payload -cne 'bundled') { throw 'Installed payload provenance mismatch' }
    $xml = Get-AppxPackageManifest -Package $pkg.PackageFullName
    $application = @($xml.Package.Applications.Application | Where-Object { $_.Id -ceq $Side.applicationId })
    if ($application.Count -ne 1) { throw 'Installed applicationId missing' }
    $exe = Join-Path $pkg.InstallLocation $application[0].Executable
    if (-not (Test-Path -LiteralPath $exe)) { throw 'Installed executable missing' }
    return @{ package=$pkg; exe=$exe; stamp=$stamp }
}
function Main-Processes([string]$Exe) {
    return @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -ieq $Exe })
}
$Feed = Join-Path $Work 'feed'
New-Item -ItemType Directory -Path $Feed | Out-Null
Copy-Item -LiteralPath $m.old.artifact.path -Destination (Join-Path $Feed 'old.msixbundle')
Copy-Item -LiteralPath $m.new.artifact.path -Destination (Join-Path $Feed 'new.msixbundle')
$PortFile = Join-Path $Work 'feed-url'
$Helper = Join-Path $Assets 'windows-bundled-helpers.mjs'
$server = Start-Process -FilePath $Node -ArgumentList @('"'+$Helper+'"', 'serve', '--feed', '"'+$Feed+'"', '--port-file', '"'+$PortFile+'"') -PassThru -RedirectStandardOutput (Join-Path $Proof 'feed.log') -RedirectStandardError (Join-Path $Proof 'feed-error.log')
$installed = $false
try {
    $deadline = (Get-Date).AddSeconds(20)
    while (-not (Test-Path $PortFile)) {
        if ($server.HasExited -or (Get-Date) -gt $deadline) { throw 'Feed failed to start' }
        Start-Sleep -Milliseconds 200
    }
    $baseUrl = (Get-Content -Raw $PortFile).Trim()
    function Descriptor($Side, [string]$File) {
        Run-Node @($Helper, 'descriptor', '--feed', $Feed, '--base-url', $baseUrl, '--identity', $Side.identity, '--version', $Side.version, '--bundle', $File, '--descriptor-filename', 'update.appinstaller')
    }
    Descriptor $m.old 'old.msixbundle'
    $descriptor = Join-Path $Feed 'update.appinstaller'
    Add-AppxPackage -AppInstallerFile $descriptor
    $installed = $true
    $old = Installed $m.old
    # Pin a real OS-registered update source before launching the application.
    $old.package | Select-Object Name, Version, Publisher, Architecture, InstallLocation | ConvertTo-Json | Set-Content (Join-Path $Proof 'old-package.json')
    $marker = Join-Path $env:HERMES_HOME 'bundle-state-marker'
    $witness = [Guid]::NewGuid().ToString()
    Set-Content -LiteralPath $marker -Value $witness
    $python = (Get-Command python.exe).Source
    $verifier = Join-Path $Assets 'verify-plugin-preservation.py'
    & $python $verifier seed --home $env:HERMES_HOME --external (Join-Path $Work 'external-plugin')
    if ($LASTEXITCODE -ne 0) { throw 'Plugin seed failed' }
    & $python $verifier snapshot --home $env:HERMES_HOME --out (Join-Path $Work 'plugins-before.json')
    if ($LASTEXITCODE -ne 0) { throw 'Plugin snapshot failed' }
    # This is the only app launch performed by the driver: OLD, from its registered package.
    Start-Process -FilePath $old.exe -ArgumentList '--force-renderer-accessibility' | Out-Null
    $deadline = (Get-Date).AddMinutes(3)
    do {
        $oldRows = Main-Processes $old.exe
        if ($oldRows.Count) { break }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    if ($oldRows.Count -ne 1) { throw 'Could not identify the old installed app process' }
    $oldProcess = $oldRows[0]
    $oldProcess | Select-Object ProcessId, CreationDate, ExecutablePath | ConvertTo-Json | Set-Content (Join-Path $Proof 'old-process.json')
    Descriptor $m.new 'new.msixbundle'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Assets 'windows-bundled-drive-update.ps1') -OldProcessId $oldProcess.ProcessId -ProofDir $Proof -ResultPath (Join-Path $Proof 'click.json')
    if ($LASTEXITCODE -ne 0) { throw 'Real in-app update trigger failed' }
    $click = Get-Content -Raw (Join-Path $Proof 'click.json') | ConvertFrom-Json
    if (-not $click.ok -or -not $click.exited) { throw 'In-app trigger receipt is not successful' }
    $deadline = (Get-Date).AddMinutes(15)
    $new = $null; $newRows = @()
    do {
        $pkg = Get-AppxPackage -Name $m.new.identity
        if ($pkg -and $pkg.Version.ToString() -ceq $m.new.version) {
            $new = Installed $m.new
            $newRows = Main-Processes $new.exe
            if ($newRows.Count -eq 1) { break }
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    if (-not $new -or $newRows.Count -ne 1) { throw 'Native update did not automatically relaunch the new package' }
    $newProcess = $newRows[0]
    if ($newProcess.CreationDate -le $oldProcess.CreationDate) { throw 'New process was not created after OLD' }
    $stale = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($old.package.InstallLocation + '\', [StringComparison]::OrdinalIgnoreCase) })
    if ($stale.Count) { throw 'Old payload processes remain alive' }
    $healthy = $false
    $deadline = (Get-Date).AddMinutes(3)
    do {
        $payloadRoot = Join-Path $new.package.InstallLocation 'app\resources\agent-payload'
        $payloadProcesses = @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($payloadRoot + '\', [StringComparison]::OrdinalIgnoreCase) })
        foreach ($proc in $payloadProcesses) {
            foreach ($connection in @(Get-NetTCPConnection -State Listen -OwningProcess $proc.ProcessId -ErrorAction SilentlyContinue)) {
                try {
                    $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$($connection.LocalPort)/api/health" -TimeoutSec 2
                    if ($response.StatusCode -eq 200) { $healthy=$true; $response.Content | Set-Content (Join-Path $Proof 'backend-health.json'); break }
                } catch { }
            }
            if ($healthy) { break }
        }
        if (-not $healthy) { Start-Sleep -Seconds 2 }
    } while (-not $healthy -and (Get-Date) -lt $deadline)
    if (-not $healthy) { throw 'New packaged backend never returned HTTP 200 health' }
    & $python $verifier verify --home $env:HERMES_HOME --snapshot (Join-Path $Work 'plugins-before.json') --report (Join-Path $Proof 'plugin-preservation.json')
    if ($LASTEXITCODE -ne 0) { throw 'Plugin preservation failed' }
    if ((Get-Content -Raw $marker).Trim() -cne $witness) { throw 'User-state witness changed' }
    @{ ok=$true; oldVersion=$m.old.version; newVersion=$m.new.version; oldPid=$oldProcess.ProcessId; oldBirth=$oldProcess.CreationDate; newPid=$newProcess.ProcessId; newBirth=$newProcess.CreationDate; newPath=$newProcess.ExecutablePath; stamp=$new.stamp; automaticRelaunch=$true } | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $Proof 'acceptance.json')
} finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -ErrorAction SilentlyContinue }
    # Disposable runner teardown only, scoped to the package installed by this leg.
    if ($installed) {
        $pkg = Get-AppxPackage -Name $m.old.identity
        if ($pkg) {
            Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($pkg.InstallLocation + '\', [StringComparison]::OrdinalIgnoreCase) } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
            Remove-AppxPackage -Package $pkg.PackageFullName
        }
    }
}
