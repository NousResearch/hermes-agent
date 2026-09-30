# Real deployment of receipt-verified bytes. Never run on a developer desktop.
param(
    [Parameter(Mandatory=$true)][string]$Artifact,
    [Parameter(Mandatory=$true)][ValidateSet('x64','arm64')][string]$Arch,
    [Parameter(Mandatory=$true)][string]$Commit,
    [string]$Tag = '',
    [string]$ChannelRequest = '',
    [Parameter(Mandatory=$true)][string]$Work,
    [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($env:GITHUB_ACTIONS -cne 'true' -or $env:RUNNER_ENVIRONMENT -cne 'github-hosted' -or $env:OS -cne 'Windows_NT') {
    throw 'Disposable native GitHub-hosted Windows runner required'
}
if ($Commit -cnotmatch '^[a-f0-9]{40}$') { throw 'Expected exact lowercase full commit SHA' }
if (-not [IO.Path]::IsPathRooted($Artifact) -or -not (Test-Path -LiteralPath $Artifact -PathType Leaf)) { throw 'Absolute artifact path required' }
$Artifact = (Resolve-Path -LiteralPath $Artifact).Path
$processors = @(Get-CimInstance Win32_Processor)
$nativeCode = @{ x64=9; arm64=12 }[$Arch]
if (-not $processors.Count -or @($processors | Where-Object { $_.Architecture -ne $nativeCode }).Count) {
    throw "Host is not native $Arch; emulated x64 on ARM is not coverage"
}
$Assets = Join-Path $PSScriptRoot 'e2e-assets'
$Node = (Get-Command node.exe -ErrorAction Stop).Source
$Metadata = Join-Path $Assets 'bundle-smoke-metadata.mjs'
function Run-Node([string[]]$Argv) {
    & $Node @Argv
    if ($LASTEXITCODE -ne 0) { throw "Node exited $LASTEXITCODE ($($Argv[0]))" }
}
$nodeArch = Run-Node @('-p', 'process.arch')
if ($nodeArch -cne $Arch) { throw 'Driver Node must also use the native architecture' }
$identityArgs = @('--commit', $Commit)
if ($Tag) { $identityArgs += @('--tag', $Tag) }
if ($ChannelRequest) { $identityArgs += @('--channel-request', $ChannelRequest) }
$Expected = (Run-Node (@($Metadata, 'identity') + $identityArgs)) | ConvertFrom-Json
Run-Node @($Metadata, 'prepare', '--work', $Work, '--out', $Out)
. (Join-Path $Assets 'windows-bundle-metadata.ps1')
Add-Type -AssemblyName System.IO.Compression.FileSystem
$HomeDir = Join-Path $Work 'home'
$UserData = Join-Path $Work 'user-data'
New-Item -ItemType Directory -Path $HomeDir, $UserData | Out-Null
Start-Transcript -Path (Join-Path $Out 'native-install.log') | Out-Null
$attempted = $false
$installedFullName = $null
$failed = $false
try {
    $beforeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact).Hash.ToLowerInvariant()
    @{ path=$Artifact; sha256=$beforeHash } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'artifact.json')
    $signature = Get-AuthenticodeSignature -LiteralPath $Artifact
    if ($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -cne $Expected.publisher) { throw 'Artifact signature/publisher invalid' }
    $inputMetadata = Read-BundleSmokeMetadata $Artifact $Arch $Expected
    $inputMetadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'artifact-metadata.json')
    if (@(Get-AppxPackage -AllUsers -Name $Expected.msixIdentity).Count) { throw 'Package identity already exists; refusing to replace it' }
    $attempted = $true
    Add-AppxPackage -Path $Artifact -ErrorAction Stop
    $packages = @(Get-AppxPackage -Name $Expected.msixIdentity)
    if ($packages.Count -ne 1) { throw 'Expected exactly one deployed package' }
    $pkg = $packages[0]
    $installedFullName = $pkg.PackageFullName
    if ($pkg.Publisher -cne $Expected.publisher -or $pkg.Architecture.ToString() -ine $Arch -or
        $pkg.Version.ToString() -cne $inputMetadata.Version -or $pkg.Status.ToString() -cne 'Ok') {
        throw 'Deployed package identity/version/native architecture/status mismatch'
    }
    $manifest = Get-AppxPackageManifest -Package $pkg.PackageFullName
    $installed = Read-SmokePackageManifest $manifest $Arch $Expected
    if ($installed.Version -cne $inputMetadata.Version) { throw 'Installed manifest version mismatch' }
    if ($inputMetadata.Executable -and $installed.Executable -cne $inputMetadata.Executable) { throw 'Installed executable differs from input manifest' }
    $exe = [IO.Path]::GetFullPath((Join-Path $pkg.InstallLocation $installed.Executable))
    if (-not $exe.StartsWith($pkg.InstallLocation.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'Installed manifest executable is absent or outside package' }
    $resources = Join-Path (Split-Path -Parent $exe) 'resources'
    $root = Join-Path $resources 'agent-payload'
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { throw 'Installed agent-payload missing' }
    $stampPath = Join-Path $resources 'install-stamp.json'
    $stamp = Get-Content -Raw -LiteralPath $stampPath | ConvertFrom-Json
    $displayVersion = (Run-Node (@($Metadata, 'stamp', '--platform', 'win32', '--stamp', $stampPath) + $identityArgs)) | ConvertFrom-Json
    if ($ChannelRequest) {
        if ($pkg.Version.ToString() -cne $displayVersion) { throw 'Package version disagrees with channel request' }
    } else {
        $semverBase = ($displayVersion -split '-')[0]
        if (-not $pkg.Version.ToString().StartsWith($semverBase + '.', [StringComparison]::Ordinal)) { throw 'Package version disagrees with stamped semver' }
        if (-not $Tag -and $pkg.Version.ToString() -cne ($displayVersion + '.0')) { throw 'Commit package must use stamped semver with zero revision' }
    }
    @{ packageFullName=$pkg.PackageFullName; publisher=$pkg.Publisher; architecture=$pkg.Architecture.ToString();
        version=$pkg.Version.ToString(); exe=$exe; root=$root; stamp=$stamp } |
        ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $Out 'installed-identity.json')
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Artifact).Hash.ToLowerInvariant() -cne $beforeHash) { throw 'Input artifact changed during installation' }
    Push-Location $Work
    try {
        Run-Node @((Join-Path $Assets 'desktop-smoke.ts'), '--exe', $exe, '--root', $root, '--origin', 'bundled',
            '--home', $HomeDir, '--user-data', $UserData, '--out', $Out, '--phase', 'installed', '--expect-commit', $Commit)
    } finally { Pop-Location }
} catch {
    $failed = $true
    $_ | Out-String | Set-Content -LiteralPath (Join-Path $Out 'native-install-error.txt')
    Write-Error $_ -ErrorAction Continue
} finally {
    try {
        # A failed deployment can still have registered the package. Admission
        # ruled out prior identities, so only this invocation's exact package is eligible.
        if ($attempted) {
            $remaining = @(Get-AppxPackage -Name $Expected.msixIdentity)
            foreach ($pkg in $remaining) {
                if ($pkg.Publisher -cne $Expected.publisher -or $pkg.Version.ToString() -cne $inputMetadata.Version -or
                    ($installedFullName -and $pkg.PackageFullName -cne $installedFullName)) { throw 'Refusing cleanup of an unowned package' }
                Remove-AppxPackage -Package $pkg.PackageFullName -ErrorAction Stop
                if (Get-AppxPackage -Name $Expected.msixIdentity) { throw 'Package removal did not complete' }
            }
        }
    } catch {
        $failed = $true
        $_ | Out-String | Set-Content -LiteralPath (Join-Path $Out 'native-cleanup-error.txt')
        Write-Error $_ -ErrorAction Continue
    } finally {
        @{ exitCode=[int]$failed } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Out 'native-install-exit.json')
        Stop-Transcript | Out-Null
    }
}
if ($failed) { exit 1 }