# Pure archive/XML admission tests: no package deployment or host impersonation.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'windows-bundle-metadata.ps1')
Add-Type -AssemblyName System.IO.Compression.FileSystem
$temp = Join-Path ([IO.Path]::GetTempPath()) ('bundle-metadata-' + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $temp | Out-Null
$expected = [pscustomobject]@{ msixIdentity='Fixture.Hermes'; publisher='CN=Fixture'; applicationId='Hermes' }
function Archive([string]$Name, [string]$Entry, [string]$Xml, [string[]]$Extra = @()) {
    $file = Join-Path $temp $Name
    $zip = [IO.Compression.ZipFile]::Open($file, [IO.Compression.ZipArchiveMode]::Create)
    try {
        $writer = [IO.StreamWriter]::new($zip.CreateEntry($Entry).Open())
        try { $writer.Write($Xml) } finally { $writer.Dispose() }
        foreach ($name in $Extra) { $zip.CreateEntry($name) | Out-Null }
    } finally { $zip.Dispose() }
    return $file
}
function Reject([scriptblock]$Action) {
    $rejected = $false
    try { & $Action | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Admission unexpectedly accepted invalid metadata' }
}
try {
    $xml = '<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"><Identity Name="Fixture.Hermes" Publisher="CN=Fixture" Version="1.2.3.0" ProcessorArchitecture="arm64"/><Applications><Application Id="Hermes" Executable="app\Hermes.exe"/></Applications></Package>'
    $file = Archive 'native.msix' 'AppxManifest.xml' $xml
    $got = Read-BundleSmokeMetadata $file 'arm64' $expected
    if ($got.Version -cne '1.2.3.0' -or $got.Executable -cne 'app\Hermes.exe') { throw 'MSIX metadata lost' }
    Reject { Read-BundleSmokeMetadata $file 'x64' $expected }
    $wrong = Archive 'wrong.msix' 'AppxManifest.xml' ($xml.Replace('CN=Fixture', 'CN=Other'))
    Reject { Read-BundleSmokeMetadata $wrong 'arm64' $expected }
    $escape = Archive 'escape.msix' 'AppxManifest.xml' ($xml.Replace('app\Hermes.exe', '..\outside.exe'))
    Reject { Read-BundleSmokeMetadata $escape 'arm64' $expected }
    $bundle = '<Bundle xmlns="http://schemas.microsoft.com/appx/2013/bundle"><Identity Name="Fixture.Hermes" Publisher="CN=Fixture" Version="1.2.3.0"/><Packages><Package Type="application" Architecture="x64" Version="1.2.3.0" FileName="x64.msix"/><Package Type="application" Architecture="arm64" Version="1.2.3.0" FileName="arm64.msix"/></Packages></Bundle>'
    $file = Archive 'universal.msixbundle' 'AppxMetadata/AppxBundleManifest.xml' $bundle @('x64.msix','arm64.msix')
    foreach ($arch in @('arm64','x64')) {
        $got = Read-BundleSmokeMetadata $file $arch $expected
        if ($got.Slice -cne "$arch.msix") { throw 'Wrong native slice selected' }
    }
    $missing = Archive 'missing.msixbundle' 'AppxMetadata/AppxBundleManifest.xml' ($bundle.Replace('Architecture="arm64"', 'Architecture="x86"')) @('x64.msix','arm64.msix')
    Reject { Read-BundleSmokeMetadata $missing 'arm64' $expected }
    $duplicate = Archive 'duplicate.msixbundle' 'AppxMetadata/AppxBundleManifest.xml' ($bundle.Replace('Architecture="x64"', 'Architecture="arm64"')) @('x64.msix','arm64.msix')
    Reject { Read-BundleSmokeMetadata $duplicate 'arm64' $expected }
    $channel = [pscustomobject]@{ msixIdentity=$expected.msixIdentity; publisher=$expected.publisher;
        applicationId=$expected.applicationId; windowsVersion='0.1.1.0' }
    $channelFile = Archive 'channel.msix' 'AppxManifest.xml' ($xml.Replace('1.2.3.0', $channel.windowsVersion))
    if ((Read-BundleSmokeMetadata $channelFile 'arm64' $channel).Version -cne $channel.windowsVersion) { throw 'Channel version lost' }
    Reject { Read-BundleSmokeMetadata $file 'arm64' $channel }
    Reject { Read-SmokePackageManifest ([xml]$xml) 'arm64' $channel }
    $channelBundle = Archive 'channel.msixbundle' 'AppxMetadata/AppxBundleManifest.xml' ($bundle.Replace('1.2.3.0', $channel.windowsVersion)) @('x64.msix','arm64.msix')
    if ((Read-BundleSmokeMetadata $channelBundle 'arm64' $channel).Version -cne $channel.windowsVersion) { throw 'Channel bundle version lost' }
    Write-Output 'PASS: MSIX identity/executable and MSIXBUNDLE native-slice admission'
} finally { Remove-Item -LiteralPath $temp -Recurse -Force }