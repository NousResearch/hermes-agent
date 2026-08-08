[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseDirectory,

    [Parameter(Mandatory = $true)]
    [string]$PublisherName
)

$ErrorActionPreference = 'Stop'
$releaseRoot = (Resolve-Path -LiteralPath $ReleaseDirectory).Path
$packagedExecutable = Join-Path $releaseRoot 'win-unpacked\Hermes.exe'
$installers = @(
    Get-ChildItem -LiteralPath $releaseRoot -File |
        Where-Object { $_.Extension -in '.exe', '.msi' }
)
$targets = @($packagedExecutable) + @($installers.FullName)

if (-not (Test-Path -LiteralPath $packagedExecutable -PathType Leaf)) {
    throw "Packaged executable not found: $packagedExecutable"
}
if (-not ($installers.Extension -contains '.exe')) {
    throw "Signed NSIS installer was not produced under $releaseRoot"
}
if (-not ($installers.Extension -contains '.msi')) {
    throw "Signed MSI installer was not produced under $releaseRoot"
}

$signtool = Get-Command signtool.exe -ErrorAction SilentlyContinue
if (-not $signtool) {
    $kitsRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    $signtool = Get-ChildItem -LiteralPath $kitsRoot -Filter signtool.exe -Recurse |
        Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
        Sort-Object FullName -Descending |
        Select-Object -First 1
}
if (-not $signtool) {
    throw 'signtool.exe was not found on the release runner'
}
$signtoolPath = if ($signtool.Source) {
    $signtool.Source
} else {
    $signtool.FullName
}

foreach ($target in $targets) {
    $signature = Get-AuthenticodeSignature -LiteralPath $target
    if ($signature.Status -ne 'Valid') {
        throw "Invalid Authenticode signature on $target ($($signature.Status))"
    }

    $actualPublisher = $signature.SignerCertificate.GetNameInfo(
        [System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName,
        $false
    )
    if (-not [string]::Equals(
        $actualPublisher,
        $PublisherName,
        [System.StringComparison]::Ordinal
    )) {
        throw "Unexpected publisher on $target. Expected '$PublisherName', found '$actualPublisher'"
    }
    if (-not $signature.TimeStamperCertificate) {
        throw "Authenticode timestamp missing on $target"
    }

    & $signtoolPath verify /pa /all /v $target
    if ($LASTEXITCODE -ne 0) {
        throw "signtool verification failed for $target"
    }
}

Write-Host "Verified $($targets.Count) signed Hermes artifact(s) for publisher '$PublisherName'."
