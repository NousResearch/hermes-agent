# Metadata-only readers shared by native admission and pure XML/archive tests.
function Read-SmokeXmlEntry($Zip, [string]$Name) {
    $entries = @($Zip.Entries | Where-Object { $_.FullName -ceq $Name })
    if ($entries.Count -ne 1) { throw "Expected exactly one $Name" }
    $settings = [Xml.XmlReaderSettings]::new()
    $settings.DtdProcessing = [Xml.DtdProcessing]::Prohibit
    $settings.XmlResolver = $null
    $stream = $entries[0].Open()
    $reader = [Xml.XmlReader]::Create($stream, $settings)
    try {
        $xml = [Xml.XmlDocument]::new()
        $xml.XmlResolver = $null
        $xml.Load($reader)
        return ,$xml
    } finally { $reader.Dispose(); $stream.Dispose() }
}

function Assert-SmokeIdentity($Identity, $Expected) {
    if (-not $Identity -or $Identity.GetAttribute('Name') -cne $Expected.msixIdentity -or
        $Identity.GetAttribute('Publisher') -cne $Expected.publisher) { throw 'Package identity/publisher mismatch' }
    $version = $Identity.GetAttribute('Version')
    if ($version -cnotmatch '^\d+\.\d+\.\d+\.\d+$' -or
        @($version.Split('.') | Where-Object { [int]$_ -gt 65535 }).Count) { throw 'Invalid package version' }
    if ($Expected.PSObject.Properties['windowsVersion'] -and $version -cne $Expected.windowsVersion) {
        throw 'Package version disagrees with channel request'
    }
    return $version
}

function Assert-SmokeRelativePath([string]$Path) {
    if (-not $Path -or $Path -match '^[\\/]|:|(^|[\\/])\.\.?([\\/]|$)') {
        throw "Unsafe package relative path: $Path"
    }
}

function Read-SmokePackageManifest([xml]$Xml, [string]$Arch, $Expected) {
    $identities = @($Xml.SelectNodes('/*[local-name()="Package"]/*[local-name()="Identity"]'))
    if ($identities.Count -ne 1) { throw 'Expected one package identity' }
    $identity = $identities[0]
    $version = Assert-SmokeIdentity $identity $Expected
    if ($identity.GetAttribute('ProcessorArchitecture') -cne $Arch) { throw 'Package lacks requested native architecture' }
    $applications = @($Xml.SelectNodes('/*[local-name()="Package"]/*[local-name()="Applications"]/*[local-name()="Application"]') |
        Where-Object { $_.GetAttribute('Id') -ceq $Expected.applicationId })
    if ($applications.Count -ne 1) { throw 'Expected exactly one desktop applicationId' }
    $exe = $applications[0].GetAttribute('Executable')
    Assert-SmokeRelativePath $exe
    return [pscustomobject]@{ Version=$version; Architecture=$Arch; Executable=$exe; Slice=$null }
}

function Read-BundleSmokeMetadata([string]$Artifact, [string]$Arch, $Expected) {
    $zip = [IO.Compression.ZipFile]::OpenRead($Artifact)
    try {
        switch ([IO.Path]::GetExtension($Artifact).ToLowerInvariant()) {
            '.msix' {
                return Read-SmokePackageManifest (Read-SmokeXmlEntry $zip 'AppxManifest.xml') $Arch $Expected
            }
            '.msixbundle' {
                $xml = Read-SmokeXmlEntry $zip 'AppxMetadata/AppxBundleManifest.xml'
                $identities = @($xml.SelectNodes('/*[local-name()="Bundle"]/*[local-name()="Identity"]'))
                if ($identities.Count -ne 1) { throw 'Expected one bundle identity' }
                $version = Assert-SmokeIdentity $identities[0] $Expected
                $slices = @($xml.SelectNodes('/*[local-name()="Bundle"]/*[local-name()="Packages"]/*[local-name()="Package"]') |
                    Where-Object { $_.GetAttribute('Type') -ceq 'application' -and $_.GetAttribute('Architecture') -ceq $Arch })
                if ($slices.Count -ne 1) { throw 'Expected exactly one native application slice (no emulation fallback)' }
                $slice = $slices[0].GetAttribute('FileName')
                Assert-SmokeRelativePath $slice
                if (@($zip.Entries | Where-Object { $_.FullName -ceq $slice }).Count -ne 1) { throw 'Native package bytes missing or ambiguous' }
                if ($slices[0].GetAttribute('Version') -cne $version) { throw 'Native slice and bundle version mismatch' }
                return [pscustomobject]@{ Version=$version; Architecture=$Arch; Executable=$null; Slice=$slice }
            }
            default { throw 'Expected a receipt-selected MSIX or MSIXBUNDLE' }
        }
    } finally { $zip.Dispose() }
}