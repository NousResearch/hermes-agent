import { mkdir, readdir, rm } from 'node:fs/promises'
import path from 'node:path'

import { downloadPinnedArtifact, PINNED_DOWNLOAD_POLICY, type PinnedDownloadProgress } from './artifact'
import type { ChannelTarget } from './channel'
import { type NativeCommandResult, runChannelPowerShell } from './channel-native'

/** Verify downloaded bytes and native metadata before App Installer can stop the backend. */
export async function verifyPreparedChannelInstaller(
  file: string,
  target: ChannelTarget,
  command: (script: string, input: string) => Promise<NativeCommandResult> = runChannelPowerShell,
  onBytes?: PinnedDownloadProgress
): Promise<void> {
  const pkg = target.package

  if (pkg.platform !== 'win32' || !pkg.publisher) {
    throw new Error('Expected a publisher-bound Windows package')
  }

  if (!/^[0-9a-f]{64}$/.test(pkg.artifact.sha256)) {
    throw new Error('Expected a sha256-pinned Windows package')
  }

  // Keyed by the pinned digest, not a fresh temp dir: a download that stalled
  // or dropped keeps its partial bytes here and the next attempt resumes them.
  const name: string = `.channel-artifact-${pkg.artifact.sha256}`
  const directory: string = path.join(path.dirname(file), name)

  // Partial bytes of any other (superseded) package can never be resumed.
  for (const entry of await readdir(path.dirname(file))) {
    if (entry.startsWith('.channel-artifact-') && entry !== name) {
      await rm(path.join(path.dirname(file), entry), { recursive: true, force: true })
    }
  }

  await mkdir(directory, { recursive: true, mode: 0o700 })

  const artifact: string = await downloadPinnedArtifact(
    directory,
    {
      url: target.artifactUrl,
      sha256: pkg.artifact.sha256,
      size: pkg.artifact.size,
      format: pkg.artifact.key.endsWith('.msixbundle') ? 'msixbundle' : 'msix'
    },
    PINNED_DOWNLOAD_POLICY,
    onBytes
  )

  try {
    await command(
      String.raw`
$ErrorActionPreference='Stop'
$p=[Console]::In.ReadToEnd() | ConvertFrom-Json
function Read-SafeXml($stream) {
  $settings=[Xml.XmlReaderSettings]::new(); $settings.DtdProcessing=[Xml.DtdProcessing]::Prohibit; $settings.XmlResolver=$null
  $reader=[Xml.XmlReader]::Create($stream,$settings)
  try { $xml=[Xml.XmlDocument]::new(); $xml.XmlResolver=$null; $xml.Load($reader); return ,$xml } finally { $reader.Dispose() }
}
$stream=[IO.File]::OpenRead($p.file)
try { $xml=Read-SafeXml $stream } finally { $stream.Dispose() }
$root=$xml.DocumentElement
if ($root.LocalName -cne 'AppInstaller' -or $root.GetAttribute('Uri') -cne $p.feedUrl -or $root.GetAttribute('Version') -cne $p.version) { throw 'Channel descriptor binding mismatch' }
$main=@($root.ChildNodes | Where-Object { $_.LocalName -in @('MainPackage','MainBundle') })
if ($main.Count -ne 1 -or $main[0].GetAttribute('Name') -cne $p.identity -or $main[0].GetAttribute('Publisher') -cne $p.publisher -or $main[0].GetAttribute('Version') -cne $p.version -or $main[0].GetAttribute('Uri') -cne $p.artifactUrl) { throw 'Descriptor does not reference the pinned native package' }
if (@($root.ChildNodes | Where-Object { $_.LocalName -notin @('MainPackage','MainBundle') }).Count) { throw 'Pinned channel descriptor must not add dependencies, optional packages or background update policies' }
$signature=Get-AuthenticodeSignature -LiteralPath $p.artifact
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cne $p.publisher) { throw 'Native artifact signature/publisher mismatch' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::OpenRead($p.artifact)
try {
  $name=if ($p.artifact.EndsWith('.msixbundle')) { 'AppxMetadata/AppxBundleManifest.xml' } else { 'AppxManifest.xml' }
  $entry=$zip.GetEntry($name)
  if (-not $entry) { throw 'Native manifest missing' }
  $stream=$entry.Open()
  try { $native=Read-SafeXml $stream } finally { $stream.Dispose() }
  if ($name -ceq 'AppxManifest.xml') { $identity=$native.Package.Identity; if ($identity.ProcessorArchitecture -cne $p.arch) { throw 'Package architecture mismatch' } }
  else { $identity=$native.Bundle.Identity; $slices=@($native.Bundle.Packages.Package | Where-Object { $_.Type -ceq 'application' -and $_.Architecture -ceq $p.arch -and $_.Version -ceq $p.version }); if ($slices.Count -ne 1) { throw 'Bundle architecture/version mismatch' } }
  if ($identity.Name -cne $p.identity -or $identity.Publisher -cne $p.publisher -or $identity.Version -cne $p.version) { throw 'Native manifest binding mismatch' }
} finally { $zip.Dispose() }
`,
      JSON.stringify({
        file,
        artifact,
        feedUrl: target.feedUrl,
        artifactUrl: target.artifactUrl,
        identity: pkg.identity,
        publisher: pkg.publisher,
        version: pkg.version,
        arch: pkg.arch
      })
    )
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
}

/** `verifyPreparedChannelInstaller` with the real PowerShell, reporting the
 *  pinned package's download progress to the update overlay. */
export function verifyPreparedChannelDownload(
  file: string,
  target: ChannelTarget,
  onBytes?: PinnedDownloadProgress
): Promise<void> {
  return verifyPreparedChannelInstaller(file, target, runChannelPowerShell, onBytes)
}
