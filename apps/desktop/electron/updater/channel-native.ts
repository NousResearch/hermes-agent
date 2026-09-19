import { execFile } from 'node:child_process'
import { createHash } from 'node:crypto'
import { open, realpath } from 'node:fs/promises'
import path from 'node:path'

import type { ChannelBuild, ChannelPackage } from './channel-protocol'

/** Native feeds carry their own hashes; also require the admitted build's hash. */
export async function verifyChannelDownload(files: string[], artifact: ChannelPackage['artifact']): Promise<void> {
  if (files.length !== 1) { throw new Error('Expected exactly one channel artifact') }
  const file = await open(files[0], 'r')
  try {
    const stat = await file.stat()
    if (!stat.isFile() || stat.size !== artifact.size) { throw new Error('Channel artifact size mismatch') }
    const digest = createHash('sha256')
    for await (const chunk of file.createReadStream({ autoClose: false })) { digest.update(chunk) }
    if (digest.digest('hex') !== artifact.sha256) { throw new Error('Channel artifact digest mismatch') }
  } finally { await file.close() }
}

export interface NativeCommandResult {
  stdout: string
  stderr: string
}

/** Run one native (codesign / PowerShell) command with bounded output and timeout. */
export function runChannelCommand(
  file: string,
  args: readonly string[],
  timeout: number = 120_000,
  input: string = ''
): Promise<NativeCommandResult> {
  return new Promise<NativeCommandResult>((resolve, reject): void => {
    const child: ReturnType<typeof execFile> = execFile(
      file,
      [...args],
      { encoding: 'utf8', windowsHide: true, timeout, maxBuffer: 4 * 1024 * 1024 },
      (error: Error | null, stdout: string, stderr: string): void => {
        if (error) {
          reject(new Error(`Native channel operation failed: ${file}: ${stderr}`, { cause: error }))

          return
        }

        resolve({ stdout, stderr })
      }
    )

    child.stdin?.end(input)
  })
}

export async function runChannelPowerShell(script: string, input: string = ''): Promise<NativeCommandResult> {
  if (process.platform !== 'win32') { throw new Error('This operation requires Windows') }
  return runChannelCommand(
    path.join(process.env.SystemRoot || 'C:\\Windows', 'System32/WindowsPowerShell/v1.0/powershell.exe'),
    ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(script, 'utf16le').toString('base64')],
    600_000, input
  )
}

/** Resolve symlinks to the canonical on-disk path, tolerating a not-yet-created tail. */
export async function canonicalChannelPath(candidate: string): Promise<string> {
  if (!path.isAbsolute(candidate) || candidate.includes('\0')) {
    throw new Error('Channel operations require absolute local paths')
  }

  try {
    return await realpath(candidate)
  } catch (error) {
    if (!(error instanceof Error) || !('code' in error) || error.code !== 'ENOENT') {
      throw error
    }

    const parent: string = path.dirname(candidate)

    if (parent === candidate) {
      throw error
    }

    return path.join(await canonicalChannelPath(parent), path.basename(candidate))
  }
}

export interface RunningChannelApp {
  identity: string
  signer: string
  appPath: string
  nativeVersion: string
  executable: string
  applicationId: string | null
  packageFullName: string | null
  packageFamilyName: string | null
  removalRoots: string[]
}

/** Derive trust from the signed running application, never from an R2 document. */
export async function inspectRunningChannelApp(build: ChannelBuild): Promise<RunningChannelApp> {
  const executable: string = await canonicalChannelPath(process.execPath)
  if (process.platform === 'darwin') {
    const appPath: string = path.resolve(executable, '../../..')
    await runChannelCommand('/usr/bin/codesign', ['--verify', '--deep', '--strict', appPath])
    const signature: NativeCommandResult = await runChannelCommand('/usr/bin/codesign', ['-dv', '--verbose=4', appPath])
    const signer: string = /^TeamIdentifier=(.+)$/m.exec(signature.stderr)?.[1] || ''
    const identity: string = /^Identifier=(.+)$/m.exec(signature.stderr)?.[1] || ''
    if (!/^[A-Z0-9]{10}$/.test(signer) || identity !== build.identity.appId) {
      throw new Error('Running channel application signature does not match its baked identity')
    }
    return { identity, signer, appPath, executable, nativeVersion: build.version,
      applicationId: null, packageFullName: null, packageFamilyName: null, removalRoots: [appPath] }
  }
  const result: NativeCommandResult = await runChannelPowerShell(String.raw`
$ErrorActionPreference='Stop'
$p=[Console]::In.ReadToEnd() | ConvertFrom-Json
$packages=@(Get-AppxPackage -Name $p.identity | Where-Object { $_.Name -ceq $p.identity })
if ($packages.Count -ne 1) { throw 'Running channel package is not uniquely registered' }
$pkg=$packages[0]
if ($pkg.Status.ToString() -cne 'Ok' -or $pkg.SignatureKind.ToString() -cne 'Developer' -or $pkg.IsDevelopmentMode -or $pkg.NonRemovable) { throw 'Channel package is not a trusted current-user sideload' }
if (-not $p.executable.StartsWith($pkg.InstallLocation.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Running executable is outside registered package' }
$manifest=Get-AppxPackageManifest -Package $pkg.PackageFullName
$apps=@($manifest.Package.Applications.Application | Where-Object { [IO.Path]::GetFullPath((Join-Path $pkg.InstallLocation $_.Executable)) -ieq $p.executable })
if ($apps.Count -ne 1) { throw 'Running executable is not the registered application' }
@{ identity=$pkg.Name; signer=$pkg.Publisher; appPath=$pkg.InstallLocation; nativeVersion=$pkg.Version.ToString(); executable=$p.executable; applicationId=$apps[0].Id; packageFullName=$pkg.PackageFullName; packageFamilyName=$pkg.PackageFamilyName; removalRoots=@($pkg.InstallLocation,(Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) ('Packages\'+$pkg.PackageFamilyName))) } | ConvertTo-Json -Compress
`, JSON.stringify({ identity: build.identity.msixAppIdWithOrg, executable }))
  const installed: RunningChannelApp = JSON.parse(result.stdout)
  if (installed.nativeVersion !== build.windowsVersion || installed.identity !== build.identity.msixAppIdWithOrg || !installed.signer) {
    throw new Error('Running package does not match its baked channel version')
  }
  return installed
}
