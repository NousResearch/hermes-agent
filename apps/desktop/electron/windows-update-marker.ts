import { execFile } from 'node:child_process'
import path from 'node:path'

import type { UpdateMarkerClaim } from './update-marker'
import { windowsPowerShellExecutable } from './windows-powershell-path'

/** Transfer a repair claim without exposing an absent marker to another updater. */
export async function transferUpdateMarkerIfOwnedBy(
  hermesHome: string,
  expected: UpdateMarkerClaim,
  successor: UpdateMarkerClaim
): Promise<boolean> {
  if (process.platform !== 'win32' || expected.pid !== process.pid ||
    ![expected.pid, expected.startedAt, successor.pid, successor.startedAt]
      .every(value => Number.isSafeInteger(value) && value > 0)) { return false }

  const encode = (value: string) => Buffer.from(value, 'utf8').toString('base64')
  const marker = path.join(hermesHome, '.hermes-update-in-progress')

  const script = `
$ErrorActionPreference = 'Stop'
$stream = $null
try {
  $file = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${encode(marker)}'))
  $expected = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${encode(`${expected.pid}\n${expected.startedAt}\n`)}'))
  $next = [Convert]::FromBase64String('${encode(`${successor.pid}\n${successor.startedAt}\n`)}')
  $stream = [IO.File]::Open($file, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
  $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true, 4096, $true)
  $current = $reader.ReadToEnd()
  $reader.Dispose()
  if ($current -cne $expected) { exit 2 }
  # Write BEFORE truncating. Truncating first leaves a zero-length marker if
  # this process dies mid-transaction, and an empty marker blocks every later
  # launch until the dwell-and-CAS self-heal notices it (B2). Growing then
  # shrinking never exposes an empty file at any instant.
  $stream.Position = 0
  $stream.Write($next, 0, $next.Length)
  $stream.Flush($true)
  $stream.SetLength($next.Length)
  $stream.Flush($true)
} catch { exit 2 } finally { if ($stream) { $stream.Dispose() } }
`

  return new Promise(resolve => {
    execFile(
      windowsPowerShellExecutable(),
      ['-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(script, 'utf16le').toString('base64')],
      { windowsHide: true, timeout: 5_000 },
      error => resolve(!error)
    )
  })
}
