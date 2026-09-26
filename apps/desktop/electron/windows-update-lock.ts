/**
 * windows-update-lock.ts
 *
 * Scan for EXTERNAL processes that hold the venv (the messaging gateway launched
 * via Scheduled Task / Startup, a dashboard, a user terminal) and return their
 * PIDs so the pre-update lock release can tree-kill them (#62311).
 *
 * The desktop's own teardown only reaches the backends it spawned; the lock
 * gate then waits 15s on a shim those external holders keep locked and the
 * update aborts with "venv shim still locked" — on any install that runs the
 * gateway, that means "always". `stopGatewayBeforeUpdate` drains gateways
 * through the CLI, but a worker that ignores the drain (or any other venv
 * holder) still needs the process-table sweep here.
 *
 * PowerShell-based, mirroring the existing hidden-children contract: the scan
 * runs under the app's injected environment with `windowsHide`, never shows a
 * window, and returns `[]` whenever it cannot run (no PowerShell, not on
 * Windows, scan failure) — a failed scan degrades to the pre-fix behavior.
 */

import type { ExecFileSyncOptionsWithStringEncoding } from 'node:child_process'
import { execFileSync } from 'node:child_process'
import path from 'node:path'

type ExecFileSyncLike = typeof execFileSync
type ChildOptionsBuilder = (
  options?: ExecFileSyncOptionsWithStringEncoding
) => ExecFileSyncOptionsWithStringEncoding

type ListExternalVenvHolderPidsOptions = {
  childOptions?: ChildOptionsBuilder
  currentPid?: number
  exec?: ExecFileSyncLike
  isWindows: boolean
  ownedPids?: number[]
  powerShellPath?: string | null
  updateRoot: string
}

function normalizeOwnedPids(ownedPids: number[] = [], currentPid?: number): number[] {
  const values = currentPid ? [...ownedPids, currentPid] : [...ownedPids]
  return [...new Set(values.filter(pid => Number.isInteger(pid) && pid > 0))]
}

function powerShellVenvHolderScript(): string {
  return `
$ErrorActionPreference = 'Stop'
$root = [Environment]::GetEnvironmentVariable('HERMES_UPDATE_ROOT')
if ([string]::IsNullOrWhiteSpace($root)) {
  Write-Output ''
  exit 0
}
$ownedRaw = [Environment]::GetEnvironmentVariable('HERMES_OWNED_PIDS')
$owned = @()
if (-not [string]::IsNullOrWhiteSpace($ownedRaw)) {
  $owned = @(
    $ownedRaw -split ',' |
      ForEach-Object { $_.Trim() } |
      Where-Object { $_ -match '^[0-9]+$' } |
      ForEach-Object { [int]$_ }
  )
}
$rootLower = $root.ToLowerInvariant()
$venvScriptsLower = (Join-Path $root 'venv\\Scripts').ToLowerInvariant()
$pids = @(
  Get-CimInstance Win32_Process | Where-Object {
    $pid = [int]$_.ProcessId
    if ($owned -contains $pid -or $pid -eq $PID) {
      return $false
    }
    $exe = if ($_.ExecutablePath) { $_.ExecutablePath.ToLowerInvariant() } else { '' }
    $cmd = if ($_.CommandLine) { $_.CommandLine.ToLowerInvariant() } else { '' }
    return $exe.Contains($rootLower) -or $exe.Contains($venvScriptsLower) -or $cmd.Contains($rootLower)
  } | ForEach-Object { [int]$_.ProcessId } | Sort-Object -Unique
)
Write-Output ($pids -join ',')
`.trim()
}

export function parseWindowsPidList(output: string): number[] {
  const text = typeof output === 'string' ? output : ''
  return [...new Set(
    text
      .split(/[,\r\n]+/)
      .map(part => Number.parseInt(part.trim(), 10))
      .filter(pid => Number.isInteger(pid) && pid > 0)
  )]
}

export function listExternalVenvHolderPids({
  childOptions = options => (options || {}) as ExecFileSyncOptionsWithStringEncoding,
  currentPid,
  exec = execFileSync,
  isWindows,
  ownedPids = [],
  powerShellPath,
  updateRoot
}: ListExternalVenvHolderPidsOptions): number[] {
  if (!isWindows || !powerShellPath || !updateRoot) {
    return []
  }

  const owned = normalizeOwnedPids(ownedPids, currentPid)
  const stdout = exec(
    powerShellPath,
    ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', powerShellVenvHolderScript()],
    childOptions({
      encoding: 'utf8',
      env: {
        ...process.env,
        HERMES_OWNED_PIDS: owned.join(','),
        HERMES_UPDATE_ROOT: path.win32.normalize(updateRoot)
      },
      stdio: ['ignore', 'pipe', 'ignore']
    })
  )

  return parseWindowsPidList(stdout)
}
