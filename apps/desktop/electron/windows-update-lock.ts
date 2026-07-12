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

function powerShellVenvHolderScript() {
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
  const text = typeof output === "string" ? output : ''
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
