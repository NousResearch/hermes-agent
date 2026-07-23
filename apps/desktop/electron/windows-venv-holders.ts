/**
 * Windows venv-holder discovery + force-kill for self-update hand-off.
 *
 * The desktop used to only tree-kill backends it owns (hermesProcess /
 * backendPool). Gateways, slash workers, stray terminals and other profiles
 * keep `venv\Scripts\hermes.exe` / `.pyd` files mandatory-locked on Windows,
 * so the Update button aborted after 15s with "another process is holding the
 * Hermes install open" — even though the user had no second Hermes window.
 *
 * This module enumerates every process whose executable lives under the
 * install's venv (or whose command line / cwd references that venv or the
 * install root with hermes_cli), then taskkill /T /F each PID. Pure helpers
 * are injectable for unit tests; live discovery uses PowerShell CIM.
 */

import { execFileSync } from 'node:child_process'
import path from 'node:path'

import { hiddenWindowsChildOptions } from './windows-child-options'

export type VenvHolder = {
  pid: number
  name: string
  exe: string
  cmdline: string
}

/**
 * Normalize a Windows path for prefix comparison: lower-case, forward slashes
 * collapsed to backslashes, trailing separator stripped.
 */
export function normalizeWinPath(p: string): string {
  return String(p || '')
    .replace(/\//g, '\\')
    .replace(/\\+$/, '')
    .toLowerCase()
}

/**
 * Decide whether a process is holding the given Hermes install open.
 * Mirrors hermes_cli.main._detect_venv_python_processes matching rules.
 */
export function isVenvHolderProcess(
  {
    pid,
    exe,
    cmdline,
    cwd
  }: {
    pid: number
    exe?: string | null
    cmdline?: string | null
    cwd?: string | null
  },
  installRoot: string,
  skipPids: Set<number> = new Set()
): boolean {
  if (!Number.isInteger(pid) || pid <= 0 || skipPids.has(pid)) {
    return false
  }

  const root = normalizeWinPath(installRoot)

  if (!root) {
    return false
  }

  const venvPrefix = root + '\\venv\\'
  const rootPrefix = root + '\\'
  const exeNorm = normalizeWinPath(exe || '')

  const cmdlineLow = String(cmdline || '')
    .toLowerCase()
    .replace(/\//g, '\\')

  const cwdLow = normalizeWinPath(cwd || '') + '\\'

  if (exeNorm && (exeNorm === root + '\\venv' || exeNorm.startsWith(venvPrefix))) {
    return true
  }

  if (venvPrefix.slice(0, -1) && cmdlineLow.includes(venvPrefix.slice(0, -1))) {
    return true
  }

  if (cmdlineLow.includes('hermes_cli.main') || cmdlineLow.includes('tui_gateway')) {
    if (cmdlineLow.includes(rootPrefix.slice(0, -1)) || cwdLow.startsWith(rootPrefix)) {
      return true
    }
  }

  // Packaged/console hermes.exe living under the install (not only the venv).
  if (exeNorm.endsWith('\\hermes.exe') && exeNorm.startsWith(rootPrefix)) {
    return true
  }

  return false
}

/**
 * Parse a PowerShell-emitted JSON array of {pid,name,exe,cmdline,cwd} into
 * holder records for this install.
 */
export function selectVenvHolders(
  rows: Array<{
    pid?: number | string
    name?: string
    exe?: string
    cmdline?: string
    cwd?: string
  }>,
  installRoot: string,
  skipPids: Iterable<number> = []
): VenvHolder[] {
  const skip = new Set(Array.from(skipPids).filter(n => Number.isInteger(n)))
  const out: VenvHolder[] = []

  for (const row of rows || []) {
    const pid = Number(row.pid)

    if (
      !isVenvHolderProcess(
        {
          pid,
          exe: row.exe,
          cmdline: row.cmdline,
          cwd: row.cwd
        },
        installRoot,
        skip
      )
    ) {
      continue
    }

    out.push({
      pid,
      name: String(row.name || path.basename(String(row.exe || '')) || 'unknown'),
      exe: String(row.exe || ''),
      cmdline: String(row.cmdline || '').slice(0, 200)
    })
  }

  // Stable, unique by pid.
  const seen = new Set<number>()

  return out.filter(h => {
    if (seen.has(h.pid)) {
      return false
    }

    seen.add(h.pid)

    return true
  })
}

const LIST_HOLDERS_PS = `
$ErrorActionPreference = 'SilentlyContinue'
$procs = Get-CimInstance Win32_Process | ForEach-Object {
  [pscustomobject]@{
    pid = $_.ProcessId
    name = $_.Name
    exe = $_.ExecutablePath
    cmdline = $_.CommandLine
    cwd = $null
  }
}
$procs | ConvertTo-Json -Compress -Depth 3
`.trim()

/**
 * Live enumeration of processes holding `installRoot`'s venv open.
 * Returns [] off Windows or on discovery failure (caller still has the
 * shim lock probe as the real gate).
 */
export function listVenvHolders(
  installRoot: string,
  {
    skipPids = [],
    isWindows = process.platform === 'win32',
    execFileSyncImpl = execFileSync
  }: {
    skipPids?: Iterable<number>
    isWindows?: boolean
    execFileSyncImpl?: typeof execFileSync
  } = {}
): VenvHolder[] {
  if (!isWindows || !installRoot) {
    return []
  }

  let raw = ''

  try {
    raw = execFileSyncImpl(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', LIST_HOLDERS_PS],
      hiddenWindowsChildOptions({
        encoding: 'utf8',
        timeout: 20000,
        windowsHide: true,
        maxBuffer: 16 * 1024 * 1024
      }) as any
    ) as unknown as string
  } catch {
    return []
  }

  raw = String(raw || '').trim()

  if (!raw) {
    return []
  }

  let parsed: any

  try {
    parsed = JSON.parse(raw)
  } catch {
    return []
  }

  const rows = Array.isArray(parsed) ? parsed : parsed ? [parsed] : []

  return selectVenvHolders(rows, installRoot, skipPids)
}

/**
 * Force-kill every holder process tree. Best-effort; failures are ignored —
 * the caller's shim-unlock poll is the real gate.
 */
export function forceKillVenvHolders(
  holders: VenvHolder[],
  {
    isWindows = process.platform === 'win32',
    execFileSyncImpl = execFileSync,
    forceKillProcessTree
  }: {
    isWindows?: boolean
    execFileSyncImpl?: typeof execFileSync
    forceKillProcessTree?: (pid: number) => void
  } = {}
): number[] {
  if (!isWindows) {
    return []
  }

  const killed: number[] = []

  for (const h of holders) {
    try {
      if (forceKillProcessTree) {
        forceKillProcessTree(h.pid)
      } else {
        execFileSyncImpl(
          'taskkill',
          ['/PID', String(h.pid), '/T', '/F'],
          hiddenWindowsChildOptions({ stdio: 'ignore', windowsHide: true }) as any
        )
      }

      killed.push(h.pid)
    } catch {
      // Already gone / access denied — unlock wait decides success.
    }
  }

  return killed
}
