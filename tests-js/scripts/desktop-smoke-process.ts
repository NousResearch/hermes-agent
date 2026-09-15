import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { z } from 'zod'

export interface NativeProcess {
  pid: number
  parentPid: number
  executable: string
  command: string
  sourceRoot?: string
  cwd?: string
}

export function within(root: string, candidate: string): boolean {
  const relative = path.relative(fs.realpathSync(root), fs.realpathSync(candidate))
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
}

function nativeText(command: string, args: string[]): string {
  return execFileSync(command, args, { encoding: 'utf8', timeout: 30_000, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] }).trim()
}

/** Call only after binding the live listener (and bundled resources) to root. */
export function readInstallationCommit(root: string, origin: 'source' | 'bundled'): string {
  const commit = origin === 'source' ? nativeText('git', ['-C', root, 'rev-parse', 'HEAD'])
    : z.object({ payload: z.literal('bundled'), commit: z.string() }).parse(
      JSON.parse(fs.readFileSync(path.join(root, '..', 'install-stamp.json'), 'utf8')),
    ).commit
  return z.string().regex(/^[0-9a-f]{40}$/).parse(commit)
}

const windowsProcesses = z.array(z.object({
  ProcessId: z.number(), ParentProcessId: z.number(), ExecutablePath: z.string().nullable(), CommandLine: z.string().nullable(),
}))

export function readNativeProcesses(): NativeProcess[] {
  if (process.platform === 'win32') {
    const rows = windowsProcesses.parse(JSON.parse(nativeText('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
      '[Console]::OutputEncoding = [Text.UTF8Encoding]::new(); ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine)'])))
    return rows.map((row): NativeProcess => ({ pid: row.ProcessId, parentPid: row.ParentProcessId, executable: row.ExecutablePath ?? '', command: row.CommandLine ?? '' }))
  }
  const table = nativeText('ps', ['-axo', 'pid=,ppid=,comm='])
  return table.split('\n').flatMap((line: string): NativeProcess[] => {
    const match = /^\s*(\d+)\s+(\d+)\s+(.+)$/.exec(line)
    if (!match) {
      return []
    }
    return [{ pid: Number(match[1]), parentPid: Number(match[2]), executable: match[3], command: '' }]
  })
}

export function descendants(processes: NativeProcess[], parentPid: number): NativeProcess[] {
  const owned = new Set<number>([parentPid])
  for (;;) {
    const priorSize = owned.size
    for (const child of processes) {
      if (owned.has(child.parentPid)) {
        owned.add(child.pid)
      }
    }
    if (priorSize === owned.size) {
      return processes.filter((child: NativeProcess): boolean => child.pid !== parentPid && owned.has(child.pid))
    }
  }
}

function linuxListeningPid(port: number, candidates: NativeProcess[]): number[] {
  const inodes = new Set<string>()
  for (const filename of ['/proc/net/tcp', '/proc/net/tcp6']) {
    for (const line of fs.readFileSync(filename, 'utf8').split('\n').slice(1)) {
      const fields = line.trim().split(/\s+/)
      if (fields[3] === '0A' && Number.parseInt(fields[1].split(':')[1], 16) === port) {
        inodes.add(`socket:[${fields[9]}]`)
      }
    }
  }
  return candidates.filter((candidate: NativeProcess): boolean => {
    try {
      const dir = `/proc/${candidate.pid}/fd`
      return fs.readdirSync(dir).some((fd: string): boolean => {
        try { return inodes.has(fs.readlinkSync(path.join(dir, fd))) } catch { return false }
      })
    } catch { return false }
  }).map((candidate: NativeProcess): number => candidate.pid)
}

/** Identify the actual listener, not a healthy helper or a command recorded before spawn. */
export function localBackendProcess(port: number, electronPid: number): NativeProcess {
  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new Error('Invalid backend port')
  }
  const children = descendants(readNativeProcesses(), electronPid)
  let pids: number[]
  if (process.platform === 'win32') {
    pids = z.array(z.number()).parse(JSON.parse(nativeText('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
      `ConvertTo-Json -Compress -InputObject @((Get-NetTCPConnection -State Listen -LocalPort ${port} -ErrorAction Stop).OwningProcess | Sort-Object -Unique)`])))
  } else if (process.platform === 'darwin') {
    pids = nativeText('/usr/sbin/lsof', ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN', '-Fp']).split('\n')
      .filter((line: string): boolean => /^p\d+$/.test(line)).map((line: string): number => Number(line.slice(1)))
  } else {
    pids = linuxListeningPid(port, children)
  }
  const matches = children.filter((child: NativeProcess): boolean => pids.includes(child.pid))
  if (matches.length !== 1) {
    throw new Error(`Expected one Electron-owned backend listener on port ${port}; found ${matches.length}`)
  }
  const backend = matches[0]
  if (process.platform === 'linux') {
    backend.executable = fs.readlinkSync(`/proc/${backend.pid}/exe`)
    backend.cwd = fs.readlinkSync(`/proc/${backend.pid}/cwd`)
    backend.command = fs.readFileSync(`/proc/${backend.pid}/cmdline`, 'utf8').split('\0').filter(Boolean).map((arg: string): string => JSON.stringify(arg)).join(' ')
    backend.sourceRoot = fs.readFileSync(`/proc/${backend.pid}/environ`, 'utf8').split('\0')
      .find((entry: string): boolean => entry.startsWith('HERMES_PYTHON_SRC_ROOT='))?.slice('HERMES_PYTHON_SRC_ROOT='.length)
  } else if (process.platform === 'darwin') {
    backend.command = nativeText('ps', ['-p', String(backend.pid), '-o', 'args='])
    backend.cwd = nativeText('/usr/sbin/lsof', ['-a', '-p', String(backend.pid), '-d', 'cwd', '-Fn'])
      .split('\n').find((line: string): boolean => line.startsWith('n'))?.slice(1)
    backend.sourceRoot = /(?:^|\s)HERMES_PYTHON_SRC_ROOT=(.*?)(?=\s+[A-Za-z_][A-Za-z_0-9]*=|$)/
      .exec(nativeText('ps', ['eww', '-p', String(backend.pid), '-o', 'args=']))?.[1]
  }
  return backend
}

export function assertBackendOrigin(backend: NativeProcess, root: string, origin: 'source' | 'bundled'): void {
  if (origin === 'bundled') {
    if (path.basename(root) !== 'agent-payload' || !within(root, backend.executable)) {
      throw new Error('Bundled backend listener is not running the installed agent-payload interpreter')
    }
    return
  }
  // A Python -m entry has no source path in argv. Its live import root is
  // either the captured editable root (Nix) or the process's working directory.
  const moduleLaunch = /(?:^|\s)"?-m"?\s+"?hermes_cli\.main"?(?:\s|$)/.test(backend.command)
    && /^python(?:w|\d+(?:\.\d+)*)?(?:\.exe)?$/i.test(path.basename(backend.executable))
  const importRoot = backend.sourceRoot || backend.cwd
  if (moduleLaunch && importRoot) {
    if (fs.realpathSync(importRoot) !== fs.realpathSync(root)) {
      throw new Error('Source backend listener imports a different source tree')
    }
    return
  }
  // PM's Windows .cmd fallback encodes the installation-bound bootstrap;
  // POSIX launchers pass that same script directly as Python's -c argument.
  const encoded = /base64\.b64decode\('([A-Za-z0-9+/=]+)'\)/.exec(backend.command)?.[1]
  const command = encoded ? Buffer.from(encoded, 'base64').toString('utf8') : backend.command
  // Source venvs may resolve to system Python; their entry script still lives in the installation.
  const spellings = [root, fs.realpathSync(root)].flatMap((value: string): string[] => [value, JSON.stringify(value).slice(1, -1)])
  const fromInstall = spellings.some((value: string): boolean => {
    const escaped = value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    return new RegExp(`(?:^|[\\s"'])${escaped}(?:[/\\\\]|[\\s"']|$)`, process.platform === 'win32' ? 'i' : '').test(command)
  })
  if (!fromInstall) {
    throw new Error('Source backend listener command does not name the expected installed source tree')
  }
}