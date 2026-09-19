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
  // The app binds its backend to a tree by environment, not only by argv: the
  // installation root leads PYTHONPATH and VIRTUAL_ENV names the venv. Only the
  // platforms that can read a process environment populate these.
  pythonPath?: string
  virtualEnv?: string
  cwd?: string
}

/** What the caller already knows about the app that owns this listener. */
export interface OriginEvidence {
  /** The root the app itself reported resolving, from its own UI/identity channel. */
  appReportedRoot?: string
}

export function within(root: string, candidate: string): boolean {
  const relative = path.relative(fs.realpathSync(root), fs.realpathSync(candidate))
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
}

function nativeText(command: string, args: string[]): string {
  return execFileSync(command, args, { encoding: 'utf8', timeout: 30_000, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] }).trim()
}

/** One NAME=value from `ps eww`'s environment tail (values may contain spaces). */
function psEnvValue(environment: string, name: string): string | undefined {
  return new RegExp(`(?:^|\\s)${name}=(.*?)(?=\\s+[A-Za-z_][A-Za-z_0-9]*=|$)`).exec(environment)?.[1]
}

/** Call only after binding the live listener (and bundled resources) to root. */
export function readInstallationCommit(root: string, origin: 'source' | 'bundled'): string {
  const commit = origin === 'source' ? nativeText('git', ['-C', root, 'rev-parse', 'HEAD'])
    : z.object({ payload: z.literal('bundled'), commit: z.string() }).parse(
      JSON.parse(fs.readFileSync(path.join(root, '..', 'install-stamp.json'), 'utf8')),
    ).commit
  return z.string().regex(/^[0-9a-f]{40}$/).parse(commit)
}

const bundleEnvSchema = z.record(z.string(), z.string().nullable())

/** The baked runtime defaults/clears of a bundled artifact, recorded in the
 * install stamp so the smoke driver can predict the app's resolved Hermes home
 * without reimplementing the banner. Absent for artifacts built before the
 * stamp carried it, and for source checkpoints. */
export function readBundledBundleEnv(root: string): Record<string, string | null> | undefined {
  const stampPath = path.join(root, '..', 'install-stamp.json')
  // Absent for source checkpoints and artifacts built before the stamp carried
  // bundleEnv; treat as "no baked env" and fall back to the pinned home.
  if (!fs.existsSync(stampPath)) {
    return undefined
  }
  const stamp = z.object({ payload: z.literal('bundled'), bundleEnv: bundleEnvSchema.optional() }).parse(
    JSON.parse(fs.readFileSync(stampPath, 'utf8')),
  )
  return stamp.bundleEnv
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
    const environment = nativeText('ps', ['eww', '-p', String(backend.pid), '-o', 'args='])
    backend.sourceRoot = psEnvValue(environment, 'HERMES_PYTHON_SRC_ROOT')
    backend.pythonPath = psEnvValue(environment, 'PYTHONPATH')
    backend.virtualEnv = psEnvValue(environment, 'VIRTUAL_ENV')
  }
  return backend
}

export function assertBackendOrigin(backend: NativeProcess, root: string, origin: 'source' | 'bundled',
  evidence: OriginEvidence = {}): void {
  if (origin === 'bundled') {
    if (path.basename(root) !== 'agent-payload' || !within(root, backend.executable)) {
      throw new Error('Bundled backend listener is not running the installed agent-payload interpreter')
    }
    return
  }
  // PM's Windows .cmd fallback encodes the installation-bound bootstrap;
  // POSIX launchers pass that same script directly as Python's -c argument.
  const encoded = /base64\.b64decode\('([A-Za-z0-9+/=]+)'\)/.exec(backend.command)?.[1]
  const command = encoded ? Buffer.from(encoded, 'base64').toString('utf8') : backend.command
  // Source venvs may resolve to system Python; their entry script still lives in the installation.
  const spellings = [root, fs.realpathSync(root)].flatMap((value: string): string[] => [value, JSON.stringify(value).slice(1, -1)])
  const namesInstallRoot = spellings.some((value: string): boolean => {
    const escaped = value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    return new RegExp(`(?:^|[\\s"'])${escaped}(?:[/\\\\]|[\\s"']|$)`, process.platform === 'win32' ? 'i' : '').test(command)
  })
  const sameTree = (value?: string): boolean => {
    if (value === undefined) return false
    // An unreadable path is no evidence, not a crash: a process cwd can be gone.
    try { return fs.realpathSync(value) === fs.realpathSync(root) } catch { return false }
  }
  // A Python -m entry has no source path in argv. Its live import root is
  // either the captured editable root (Nix) or the process's working directory.
  const moduleLaunch = /(?:^|\s)"?-m"?\s+"?hermes_cli\.main"?(?:\s|$)/.test(command)
    && /^python(?:w|\d+(?:\.\d+)*)?(?:\.exe)?$/i.test(path.basename(backend.executable))
  if (moduleLaunch) {
    // A captured root is authoritative: the launcher told us which tree it bound.
    if (backend.sourceRoot !== undefined) {
      if (!sameTree(backend.sourceRoot)) {
        throw new Error('Source backend listener imports a different source tree'
          + ` (HERMES_PYTHON_SRC_ROOT=${backend.sourceRoot}, expected=${root}, command=${backend.command})`)
      }
      return
    }
    // Without a captured root the backend's cwd is NOT sufficient on its own --
    // the app hands the backend ITS OWN directory (the smoke's home), so a
    // cwd-only inference calls a backend running the installation's own venv
    // interpreter "a different source tree". Accept the launcher cd'ing into the
    // tree, or a command that names it (`<root>/venv/bin/python -m hermes_cli.main`).
    // The app binds the backend to the tree by environment as well as argv:
    // `main.ts` puts the installation root first on the backend's PYTHONPATH and
    // VIRTUAL_ENV names its venv, which is how `import hermes_cli` resolves from
    // the installation. A platform that can read that environment needs no
    // spelling in argv (macOS resolves the venv symlink before spawning); a
    // platform that cannot (Windows) stays strict.
    const joinsInstall = (value?: string): boolean => value !== undefined
      && value.split(path.delimiter).some((entry: string): boolean => entry !== '' && sameTree(entry))
    const usesInstallEnvironment = joinsInstall(backend.pythonPath)
      || (backend.virtualEnv !== undefined && backend.virtualEnv.trim() !== ''
          && sameTree(path.dirname(backend.virtualEnv)))
    // Some platforms expose no way to read another process's environment or cwd, so a
    // module-launched backend there can never name its tree in argv (Windows: the venv
    // launcher hands the interpreter over as a system python). The listener has already
    // been tied to the app process that owns it, and the caller has already asserted the
    // root that app reported resolving, so those two facts together are the evidence.
    // Requiring the process evidence to be absent keeps this from loosening a platform
    // that can read one.
    const processEvidenceUnreadable = backend.cwd === undefined
      && backend.pythonPath === undefined && backend.virtualEnv === undefined
    const appOwnsBackend = processEvidenceUnreadable
      && evidence.appReportedRoot !== undefined && sameTree(evidence.appReportedRoot)
    if (!sameTree(backend.cwd) && !namesInstallRoot && !usesInstallEnvironment && !appOwnsBackend) {
      throw new Error('Source backend listener imports a different source tree'
        + ` (cwd=${backend.cwd ?? '(unreadable)'}, HERMES_PYTHON_SRC_ROOT=(unset),`
        + ` executable=${backend.executable}, expected=${root},`
        + ` appReportedRoot=${evidence.appReportedRoot ?? '(unreported)'}, command=${backend.command})`)
    }
    return
  }
  if (!namesInstallRoot) {
    throw new Error('Source backend listener command does not name the expected installed source tree')
  }
}