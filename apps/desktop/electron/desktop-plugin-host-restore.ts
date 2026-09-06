/**
 * desktop-plugin-host-restore.ts
 *
 * A Desktop plugin service on Windows is supervised by a Windows Script Host
 * loop (`<plugin>\service-host.vbs`) that relaunches the service ten seconds
 * after it exits and is itself started only at login. The update preflight
 * must stop that supervisor first (or it reopens the venv mid-update), which
 * would strand the plugin service until the next login.
 *
 * This module is the ledger that closes the loop: the preflight records each
 * supervisor it stopped, and the Desktop relaunches the recorded supervisors
 * once the update finishes (next boot) or aborts (same process). Pure and
 * dependency-injected so the relaunch policy is testable without Electron.
 */

import { spawn as nodeSpawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

export const STOPPED_PLUGIN_HOSTS_FILE = 'update-stopped-plugin-hosts.json'
export const STOPPED_PLUGIN_HOSTS_SCHEMA_VERSION = 1

/** The supervisor identity the scanner reports for a stopped plugin service. */
export interface TerminatedPluginServiceHost {
  pid: number
  createdAt: number
  argv: string[]
  cwd: string | null
}

export interface StoppedDesktopPluginHost extends TerminatedPluginServiceHost {
  stoppedAt: number
}

export interface PluginHostRestoreDeps {
  isWindows?: boolean
  now?: () => number
  existsSync?: (target: string) => boolean
  readFileSync?: (target: string) => string
  writeFileSync?: (target: string, contents: string) => void
  mkdirSync?: (target: string) => void
  rmSync?: (target: string) => void
  spawn?: (command: string, args: string[], options: { cwd?: string }) => void
  log?: (line: string) => void
}

export function stoppedPluginHostsPath(hermesHome: string): string {
  return path.join(hermesHome, STOPPED_PLUGIN_HOSTS_FILE)
}

function defaultDeps(deps: PluginHostRestoreDeps) {
  return {
    isWindows: deps.isWindows ?? process.platform === 'win32',
    now: deps.now ?? Date.now,
    existsSync: deps.existsSync ?? ((target: string) => fs.existsSync(target)),
    readFileSync: deps.readFileSync ?? ((target: string) => fs.readFileSync(target, 'utf8')),
    writeFileSync:
      deps.writeFileSync ?? ((target: string, contents: string) => fs.writeFileSync(target, contents, { encoding: 'utf8', mode: 0o600 })),
    mkdirSync: deps.mkdirSync ?? ((target: string) => fs.mkdirSync(target, { recursive: true })),
    rmSync: deps.rmSync ?? ((target: string) => fs.rmSync(target, { force: true })),
    spawn:
      deps.spawn ??
      ((command: string, args: string[], options: { cwd?: string }) => {
        const child = nodeSpawn(command, args, {
          ...(options.cwd ? { cwd: options.cwd } : {}),
          detached: true,
          stdio: 'ignore',
          windowsHide: true
        })

        child.unref()
      }),
    log: deps.log ?? (() => {})
  }
}

// The launch line is a Windows Script Host argv, so its paths are Windows
// paths regardless of where this code runs (CI lints these tests on Linux).
const winPath = path.win32

function isPathUnder(target: string, root: string): boolean {
  const relative = winPath.relative(winPath.resolve(root), winPath.resolve(target))

  return relative.length > 0 && !relative.startsWith('..') && !winPath.isAbsolute(relative)
}

function scriptArgument(argv: readonly string[]): string | null {
  const candidate = argv.find((value, index) => index > 0 && !value.startsWith('/') && !value.startsWith('-'))

  return candidate ? candidate.replace(/^"|"$/g, '') : null
}

export function parseTerminatedPluginServiceHost(value: unknown): TerminatedPluginServiceHost | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {return null}

  const { pid, created_at: createdAt, argv, cwd } = value as Record<string, unknown>

  if (!Number.isInteger(pid) || (pid as number) <= 0) {return null}

  if (typeof createdAt !== 'number' || !Number.isFinite(createdAt) || createdAt <= 0) {return null}

  if (!Array.isArray(argv) || argv.length === 0 || !argv.every(entry => typeof entry === 'string' && entry.length > 0)) {
    return null
  }

  if (cwd !== null && cwd !== undefined && typeof cwd !== 'string') {return null}

  return { pid: pid as number, createdAt, argv: [...(argv as string[])], cwd: typeof cwd === 'string' && cwd ? cwd : null }
}

/**
 * Only the exact supervisor shape the scanner proved is ever relaunched: a
 * Windows Script Host binary running a `.vbs` that lives under this
 * HERMES_HOME's `desktop-plugins` directory. The ledger lives in HERMES_HOME
 * too, so this is a shape check, not a trust boundary; it keeps a corrupt or
 * stale entry from turning into an arbitrary process launch.
 */
export function isRelaunchableDesktopPluginHost(
  hermesHome: string,
  host: Pick<TerminatedPluginServiceHost, 'argv'>,
  existsSync: (target: string) => boolean = fs.existsSync
): boolean {
  const [executable] = host.argv

  if (!executable) {return false}

  const binary = winPath.basename(executable.replace(/^"|"$/g, '')).toLowerCase()

  if (binary !== 'wscript.exe' && binary !== 'cscript.exe') {return false}

  const script = scriptArgument(host.argv)

  if (!script || winPath.extname(script).toLowerCase() !== '.vbs') {return false}

  if (!isPathUnder(script, winPath.join(hermesHome, 'desktop-plugins'))) {return false}

  try {
    return existsSync(script)
  } catch {
    return false
  }
}

function readLedger(target: string, io: ReturnType<typeof defaultDeps>): StoppedDesktopPluginHost[] {
  let raw: string

  try {
    if (!io.existsSync(target)) {return []}
    raw = io.readFileSync(target)
  } catch {
    return []
  }

  try {
    const parsed = JSON.parse(raw)

    if (!parsed || parsed.schemaVersion !== STOPPED_PLUGIN_HOSTS_SCHEMA_VERSION || !Array.isArray(parsed.hosts)) {
      return []
    }

    const hosts: StoppedDesktopPluginHost[] = []

    for (const entry of parsed.hosts) {
      const host = parseTerminatedPluginServiceHost({
        pid: entry?.pid,
        created_at: entry?.createdAt,
        argv: entry?.argv,
        cwd: entry?.cwd ?? null
      })

      if (!host) {continue}

      const stoppedAt = typeof entry?.stoppedAt === 'number' && Number.isFinite(entry.stoppedAt) ? entry.stoppedAt : 0
      hosts.push({ ...host, stoppedAt })
    }

    return hosts
  } catch {
    return []
  }
}

function ledgerKey(host: Pick<TerminatedPluginServiceHost, 'argv'>): string {
  const script = scriptArgument(host.argv) ?? host.argv.join(' ')

  return script.toLowerCase()
}

/**
 * Remember one stopped supervisor. Idempotent per script: stopping the same
 * plugin service twice in one preflight (worker record, then wrapper record)
 * records it once.
 */
export function recordStoppedDesktopPluginHost(
  hermesHome: string,
  host: TerminatedPluginServiceHost,
  deps: PluginHostRestoreDeps = {}
): boolean {
  const io = defaultDeps(deps)

  if (!isRelaunchableDesktopPluginHost(hermesHome, host, io.existsSync)) {
    io.log(`[updates] not recording plugin service host PID ${host.pid}: launch line is not a desktop-plugins script host`)

    return false
  }

  const target = stoppedPluginHostsPath(hermesHome)
  const existing = readLedger(target, io).filter(entry => ledgerKey(entry) !== ledgerKey(host))
  const hosts = [...existing, { ...host, stoppedAt: io.now() }]

  try {
    io.mkdirSync(path.dirname(target))
    io.writeFileSync(target, JSON.stringify({ schemaVersion: STOPPED_PLUGIN_HOSTS_SCHEMA_VERSION, hosts }, null, 2))
  } catch (error) {
    io.log(`[updates] could not record stopped plugin service host: ${error instanceof Error ? error.message : String(error)}`)

    return false
  }

  return true
}

/**
 * Relaunch every recorded supervisor and clear the ledger. Best-effort and
 * never throws: a supervisor that cannot be relaunched is logged and dropped
 * (the user can relaunch it from the Startup folder; the next login does too).
 */
export function restoreStoppedDesktopPluginHosts(
  hermesHome: string,
  deps: PluginHostRestoreDeps = {}
): { relaunched: string[]; skipped: string[] } {
  const io = defaultDeps(deps)
  const outcome = { relaunched: [] as string[], skipped: [] as string[] }

  if (!io.isWindows) {return outcome}

  const target = stoppedPluginHostsPath(hermesHome)
  const hosts = readLedger(target, io)

  try {
    io.rmSync(target)
  } catch {
    void 0
  }

  for (const host of hosts) {
    const label = scriptArgument(host.argv) ?? host.argv.join(' ')

    if (!isRelaunchableDesktopPluginHost(hermesHome, host, io.existsSync)) {
      outcome.skipped.push(label)
      io.log(`[updates] not relaunching plugin service host ${label}: launch line no longer qualifies`)

      continue
    }

    try {
      const [command, ...args] = host.argv
      io.spawn(command!, args, host.cwd ? { cwd: host.cwd } : {})
      outcome.relaunched.push(label)
      io.log(`[updates] relaunched plugin service host ${label}`)
    } catch (error) {
      outcome.skipped.push(label)
      io.log(`[updates] could not relaunch plugin service host ${label}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  return outcome
}
