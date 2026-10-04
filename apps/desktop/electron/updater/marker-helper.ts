/**
 * The checkout script's marker helper (A7 rule 3, SPEC section 6): Electron's
 * ONLY way to mutate the update marker. The script holds the `<marker>.lock`
 * kernel lock for the whole read-judge-mutate; Node has no portable kernel
 * lock, so Electron never deletes or rewrites the marker itself.
 *
 *   posix:   bash <root>/scripts/desktop-update/posix.sh --marker-op <op> --install-root <root>
 *              [--desktop-pid P] [--handoff-run R]
 *   Windows: powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File windows.ps1
 *              -MarkerOp <op> -InstallRoot <root> [-DesktopPid P] [-HandoffRun R]
 *
 * One stdout line: `absent | reclaimed | held | busy | live <pid> | withdrawn |
 * taken <pid> | foreign`. Anything else, a nonzero exit, a timeout or a
 * missing script is `unsupported` (an older checkout).
 */

import { type ChildProcess, spawn } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import * as path from 'node:path'

import { resolveUpdateScriptHandoff } from '../updater-process'

export type MarkerHelperOp = 'reclaim' | 'withdraw'

export type MarkerHelperVerdict =
  | { kind: 'absent' | 'reclaimed' | 'held' | 'busy' | 'withdrawn' | 'foreign' | 'unsupported' }
  | { kind: 'live' | 'taken'; pid: number }

export const MARKER_HELPER_TIMEOUT_MS = 20_000

/** The protocol line a hand-off script advertises (SPEC 4). */
const PROTOCOL_RE = /^# hermes-handoff-protocol: (\d+)\s*$/m

export interface HelperSpawnResult {
  /** Exit code; null when killed (timeout) or the spawn failed. */
  code: number | null
  stdout: string
}

export type HelperSpawn = (
  command: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv; timeout: number; windowsHide: boolean }
) => Promise<HelperSpawnResult>

export interface MarkerHelperOptions {
  updateRoot: string
  hermesHome: string
  desktopPid?: number | null
  runId?: string | null
  isWindows: boolean
  spawn?: HelperSpawn
  timeoutMs?: number
}

/** The checkout's hand-off script, or null when the checkout predates it. */
export function handoffScriptPath(updateRoot: string, isWindows: boolean): string | null {
  if (isWindows) {
    return resolveUpdateScriptHandoff(updateRoot, { isWindows: true })?.scriptPath ?? null
  }

  const posix = path.join(updateRoot, 'scripts', 'desktop-update', 'posix.sh')

  return existsSync(posix) ? posix : null
}

/** Hand-off protocol a script speaks: 2+ from its protocol line, else 1 (legacy). */
export function readHandoffProtocol(scriptPath: string | null | undefined): number {
  if (!scriptPath) {
    return 1
  }

  try {
    const match = PROTOCOL_RE.exec(readFileSync(scriptPath, 'utf8'))
    const version = match ? Number(match[1]) : 1

    return Number.isSafeInteger(version) && version >= 2 ? version : 1
  } catch {
    return 1
  }
}

/**
 * Spawn the helper and collect stdout, hard-bounded by `timeout`: on POSIX the
 * helper runs in its own process group and the whole group is killed on
 * timeout, so a stuck grandchild holding the pipe cannot extend the wait.
 */
const defaultSpawn: HelperSpawn = (command, args, { env, timeout, windowsHide }) =>
  new Promise(resolve => {
    const posix = process.platform !== 'win32'
    let stdout = ''
    let settled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const finish = (code: number | null) => {
      if (!settled) {
        settled = true
        clearTimeout(timer)
        resolve({ code, stdout })
      }
    }

    let child: ChildProcess

    try {
      child = spawn(command, args, { env, windowsHide, detached: posix, stdio: ['ignore', 'pipe', 'ignore'] })
    } catch {
      finish(null)

      return
    }

    child.stdout?.setEncoding('utf8')
    child.stdout?.on('data', (chunk: string) => {
      if (stdout.length < 64 * 1024) {
        stdout += chunk
      }
    })
    child.on('error', () => finish(null))
    child.on('close', code => finish(typeof code === 'number' ? code : null))
    timer = setTimeout(() => {
      try {
        if (posix && child.pid) {
          process.kill(-child.pid, 'SIGKILL')
        } else {
          child.kill()
        }
      } catch {
        // Already gone.
      }

      finish(null)
    }, timeout)
  })

const BARE_VERDICTS = new Set(['absent', 'reclaimed', 'held', 'busy', 'withdrawn', 'foreign'])

/** Parse the helper's one verdict line; anything else is `unsupported`. */
export function parseMarkerHelperVerdict(stdout: string): MarkerHelperVerdict {
  const lines = stdout.split(/\r?\n/).map(line => line.trim()).filter(Boolean)

  if (lines.length !== 1) {
    return { kind: 'unsupported' }
  }

  const [word, arg, ...extra] = lines[0].split(/\s+/)

  if (extra.length) {
    return { kind: 'unsupported' }
  }

  if (arg === undefined) {
    return BARE_VERDICTS.has(word) ? ({ kind: word } as MarkerHelperVerdict) : { kind: 'unsupported' }
  }

  const pid = /^\d+$/.test(arg) ? Number(arg) : NaN

  if ((word === 'live' || word === 'taken') && Number.isSafeInteger(pid) && pid > 0) {
    return { kind: word, pid }
  }

  return { kind: 'unsupported' }
}

/** The exact command line for one helper op (exported for tests). */
export function markerHelperCommand(
  op: MarkerHelperOp,
  scriptPath: string,
  { updateRoot, desktopPid, runId, isWindows }: Pick<MarkerHelperOptions, 'updateRoot' | 'desktopPid' | 'runId' | 'isWindows'>
): { command: string; args: string[] } {
  const hasPid = typeof desktopPid === 'number' && Number.isInteger(desktopPid) && desktopPid > 0

  if (isWindows) {
    return {
      command: 'powershell.exe',
      args: [
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-File',
        scriptPath,
        '-MarkerOp',
        op,
        '-InstallRoot',
        updateRoot,
        ...(hasPid ? ['-DesktopPid', String(desktopPid)] : []),
        ...(runId ? ['-HandoffRun', runId] : [])
      ]
    }
  }

  return {
    command: 'bash',
    args: [
      scriptPath,
      '--marker-op',
      op,
      '--install-root',
      updateRoot,
      ...(hasPid ? ['--desktop-pid', String(desktopPid)] : []),
      ...(runId ? ['--handoff-run', runId] : [])
    ]
  }
}

/** Run one marker op through the checkout's script under its sidecar lock. */
export async function runMarkerHelper(op: MarkerHelperOp, options: MarkerHelperOptions): Promise<MarkerHelperVerdict> {
  const scriptPath = handoffScriptPath(options.updateRoot, options.isWindows)

  if (!scriptPath) {
    return { kind: 'unsupported' }
  }

  const { command, args } = markerHelperCommand(op, scriptPath, options)

  try {
    const result = await (options.spawn ?? defaultSpawn)(command, args, {
      env: { ...process.env, HERMES_HOME: options.hermesHome },
      timeout: options.timeoutMs ?? MARKER_HELPER_TIMEOUT_MS,
      windowsHide: true
    })

    return result.code === 0 ? parseMarkerHelperVerdict(result.stdout) : { kind: 'unsupported' }
  } catch {
    return { kind: 'unsupported' }
  }
}
