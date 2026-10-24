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
 * One stdout line: reclaim `absent | reclaimed | held | busy | live <pid>`,
 * withdraw `absent | busy | withdrawn | taken <pid> | foreign`. Anything else
 * (another op's word included), a nonzero exit, a timeout, malformed output
 * or a script that exists but cannot be read is `error` (retry without
 * clearance). Only a missing or pre-protocol-2 script is `unsupported` (an
 * older checkout).
 */

import { type ChildProcess, spawn } from 'node:child_process'
import { existsSync, readFileSync, statSync } from 'node:fs'
import * as path from 'node:path'

import { resolveUpdateScriptHandoff } from '../updater-process'

export type MarkerHelperOp = 'reclaim' | 'withdraw'

export type MarkerHelperVerdict =
  | { kind: 'absent' | 'reclaimed' | 'held' | 'busy' | 'withdrawn' | 'foreign' | 'unsupported' | 'error' }
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

/**
 * Hand-off protocol a script speaks: 2+ from its protocol line, else 1
 * (legacy, a missing script included). Null when the script exists but cannot
 * be read right now (a sharing violation, permissions): that is not an older
 * checkout (review R8 M5).
 */
export function readHandoffProtocol(scriptPath: string | null | undefined): number | null {
  if (!scriptPath) {
    return 1
  }

  try {
    const match = PROTOCOL_RE.exec(readFileSync(scriptPath, 'utf8'))
    const version = match ? Number(match[1]) : 1

    return Number.isSafeInteger(version) && version >= 2 ? version : 1
  } catch (error) {
    return (error as NodeJS.ErrnoException)?.code === 'ENOENT' ? 1 : null
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

/** The words each op can answer (marker.sh marker_op, marker-claim.ps1 Invoke-MarkerOp). */
const OP_VERDICTS: Record<MarkerHelperOp, ReadonlySet<string>> = {
  reclaim: new Set(['absent', 'reclaimed', 'held', 'busy', 'live']),
  withdraw: new Set(['absent', 'busy', 'withdrawn', 'taken', 'foreign'])
}

/** Parse the helper's one verdict line; anything else is an operational error. */
export function parseMarkerHelperVerdict(stdout: string): MarkerHelperVerdict {
  const lines = stdout
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean)

  if (lines.length !== 1) {
    return { kind: 'error' }
  }

  const [word, arg, ...extra] = lines[0].split(/\s+/)

  if (extra.length) {
    return { kind: 'error' }
  }

  if (arg === undefined) {
    return BARE_VERDICTS.has(word) ? ({ kind: word } as MarkerHelperVerdict) : { kind: 'error' }
  }

  const pid = /^\d+$/.test(arg) ? Number(arg) : NaN

  if ((word === 'live' || word === 'taken') && Number.isSafeInteger(pid) && pid > 0) {
    return { kind: word, pid }
  }

  return { kind: 'error' }
}

/** The exact command line for one helper op (exported for tests). */
export function markerHelperCommand(
  op: MarkerHelperOp,
  scriptPath: string,
  {
    updateRoot,
    desktopPid,
    runId,
    isWindows
  }: Pick<MarkerHelperOptions, 'updateRoot' | 'desktopPid' | 'runId' | 'isWindows'>
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

/** First line as the scripts clean it: marker.sh `marker_line` (one CR, spaces/tabs); marker.ps1 `.Trim()`. */
function scriptLine(text: string, isWindows: boolean): string {
  const line = text.replace(/^\uFEFF/, '').split('\n')[0]

  return isWindows ? line.trim() : line.replace(/\r$/, '').replace(/^[ \t]+|[ \t]+$/g, '')
}

/**
 * The checkout lock exactly as marker.sh `checkout_lock_path` / marker.ps1
 * `Get-CheckoutLockPath` resolve it (hermes_cli/update_lock.py::checkout_lock_path):
 * `<git common dir>/hermes-update.lock`, else `<root>/.hermes-update.lock`.
 * Throws when `.git` or `commondir` exists but cannot be read.
 */
export function checkoutLockPath(updateRoot: string, isWindows: boolean): string {
  const dot = path.join(updateRoot, '.git')
  const plain = path.join(updateRoot, '.hermes-update.lock')
  const kind = statSync(dot, { throwIfNoEntry: false })
  let gitdir = dot

  if (kind?.isFile()) {
    const line = scriptLine(readFileSync(dot, 'utf8'), isWindows)

    // bash `case gitdir:*` is case-sensitive; PowerShell `-like` is not.
    if (!(isWindows ? /^gitdir:/i : /^gitdir:/).test(line)) {
      return plain
    }

    const named = scriptLine(line.slice(7), isWindows)

    // marker.ps1 treats an empty `gitdir:` as no git dir; bash resolves it to the root.
    if (isWindows && !named) {
      return plain
    }

    gitdir = path.resolve(updateRoot, named)
  } else if (!kind?.isDirectory()) {
    return plain
  }

  const commondir = path.join(gitdir, 'commondir')

  if (statSync(commondir, { throwIfNoEntry: false })?.isFile()) {
    const common = scriptLine(readFileSync(commondir, 'utf8'), isWindows)
    gitdir = common ? path.resolve(gitdir, common) : gitdir
  }

  return path.join(gitdir, 'hermes-update.lock')
}

/**
 * False only when the scripts' `checkout_lock_held` would answer "not held"
 * without trying the lock: no regular file at the checkout lock path
 * (`[ -f ]` / `[IO.File]::Exists`). Anything it cannot settle (an unreadable
 * `.git`, a stat error other than "missing") asks the script.
 */
export function checkoutLockMayBeHeld(updateRoot: string, isWindows: boolean): boolean {
  try {
    return statSync(checkoutLockPath(updateRoot, isWindows), { throwIfNoEntry: false })?.isFile() ?? false
  } catch (error) {
    return (error as NodeJS.ErrnoException)?.code !== 'ENOTDIR'
  }
}

/** Run one marker op through the checkout's script under its sidecar lock. */
export async function runMarkerHelper(op: MarkerHelperOp, options: MarkerHelperOptions): Promise<MarkerHelperVerdict> {
  const scriptPath = handoffScriptPath(options.updateRoot, options.isWindows)
  const protocol = readHandoffProtocol(scriptPath)

  if (protocol === null) {
    return { kind: 'error' }
  }

  if (!scriptPath || protocol < 2) {
    return { kind: 'unsupported' }
  }

  const { command, args } = markerHelperCommand(op, scriptPath, options)

  try {
    const result = await (options.spawn ?? defaultSpawn)(command, args, {
      env: { ...process.env, HERMES_HOME: options.hermesHome },
      timeout: options.timeoutMs ?? MARKER_HELPER_TIMEOUT_MS,
      windowsHide: true
    })

    const verdict: MarkerHelperVerdict = result.code === 0 ? parseMarkerHelperVerdict(result.stdout) : { kind: 'error' }

    return OP_VERDICTS[op].has(verdict.kind) || verdict.kind === 'error' ? verdict : { kind: 'error' }
  } catch {
    return { kind: 'error' }
  }
}
