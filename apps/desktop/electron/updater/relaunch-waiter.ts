// Stage the waiter outside the package and await its old-version snapshot.
// A failed handshake cancels our child instead of leaving a delayed relaunch.

import { type ChildProcess, spawn as nodeSpawn } from 'node:child_process'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

export const RELAUNCH_WAITER_SCRIPT = 'update-relaunch-waiter.ps1'
export const RELAUNCH_WAITER_READY_FILENAME = 'ready.txt'
export const DEFAULT_RELAUNCH_WAITER_TIMEOUT_SECONDS = 900
export const DEFAULT_RELAUNCH_WAITER_HANDSHAKE_MS = 10_000

/** Absolute path — never resolved via inherited PATH (it can point into the package). */
export const POWERSHELL_PATH = path.win32.join(process.env.SystemRoot || process.env.SYSTEMROOT || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')

export interface RelaunchWaiterOptions {
  /** The quitting app's pid — the waiter waits for this process to exit. */
  processId: number
  /** Wall-clock ms the parent process started (PID-reuse guard). */
  processStartTimeMs: number
  /** The MSIX identity name of this install (e.g. NousResearch.HermesBundled). */
  identityName: string
  /** Absolute path to the waiter script inside the payload repo snapshot. */
  scriptPath: string
  timeoutSeconds?: number
}

export interface WaiterStaging {
  stageDir: string
  scriptPath: string
  readyFile: string
}

export type SpawnWaiter = (
  command: string,
  args: string[],
  opts: { detached: boolean; stdio: 'ignore'; windowsHide: boolean; cwd: string }
) => ChildProcess

export interface RelaunchWaiterDeps {
  spawn?: SpawnWaiter
  /** Handshake deadline in ms (tests shrink this). */
  handshakeTimeoutMs?: number
  /** Poll interval for the ready file (tests shrink this). */
  pollMs?: number
  cancelTimeoutMs?: number
}

/** Pure: the exact argv the waiter is spawned with. */
export function buildRelaunchWaiterArgs(options: RelaunchWaiterOptions, readyFile: string): string[] {
  return [
    '-NoProfile',
    '-NonInteractive',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    options.scriptPath,
    '-ProcessId',
    String(options.processId),
    '-ProcessStartTimeMs',
    String(options.processStartTimeMs),
    '-IdentityName',
    options.identityName,
    '-ReadyFile',
    readyFile,
    '-TimeoutSeconds',
    String(options.timeoutSeconds ?? DEFAULT_RELAUNCH_WAITER_TIMEOUT_SECONDS)
  ]
}

/** Stage outside the package so its replacement does not invalidate the waiter. */
async function stageRelaunchWaiter(options: RelaunchWaiterOptions): Promise<WaiterStaging | undefined> {
  let stageDir: string

  try {
    stageDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'hermes-relaunch-'))
  } catch {
    return undefined
  }

  const scriptPath = path.join(stageDir, RELAUNCH_WAITER_SCRIPT)
  const readyFile = path.join(stageDir, RELAUNCH_WAITER_READY_FILENAME)

  try {
    await fs.promises.copyFile(options.scriptPath, scriptPath)
  } catch (error) {
    try {
      await fs.promises.rm(stageDir, { recursive: true, force: true })
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], 'Relaunch waiter staging cleanup failed')
    }

    return undefined
  }

  return { stageDir, scriptPath, readyFile }
}

export interface RelaunchWaiterHandle {
  cancel: () => Promise<void>
}

/** Return ownership after readiness, or no handle after a safely stopped failure. */
export async function startRelaunchWaiter(
  options: RelaunchWaiterOptions,
  deps: RelaunchWaiterDeps = {}
): Promise<RelaunchWaiterHandle | undefined> {
  const spawn = deps.spawn ?? (nodeSpawn as unknown as SpawnWaiter)
  const handshakeTimeoutMs = deps.handshakeTimeoutMs ?? DEFAULT_RELAUNCH_WAITER_HANDSHAKE_MS
  const pollMs = deps.pollMs ?? 250
  const cancelTimeoutMs = deps.cancelTimeoutMs ?? 10_000
  const staging = await stageRelaunchWaiter(options)

  if (!staging) { return undefined }

  const cleanup = () => fs.promises.rm(staging.stageDir, { recursive: true, force: true })
  let child: ChildProcess

  try {
    child = spawn(POWERSHELL_PATH,
      buildRelaunchWaiterArgs({ ...options, scriptPath: staging.scriptPath }, staging.readyFile),
      { detached: true, stdio: 'ignore', windowsHide: true, cwd: staging.stageDir })
  } catch (error) {
    try {
      await cleanup()
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], 'Relaunch waiter spawn cleanup failed')
    }

    return undefined
  }

  let closed = false

  const closedPromise = new Promise<void>(resolve => {
    child.once('close', () => { closed = true; resolve() })
  })

  let cancellation: Promise<void> | undefined

  const cancel = (): Promise<void> => {
    cancellation ??= (async () => {
      if (!closed) {
        if (child.pid !== undefined) { child.kill() }

        let timeout: ReturnType<typeof setTimeout> | undefined

        try {
          await Promise.race([
            closedPromise,
            new Promise<never>((_, reject) => {
              timeout = setTimeout(() => reject(new Error('Relaunch waiter did not exit after cancellation')), cancelTimeoutMs)
            })
          ])
        } finally {
          clearTimeout(timeout)
        }
      }

      await cleanup()
    })()

    return cancellation
  }

  const ready = await new Promise<boolean>(resolve => {
    let settled = false
    let timer: ReturnType<typeof setTimeout>
    const deadline = Date.now() + handshakeTimeoutMs

    const finish = (ready: boolean) => {
      if (settled) { return }
      settled = true
      clearTimeout(timer)
      resolve(ready)
    }

    child.once('error', () => finish(false))
    child.once('close', () => finish(false))

    const check = () => {
      if (settled) { return }

      if (fs.existsSync(staging.readyFile)) {
        finish(true)
      } else if (Date.now() >= deadline) {
        finish(false)
      } else {
        timer = setTimeout(check, pollMs)
      }
    }

    check()
  })

  if (!ready) {
    await cancel()

    return undefined
  }

  child.unref()

  return { cancel }
}
