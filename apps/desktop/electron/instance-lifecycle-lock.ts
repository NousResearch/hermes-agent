/**
 * instance-lifecycle-lock.ts
 *
 * Pure, no-electron-import gate for #107671: Chromium releases Electron's
 * single-instance lock when quit *starts*, but `before-quit` + async backend
 * / SSH teardown keeps the old Node process alive for seconds to ~30s. A new
 * `.desktop` launch can win the lock and become a second GNOME-visible app
 * while the prior PID is still in the session's START/STOP state machine
 * (`shell_app_dispose` assertion → session crash).
 *
 * `requestSingleInstanceLock` still covers the live-primary case. This helper
 * covers the overlapping-teardown case: after THIS process won the Electron
 * lock, park until the userData pidfile holder is gone — before createWindow
 * / startHermes — then claim the pidfile. Timeout while the holder is still
 * alive exits as a secondary (same as losing the lock). Never kill the prior
 * PID (#87295 reapOrphans footgun). Fail open on missing / stale / unreadable
 * pidfiles so a crash cannot brick the next launch.
 *
 * The on-disk update marker (`update-marker.ts` / `update-gate.ts`) is a
 * different signal: it parks backend spawn while `.hermes-update-in-progress`
 * is live. After `hermes update` exits that marker is gone, but Electron can
 * still be mid-teardown. Do not overload it.
 */

import fs from 'fs'
import path from 'path'

import { isPidAlive as probePidAlive } from './update-marker'

export const INSTANCE_PID_FILENAME = 'hermes-desktop-instance.pid'
export const INSTANCE_LIFECYCLE_WAIT_MS = 45_000
export const INSTANCE_LIFECYCLE_POLL_MS = 250

export type LifecycleDecision = 'proceed' | 'exit-as-secondary'

export function parseInstancePid(raw: string | null | undefined): number | null {
  if (raw == null) {
    return null
  }

  const first = String(raw).split('\n')[0]?.trim() ?? ''

  if (!/^\d+$/.test(first)) {
    return null
  }

  const pid = Number.parseInt(first, 10)

  if (!Number.isInteger(pid) || pid <= 0) {
    return null
  }

  return pid
}

export function instancePidPath(userDataDir: string): string {
  return path.join(userDataDir, INSTANCE_PID_FILENAME)
}

/**
 * `second-instance` is registered at module load, before the pidfile wait
 * in whenReady. Electron can already be ready while mainWindow is still
 * null — ensureMainWindow would then createWindow / startHermes while the
 * foreign holder PID is still alive. Both bits must be true.
 */
export function lifecycleAllowsCreateWindow(electronReady: boolean, lifecycleReady: boolean): boolean {
  return electronReady && lifecycleReady
}

export async function awaitPriorDesktopInstance(opts: {
  userDataDir: string
  selfPid: number
  timeoutMs?: number
  pollMs?: number
  now?: () => number
  sleep?: (ms: number) => Promise<void>
  readFile?: (path: string) => string | null
  writeFile?: (path: string, contents: string) => void
  isPidAlive?: (pid: number) => boolean
  log?: (line: string) => void
}): Promise<LifecycleDecision> {
  const pidPath = instancePidPath(opts.userDataDir)
  const now = opts.now ?? Date.now
  const sleep = opts.sleep ?? ((ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms)))
  const readFile = opts.readFile ?? defaultReadFile
  const writeFile = opts.writeFile ?? defaultWriteFile
  const isPidAlive = opts.isPidAlive ?? ((pid: number) => probePidAlive(pid))
  const log = opts.log ?? (() => undefined)
  const timeoutMs = opts.timeoutMs ?? INSTANCE_LIFECYCLE_WAIT_MS
  const pollMs = opts.pollMs ?? INSTANCE_LIFECYCLE_POLL_MS

  const claimAsPrimary = (): LifecycleDecision => {
    try {
      writeFile(pidPath, `${opts.selfPid}\n`)
    } catch (err) {
      log(`[instance-lock] failed to write pidfile ${pidPath}: ${err}`)
    }

    return 'proceed'
  }

  let raw: string | null

  try {
    raw = readFile(pidPath)
  } catch (err) {
    log(`[instance-lock] failed to read pidfile ${pidPath}: ${err}`)

    return claimAsPrimary()
  }

  const holderPid = parseInstancePid(raw)

  // Missing / malformed / non-positive / self / dead → fail-open (cold start
  // or crash-stale). Never wait on a pid that is not a live foreign holder.
  if (holderPid == null || holderPid === opts.selfPid || !isPidAlive(holderPid)) {
    return claimAsPrimary()
  }

  log(`[instance-lock] prior desktop PID ${holderPid} still alive; waiting for teardown`)

  const deadline = now() + timeoutMs

  while (isPidAlive(holderPid) && now() < deadline) {
    await sleep(pollMs)
  }

  if (isPidAlive(holderPid)) {
    log(`[instance-lock] prior desktop PID ${holderPid} still alive after ${timeoutMs}ms; exiting as secondary`)

    return 'exit-as-secondary'
  }

  log(`[instance-lock] prior desktop PID ${holderPid} exited; claiming primary`)

  return claimAsPrimary()
}

function defaultReadFile(filePath: string): string | null {
  try {
    return fs.readFileSync(filePath, 'utf8')
  } catch (err: unknown) {
    const code = err && typeof err === 'object' && 'code' in err ? (err as { code?: string }).code : undefined

    if (code === 'ENOENT') {
      return null
    }

    throw err
  }
}

function defaultWriteFile(filePath: string, contents: string): void {
  fs.writeFileSync(filePath, contents, 'utf8')
}
