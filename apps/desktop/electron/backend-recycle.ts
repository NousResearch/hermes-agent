/**
 * Recycle a Desktop-owned backend after a code-skew 503.
 *
 * Closing the local tunnel/child is not enough for SSH: `serve --isolated`
 * detaches with setsid/nohup, so a reconnect would reuse the still-alive
 * stale process via the lockfile. Kill the owned remote serve first (while
 * the SSH channel can still exec), then tear down the local child — the
 * same order as connection apply (#97046, #91668).
 *
 * #101561: also reap a leftover local dashboard/serve that served the 503
 * when it is not the Electron-owned child. Never treat a remote pid number
 * as a local kill target without a hermes serve/dashboard cmdline match.
 */

import { backendCommandMatches } from './backend-ownership'

export function canReapSkewServingPid(
  pid: unknown,
  opts: { command: null | string; selfPid: number }
): boolean {
  if (!Number.isInteger(pid) || (pid as number) <= 1 || pid === opts.selfPid) {
    return false
  }

  return Boolean(opts.command) && backendCommandMatches(opts.command)
}

/** Loopback TCP port the picker is talking to, or null for remote URLs. */
export function localLoopbackListenPort(baseUrl: unknown): number | null {
  if (typeof baseUrl !== 'string' || !baseUrl.trim()) {
    return null
  }

  try {
    const url = new URL(baseUrl)
    const host = url.hostname.toLowerCase()

    if (host !== '127.0.0.1' && host !== 'localhost' && host !== '::1') {
      return null
    }

    const port = Number(url.port || (url.protocol === 'https:' ? 443 : 80))

    return Number.isInteger(port) && port > 0 ? port : null
  } catch {
    return null
  }
}

/** PIDs from `lsof -t` listen output. */
export function listenPidsFromLsofT(stdout: unknown): number[] {
  const seen = new Set<number>()

  for (const part of String(stdout ?? '').split(/[\s,]+/)) {
    const pid = Number(part)

    if (Number.isInteger(pid) && pid > 1 && !seen.has(pid)) {
      seen.add(pid)
    }
  }

  return [...seen]
}

export type RecycleOwnedBackendTarget = 'pool' | 'primary'

export interface RecycleOwnedBackendDeps {
  notifyApplied: () => void
  primaryProfile: string
  profile?: null | string
  /** Local leftover dashboard/serve that actually served the 503 (#101561). */
  servingPid?: null | number
  /** Loopback port the picker talks to — used when the leftover predates pid= in 503. */
  listenPort?: null | number
  teardownPool: (profile: string) => Promise<void>
  teardownPrimary: () => Promise<void>
  teardownServingPid?: (pid: number) => Promise<void>
  teardownListenPort?: (port: number) => Promise<void>
  teardownSsh: (profile: string) => Promise<void>
}

export function recycleOwnedBackendTarget(
  profile: null | string | undefined,
  primaryProfile: string
): RecycleOwnedBackendTarget {
  const key = String(profile ?? '').trim()

  return !key || key === primaryProfile ? 'primary' : 'pool'
}

export async function recycleOwnedBackend(deps: RecycleOwnedBackendDeps): Promise<RecycleOwnedBackendTarget> {
  const target = recycleOwnedBackendTarget(deps.profile, deps.primaryProfile)
  const profile = String(deps.profile ?? '').trim()
  const servingPid = Number(deps.servingPid)
  const listenPort = Number(deps.listenPort)
  const reapLeftover = async () => {
    try {
      if (Number.isInteger(servingPid) && servingPid > 1) {
        await deps.teardownServingPid?.(servingPid)
        return
      }

      if (Number.isInteger(listenPort) && listenPort > 0) {
        await deps.teardownListenPort?.(listenPort)
      }
    } catch {
      // Leftover reap must not skip owned-child recycle (#101561).
    }
  }

  if (target === 'primary') {
    await deps.teardownSsh('')
    await reapLeftover()
    await deps.teardownPrimary()
    deps.notifyApplied()

    return target
  }

  await deps.teardownSsh(profile)
  await reapLeftover()
  await deps.teardownPool(profile)

  return target
}
