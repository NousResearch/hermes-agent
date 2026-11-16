/**
 * Recycle the backend behind a code-skew 503 ("Restart required" after `hermes update`).
 *
 * Closing the local tunnel/child is not enough for SSH: `serve --isolated`
 * detaches with setsid/nohup, so a reconnect would reuse the still-alive
 * stale process via the lockfile. Kill the owned remote serve first (while
 * the SSH channel can still exec), then tear down the local child — the
 * same order as connection apply (#97046, #91668).
 *
 * A LOCAL profile attaches to the per-host gateway (`hermes gateway ensure`),
 * which Desktop never owns: dropping its descriptor only re-attached the same
 * stale process. `restartLocalGateway` runs a supervised
 * `hermes --profile P gateway restart` (a no-op for a remote profile) before
 * the descriptor drop, so the next ensure dials the replacement.
 */

export type RecycleOwnedBackendTarget = 'pool' | 'primary'

export interface RecycleOwnedBackendDeps {
  notifyApplied: () => void
  primaryProfile: string
  profile?: null | string
  restartLocalGateway: (profile: string) => Promise<void>
  teardownPool: (profile: string) => Promise<void>
  teardownPrimary: () => Promise<void>
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

  if (target === 'primary') {
    await deps.teardownSsh('')
    await deps.restartLocalGateway(deps.primaryProfile)
    await deps.teardownPrimary()
    deps.notifyApplied()

    return target
  }

  await deps.teardownSsh(profile)
  await deps.restartLocalGateway(profile)
  await deps.teardownPool(profile)

  return target
}
