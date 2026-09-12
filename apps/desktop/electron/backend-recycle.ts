/**
 * Recycle a Desktop-owned backend after a code-skew 503.
 *
 * Closing the local tunnel/child is not enough for SSH: `serve --isolated`
 * detaches with setsid/nohup, so a reconnect would reuse the still-alive
 * stale process via the lockfile. Kill the owned remote serve first (while
 * the SSH channel can still exec), then tear down the local child — the
 * same order as connection apply (#97046, #91668).
 */

import { assertLegacyConnectionOwner, type ProfileRouteOptions, resolveProfileBackendRoute } from './connection-config'
import {
  backendScopeKey,
  type ConnectionRegistry,
  type RegistryConnection,
  type ResolvedConnectionDescriptor,
  resolvedConnectionId,
  resolveRegistryLocalRoute,
  reuseMatchingPrimarySshBackend
} from './connection-registry'

export type RecycleOwnedBackendTarget = 'pool' | 'primary'

/** Resolve the same registry/local routing ladder as requests, without booting anything. */
export async function recyclePinnedBackend(
  owner: { connectionId?: null | string; profile?: null | string; legacyConnection?: ResolvedConnectionDescriptor },
  deps: {
    registry: ConnectionRegistry
    routeOptions: ProfileRouteOptions
    primarySshKey: string
    effectiveSshFingerprint: (source: RegistryConnection) => Promise<string>
    primaryPromise: () => Promise<ResolvedConnectionDescriptor> | null
    pool: Map<string, { connectionPromise: Promise<ResolvedConnectionDescriptor> }>
    sshState: (key: string) => { remotePlatform?: string } | undefined
    teardownSsh: (key: string) => Promise<void>
    teardownPool: (key: string) => Promise<void>
    teardownPrimary: () => Promise<void>
    notifyApplied: () => void
  }
): Promise<RecycleOwnedBackendTarget> {
  const { connectionId, profile } = owner

  const legacy = connectionId == null && owner.legacyConnection

  if (
    (!legacy && (typeof connectionId !== 'string' || !connectionId.trim())) ||
    typeof profile !== 'string' ||
    !profile.trim()
  ) {
    throw new Error('Restart requires an exact connection and profile.')
  }

  const source = deps.registry.connections.find(row => row.id === connectionId)

  if (!legacy && (!source || (source.kind !== 'local' && source.kind !== 'ssh'))) {
    throw new Error('This backend is not managed by Desktop. Restart it on its host.')
  }

  let key = backendScopeKey(connectionId, profile)
  let primary = false

  if (legacy) {
    primary = resolveProfileBackendRoute(profile, deps.routeOptions).backend === 'primary'
    key = profile
  } else if (source!.kind === 'local') {
    const local = resolveRegistryLocalRoute(profile, deps.routeOptions)
    key = local.poolKey
    primary = local.delegate && resolveProfileBackendRoute(profile, deps.routeOptions).backend === 'primary'
  } else {
    const legacyPrimary = resolveProfileBackendRoute(profile, deps.routeOptions).backend === 'primary'
    const legacyEntry = deps.pool.get(profile)
    const legacyPromise = legacyPrimary ? deps.primaryPromise() : legacyEntry?.connectionPromise

    if (legacyPromise) {
      const reused = await reuseMatchingPrimarySshBackend({
        connectionId: connectionId!,
        profile,
        source: source!,
        registry: deps.registry,
        effectiveFingerprint: deps.effectiveSshFingerprint,
        ensurePrimary: () => legacyPromise
      })

      if (reused) {
        if (legacyPrimary ? deps.primaryPromise() !== legacyPromise : deps.pool.get(profile) !== legacyEntry) {
          throw new Error('Backend changed during restart. Retry for the current owner.')
        }

        primary = legacyPrimary
        key = profile
      }
    }
  }

  const entry = primary ? null : deps.pool.get(key)
  const promise = primary ? deps.primaryPromise() : entry?.connectionPromise
  const sshKey = primary ? deps.primarySshKey : key
  const ssh = deps.sshState(sshKey)

  const assertCurrent = () => {
    if (!promise || (primary ? deps.primaryPromise() !== promise : deps.pool.get(key) !== entry)) {
      throw new Error('Backend changed during restart. Retry for the current owner.')
    }
  }

  assertCurrent()
  const descriptor = await promise!
  assertCurrent()

  if (legacy) {
    assertLegacyConnectionOwner(legacy, descriptor)
  }

  if (source?.kind === 'ssh' && resolvedConnectionId(deps.registry, descriptor) !== connectionId) {
    throw new Error('The running backend does not belong to the requested connection.')
  }

  if (descriptor.mode !== 'local' && (descriptor.remoteKind !== 'ssh' || !ssh || ssh.remotePlatform === 'Windows')) {
    throw new Error('This backend is not managed by Desktop. Restart it on its host.')
  }

  // ponytail: retain the existing ordered SSH/child teardown and pool recovery.
  return recycleOwnedBackend({
    profile: primary ? '' : key,
    primaryProfile: '',
    notifyApplied: deps.notifyApplied,
    teardownSsh: async () => {
      assertCurrent()

      if (descriptor.mode === 'remote') {
        if (deps.sshState(sshKey) !== ssh) {
          throw new Error('SSH owner changed during restart.')
        }

        await deps.teardownSsh(sshKey)
      }
    },
    teardownPool: async () => {
      assertCurrent()
      await deps.teardownPool(key)
    },
    teardownPrimary: async () => {
      assertCurrent()
      await deps.teardownPrimary()
    }
  })
}

export interface RecycleOwnedBackendDeps {
  notifyApplied: () => void
  primaryProfile: string
  profile?: null | string
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
    await deps.teardownPrimary()
    deps.notifyApplied()

    return target
  }

  await deps.teardownSsh(profile)
  await deps.teardownPool(profile)

  return target
}
