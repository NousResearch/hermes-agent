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

import {
  assertConnectionOwner,
  assertLegacyConnectionOwner,
  type ProfileRouteOptions,
  resolveProfileBackendRoute
} from './connection-config'
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

interface RecyclePinnedBackendOwner {
  connectionId?: null | string
  connectionOwner?: ResolvedConnectionDescriptor
  profile?: null | string
  legacyConnection?: ResolvedConnectionDescriptor
}

interface RecyclePinnedBackendDeps {
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
  restartLocalGateway?: (profile: string) => Promise<void>
  notifyApplied: () => void
}

function resolvePinnedOwner(owner: RecyclePinnedBackendOwner, deps: RecyclePinnedBackendDeps) {
  const { connectionId } = owner
  // A custom HERMES_HOME has no real profile name. Electron's primary-profile
  // route still identifies its owned process without sending a synthetic
  // `custom` profile back through REST.
  const profile = owner.profile?.trim() || deps.routeOptions.primaryProfile?.trim()
  const legacy = connectionId == null ? owner.legacyConnection : undefined

  if (
    (!legacy &&
      (typeof connectionId !== 'string' || !connectionId.trim() || !Object.hasOwn(owner, 'connectionOwner'))) ||
    !profile
  ) {
    throw new Error('Restart requires an exact connection and profile.')
  }

  const source = deps.registry.connections.find(row => row.id === connectionId)

  if (!legacy && (!source || (source.kind !== 'local' && source.kind !== 'ssh'))) {
    throw new Error('This backend is not managed by Desktop. Restart it on its host.')
  }

  return { connectionId, legacy, profile, source }
}

async function resolvePinnedRoute(
  connectionId: null | string | undefined,
  legacy: ResolvedConnectionDescriptor | undefined,
  profile: string,
  source: RegistryConnection | undefined,
  deps: RecyclePinnedBackendDeps
) {
  let key = backendScopeKey(connectionId, profile)
  let primary = false

  if (legacy) {
    return {
      key: profile,
      primary: resolveProfileBackendRoute(profile, deps.routeOptions).backend === 'primary'
    }
  }

  if (source?.kind === 'local') {
    const local = resolveRegistryLocalRoute(profile, deps.routeOptions)

    return {
      key: local.poolKey,
      primary: local.delegate && resolveProfileBackendRoute(profile, deps.routeOptions).backend === 'primary'
    }
  }

  if (!source) {
    throw new Error('This backend is not managed by Desktop. Restart it on its host.')
  }

  const legacyPrimary = resolveProfileBackendRoute(profile, deps.routeOptions).backend === 'primary'
  const legacyEntry = deps.pool.get(profile)
  const legacyPromise = legacyPrimary ? deps.primaryPromise() : legacyEntry?.connectionPromise

  if (legacyPromise) {
    const reused = await reuseMatchingPrimarySshBackend({
      connectionId: connectionId!,
      profile,
      source,
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

  return { key, primary }
}

function assertPinnedDescriptor(
  owner: RecyclePinnedBackendOwner,
  connectionId: null | string | undefined,
  legacy: ResolvedConnectionDescriptor | undefined,
  source: RegistryConnection | undefined,
  descriptor: ResolvedConnectionDescriptor,
  ssh: { remotePlatform?: string } | undefined,
  registry: ConnectionRegistry
) {
  if (legacy) {
    assertLegacyConnectionOwner(legacy, descriptor)
  } else {
    assertConnectionOwner(owner.connectionOwner, descriptor)
  }

  if (source?.kind === 'ssh' && resolvedConnectionId(registry, descriptor) !== connectionId) {
    throw new Error('The running backend does not belong to the requested connection.')
  }

  if (descriptor.mode !== 'local' && (descriptor.remoteKind !== 'ssh' || !ssh || ssh.remotePlatform === 'Windows')) {
    throw new Error('This backend is not managed by Desktop. Restart it on its host.')
  }
}

/** Resolve the same registry/local routing ladder as requests, without booting anything. */
export async function recyclePinnedBackend(
  owner: RecyclePinnedBackendOwner,
  deps: RecyclePinnedBackendDeps
): Promise<RecycleOwnedBackendTarget> {
  const { connectionId, legacy, profile, source } = resolvePinnedOwner(owner, deps)
  const { key, primary } = await resolvePinnedRoute(connectionId, legacy, profile, source, deps)

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

  assertPinnedDescriptor(owner, connectionId, legacy, source, descriptor, ssh, deps.registry)

  // ponytail: retain the existing ordered SSH/child teardown and pool recovery.
  return recycleOwnedBackend({
    profile: primary ? profile : key,
    primaryProfile: primary ? profile : '',
    notifyApplied: deps.notifyApplied,
    restartLocalGateway: deps.restartLocalGateway ?? (async () => undefined),
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
