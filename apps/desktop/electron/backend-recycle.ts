/**
 * Recycle a Desktop-owned backend after a code-skew 503.
 *
 * Closing the local tunnel/child is not enough for SSH: `serve --isolated`
 * detaches with setsid/nohup, so a reconnect would reuse the still-alive
 * stale process via the lockfile. Kill the owned remote serve first (while
 * the SSH channel can still exec), then tear down the local child — the
 * same order as connection apply (#97046, #91668).
 */

import type {
  BackendRestartCapability,
  BackendRestartTarget,
  BackendRestartUnsupportedReason
} from './backend-restart-contract'
import { type ProfileRouteOptions, resolveProfileBackendRoute } from './connection-config'
import { type ConnectionRegistry, resolveRegistryLocalRoute } from './connection-registry'

export type BackendRecycleResult =
  | { status: 'recycled'; connectionId: string; profile: string }
  | { status: 'unsupported'; reason: BackendRestartUnsupportedReason }

interface OwnedLocalBackend {
  process: { killed: boolean; exitCode: number | null } | null
  connectionPromise: Promise<{ mode?: string; profile?: string; connectionId?: string }> | null
}

interface UnsupportedBackend {
  reason: BackendRestartUnsupportedReason
}

interface LocalBackendRoute {
  primary: boolean
  key: string
  entry: OwnedLocalBackend | undefined
  target: BackendRestartTarget
}

interface OwnedBackendSelection extends LocalBackendRoute {
  process: NonNullable<OwnedLocalBackend['process']>
  connectionPromise: NonNullable<OwnedLocalBackend['connectionPromise']>
}

export interface ScopedBackendRecycleState {
  registry: ConnectionRegistry
  routeOptions: ProfileRouteOptions
  primary: OwnedLocalBackend
  pool: ReadonlyMap<string, OwnedLocalBackend>
}

export interface ScopedBackendRecycleDeps {
  readState: (profile: string) => ScopedBackendRecycleState
  stopPrimary: () => Promise<void>
  stopPool: (key: string) => Promise<void>
}

function isBackendRestartTarget(value: unknown): value is BackendRestartTarget {
  if (!value || typeof value !== 'object') {
    return false
  }

  const { connectionId, profile } = value as BackendRestartTarget

  return (
    typeof connectionId === 'string' &&
    Boolean(connectionId) &&
    connectionId === connectionId.trim() &&
    typeof profile === 'string' &&
    /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(profile)
  )
}

/** Explicit registry targets only: never fall back to the ambient primary. */
export function createScopedBackendRecycler(deps: ScopedBackendRecycleDeps) {
  function selectRoute(target: unknown): UnsupportedBackend | LocalBackendRoute {
    if (!isBackendRestartTarget(target)) {
      return { reason: 'invalid-target' as const }
    }

    // IPC values are copied, but keep the internal contract safe for callers too.
    target = { connectionId: target.connectionId, profile: target.profile }
    const requested = target as BackendRestartTarget
    const state = deps.readState(requested.profile)
    const source = state.registry.connections.find(connection => connection.id === requested.connectionId)

    if (!source) {
      return { reason: 'unknown-connection' as const }
    }

    if (source.kind === 'ssh') {
      return { reason: 'ssh-ownership-unverified' as const }
    }

    if (source.kind !== 'local') {
      return { reason: 'externally-managed' as const }
    }

    // The local registry entry can be a v1 per-profile remote alias. Never
    // operate on that bare-name slot, even if an old local child remains there.
    if (state.routeOptions.profileRemoteOverride) {
      return { reason: 'not-owned' as const }
    }

    const local = resolveRegistryLocalRoute(requested.profile, state.routeOptions)

    const primary =
      local.delegate && resolveProfileBackendRoute(requested.profile, state.routeOptions).backend === 'primary'

    const entry = primary ? state.primary : state.pool.get(local.poolKey)

    if (
      primary &&
      (state.routeOptions.primaryRemoteActive || state.routeOptions.primaryProfile !== requested.profile)
    ) {
      return { reason: 'not-owned' as const }
    }

    return { primary, key: local.poolKey, entry, target: requested }
  }

  function select(target: unknown): UnsupportedBackend | OwnedBackendSelection {
    const route = selectRoute(target)

    if ('reason' in route) {
      return { reason: route.reason }
    }

    const { entry } = route

    if (!entry?.process || entry.process.killed || entry.process.exitCode !== null || !entry.connectionPromise) {
      return { reason: 'not-owned' as const }
    }

    return { ...route, process: entry.process, connectionPromise: entry.connectionPromise }
  }

  async function inspect(target: unknown, recycle: boolean): Promise<BackendRestartCapability | BackendRecycleResult> {
    const result = select(target)

    const unsupported = (reason: BackendRestartUnsupportedReason) =>
      recycle ? { status: 'unsupported' as const, reason } : { supported: false as const, reason }

    if ('reason' in result) {
      return unsupported(result.reason)
    }

    const descriptor = await result.connectionPromise

    if (
      descriptor.mode !== 'local' ||
      descriptor.profile !== result.target.profile ||
      ('connectionId' in descriptor && descriptor.connectionId !== result.target.connectionId)
    ) {
      return unsupported('not-owned')
    }

    // No async gap between this ownership check and the synchronous eviction
    // inside the stop dependency. Never let a late descriptor stop its successor.
    const current = select(result.target)

    if (
      'reason' in current ||
      current.primary !== result.primary ||
      current.key !== result.key ||
      current.process !== result.process ||
      current.connectionPromise !== result.connectionPromise
    ) {
      return unsupported('target-changed')
    }

    if (!recycle) {
      return { supported: true }
    }

    if (result.primary) {
      await deps.stopPrimary()
    } else {
      await deps.stopPool(result.key)
    }

    return { status: 'recycled', ...result.target }
  }

  return {
    capability: (target: unknown) => inspect(target, false) as Promise<BackendRestartCapability>,
    recycle: (target: unknown) => inspect(target, true) as Promise<BackendRecycleResult>,
    async restart(target: unknown, startLocal: ScopedBackendRestartDeps['startLocal']) {
      const selected = select(target)

      if ('reason' in selected) {
        throw new Error(`Backend restart unsupported: ${selected.reason}`)
      }

      const result = (await inspect(target, true)) as BackendRecycleResult

      if (result.status === 'unsupported') {
        throw new Error(`Backend restart unsupported: ${result.reason}`)
      }

      const current = selectRoute(selected.target)

      if ('reason' in current || current.primary !== selected.primary || current.key !== selected.key) {
        throw new Error('Backend restart unsupported: target-changed')
      }

      return startLocal(selected.target, { primary: selected.primary, key: selected.key })
    }
  }
}

export interface ScopedBackendRestartDeps extends ScopedBackendRecycleDeps {
  /** Must start locally in this captured slot, without consulting remote routing. */
  startLocal: (target: BackendRestartTarget, slot: { primary: boolean; key: string }) => Promise<unknown>
}

export function registerScopedBackendRecycleIpc(
  ipc: { handle: (channel: string, handler: (_event: unknown, target: unknown) => unknown) => void },
  deps: ScopedBackendRestartDeps
): void {
  const recycler = createScopedBackendRecycler(deps)

  ipc.handle('hermes:backend:restart-capability', (_event, target) => recycler.capability(target))
  ipc.handle('hermes:backend:restart-for', (_event, target) => recycler.restart(target, deps.startLocal))
}

export type RecycleOwnedBackendTarget = 'pool' | 'primary'

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
