import { bindSessionRowsToRoute } from './connection-config'
import {
  backendScopePrefix,
  type ConnectionRegistry,
  registrySourceOwnsPrimaryBackend,
  type ResolvedConnectionDescriptor
} from './connection-registry'
import {
  applyRemoteProfileSessionOutcomes,
  composeUnifiedSessionResponse,
  fetchPrimaryProfileSessions,
  mergeRegistrySessionRows,
  type RegistrySessionSource,
  settleRemoteProfileSessions,
  tagPrimaryAggregateRows
} from './profile-session-routing'
import { collectRegistrySessionSources } from './registry-session-sources'
import { sharedSshProfiles } from './ssh-shared-profile-backend'

interface BackendPoolEntry {
  connectionPromise: Promise<ResolvedConnectionDescriptor>
}

export function createPooledRegistrySessionSources(deps: {
  backendIdForConnection: (connectionId: string, descriptor: ResolvedConnectionDescriptor) => Promise<null | string>
  backendPool: Map<string, BackendPoolEntry>
  primaryBackend: () => null | Promise<ResolvedConnectionDescriptor>
  readRegistry: () => ConnectionRegistry
  sharedProfiles?: (connectionId: string, backends: RegistrySessionSource['backends']) => string[] | undefined
}) {
  return async (includeLocal = false): Promise<RegistrySessionSource[]> => {
    return collectRegistrySessionSources(
      {
        connections: deps.readRegistry().connections,
        installId: async (connectionId, descriptor) =>
          (await deps.backendIdForConnection(connectionId, descriptor as ResolvedConnectionDescriptor)) ?? undefined,
        pool: deps.backendPool.entries(),
        reusedPrimary: async connectionId => {
          const pending = deps.primaryBackend()

          if (!pending) {
            return null
          }

          try {
            const descriptor = await Promise.race([
              pending,
              new Promise<never>((_, reject) => setTimeout(() => reject(new Error('pending')), 2_000))
            ])

            return registrySourceOwnsPrimaryBackend(deps.readRegistry(), connectionId, descriptor) ? descriptor : null
          } catch {
            return null
          }
        },
        scopePrefix: backendScopePrefix,
        sharedProfiles: deps.sharedProfiles ?? sharedSshProfiles
      },
      includeLocal
    )
  }
}

export function createRemoteProfileSessionMerger(deps: {
  fetchJsonForProfile: (profile: null | string, path: string) => Promise<unknown>
  getJsonForBackend: (descriptor: unknown, path: string, options: { timeoutMs: number }) => Promise<unknown>
  pooledRegistrySessionSources: (includeLocal?: boolean) => Promise<RegistrySessionSource[]>
  primaryBackend: () => Promise<unknown>
  primaryBackendId: () => Promise<null | string>
  remoteSessionList: (profile: string, params: URLSearchParams) => Promise<unknown>
}) {
  return async (
    searchParams: URLSearchParams,
    remoteProfiles: string[],
    registrySourcesOverride: RegistrySessionSource[] | null = null,
    pinnedConnectionId: null | string = null
  ) => {
    const limit = Math.max(1, Number(searchParams.get('limit')) || 20)
    const offset = Math.max(0, Number(searchParams.get('offset')) || 0)
    const order = searchParams.get('order') === 'created' ? 'started_at' : 'last_active'
    const base = (await fetchPrimaryProfileSessions(searchParams, deps.fetchJsonForProfile, { prefixForMerge: true })) as any

    if (pinnedConnectionId) {
      await tagPrimaryAggregateRows(base, deps.primaryBackend, pinnedConnectionId)
    }

    const remoteParams = new URLSearchParams(searchParams)
    remoteParams.set('limit', String(limit + offset))
    remoteParams.set('offset', '0')

    const remoteSet = new Set(remoteProfiles)

    const merged: any[] = (Array.isArray(base?.sessions) ? base.sessions : []).filter(
      (row: any) => !remoteSet.has(row?.profile)
    )

    const profileTotals = { ...(base.profile_totals || {}) }
    let total = (Number(base.total) || 0) - remoteProfiles.reduce((count, profile) => count + (profileTotals[profile] || 0), 0)

    const remoteOutcomes = await settleRemoteProfileSessions(remoteProfiles, name =>
      deps.remoteSessionList(name, remoteParams)
    )

    const { total: remoteTotal, errors: remoteErrors } = applyRemoteProfileSessionOutcomes(
      remoteOutcomes,
      merged,
      profileTotals,
      total
    )

    total = remoteTotal

    const registrySources = registrySourcesOverride || (await deps.pooledRegistrySessionSources())

    if (registrySources.length) {
      total += await mergeRegistrySessionRows(
        registrySources,
        remoteParams,
        (descriptor, path) => deps.getJsonForBackend(descriptor, path, { timeoutMs: 10_000 }),
        merged,
        profileTotals,
        { primaryBackendId: await deps.primaryBackendId(), unknownPrimaryProfiles: remoteProfiles },
        bindSessionRowsToRoute
      )
    }

    const recency = (row: any) => row?.[order] ?? row?.started_at ?? 0
    merged.sort((a: any, b: any) => recency(b) - recency(a))

    return composeUnifiedSessionResponse(base, merged, offset, limit, total, profileTotals, remoteErrors, {
      archivedOnly: searchParams.get('archived') === 'only'
    })
  }
}
