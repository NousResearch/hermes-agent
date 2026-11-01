import { bindSessionRowsToRoute } from './connection-config'
import { backendScopePrefix, type ConnectionRegistry, type ResolvedConnectionDescriptor } from './connection-registry'
import {
  applyRemoteProfileSessionOutcomes,
  composeUnifiedSessionResponse,
  fetchPrimaryProfileSessions,
  mergeRegistrySessionRows,
  type RegistrySessionSource,
  settleRemoteProfileSessions
} from './profile-session-routing'

interface BackendPoolEntry {
  connectionPromise: Promise<ResolvedConnectionDescriptor>
}

export function createPooledRegistrySessionSources(deps: {
  backendIdForConnection: (connectionId: string, descriptor: ResolvedConnectionDescriptor) => Promise<null | string>
  backendPool: Map<string, BackendPoolEntry>
  readRegistry: () => ConnectionRegistry
}) {
  return async (includeLocal = false): Promise<RegistrySessionSource[]> => {
    const sources: RegistrySessionSource[] = []

    for (const connection of deps.readRegistry().connections) {
      if (connection.kind === 'local' && !includeLocal) {
        continue
      }

      const prefix = backendScopePrefix(connection.id)

      const pooled = [...deps.backendPool.entries()].filter(
        ([key, entry]) => key.startsWith(prefix) && entry.connectionPromise
      )

      if (pooled.length === 0) {
        continue
      }

      const backends: RegistrySessionSource['backends'] = []
      const perProfile = connection.kind === 'ssh' || (includeLocal && connection.kind === 'local')

      for (const [key, entry] of perProfile ? pooled : pooled.slice(0, 1)) {
        try {
          const descriptor = await Promise.race([
            entry.connectionPromise,
            new Promise<never>((_, reject) => setTimeout(() => reject(new Error('pending')), 2_000))
          ])

          backends.push({
            descriptor,
            profileLabel: perProfile ? key.slice(prefix.length) || 'default' : null
          })
        } catch {
          // Dead or still-connecting backend contributes nothing this refresh.
        }
      }

      if (backends.length) {
        const backendId = await deps.backendIdForConnection(connection.id, backends[0].descriptor)
        sources.push({ backendId, backends, connectionId: connection.id, kind: connection.kind })
      }
    }

    return sources
  }
}

export function createRemoteProfileSessionMerger(deps: {
  fetchJsonForProfile: (profile: null | string, path: string) => Promise<unknown>
  getJsonForBackend: (descriptor: unknown, path: string, options: { timeoutMs: number }) => Promise<unknown>
  pooledRegistrySessionSources: (includeLocal?: boolean) => Promise<RegistrySessionSource[]>
  primaryBackendId: () => Promise<null | string>
  remoteSessionList: (profile: string, params: URLSearchParams) => Promise<unknown>
}) {
  return async (
    searchParams: URLSearchParams,
    remoteProfiles: string[],
    registrySourcesOverride: RegistrySessionSource[] | null = null
  ) => {
    const limit = Math.max(1, Number(searchParams.get('limit')) || 20)
    const offset = Math.max(0, Number(searchParams.get('offset')) || 0)
    const order = searchParams.get('order') === 'created' ? 'started_at' : 'last_active'
    const base = (await fetchPrimaryProfileSessions(searchParams, deps.fetchJsonForProfile, { prefixForMerge: true })) as any
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
