/**
 * Every CONNECTED registry gateway as a unified-session-list source: resolved
 * descriptors straight from the backend pool, never dialing. SSH sources
 * contribute one backend per pooled (connection, profile) scope; remote/cloud
 * sources are one shared host (any pooled scope's descriptor serves the
 * cross-profile read). The primary local connection is excluded for the legacy
 * unpinned path — the primary aggregate carries local rows there. A
 * registry-pinned aggregate opts in so forced-local backends remain visible
 * when the legacy primary is remote.
 *
 * Extracted from main.ts (behaviour unchanged) so the pool, registry and
 * identity probe are injected and the selection logic is unit-testable.
 */

import { backendScopePrefix } from './connection-registry'
import type { RegistrySessionSource } from './profile-session-routing'
import { boundedInstallId } from './registry-backend-identity'

/** How long a still-dialing pooled backend may hold up a sidebar refresh. */
export const PENDING_BACKEND_BUDGET_MS = 2_000

export interface PooledRegistrySessionSourceDeps {
  readRegistry: () => { connections: ReadonlyArray<{ id: string; kind: string }> }
  /** The main process's backend pool: scope key → entry. */
  pool: ReadonlyMap<string, { connectionPromise?: unknown }>
  /** The connection's gateway install_id, probed against a resolved descriptor. */
  probeInstallId: (connectionId: string, descriptor: unknown) => Promise<string | undefined>
}

export function createPooledRegistrySessionSources(deps: PooledRegistrySessionSourceDeps) {
  return async function pooledRegistrySessionSources(includeLocal = false): Promise<RegistrySessionSource[]> {
    const registry = deps.readRegistry()
    const sources: RegistrySessionSource[] = []

    for (const connection of registry.connections) {
      if (connection.kind === 'local' && !includeLocal) {
        continue
      }

      const prefix = backendScopePrefix(connection.id)

      const pooled = [...deps.pool.entries()].filter(
        ([key, entry]) => key.startsWith(prefix) && entry.connectionPromise
      )

      if (pooled.length === 0) {
        continue
      }

      const backends: Array<{ descriptor: unknown; profileLabel: null | string }> = []

      const perProfile = connection.kind === 'ssh' || (includeLocal && connection.kind === 'local')

      for (const [key, entry] of perProfile ? pooled : pooled.slice(0, 1)) {
        let timer: ReturnType<typeof setTimeout> | undefined

        try {
          // Already-resolved for a connected backend; a still-dialing entry is
          // skipped via the timeout guard rather than blocking the sidebar.
          const descriptor = await Promise.race([
            entry.connectionPromise,
            new Promise((_, reject) => {
              timer = setTimeout(() => reject(new Error('pending')), PENDING_BACKEND_BUDGET_MS)
            })
          ])

          backends.push({
            descriptor,
            profileLabel: perProfile ? key.slice(prefix.length) || 'default' : null
          })
        } catch {
          // Dead or still-connecting backend — contributes nothing this refresh.
        } finally {
          clearTimeout(timer)
        }
      }

      if (backends.length) {
        const backendId = await boundedInstallId(deps.probeInstallId(connection.id, backends[0].descriptor))
        sources.push({ backendId, backends, connectionId: connection.id, kind: connection.kind })
      }
    }

    return sources
  }
}
