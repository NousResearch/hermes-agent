/**
 * Wiring for the federated session index and its background enumerator
 * (federated-session-index.ts, registry-session-enumerator.ts).
 *
 * main.ts owns the machine — the backend pool, the dial claims, the registry
 * file — so everything it owns is injected; this module only decides what an
 * enumeration pass reads and what the unified list may backfill from the index.
 * That keeps the Electron main file from growing and lets both decisions be
 * unit-tested without Electron.
 */

import {
  coldRegistryConnectionIds,
  FederatedSessionIndexStore,
  indexedRowsForColdConnections,
  type IndexedSessionRow
} from './federated-session-index'
import { fetchRegistrySessionRows, type RegistrySessionSource } from './profile-session-routing'
import {
  ENUMERATION_FETCH_TIMEOUT_MS,
  ENUMERATION_PAGE_LIMIT,
  RegistrySessionEnumerator
} from './registry-session-enumerator'

export interface RegistrySessionEnumerationDeps {
  /**
   * Resolve (dialing if needed) the backend descriptor for a connection's
   * default scope, through the claim-guarded path an interactive open uses
   * (#90812), so a pass coalesces with a renderer dial for the same scope
   * instead of opening a second tunnel.
   */
  dial: (connectionId: string) => Promise<unknown>
  getJson: (descriptor: unknown, path: string, timeoutMs: number) => Promise<unknown>
  readRegistry: () => { connections: ReadonlyArray<{ id: string; kind: string }>; primary: string }
  userDataDir: () => string
}

export function createRegistrySessionEnumeration(deps: RegistrySessionEnumerationDeps) {
  let store: FederatedSessionIndexStore | null = null

  // Lazy: userData is only asked for once there is something to read or write.
  const index = (): FederatedSessionIndexStore => (store ??= new FederatedSessionIndexStore(deps.userDataDir()))

  /**
   * One connection's cross-profile session rows. Null on any failure —
   * including an EMPTY read that carries no total, which a dead or too-old
   * host is indistinguishable from — so a failed read can never clear a
   * connection's remembered rows. Reads omit include_hidden, same as the live
   * merge, so Bot Mode's hidden canonical chats stay out.
   */
  async function fetchSessions(connectionId: string): Promise<{ sessions: unknown[] } | null> {
    try {
      const kind = deps.readRegistry().connections.find(connection => connection.id === connectionId)?.kind

      if (!kind) {
        return null
      }

      const descriptor = await deps.dial(connectionId)

      // An ssh connection runs one backend per profile (this one is its
      // default); a shared remote/cloud host serves every profile.
      const source: RegistrySessionSource = {
        backends: [{ descriptor, profileLabel: kind === 'ssh' ? 'default' : null }],
        connectionId,
        kind
      }

      const params = new URLSearchParams({
        archived: 'exclude',
        limit: String(ENUMERATION_PAGE_LIMIT),
        offset: '0',
        order: 'recent'
      })

      const reportedTotals = new Map<string, number>()

      const sessions = await fetchRegistrySessionRows(
        [source],
        params,
        (backend, path) => deps.getJson(backend, path, ENUMERATION_FETCH_TIMEOUT_MS),
        reportedTotals
      )

      return sessions.length === 0 && reportedTotals.size === 0 ? null : { sessions }
    } catch {
      // Dead host, auth failure, dial timeout — a pass is best-effort.
      return null
    }
  }

  const enumerator = new RegistrySessionEnumerator({
    cancel: handle => clearTimeout(handle as ReturnType<typeof setTimeout>),
    connections: () => {
      const registry = deps.readRegistry()

      return registry.connections.map(connection => ({
        connectionId: connection.id,
        kind: connection.kind,
        primary: connection.id === registry.primary
      }))
    },
    fetchSessions,
    remember: (connectionId, sessions) => index().remember(connectionId, { sessions }),
    retain: connectionIds => index().retainConnections(connectionIds),
    schedule: (fn, ms) => setTimeout(fn, ms)
  })

  return {
    /**
     * The rows the unified list may backfill for one read: indexed rows of
     * registry connections that have no readable backend this refresh,
     * narrowed by the request's own filters.
     */
    coldRows(searchParams: URLSearchParams, readableConnectionIds: ReadonlySet<string>): IndexedSessionRow[] {
      const registry = deps.readRegistry()

      return indexedRowsForColdConnections(
        index().load(),
        coldRegistryConnectionIds(registry.connections, registry.primary, readableConnectionIds),
        searchParams
      )
    },
    fetchSessions,
    start: () => enumerator.start(),
    stop: () => enumerator.stop()
  }
}
