/**
 * Every CONNECTED registry gateway as a session source for the unified
 * session list: resolved descriptors straight from the backend pool, never
 * dialing. SSH sources list every pooled (connection, profile) backend: the
 * default one reads every profile through the remote aggregate, the rest serve
 * older runtimes' per-profile lists; remote/cloud sources are one shared host.
 *
 * The SSH connection that IS the window's primary has no pooled default
 * backend: its registry routes reuse the primary descriptor (and named profiles
 * ride that one remote serve), so the primary descriptor stands in as the
 * source's default backend when it is already up. Without it a pinned
 * aggregate for that connection finds no source and falls back to a
 * single-gateway read.
 *
 * The primary local connection is excluded for the legacy unpinned path: the
 * primary aggregate carries local rows there. A registry-pinned aggregate opts
 * in so forced-local backends remain visible when the legacy primary is remote.
 */

import type { RegistrySessionSource } from './profile-session-routing'

type SessionBackend = RegistrySessionSource['backends'][number]

export interface RegistrySessionSourceDeps {
  connections: ReadonlyArray<{ id: string; kind: string }>
  pool: Iterable<[string, { connectionPromise?: null | Promise<unknown> }]>
  scopePrefix: (connectionId: string) => string
  /** The already-running primary descriptor when it serves this SSH connection
   *  without a pooled backend of its own; null otherwise. Never starts it. */
  reusedPrimary: (connectionId: string) => Promise<unknown>
  installId: (connectionId: string, descriptor: unknown) => Promise<string | undefined>
  /** SSH: named profiles the connection's default backend serves. */
  sharedProfiles: (connectionId: string, backends: SessionBackend[]) => string[] | undefined
  pendingMs?: number
}

export async function collectRegistrySessionSources(
  deps: RegistrySessionSourceDeps,
  includeLocal = false
): Promise<RegistrySessionSource[]> {
  const pool = [...deps.pool]
  const sources: RegistrySessionSource[] = []

  for (const connection of deps.connections) {
    if (connection.kind === 'local' && !includeLocal) {
      continue
    }

    const prefix = deps.scopePrefix(connection.id)
    const pooled = pool.filter(([key, entry]) => key.startsWith(prefix) && entry.connectionPromise)
    const perProfile = connection.kind === 'ssh' || (includeLocal && connection.kind === 'local')
    const backends: SessionBackend[] = []

    if (connection.kind === 'ssh' && !pooled.some(([key]) => (key.slice(prefix.length) || 'default') === 'default')) {
      const primary = await deps.reusedPrimary(connection.id)

      if (primary) {
        backends.push({ descriptor: primary, profileLabel: 'default' })
      }
    }

    for (const [key, entry] of perProfile ? pooled : pooled.slice(0, 1)) {
      try {
        // Already-resolved for a connected backend; a still-dialing entry is
        // skipped via the timeout guard rather than blocking the sidebar.
        const descriptor = await Promise.race([
          entry.connectionPromise,
          new Promise((_, reject) => setTimeout(() => reject(new Error('pending')), deps.pendingMs ?? 2_000))
        ])

        backends.push({ descriptor, profileLabel: perProfile ? key.slice(prefix.length) || 'default' : null })
      } catch {
        // Dead or still-connecting backend: contributes nothing this refresh.
      }
    }

    if (backends.length) {
      const backendId = await deps.installId(connection.id, backends[0].descriptor)
      const sharedProfiles = connection.kind === 'ssh' ? deps.sharedProfiles(connection.id, backends) : undefined
      sources.push({ backendId, backends, connectionId: connection.id, kind: connection.kind, sharedProfiles })
    }
  }

  return sources
}
