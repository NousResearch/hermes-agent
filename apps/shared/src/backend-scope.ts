/**
 * Composite backend scope keys for the multi-connection registry — shared
 * between the Electron main process (backend pool keying) and the renderer
 * (secondary socket registry), so both sides derive identical keys.
 *
 * The local/primary connection keeps the BARE profile key: every legacy pool
 * entry, reaper log line, and touch call stays byte-identical for
 * single-source users. Non-local connections get `conn:<id>::<profile>`,
 * which cannot collide with a plain profile name (colons are invalid in
 * profile names).
 */

export const LOCAL_CONNECTION_ID = 'local'

export function backendScopeKey(connectionId: null | string | undefined, profile: null | string | undefined): string {
  const profileKey = String(profile ?? '').trim() || 'default'
  const connection = String(connectionId ?? '').trim()

  if (!connection || connection === LOCAL_CONNECTION_ID) {
    return profileKey
  }

  return `conn:${connection}::${profileKey}`
}

/** Scope a registry route without collapsing its explicit `local` source id.
 * Null/empty ids still identify the legacy profile-only route. */
export function registryBackendScopeKey(
  connectionId: null | string | undefined,
  profile: null | string | undefined
): string {
  const profileKey = String(profile ?? '').trim() || 'default'
  const connection = String(connectionId ?? '').trim()

  return connection ? `conn:${connection}::${profileKey}` : profileKey
}

/** All pool keys owned by a connection share this prefix (teardown on remove). */
export function backendScopePrefix(connectionId: string): string {
  return `conn:${String(connectionId).trim()}::`
}
