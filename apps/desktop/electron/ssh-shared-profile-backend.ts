/**
 * One remote backend per SSH connection, every profile scoped per request.
 *
 * A registry SSH source used to spawn `hermes --profile <p> serve --isolated`
 * on the remote for EACH profile opened (one tunnel + one remote process per
 * (connection, profile)). The remote `hermes serve` already multiplexes every
 * profile under its HERMES_HOME the way a URL remote or the local host backend
 * does (REST `?profile=`, a host-scoped `/api/ws`), so a named profile rides the
 * connection's default backend as a `sharedRemote` descriptor instead: the
 * first click needs no remote spawn, and a hover/roster sweep cannot fan out
 * into one process per bot (the #89756 crash class).
 *
 * The isolated per-profile path stays the fallback for exactly the case the
 * shared backend cannot serve: a source pinned to one remote profile
 * (`remoteProfile`), or a remote whose `/api/profiles` does not list the
 * profile (an older runtime without the endpoint, or a profile it cannot see).
 */

import { normalizeSshConfig } from './connection-config'
import type { RegistryConnection } from './connection-registry'

export interface SharedSshProfileScope {
  connectionId: string
  profile: string
  remoteProfile: ''
  sharedRemote: true
}

export interface SharedSshProfileDeps<T extends { baseUrl?: string }> {
  /** The connection's default-profile backend (pooled, or dialed once). */
  ensureDefault: () => Promise<T>
  /** GET /api/profiles against that backend. */
  listProfiles: (descriptor: T) => Promise<unknown>
}

/** The SSH config that dials `profile`'s own remote serve (the isolated path). */
export function managedSshConfig(source: RegistryConnection, profile: null | string = '') {
  const profileKey = String(profile ?? '').trim() || 'default'

  return normalizeSshConfig({
    mode: 'ssh',
    host: source.host,
    user: source.user,
    port: source.port,
    keyPath: source.keyPath,
    remoteHermesPath: source.remoteHermesPath,
    remoteProfile: source.remoteProfile || (profileKey === 'default' ? '' : profileKey)
  })
}

// Profiles each default backend has been seen serving, keyed by connection +
// base URL: a reconnect lands on a new forwarded port and re-checks the roster.
const servedProfiles = new Map<string, Set<string>>()

function profileNames(body: unknown): string[] {
  const profiles = (body as { profiles?: unknown } | null)?.profiles

  if (!Array.isArray(profiles)) {
    return []
  }

  return profiles
    .map(entry => String((entry as { name?: unknown } | null)?.name ?? '').trim().toLowerCase())
    .filter(Boolean)
}

/**
 * The shared-backend descriptor for `profile` on an SSH source, or null when
 * the caller must keep the per-profile isolated backend.
 */
export async function resolveSharedSshProfileBackend<T extends { baseUrl?: string }>(
  source: Pick<RegistryConnection, 'id' | 'kind' | 'remoteProfile'>,
  profile: null | string | undefined,
  deps: SharedSshProfileDeps<T>
): Promise<(T & SharedSshProfileScope) | null> {
  const key = String(profile ?? '').trim().toLowerCase()

  if (source.kind !== 'ssh' || String(source.remoteProfile ?? '').trim() || !key || key === 'default') {
    return null
  }

  const base = await deps.ensureDefault()
  const cacheKey = `${source.id}\0${base.baseUrl ?? ''}`
  let served = servedProfiles.get(cacheKey)

  if (!served?.has(key)) {
    let body: unknown = null

    try {
      body = await deps.listProfiles(base)
    } catch {
      // No roster from this backend: it cannot be trusted to scope requests.
    }

    const names = profileNames(body)

    if (!names.includes(key)) {
      return null
    }

    served = new Set(names)
    servedProfiles.set(cacheKey, served)
  }

  return { ...base, profile: key, connectionId: source.id, sharedRemote: true, remoteProfile: '' }
}
