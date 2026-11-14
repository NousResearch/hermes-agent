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
 * (`remoteProfile`), or a remote with no `/api/profiles` roster (an older
 * runtime answering 404). Any other roster failure is transient and surfaces. A roster that answers without the profile
 * means the profile does not exist on that host (the endpoint lists every
 * profile on disk), so the dial fails fast instead of spawning a remote serve
 * that can only exit with the same error. Renderer scope races ask for exactly
 * that, e.g. a version read carrying the new connection with the previous
 * connection's local profile.
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

/**
 * Named profiles a pooled SSH source serves through its default backend, from
 * that backend's last roster read: the session aggregate covers them although
 * none has a pooled backend of its own.
 */
export function sharedSshProfiles(
  connectionId: string,
  backends: ReadonlyArray<{ descriptor: unknown; profileLabel: null | string }>
): string[] | undefined {
  const base = backends.find(({ profileLabel }) => (profileLabel || 'default') === 'default')?.descriptor as
    | { baseUrl?: string }
    | undefined

  const served = base ? servedProfiles.get(`${connectionId}\0${base.baseUrl ?? ''}`) : undefined

  return served ? [...served] : undefined
}

function profileNames(body: unknown): string[] {
  const profiles = (body as { profiles?: unknown } | null)?.profiles

  if (!Array.isArray(profiles)) {
    return []
  }

  return profiles
    .map(entry => String((entry as { name?: unknown } | null)?.name ?? '').trim().toLowerCase())
    .filter(Boolean)
}

// Same shape remote-lifecycle gives a remote `--profile X serve` that refuses a missing profile.
function remoteProfileMissing(profile: string): Error & { kind: string; profile: string } {
  return Object.assign(
    new Error(`The remote Hermes profile '${profile}' does not exist. Select an existing remote profile.`),
    { kind: 'remote-profile-missing', profile }
  )
}

/**
 * The shared-backend descriptor for `profile` on an SSH source, or null when
 * the caller must keep the per-profile isolated backend. Throws
 * `remote-profile-missing` when the host's roster has no such profile.
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
    let body: unknown

    try {
      body = await deps.listProfiles(base)
    } catch (error) {
      // Only a runtime without the endpoint keeps the per-profile backend. A
      // timeout or 5xx is transient: surface it rather than pool an isolated
      // serve that every later request would keep using.
      if ((error as { statusCode?: unknown } | null)?.statusCode === 404) {
        return null
      }

      throw error
    }

    const names = profileNames(body)

    if (!names.length) {
      return null
    }

    if (!names.includes(key)) {
      throw remoteProfileMissing(key)
    }

    served = new Set(names)
    servedProfiles.set(cacheKey, served)
  }

  return { ...base, profile: key, connectionId: source.id, sharedRemote: true, remoteProfile: '' }
}
