interface ResolvedPrimaryRemote {
  authMode?: 'oauth' | 'token'
  baseUrl: string
  connectionId?: string
  headers?: Record<string, string>
  remoteHermesVersion?: string
  remoteHost?: string
  remoteKind?: 'cloud' | 'ssh' | 'url'
  source?: string
  ssh?: {
    effectiveConfigFingerprint?: string
    host?: string
    keyPath?: string
    port?: number
    remoteHermesPath?: string
    remoteProfile?: string
    user?: string
  }
  token: unknown
  wsUrl: string
}

/**
 * Build the renderer-facing primary remote descriptor without dropping route
 * identity. Tests cross this same seam, so adding a field to the resolved
 * remote cannot silently disappear during primary startup.
 */
export function createPrimaryRemoteConnection<State extends object>(
  remote: ResolvedPrimaryRemote,
  logs: string[],
  windowState: State
) {
  return {
    baseUrl: remote.baseUrl,
    mode: 'remote' as const,
    source: remote.source,
    authMode: remote.authMode || 'token',
    remoteHost: remote.remoteHost,
    remoteKind: remote.remoteKind,
    remoteHermesVersion: remote.remoteHermesVersion,
    ...(remote.connectionId ? { connectionId: remote.connectionId } : {}),
    ...(remote.ssh ? { ssh: remote.ssh } : {}),
    // fetchJsonForBackend reads descriptor.headers for every REST call; the
    // WebSocket header store is keyed by exact URL and cannot stand in for it.
    headers: remote.headers,
    token: remote.token,
    wsUrl: remote.wsUrl,
    logs,
    ...windowState
  }
}

interface PrimaryRemoteStartupOptions<State extends object> {
  profile: string
  restoreRememberedProfile: () => boolean
  /** The profile this connection itself last ran, mirrored from the renderer. */
  connectionProfile?: (connectionId: string | undefined) => null | string | undefined
  signal?: AbortSignal
  assertCurrentAttempt: () => void
  advanceBootProgress: (phase: string, message: string, progress: number) => Promise<void>
  waitForRemote: (remote: ResolvedPrimaryRemote) => Promise<void>
  readProfiles: (remote: ResolvedPrimaryRemote) => Promise<unknown>
  /** `/api/status`, for the gateway's served set; absent or failed reads as unknown. */
  readStatus?: (remote: ResolvedPrimaryRemote) => Promise<unknown>
  onProfileRestored: (profile: string) => void
  updateBootProgress: (update: {
    phase: string
    message: string
    progress: number
    running: boolean
    error: null
  }) => void
  logs: () => string[]
  windowState: () => State
}

/**
 * The served profile a primary should boot as when the inherited one is not
 * served: null keeps `active`. Same ladder and served predicate as the
 * renderer's startupProfileForConnection, so main and the window land on one
 * profile.
 */
export function servedStartupProfile(
  active: string,
  remembered: null | string | undefined,
  profiles: Array<{ name: string }>,
  served: null | ReadonlySet<string> = null
): null | string {
  const names = profiles.map(profile => profile.name).filter(name => served === null || served.has(name))
  const servable = new Set(names)

  if (servable.has(active)) {
    return null
  }

  if (remembered && servable.has(remembered)) {
    return remembered
  }

  return servable.has('default') ? 'default' : (names[0] ?? null)
}

/**
 * The profiles this gateway serves, from `/api/status`: `/api/profiles` lists
 * every INSTALLED profile, parked and standalone ones included, while a
 * gateway-hosted listener refuses those with `profile_mismatch`.
 * `gateway_shared_with` is the multiplexer's served set; [] means a
 * single-profile gateway that serves only the host's own `default`. Null
 * (no gateway running, an older backend, a failed read) is unknown, and every
 * installed profile counts as served, as before the set was read.
 */
export function servedProfileSet(status: unknown): null | Set<string> {
  const shared = (status as { gateway_shared_with?: unknown } | null)?.gateway_shared_with

  if (!Array.isArray(shared)) {
    return null
  }

  return new Set(['default', ...shared.filter((name): name is string => typeof name === 'string' && Boolean(name.trim()))])
}

/** A complete profile listing, or null: a failed or partial read proves nothing. */
function installedProfiles(listing: unknown): Array<{ name: string }> | null {
  const profiles = (listing as { profiles?: unknown } | null)?.profiles

  if (!Array.isArray(profiles) || profiles.length === 0) {
    return null
  }

  return profiles.every(profile => typeof profile?.name === 'string' && profile.name.trim())
    ? (profiles as Array<{ name: string }>)
    : null
}

/** Resolve a stale last-used profile against the gateway we actually connected to. */
export async function connectPrimaryRemote<State extends object>(
  remote: ResolvedPrimaryRemote,
  options: PrimaryRemoteStartupOptions<State>
) {
  options.signal?.throwIfAborted()
  options.assertCurrentAttempt()
  await options.advanceBootProgress('backend.remote', `Connecting to remote Hermes backend at ${remote.baseUrl}`, 24)
  await options.waitForRemote(remote)
  options.signal?.throwIfAborted()
  options.assertCurrentAttempt()

  let restoredProfile: string | undefined

  // Explicit routes and per-profile remote aliases are intent, not stale
  // workspace memory. A failed/unsupported listing is not proof of deletion.
  if (
    options.restoreRememberedProfile() &&
    options.profile !== 'default' &&
    ['settings', 'env', 'registry'].includes(remote.source || '') &&
    !remote.ssh?.remoteProfile
  ) {
    const [listing, status] = await Promise.all([
      options.readProfiles(remote).catch(() => null),
      options.readStatus?.(remote).catch(() => null) ?? null
    ])

    options.signal?.throwIfAborted()
    options.assertCurrentAttempt()
    const profiles = installedProfiles(listing)

    const target =
      profiles &&
      servedStartupProfile(
        options.profile,
        options.connectionProfile?.(remote.connectionId),
        profiles,
        servedProfileSet(status)
      )

    if (target && options.restoreRememberedProfile()) {
      restoredProfile = target
      options.onProfileRestored(restoredProfile)
    }
  }

  options.updateBootProgress({
    phase: 'backend.ready',
    message: 'Remote Hermes backend is ready',
    progress: 94,
    running: true,
    error: null
  })

  return {
    ...createPrimaryRemoteConnection(remote, options.logs(), options.windowState()),
    ...(restoredProfile ? { profile: restoredProfile } : {})
  }
}
