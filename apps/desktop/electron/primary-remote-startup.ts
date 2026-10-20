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
  signal?: AbortSignal
  assertCurrentAttempt: () => void
  advanceBootProgress: (phase: string, message: string, progress: number) => Promise<void>
  waitForRemote: (remote: ResolvedPrimaryRemote) => Promise<void>
  readProfiles: (remote: ResolvedPrimaryRemote) => Promise<unknown>
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
    let listing: unknown

    try {
      listing = await options.readProfiles(remote)
    } catch {
      listing = null
    }

    options.signal?.throwIfAborted()
    options.assertCurrentAttempt()
    const profiles = (listing as { profiles?: Array<{ name?: string }> } | null)?.profiles

    if (
      Array.isArray(profiles) &&
      profiles.every(profile => typeof profile?.name === 'string' && profile.name.trim()) &&
      profiles.some(profile => profile?.name === 'default') &&
      !profiles.some(profile => profile?.name === options.profile) &&
      options.restoreRememberedProfile()
    ) {
      restoredProfile = 'default'
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
