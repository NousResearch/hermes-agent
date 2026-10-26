import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { HermesApiRequest, HermesConnection } from '@/global'
import { webappSignInRequired } from '@/lib/browser-launch-session'
import { createServedProfiles, UnservedProfileError } from '@/lib/browser-served-profiles'
import {
  type AudioSocketPath,
  type BrowserBootstrap,
  BrowserReauthRequiredError,
  endpointUrl,
  gatewaySocketUrl,
  navigateToBrowserLogin,
  ticketedWebsocketUrl
} from '@/lib/browser-transport'
import { browserLaunchProfileOrDefault } from '@/lib/platform'

// A pick waits at most this long for the served set before routing on the set it has: one
// `/api/profiles` read, then a `/api/status` read capped at the same 5 s. It sits well under the
// store's 20 s budget for a dial's URL, so a stalled gateway costs a pick 5 s and a profile that
// stays unserved is still refused and parked rather than timed out and redialed.
const SERVED_PROFILE_REFRESH_TIMEOUT_MS = 5_000

/** The serving origin is the only connection a browser-hosted Desktop has. */
export function requireBrowserConnection(connectionId?: null | string): void {
  const requested = connectionId?.trim() || LOCAL_CONNECTION_ID

  if (requested !== LOCAL_CONNECTION_ID) {
    throw new Error(`No connection with id "${requested}"`)
  }
}

// The tab's one socket answers for the profile the serving gateway runs as when a request names
// none, as its REST does. Any other profile it serves rides that socket with `profile` on every
// request.
function ridesWithProfile(profile?: null | string): boolean {
  const name = profile?.trim()

  return Boolean(name) && name !== browserLaunchProfileOrDefault()
}

function connectionFor(bootstrap: BrowserBootstrap, profile: null | string | undefined, servedHere: boolean): HermesConnection {
  if (!bootstrap.authRequired && !bootstrap.token) {throw webappSignInRequired()}

  const baseUrl = `${window.location.origin}${bootstrap.basePath}`

  return {
    authMode: bootstrap.authRequired ? 'oauth' : 'token',
    baseUrl,
    isFullscreen: Boolean(document.fullscreenElement),
    logs: [],
    mode: 'remote',
    nativeOverlayWidth: 0,
    profile: profile || undefined,
    remoteHost: window.location.host,
    remoteKind: 'url',
    sharedPrimary: ridesWithProfile(profile) && servedHere,
    source: 'settings',
    token: bootstrap.token,
    windowButtonPosition: null,
    // The serving gateway's canonical socket: credential-free here, every dial mints its own
    // ticket (getGatewayWsUrl). `native_dial` selects the canonical contract.
    wsUrl: gatewaySocketUrl(bootstrap, profile)
  }
}

function browserConnectionConfig(bootstrap: BrowserBootstrap, profile?: null | string) {
  return {
    cloudOrg: '',
    envOverride: false,
    mode: 'remote' as const,
    profile: profile || null,
    remoteAuthMode: bootstrap.authRequired ? ('oauth' as const) : ('token' as const),
    remoteOauthConnected: bootstrap.authRequired,
    remoteTokenPlainText: false,
    remoteTokenPreview: null,
    remoteTokenSet: Boolean(bootstrap.token),
    secureTokenStorage: false,
    remoteUrl: `${window.location.origin}${bootstrap.basePath}`,
    sshHost: '',
    sshKeyPath: '',
    sshPort: null,
    sshRemoteHermesPath: '',
    sshRemoteProfile: '',
    sshUser: ''
  }
}

interface BrowserConnectionOptions {
  api: <T>(request: HermesApiRequest) => Promise<T>
  bootstrap: BrowserBootstrap
}

/** Connection, gateway socket, roster and sign-in members, all bound to the serving origin. */
export function createBrowserConnectionBridge({ api, bootstrap }: BrowserConnectionOptions): Pick<
  Window['hermesDesktop'],
  | 'getAgentRoster'
  | 'getAudioWsUrl'
  | 'getConnection'
  | 'getConnectionConfig'
  | 'getConnectionFor'
  | 'getGatewayWsUrl'
  | 'getGatewayWsUrlFor'
  | 'getProfileRoutes'
  | 'oauthLoginConnectionConfig'
  | 'oauthLogoutConnectionConfig'
  | 'refreshServedProfile'
  | 'revalidateConnection'
  | 'testConnectionConfig'
> {
  const served = createServedProfiles(api)

  // The gateway's own profile, or another one it serves: the only profiles this tab's socket reaches.
  const reachesSocket = (profile?: null | string) => !ridesWithProfile(profile) || served.serves(profile)

  const getProfiles = async () => {
    const result = await api<{ profiles?: { name?: string }[] }>({ path: '/api/profiles' })

    return [...new Set((result.profiles || []).map(profile => String(profile.name || '').trim()).filter(Boolean))]
  }

  // The list read in flight. Both mounted pickers read the roster on the same list refresh, so
  // concurrent callers share it; it is dropped once it settles, so no answer outlives its read.
  let profilesRead: null | Promise<string[]> = null

  const readProfiles = (join: boolean) => {
    if (join && profilesRead) {
      return profilesRead
    }

    const read = getProfiles()
    const settle = () => {
      if (profilesRead === read) {
        profilesRead = null
      }
    }

    profilesRead = read
    read.then(settle, settle)

    return read
  }

  // Every installed profile, with the served set brought up to date for that list. `join: false`
  // starts a read of its own, for a caller that needs a list newer than the read in flight.
  const installedProfiles = async (join = true) => {
    const profiles = await readProfiles(join)
    await served.refresh(profiles)

    return profiles
  }

  const getGatewayWsUrl = async (profile?: null | string) => {
    // Its ticket would mint, then the gateway would close the dial (4403) without a reason to show.
    if (!reachesSocket(profile)) {
      throw new UnservedProfileError((profile ?? '').trim())
    }

    try {
      return {
        ok: true as const,
        wsUrl: await ticketedWebsocketUrl(bootstrap, '/api/ws', profile)
      }
    } catch (error) {
      if (error instanceof BrowserReauthRequiredError) {
        return {
          error: error.message,
          needsOauthLogin: true,
          ok: false as const
        }
      }

      throw error
    }
  }

  return {
    getAgentRoster: async () => {
      const profiles = await installedProfiles()
      const connectionLabel = window.location.host

      return {
        agents: profiles.map(profile => ({
          connectionId: LOCAL_CONNECTION_ID,
          connectionKind: 'local' as const,
          connectionLabel,
          handle: profile === 'default' ? 'hermes' : profile,
          profile,
          targetProfile: profile,
          // Installed here, so listed, but served elsewhere or nowhere.
          ...(reachesSocket(profile) ? {} : { reachable: false as const })
        })),
        primaryConnectionId: LOCAL_CONNECTION_ID,
        sources: [{
          connectionId: LOCAL_CONNECTION_ID,
          kind: 'local' as const,
          label: connectionLabel,
          reachable: true
        }]
      }
    },
    // Each voice stream dial mints its own ticket: the gateway's opens only `/api/ws`.
    getAudioWsUrl: async (payload: {
      connectionId?: null | string
      path: AudioSocketPath
      profile?: null | string
    }) => {
      requireBrowserConnection(payload.connectionId)

      return ticketedWebsocketUrl(bootstrap, payload.path, payload.profile)
    },
    // Without a profile: the tab's own socket, the one boot and every reconnect dial, whatever
    // profile the tab is on. A named profile the gateway serves, other than its own, is a view of
    // that same socket (sharedPrimary), so its requests carry `profile`. Routing reads the cached
    // served set and never fetches it.
    getConnection: async (profile?: null | string) => ({
      ...connectionFor(bootstrap, profile, served.serves(profile)),
      connectionId: LOCAL_CONNECTION_ID
    }),
    getConnectionConfig: async (profile?: null | string) => browserConnectionConfig(bootstrap, profile),
    getConnectionFor: async (payload: { connectionId?: null | string; profile?: null | string }) => {
      requireBrowserConnection(payload.connectionId)

      return {
        ...connectionFor(bootstrap, payload.profile, served.serves(payload.profile)),
        connectionId: LOCAL_CONNECTION_ID,
        registryScoped: true,
        sharedPrimary: false,
        sharedRemote: Boolean(payload.profile) && reachesSocket(payload.profile)
      }
    },
    getGatewayWsUrl,
    getGatewayWsUrlFor: async (payload: { connectionId?: null | string; profile?: null | string }) => {
      requireBrowserConnection(payload.connectionId)

      return getGatewayWsUrl(payload.profile)
    },
    getProfileRoutes: async (profiles: string[]) => {
      const available = new Set((await installedProfiles()).filter(reachesSocket))

      return [...new Set(profiles.map(profile => profile.trim() || 'default'))]
        .filter(profile => available.has(profile))
        .map(profile => ({
          connectionId: LOCAL_CONNECTION_ID,
          mode: 'local' as const,
          profile,
          targetProfile: profile
        }))
    },
    oauthLoginConnectionConfig: async (remoteUrl: string) => {
      const target = new URL(`${bootstrap.basePath}/login`, window.location.origin)
      target.searchParams.set('next', `${window.location.pathname}${window.location.search}${window.location.hash}`)
      navigateToBrowserLogin(new BrowserReauthRequiredError('Sign in required', target.href))

      return { baseUrl: remoteUrl, connected: false, ok: false }
    },
    oauthLogoutConnectionConfig: async () => {
      const response = await fetch(endpointUrl('/auth/logout', bootstrap.basePath), {
        credentials: 'same-origin',
        method: 'POST'
      })

      if (!response.ok) {
        throw new Error(`Sign out failed (${response.status})`)
      }

      const login = new URL(`${bootstrap.basePath}/login`, window.location.origin)
      navigateToBrowserLogin(new BrowserReauthRequiredError('Signed out', login.href))

      return { connected: false, ok: true }
    },
    // A profile created since the last inventory read is missing from the cached set, though the
    // multiplexer adopted it before its create returned. Selection reads the inventory once before
    // it routes, so the profile rides the tab's socket; dials keep reading the cache alone. The read
    // is its own: one already in flight may have gone out before the create.
    refreshServedProfile: async (profile: string) => {
      if (reachesSocket(profile)) {
        return
      }

      let timer: ReturnType<typeof setTimeout> | undefined

      await Promise.race([
        installedProfiles(false).then(
          () => undefined,
          () => undefined
        ),
        new Promise<void>(resolve => {
          timer = setTimeout(resolve, SERVED_PROFILE_REFRESH_TIMEOUT_MS)
        })
      ])
      clearTimeout(timer)
    },
    // Runs before every redial of the tab's socket: a gateway that restarted may serve another set.
    revalidateConnection: async () => {
      served.invalidate()

      return { ok: true, rebuilt: false }
    },
    testConnectionConfig: async () => ({
      baseUrl: `${window.location.origin}${bootstrap.basePath}`,
      ok: true,
      reachable: true,
      version: null
    })
  }
}
