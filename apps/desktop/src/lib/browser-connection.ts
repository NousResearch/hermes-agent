import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { DesktopOauthLoginOptions, HermesApiRequest, HermesConnection } from '@/global'
import { webappSignInRequired } from '@/lib/browser-launch-session'
import { signInSavedConnection, signOutSavedConnection } from '@/lib/browser-saved-connection-oauth'
import type { BrowserSavedConnections } from '@/lib/browser-saved-connections'
import { createServedProfiles, SERVED_SET_READ_TIMEOUT_MS, UnservedProfileError } from '@/lib/browser-served-profiles'
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
import { withTimeout } from '@/lib/with-timeout'

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

function connectionFor(
  bootstrap: BrowserBootstrap,
  profile: null | string | undefined,
  servedHere: boolean
): HermesConnection {
  if (!bootstrap.authRequired && !bootstrap.token) {
    throw webappSignInRequired()
  }

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
  /** This machine's saved gateway connections, reached through the serving gateway
   *  (`browser-saved-connections.ts`). Absent: only the serving gateway. */
  relays?: BrowserSavedConnections | null
}

const relayed = (relays: BrowserSavedConnections | null | undefined, connectionId?: null | string) =>
  relays && connectionId && connectionId !== LOCAL_CONNECTION_ID ? relays : null

function sameGateway(a: string, b: string): boolean {
  try {
    const [x, y] = [new URL(a), new URL(b)]

    return x.origin === y.origin && x.pathname.replace(/\/+$/, '') === y.pathname.replace(/\/+$/, '')
  } catch {
    return false
  }
}

/** Connection, gateway socket, roster and sign-in members, all bound to the serving origin. */
export function createBrowserConnectionBridge({
  api,
  bootstrap,
  relays
}: BrowserConnectionOptions): Pick<
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

  // Every saved gateway's profiles, read through the serving gateway, each its own source keyed by
  // its registry id (the id native Desktop and group-chat records route by); one that cannot
  // answer is listed unreachable rather than failing the roster.
  const relayedRoster = async () => {
    if (!relays) {
      return { agents: [], sources: [] }
    }

    const rows = await relays.gateways().catch(() => [])

    const read = await Promise.all(
      rows.map(async row => {
        try {
          return { agents: await relays.rosterAgents(row.id), row }
        } catch (error) {
          return { agents: [], error: error instanceof Error ? error.message : String(error), row }
        }
      })
    )

    return {
      agents: read.flatMap(entry => entry.agents),
      sources: read.map(entry => ({
        connectionId: entry.row.id,
        kind: entry.row.kind,
        label: entry.row.label,
        reachable: !('error' in entry),
        ...('error' in entry ? { error: entry.error } : {})
      }))
    }
  }

  const getRelayedWsUrl = async (connectionId: string, profile?: null | string) => {
    try {
      return { ok: true as const, wsUrl: await relays!.socketUrl(connectionId, '/api/ws', profile) }
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error), ok: false as const }
    }
  }

  return {
    getAgentRoster: async () => {
      const [profiles, remote] = await Promise.all([installedProfiles(), relayedRoster()])
      const connectionLabel = relays?.localLabel() || window.location.host

      return {
        agents: [
          ...profiles.map(profile => ({
            connectionId: LOCAL_CONNECTION_ID,
            connectionKind: 'local' as const,
            connectionLabel,
            // A real 'hermes' profile keeps @hermes; the default answers to @default then.
            handle: profile === 'default' && !profiles.includes('hermes') ? 'hermes' : profile,
            profile,
            targetProfile: profile,
            // Installed here, so listed, but served elsewhere or nowhere.
            ...(reachesSocket(profile) ? {} : { reachable: false as const })
          })),
          ...remote.agents
        ],
        primaryConnectionId: LOCAL_CONNECTION_ID,
        sources: [
          {
            connectionId: LOCAL_CONNECTION_ID,
            kind: 'local' as const,
            label: connectionLabel,
            reachable: true
          },
          ...remote.sources
        ]
      }
    },
    // Each voice stream dial mints its own ticket: the gateway's opens only `/api/ws`.
    getAudioWsUrl: async (payload: {
      connectionId?: null | string
      path: AudioSocketPath
      profile?: null | string
    }) => {
      if (relayed(relays, payload.connectionId)) {
        return relays!.socketUrl(payload.connectionId!, payload.path, payload.profile)
      }

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
      if (relayed(relays, payload.connectionId)) {
        // A tab restored before this page read its saved connections; an unreadable list names no
        // connection, which the descriptor read below refuses by id.
        if (!relays!.gateway(payload.connectionId)) {
          await relays!.gateways().catch(() => undefined)
        }

        return relays!.connection(payload.connectionId!, payload.profile)
      }

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
      if (relayed(relays, payload.connectionId)) {
        const connectionId = payload.connectionId!

        // As for this gateway's own: the dial would open, then close without a reason to show. A
        // profile created there since the last roster read is not in the set yet, so the set is read
        // again before the profile is refused. The store parks a refused dial, so only an explicit
        // open asks again.
        if (!relays!.serves(connectionId, payload.profile)) {
          await relays!.refreshServed(connectionId)
        }

        if (!relays!.serves(connectionId, payload.profile)) {
          throw new UnservedProfileError((payload.profile ?? '').trim())
        }

        return getRelayedWsUrl(connectionId, payload.profile)
      }

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
    oauthLoginConnectionConfig: async (remoteUrl: string, identity?: DesktopOauthLoginOptions) => {
      // Another gateway's sign-in runs on the serving gateway for its saved connection.
      if (!sameGateway(remoteUrl, `${window.location.origin}${bootstrap.basePath}`)) {
        if (!relays) {
          return {
            baseUrl: remoteUrl,
            connected: false,
            error: 'Signing in to another machine is not available here.',
            ok: false
          }
        }

        return signInSavedConnection(api, relays, remoteUrl, identity)
      }

      const target = new URL(`${bootstrap.basePath}/login`, window.location.origin)
      target.searchParams.set('next', `${window.location.pathname}${window.location.search}${window.location.hash}`)
      navigateToBrowserLogin(new BrowserReauthRequiredError('Sign in required', target.href))

      return { baseUrl: remoteUrl, connected: false, ok: false }
    },
    oauthLogoutConnectionConfig: async (remoteUrl?: string) => {
      // Another gateway's sign-in is its saved connection's, kept on the serving gateway.
      if (remoteUrl && new URL(remoteUrl, window.location.origin).origin !== window.location.origin) {
        return relays
          ? signOutSavedConnection(api, relays, remoteUrl)
          : { connected: false, error: 'Only this machine can be signed out here.', ok: false }
      }

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

      // On a stall or a failed read the cached set stands, so the pick routes on it.
      await withTimeout(installedProfiles(false), SERVED_SET_READ_TIMEOUT_MS, 'served profile refresh timed out').catch(
        () => undefined
      )
    },
    // Runs before every redial of the tab's socket: a gateway that restarted may serve another set.
    revalidateConnection: async () => {
      served.invalidate()

      return { ok: true, rebuilt: false }
    },
    // Only the serving gateway is reachable before a connection is saved; a saved one is tested
    // from its row (`connections.test`), through the serving gateway with its saved credential.
    testConnectionConfig: async payload => {
      const own = `${window.location.origin}${bootstrap.basePath}`

      return !payload?.remoteUrl || sameGateway(payload.remoteUrl, own)
        ? { baseUrl: own, ok: true, reachable: true, version: null }
        : {
            baseUrl: payload.remoteUrl,
            error: 'Save this connection, then use Test on its row: the web app tests it through this machine.',
            ok: false
          }
    }
  }
}
