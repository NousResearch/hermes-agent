import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { DesktopOauthLoginOptions, HermesApiRequest, HermesConnection } from '@/global'
import type { HostedRoute } from '@/lib/browser-hosted-route'
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
import { adoptBrowserLaunchProfile, browserLaunchProfileOrDefault } from '@/lib/platform'
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
  /** The host runs no agent (`browserHostRunsNoAgent`): every request goes to one of the account's
   *  agents, picked by the route the bridge's REST door uses too. */
  hosted?: HostedRoute | null
}

/** A request's gateway when it is not the serving one: a saved connection, reached by relay. */
interface RelayTarget {
  connectionId: string
  relays: BrowserSavedConnections
}

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
  hosted = null,
  relays = null
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
  const saved = hosted?.relays ?? relays

  // Each member resolves its gateway once here, then takes the relayed or the direct path. On a
  // host with no agent every request is relayed; elsewhere only one naming a saved connection is.
  const targetOf = async (connectionId?: null | string): Promise<null | RelayTarget> => {
    if (hosted) {
      return { connectionId: await hosted.agentFor(connectionId), relays: hosted.relays }
    }

    return relays && connectionId && connectionId !== LOCAL_CONNECTION_ID ? { connectionId, relays } : null
  }

  // A profile named with no agent at all is one of the agent the window's own socket opened on.
  const homeOf = async (): Promise<null | RelayTarget> =>
    hosted ? { connectionId: await hosted.home(), relays: hosted.relays } : null

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
    if (!saved) {
      return { agents: [], sources: [] }
    }

    const rows = await saved.gateways().catch(() => [])

    const read = await Promise.all(
      rows.map(async row => {
        try {
          return { agents: await saved.rosterAgents(row.id), row }
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

  // Every saved gateway's served profiles as routes, as Electron lists every registry source's: a
  // chat on a gateway the window is not on keeps its route. `primary` marks the window socket's own.
  // A gateway that cannot answer is left out and the rest still route: a chat's retry asks whether
  // its own route is listed, and another gateway's failure says nothing about that. Boot's check of
  // the window's own profiles reads `own`, which fails instead.
  const relayedRoutes = async (primary?: string) =>
    (await relayedRoster()).agents
      .filter(row => row.reachable !== false)
      .map(row => ({
        connectionId: row.connectionId,
        mode: 'remote' as const,
        ...(row.connectionId === primary ? { primary: true as const } : {}),
        profile: row.profile,
        targetProfile: row.targetProfile || row.profile
      }))

  const getRelayedWsUrl = async ({ connectionId, relays }: RelayTarget, profile?: null | string) => {
    try {
      return { ok: true as const, wsUrl: await relays.socketUrl(connectionId, '/api/ws', profile) }
    } catch (error) {
      return { error: error instanceof Error ? error.message : String(error), ok: false as const }
    }
  }

  const relayedGatewayWsUrl = async (target: RelayTarget, profile?: null | string) => {
    const { connectionId, relays } = target

    // As for this gateway's own: the dial would open, then close without a reason to show. A
    // profile created there since the last roster read is not in the set yet, so the set is read
    // again before the profile is refused. The store parks a refused dial, so only an explicit
    // open asks again.
    if (!relays.serves(connectionId, profile)) {
      await relays.refreshServed(connectionId)
    }

    if (!relays.serves(connectionId, profile)) {
      throw new UnservedProfileError((profile ?? '').trim())
    }

    return getRelayedWsUrl(target, profile)
  }

  return {
    getAgentRoster: async () => {
      if (hosted) {
        const [remote, primaryConnectionId] = await Promise.all([relayedRoster(), hosted.agentFor()])

        return { agents: remote.agents, primaryConnectionId, sources: remote.sources }
      }

      const [profiles, remote] = await Promise.all([installedProfiles(), relayedRoster()])
      const connectionLabel = saved?.localLabel() || window.location.host

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
      const target = await targetOf(payload.connectionId)

      if (target) {
        return target.relays.socketUrl(target.connectionId, payload.path, payload.profile)
      }

      requireBrowserConnection(payload.connectionId)

      return ticketedWebsocketUrl(bootstrap, payload.path, payload.profile)
    },
    // Without a profile: the tab's own socket, the one boot and every reconnect dial, whatever
    // profile the tab is on. A named profile the gateway serves, other than its own, is a view of
    // that same socket (sharedPrimary), so its requests carry `profile`. Routing reads the cached
    // served set and never fetches it.
    getConnection: async (profile?: null | string) => {
      const home = await homeOf()

      if (!home) {
        return { ...connectionFor(bootstrap, profile, served.serves(profile)), connectionId: LOCAL_CONNECTION_ID }
      }

      // Boot adopts the profile the page's server runs as; here that is the agent's own. A read that
      // fails fails the open: the socket opens on that profile, so boot must not guess `default`.
      adoptBrowserLaunchProfile(await home.relays.launchProfile(home.connectionId))

      // As a served Webapp's: a profile the agent serves beside its own rides the window's socket, so
      // the store reuses that socket instead of dialing a second one to the same agent.
      return {
        ...home.relays.connection(home.connectionId, profile),
        sharedPrimary: ridesWithProfile(profile) && home.relays.serves(home.connectionId, profile)
      }
    },
    getConnectionConfig: async (profile?: null | string) => browserConnectionConfig(bootstrap, profile),
    getConnectionFor: async ({ connectionId, profile }: { connectionId?: null | string; profile?: null | string }) => {
      const target = await targetOf(connectionId)

      if (target) {
        // A tab restored before this page read its saved connections; an unreadable list names no
        // connection, which the descriptor read below refuses by id.
        if (!target.relays.gateway(target.connectionId)) {
          await target.relays.gateways().catch(() => undefined)
        }

        return target.relays.connection(target.connectionId, profile)
      }

      requireBrowserConnection(connectionId)

      return {
        ...connectionFor(bootstrap, profile, served.serves(profile)),
        connectionId: LOCAL_CONNECTION_ID,
        registryScoped: true,
        sharedPrimary: false,
        sharedRemote: Boolean(profile) && reachesSocket(profile)
      }
    },
    getGatewayWsUrl: async (profile?: null | string) => {
      const home = await homeOf()

      return home ? relayedGatewayWsUrl(home, profile) : getGatewayWsUrl(profile)
    },
    getGatewayWsUrlFor: async ({ connectionId, profile }: { connectionId?: null | string; profile?: null | string }) => {
      const target = await targetOf(connectionId)

      if (target) {
        return relayedGatewayWsUrl(target, profile)
      }

      requireBrowserConnection(connectionId)

      return getGatewayWsUrl(profile)
    },
    // `own` asks only for the window socket's gateway's routes (boot's check of the address's
    // profile), so another gateway that is slow to answer cannot hold that check up.
    getProfileRoutes: async (profiles: string[], options?: { own?: boolean }) => {
      const home = await homeOf()

      if (home && options?.own) {
        const { connectionId, relays } = home

        // A list or a roster that cannot be read fails the read: boot would take missing routes as
        // the agent lacking the address's profile, and drop it.
        if (!(await relays.gateways()).some(row => row.id === connectionId)) {
          throw new Error(`"${connectionId}" is not in this account's list`)
        }

        return (await relays.rosterAgents(connectionId))
          .filter(row => row.reachable !== false)
          .map(row => ({
            connectionId,
            mode: 'remote' as const,
            primary: true as const,
            profile: row.profile,
            targetProfile: row.targetProfile || row.profile
          }))
      }

      if (home) {
        return relayedRoutes(home.connectionId)
      }

      // The serving gateway's: only the named profiles it has and serves. Its own list failing fails
      // boot's own read; the full list still routes the saved gateways, as on a host with no agent.
      const [installed, remote] = await Promise.all([
        options?.own ? installedProfiles() : installedProfiles().catch(() => []),
        options?.own ? [] : relayedRoutes()
      ])

      const available = new Set(installed.filter(reachesSocket))

      return [
        ...[...new Set(profiles.map(profile => profile.trim() || 'default'))]
          .filter(profile => available.has(profile))
          .map(profile => ({
            connectionId: LOCAL_CONNECTION_ID,
            mode: 'local' as const,
            profile,
            targetProfile: profile
          })),
        ...remote
      ]
    },
    oauthLoginConnectionConfig: async (remoteUrl: string, identity?: DesktopOauthLoginOptions) => {
      // Another gateway's sign-in runs on the serving gateway for its saved connection.
      if (!sameGateway(remoteUrl, `${window.location.origin}${bootstrap.basePath}`)) {
        // The host signs in to its agents itself; it has no per-connection sign-in to run.
        if (!relays || hosted) {
          return {
            baseUrl: remoteUrl,
            connected: false,
            error: 'Signing in to another gateway is not available here.',
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
        return relays && !hosted
          ? signOutSavedConnection(api, relays, remoteUrl)
          : { connected: false, error: 'Only this gateway can be signed out here.', ok: false }
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
      const home = await homeOf()

      if (home) {
        if (!home.relays.serves(home.connectionId, profile)) {
          await home.relays.refreshServed(home.connectionId)
        }

        return
      }

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
            error: 'Save this connection, then use Test on its row: the Webapp tests it through this gateway.',
            ok: false
          }
    }
  }
}
