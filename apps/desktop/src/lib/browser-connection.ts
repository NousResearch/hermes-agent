import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { HermesApiRequest, HermesConnection } from '@/global'
import { WEBAPP_LAUNCH_REQUIRED } from '@/lib/browser-launch-session'
import {
  authenticatedWebsocketUrl,
  type BrowserBootstrap,
  BrowserReauthRequiredError,
  endpointUrl,
  navigateToBrowserLogin,
  websocketUrl
} from '@/lib/browser-transport'
import { windowProfileOverride } from '@/store/windows'

/** The serving origin is the only connection a browser-hosted Desktop has. */
export function requireBrowserConnection(connectionId?: null | string): void {
  const requested = connectionId?.trim() || LOCAL_CONNECTION_ID

  if (requested !== LOCAL_CONNECTION_ID) {
    throw new Error(`No connection with id "${requested}"`)
  }
}

function connectionFor(bootstrap: BrowserBootstrap, profile?: null | string): HermesConnection {
  if (!bootstrap.authRequired && !bootstrap.token) {throw new Error(WEBAPP_LAUNCH_REQUIRED)}

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
    sharedPrimary: Boolean(profile),
    source: 'settings',
    token: bootstrap.token,
    windowButtonPosition: null,
    wsUrl: bootstrap.authRequired
      ? ''
      : websocketUrl(bootstrap.basePath, '/api/ws', { token: bootstrap.token }, profile)
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
  | 'getConnection'
  | 'getConnectionConfig'
  | 'getConnectionFor'
  | 'getGatewayWsUrl'
  | 'getGatewayWsUrlFor'
  | 'getProfileRoutes'
  | 'oauthLoginConnectionConfig'
  | 'oauthLogoutConnectionConfig'
  | 'testConnectionConfig'
> {
  const getGatewayWsUrl = async (profile?: null | string) => {
    try {
      return {
        ok: true as const,
        wsUrl: await authenticatedWebsocketUrl(bootstrap, '/api/ws', profile)
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

  const getProfiles = async () => {
    const result = await api<{ profiles?: { name?: string }[] }>({ path: '/api/profiles' })

    return [...new Set((result.profiles || []).map(profile => String(profile.name || '').trim()).filter(Boolean))]
  }

  return {
    getAgentRoster: async () => {
      const profiles = await getProfiles()
      const connectionLabel = window.location.host

      return {
        agents: profiles.map(profile => ({
          connectionId: LOCAL_CONNECTION_ID,
          connectionKind: 'local' as const,
          connectionLabel,
          handle: profile === 'default' ? 'hermes' : profile,
          profile,
          targetProfile: profile
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
    // Reconnect belongs to the window, not whichever secondary profile is active.
    // Explicit null still selects the default backend.
    getConnection: async (profile: null | string = windowProfileOverride()) => connectionFor(bootstrap, profile),
    getConnectionConfig: async (profile?: null | string) => browserConnectionConfig(bootstrap, profile),
    getConnectionFor: async (payload: { connectionId?: null | string; profile?: null | string }) => {
      requireBrowserConnection(payload.connectionId)

      return {
        ...connectionFor(bootstrap, payload.profile),
        connectionId: LOCAL_CONNECTION_ID,
        registryScoped: true,
        sharedPrimary: false,
        sharedRemote: Boolean(payload.profile)
      }
    },
    getGatewayWsUrl,
    getGatewayWsUrlFor: async (payload: { connectionId?: null | string; profile?: null | string }) => {
      requireBrowserConnection(payload.connectionId)

      return getGatewayWsUrl(payload.profile)
    },
    getProfileRoutes: async (profiles: string[]) => {
      const available = new Set(await getProfiles())

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
    testConnectionConfig: async () => ({
      baseUrl: `${window.location.origin}${bootstrap.basePath}`,
      ok: true,
      reachable: true,
      version: null
    })
  }
}
