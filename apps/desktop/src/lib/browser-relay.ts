import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type {
  DesktopConnectionsRegistry,
  DesktopConnectionTestResult,
  DesktopRegistryConnection,
  DesktopRosterAgent,
  GatewayRelayGateway,
  GatewayRelayGrant,
  GatewayRelayRemoval,
  HermesApiRequest,
  HermesConnection
} from '@/global'
import {
  browserApi,
  type BrowserBootstrap,
  gatewaySocketUrl,
  ticketedWebsocketUrl,
  type TicketSocketPath
} from '@/lib/browser-transport'

/**
 * The gateway relay, browser side: other gateways the serving one was paired with
 * (`hermes_cli/webapp_relay.py`), offered to the renderer as registry connections the way native
 * Desktop offers the gateways it registered.
 *
 * Every request to a relayed gateway goes to this same origin under `/api/relay/<id>`, which the
 * serving gateway signs and sends on, so a relayed connection is a bootstrap with that base path:
 * REST, file links and socket ticket mints reuse the serving gateway's transport unchanged. A
 * relayed socket carries two one-use tickets: the far gateway's, minted through the relay for that
 * socket (it rides the query or the subprotocol as on a direct dial), and the serving gateway's
 * `relay_ticket`, which admits this tab to the relay itself.
 */

type RelayGateway = GatewayRelayGateway

type Api = <T>(request: HermesApiRequest) => Promise<T>

type ChangeListener = (payload: { connectionId: string; reason: 'removed' | 'saved' | 'updated' }) => void

const LAST_USED_KEY = 'hermes.webapp.relay.lastUsed'

/** The serving gateway turned the relay on (`webapp.relay.enabled`, behind the sign-in gate). */
export function relayEnabled(): boolean {
  return (window as Window & { __HERMES_GATEWAY_RELAY__?: boolean }).__HERMES_GATEWAY_RELAY__ === true
}

export function relayBasePath(bootstrap: BrowserBootstrap, connectionId: string): string {
  return `${bootstrap.basePath}/api/relay/${encodeURIComponent(connectionId)}`
}

/** *bootstrap* aimed at a relayed gateway: same origin and sign-in, the relay's base path. */
export function relayBootstrap(bootstrap: BrowserBootstrap, connectionId: string): BrowserBootstrap {
  return { ...bootstrap, basePath: relayBasePath(bootstrap, connectionId) }
}

/** The label the Webapp's own gateway goes by in the registry: its host's first name. */
export function servingGatewayLabel(): string {
  const host = window.location.hostname

  return /^[\d.]+$|:/.test(host) ? host : host.split('.')[0] || host
}

export interface BrowserRelays {
  /** A cached gateway, or undefined; the cache follows every list and change. */
  gateway: (connectionId: null | string | undefined) => RelayGateway | undefined
  gateways: () => Promise<RelayGateway[]>
  /** The bootstrap a request for *connectionId* rides: the serving gateway's, or its relay's. */
  bootstrapFor: (connectionId: null | string | undefined) => BrowserBootstrap
  /** A fresh socket URL on *connectionId* for *path*, both tickets minted for this one dial. */
  socketUrl: (connectionId: string, path: TicketSocketPath, profile?: null | string) => Promise<string>
  connection: (connectionId: string, profile?: null | string) => HermesConnection
  rosterAgents: (connectionId: string) => Promise<DesktopRosterAgent[]>
  connections: NonNullable<Window['hermesDesktop']['connections']>
  /** `window.hermesDesktop.gatewayRelay`: pairing and grants, for Settings → Gateways. */
  bridge: NonNullable<Window['hermesDesktop']['gatewayRelay']>
}

/**
 * How long the roster waits for a relayed gateway's profiles before listing it unreachable. One that
 * takes the connection and never answers would otherwise hold the switcher for a whole request timeout.
 */
export const RELAYED_ROSTER_TIMEOUT_MS = 8_000

export function createBrowserRelays({ api, bootstrap }: { api: Api; bootstrap: BrowserBootstrap }): BrowserRelays {
  let cached: RelayGateway[] = []
  const listeners = new Set<ChangeListener>()

  const gateway = (connectionId: null | string | undefined) =>
    connectionId && connectionId !== LOCAL_CONNECTION_ID ? cached.find(row => row.id === connectionId) : undefined

  const gateways = async () => {
    if (!relayEnabled()) {
      return (cached = [])
    }

    const result = await api<{ gateways?: RelayGateway[] }>({ path: '/api/webapp/relay/gateways' })
    cached = Array.isArray(result?.gateways) ? result.gateways : []

    return cached
  }

  const bootstrapFor = (connectionId: null | string | undefined) => {
    if (!connectionId || connectionId === LOCAL_CONNECTION_ID) {
      return bootstrap
    }

    if (!gateway(connectionId)) {
      throw new Error(`No connection with id "${connectionId}"`)
    }

    return relayBootstrap(bootstrap, connectionId)
  }

  const socketUrl = async (connectionId: string, path: TicketSocketPath, profile?: null | string) => {
    const url = new URL(await ticketedWebsocketUrl(bootstrapFor(connectionId), path, profile))

    const minted = await browserApi<null | { ticket?: string }>(bootstrap, {
      method: 'POST',
      path: '/api/webapp/socket-ticket?socket=relay'
    })

    const ticket = String(minted?.ticket || '').trim()

    if (!ticket) {
      throw new Error('Hermes did not return a relay ticket')
    }

    url.searchParams.set('relay_ticket', ticket)

    return url.toString()
  }

  const connection = (connectionId: string, profile?: null | string): HermesConnection => {
    const row = gateway(connectionId)
    const relayed = bootstrapFor(connectionId)

    return {
      authMode: 'oauth',
      baseUrl: `${window.location.origin}${relayed.basePath}`,
      connectionId,
      isFullscreen: Boolean(document.fullscreenElement),
      logs: [],
      mode: 'remote',
      nativeOverlayWidth: 0,
      profile: profile || undefined,
      registryScoped: true,
      remoteHost: row ? new URL(row.url).host : connectionId,
      remoteKind: 'url',
      sharedPrimary: false,
      sharedRemote: Boolean(profile),
      source: 'settings',
      token: '',
      windowButtonPosition: null,
      // Credential-free, like the serving gateway's own descriptor; every dial mints its tickets.
      wsUrl: gatewaySocketUrl(relayed, profile)
    }
  }

  const rosterAgents = async (connectionId: string): Promise<DesktopRosterAgent[]> => {
    const row = gateway(connectionId)

    const result = await browserApi<{ profiles?: { name?: string }[] }>(bootstrapFor(connectionId), {
      path: '/api/profiles',
      timeoutMs: RELAYED_ROSTER_TIMEOUT_MS
    })

    const names = [
      ...new Set((result?.profiles || []).map(profile => String(profile.name || '').trim()).filter(Boolean))
    ]

    return names.map(profile => ({
      connectionId,
      connectionKind: 'remote' as const,
      connectionLabel: row?.label ?? connectionId,
      handle: profile,
      profile,
      targetProfile: profile
    }))
  }

  const registry = (): DesktopConnectionsRegistry => {
    const local: DesktopRegistryConnection = {
      id: LOCAL_CONNECTION_ID,
      kind: 'local',
      label: servingGatewayLabel(),
      tokenPreview: null,
      tokenSet: false
    }

    const relayed = cached.map((row): DesktopRegistryConnection => ({
      authMode: 'oauth',
      id: row.id,
      kind: 'remote',
      label: row.label,
      tokenPreview: null,
      tokenSet: true,
      url: row.url
    }))

    const lastUsed = window.localStorage.getItem(LAST_USED_KEY) || LOCAL_CONNECTION_ID

    return {
      connections: [local, ...relayed],
      lastUsed: [local, ...relayed].some(entry => entry.id === lastUsed) ? lastUsed : LOCAL_CONNECTION_ID,
      launchMode: 'primary',
      primary: LOCAL_CONNECTION_ID,
      secureTokenStorage: false,
      version: 2
    }
  }

  const changed = (connectionId: string, reason: 'removed' | 'saved' | 'updated') => {
    listeners.forEach(listener => listener({ connectionId, reason }))
  }

  const removeGateway = async (id: string, { forget = false } = {}): Promise<GatewayRelayRemoval> => {
    const result = await api<GatewayRelayRemoval>({
      method: 'DELETE',
      path: `/api/webapp/relay/gateways/${encodeURIComponent(id)}${forget ? '?forget=1' : ''}`
    })

    if (result.removed) {
      await gateways()
      changed(id, 'removed')
    }

    return result
  }

  const unsupported = async (): Promise<never> => {
    throw new Error('Connect other gateways from Settings → Gateways in the Webapp')
  }

  const connections: BrowserRelays['connections'] = {
    list: async () => {
      await gateways()

      return registry()
    },
    onChanged: callback => {
      listeners.add(callback)

      return () => void listeners.delete(callback)
    },
    remove: async (id: string) => {
      const result = await removeGateway(id)

      if (!result.removed) {
        throw new Error(result.detail)
      }

      return { ok: true, registry: registry() }
    },
    save: unsupported,
    setLastUsed: async (id: string) => {
      window.localStorage.setItem(LAST_USED_KEY, id)

      return { ok: true, registry: registry() }
    },
    setPrimary: unsupported,
    test: async (id: string): Promise<DesktopConnectionTestResult> => {
      if (id === LOCAL_CONNECTION_ID) {
        return { baseUrl: window.location.origin, ok: true, reachable: true, version: null }
      }

      const result = await api<{ error?: string; ok?: boolean; reachable?: boolean }>({
        method: 'POST',
        path: `/api/webapp/relay/gateways/${encodeURIComponent(id)}/test`
      })

      return {
        baseUrl: gateway(id)?.url,
        error: result?.ok ? null : result?.error || 'Not reachable',
        ok: Boolean(result?.ok),
        reachable: Boolean(result?.reachable),
        version: null
      }
    }
  }

  const bridge: BrowserRelays['bridge'] = {
    gateways,
    grants: async () =>
      (await api<{ grants?: GatewayRelayGrant[] }>({ path: '/api/webapp/relay/grants' }))?.grants ?? [],
    removeGateway,
    revokeGrant: async (grantId: string) => {
      await api({ method: 'DELETE', path: `/api/webapp/relay/grants/${encodeURIComponent(grantId)}` })
    },
    startPairing: async (url: string, label: string) => {
      const result = await api<{ authorize_url?: string }>({
        body: { label, url },
        method: 'POST',
        path: '/api/webapp/relay/gateways'
      })

      if (!result?.authorize_url) {
        throw new Error('Hermes did not start the pairing')
      }

      return result.authorize_url
    }
  }

  return { bootstrapFor, bridge, connection, connections, gateway, gateways, rosterAgents, socketUrl }
}
