import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type {
  DesktopConnectionsRegistry,
  DesktopRegistryConnection,
  DesktopRosterAgent,
  HermesApiRequest,
  HermesConnection
} from '@/global'
import { createServedProfiles, SERVED_SET_READ_TIMEOUT_MS, type ServedProfiles } from '@/lib/browser-served-profiles'
import { browserApi, type BrowserBootstrap, websocketUrl } from '@/lib/browser-transport'
import { withTimeout } from '@/lib/with-timeout'

/**
 * This machine's saved gateway connections, browser side: the ONE list native Desktop keeps
 * (`hermes_cli/gateway_connections.py`), read and written over `/api/gateway-connections`, so the
 * Webapp's Settings → Gateways is Desktop's own Saved connections page and both see every change.
 *
 * Every request to a saved gateway goes to this same origin under `/api/relay/<connection id>`; the
 * serving gateway sends it on with the credential saved for that connection, as Desktop sends it
 * (`hermes_cli/webapp_relay_target.py`). So a saved connection is a bootstrap with that base path:
 * REST, file links and uploads reuse the serving gateway's transport unchanged. A relayed socket
 * carries only this gateway's one-use `relay_ticket`; the serving gateway mints the other gateway's
 * credential for each dial. The ids are the registry's own, the ones native Desktop routes by, so
 * profile routes, Bot Mode and group-chat records name the same gateway on both hosts.
 */

type Api = <T>(request: HermesApiRequest) => Promise<T>

type ChangeReason = 'removed' | 'saved' | 'updated'

type ChangeListener = (payload: { connectionId: string; reason: ChangeReason }) => void

type Bridge = NonNullable<Window['hermesDesktop']['connections']>

const ROUTE = '/api/gateway-connections'

/** The sockets a relayed connection opens; each is admitted here by this gateway's relay ticket. */
export type RelayedSocketPath =
  '/api/audio/speak-stream' | '/api/audio/transcribe-stream' | '/api/host-terminal' | '/api/ws'

export function relayBasePath(bootstrap: BrowserBootstrap, connectionId: string): string {
  return `${bootstrap.basePath}/api/relay/${encodeURIComponent(connectionId)}`
}

/** *bootstrap* aimed at a saved gateway: same origin and sign-in, the relay's base path. */
export function relayBootstrap(bootstrap: BrowserBootstrap, connectionId: string): BrowserBootstrap {
  return { ...bootstrap, basePath: relayBasePath(bootstrap, connectionId) }
}

// What makes a saved connection dial somewhere else: a change here disposes and re-dials its sockets.
const dialKey = (row: DesktopRegistryConnection) =>
  JSON.stringify([row.kind, row.url, row.authMode, row.tokenPreview, row.headerNames ?? [], row.host, row.port])

/** Which connections changed between two snapshots, and how (the native `connections:changed` reasons). */
export function registryChanges(
  before: DesktopConnectionsRegistry | null,
  after: DesktopConnectionsRegistry
): { connectionId: string; reason: ChangeReason }[] {
  if (!before) {
    return []
  }

  const old = new Map(before.connections.map(row => [row.id, row]))
  const now = new Map(after.connections.map(row => [row.id, row]))
  const changes: { connectionId: string; reason: ChangeReason }[] = []

  for (const [id, row] of old) {
    const next = now.get(id)

    if (!next) {
      changes.push({ connectionId: id, reason: 'removed' })
    } else if (dialKey(next) !== dialKey(row)) {
      changes.push({ connectionId: id, reason: 'updated' })
    } else if (JSON.stringify(next) !== JSON.stringify(row)) {
      changes.push({ connectionId: id, reason: 'saved' })
    }
  }

  for (const id of now.keys()) {
    if (!old.has(id)) {
      changes.push({ connectionId: id, reason: 'saved' })
    }
  }

  const settings = (reg: DesktopConnectionsRegistry) => [reg.primary, reg.launchMode, reg.lastUsed].join('\n')

  if (!changes.length && settings(before) !== settings(after)) {
    changes.push({ connectionId: after.primary, reason: 'saved' })
  }

  return changes
}

export interface BrowserSavedConnections {
  /** A saved connection this page reaches through its gateway, or undefined (`local` included). */
  gateway: (connectionId: null | string | undefined) => DesktopRegistryConnection | undefined
  /** The saved connections other than this machine's own, read fresh. */
  gateways: () => Promise<DesktopRegistryConnection[]>
  /** The bootstrap a request for *connectionId* rides: the serving gateway's, or its relay's. */
  bootstrapFor: (connectionId: null | string | undefined) => BrowserBootstrap
  /** A fresh socket URL on *connectionId* for *path*, admitted by a relay ticket minted for this dial. */
  socketUrl: (connectionId: string, path: RelayedSocketPath, profile?: null | string) => Promise<string>
  connection: (connectionId: string, profile?: null | string) => HermesConnection
  rosterAgents: (connectionId: string) => Promise<DesktopRosterAgent[]>
  /** Whether *connectionId* serves *profile*, from the set its last inventory read reported. */
  serves: (connectionId: string, profile?: null | string) => boolean
  /** Re-read *connectionId*'s inventory and, if it changed or went stale, its served set. Never throws. */
  refreshServed: (connectionId: string) => Promise<void>
  /** The label this machine's own entry (`local`) carries in the list, once read. */
  localLabel: () => string | undefined
  connections: Bridge
}

/**
 * How long the roster waits for a saved gateway's profiles before listing it unreachable. One that
 * takes the connection and never answers would otherwise hold the switcher for a whole request timeout.
 */
export const RELAYED_ROSTER_TIMEOUT_MS = 8_000

/** The connection kinds the serving gateway reaches for this page (SSH needs a tunnel it does not hold yet). */
const REACHED_KINDS = new Set(['remote', 'cloud'])

export function createBrowserSavedConnections({
  api,
  bootstrap
}: {
  api: Api
  bootstrap: BrowserBootstrap
}): BrowserSavedConnections {
  let snapshot: DesktopConnectionsRegistry | null = null
  const listeners = new Set<ChangeListener>()

  const gateway = (connectionId: null | string | undefined) =>
    connectionId && connectionId !== LOCAL_CONNECTION_ID
      ? snapshot?.connections.find(row => row.id === connectionId && row.kind !== 'local')
      : undefined

  // Every read publishes what changed since the last one, so this tab follows an edit made in
  // native Desktop or another tab the way a native window follows `connections:changed`.
  const publish = (next: DesktopConnectionsRegistry) => {
    const changes = registryChanges(snapshot, next)
    snapshot = next
    changes.forEach(change => listeners.forEach(listener => listener(change)))

    return next
  }

  const list = async () => {
    const read = await api<DesktopConnectionsRegistry | null>({ path: ROUTE })

    if (!Array.isArray(read?.connections)) {
      throw new Error('Hermes did not return the saved connections')
    }

    return publish(read)
  }

  const mutated = async <T extends { registry: DesktopConnectionsRegistry }>(request: HermesApiRequest) => {
    const result = await api<T>(request)
    publish(result.registry)

    return result
  }

  const gateways = async () => (await list()).connections.filter(row => row.kind !== 'local')

  const bootstrapFor = (connectionId: null | string | undefined) => {
    if (!connectionId || connectionId === LOCAL_CONNECTION_ID) {
      return bootstrap
    }

    if (!gateway(connectionId)) {
      throw new Error(`No connection with id "${connectionId}"`)
    }

    return relayBootstrap(bootstrap, connectionId)
  }

  const socketUrl = async (connectionId: string, path: RelayedSocketPath, profile?: null | string) => {
    const relayed = bootstrapFor(connectionId)

    const minted = await browserApi<null | { ticket?: string }>(bootstrap, {
      method: 'POST',
      path: '/api/webapp/socket-ticket?socket=relay'
    })

    const ticket = String(minted?.ticket || '').trim()

    if (!ticket) {
      throw new Error('Hermes did not return a relay ticket')
    }

    const url = new URL(websocketUrl(relayed.basePath, path, {}, profile))
    url.searchParams.set('relay_ticket', ticket)

    return url.toString()
  }

  const connection = (connectionId: string, profile?: null | string): HermesConnection => {
    const row = gateway(connectionId)
    const relayed = bootstrapFor(connectionId)

    return {
      authMode: row?.kind === 'cloud' ? 'oauth' : row?.authMode || 'token',
      baseUrl: `${window.location.origin}${relayed.basePath}`,
      connectionId,
      isFullscreen: Boolean(document.fullscreenElement),
      logs: [],
      mode: 'remote',
      nativeOverlayWidth: 0,
      profile: profile || undefined,
      registryScoped: true,
      remoteHost: row?.url ? new URL(row.url).host : connectionId,
      remoteKind: row?.kind === 'cloud' ? 'cloud' : 'url',
      sharedPrimary: false,
      sharedRemote: Boolean(profile),
      source: 'settings',
      token: '',
      windowButtonPosition: null,
      // Credential-free and the legacy dial native Desktop makes to a remote; every dial mints its ticket.
      wsUrl: websocketUrl(relayed.basePath, '/api/ws', {}, profile)
    }
  }

  // Each gateway's served profiles, kept by the serving gateway's own rules
  // (browser-served-profiles.ts): read with an inventory read, re-read when the installed list
  // changes or the answer is 30 s old. Kept per dial target, not per id alone: a connection edited
  // to point elsewhere must not inherit the old target's answer or a read still in flight.
  const served = new Map<string, ServedProfiles>()

  const servedKey = (connectionId: string) => {
    const row = gateway(connectionId)

    return row ? `${row.id}\n${dialKey(row)}` : connectionId
  }

  const servedFor = (connectionId: string) => {
    const key = servedKey(connectionId)
    let set = served.get(key)

    if (!set) {
      const relayed: Api = <T>(request: HermesApiRequest) => browserApi<T>(bootstrapFor(connectionId), request)

      set = createServedProfiles(
        relayed,
        async () =>
          (await relayed<{ current?: string }>({ path: '/api/profiles/active', timeoutMs: SERVED_SET_READ_TIMEOUT_MS }))
            ?.current
      )
      served.set(key, set)
    }

    return set
  }

  // A socket that names no profile reaches the one the gateway runs as.
  const serves = (connectionId: string, profile?: null | string) =>
    !profile?.trim() || (served.get(servedKey(connectionId))?.serves(profile) ?? true)

  const installedProfiles = async (connectionId: string, timeoutMs: number) => {
    const result = await browserApi<{ profiles?: { name?: string }[] }>(bootstrapFor(connectionId), {
      path: '/api/profiles',
      timeoutMs
    })

    const names = [
      ...new Set((result?.profiles || []).map(profile => String(profile.name || '').trim()).filter(Boolean))
    ]

    await servedFor(connectionId).refresh(names)

    return names
  }

  // A profile created there since the last roster read is missing from the set. Bounded like the
  // serving gateway's own refresh on a pick, so a profile that stays unserved is still refused.
  const refreshServed = (connectionId: string) =>
    withTimeout(
      installedProfiles(connectionId, SERVED_SET_READ_TIMEOUT_MS),
      SERVED_SET_READ_TIMEOUT_MS,
      'served profile refresh timed out'
    ).then(
      () => undefined,
      () => undefined
    )

  const rosterAgents = async (connectionId: string): Promise<DesktopRosterAgent[]> => {
    const row = gateway(connectionId)

    if (row && !REACHED_KINDS.has(row.kind)) {
      throw new Error(`${row.label} is an SSH connection, which the Webapp does not reach yet`)
    }

    const names = await withTimeout(
      installedProfiles(connectionId, RELAYED_ROSTER_TIMEOUT_MS),
      RELAYED_ROSTER_TIMEOUT_MS,
      `${row?.label ?? connectionId} did not answer`
    )

    return names.map(profile => ({
      connectionId,
      connectionKind: row?.kind ?? 'remote',
      connectionLabel: row?.label ?? connectionId,
      handle: profile,
      profile,
      targetProfile: profile,
      // Installed there, so listed, but served by another gateway or parked.
      ...(serves(connectionId, profile) ? {} : { reachable: false as const })
    }))
  }

  const id = (connectionId: string) => encodeURIComponent(connectionId)

  const connections: Bridge = {
    list,
    onChanged: callback => {
      listeners.add(callback)

      return () => void listeners.delete(callback)
    },
    remove: connectionId => mutated({ method: 'DELETE', path: `${ROUTE}/${id(connectionId)}` }),
    save: payload => mutated({ body: payload, method: 'POST', path: ROUTE }),
    setLastUsed: connectionId => mutated({ body: { id: connectionId }, method: 'POST', path: `${ROUTE}/last-used` }),
    setLaunchMode: mode => mutated({ body: { mode }, method: 'POST', path: `${ROUTE}/launch-mode` }),
    setPrimary: connectionId => mutated({ method: 'POST', path: `${ROUTE}/${id(connectionId)}/primary` }),
    test: connectionId => api({ method: 'POST', path: `${ROUTE}/${id(connectionId)}/test`, timeoutMs: 20_000 })
  }

  // Another window or native Desktop may have edited the list while this tab was in the background.
  window.addEventListener('focus', () => void list().catch(() => undefined))

  return {
    bootstrapFor,
    connection,
    connections,
    gateway,
    gateways,
    localLabel: () => snapshot?.connections.find(row => row.id === LOCAL_CONNECTION_ID)?.label,
    refreshServed,
    rosterAgents,
    serves,
    socketUrl
  }
}
