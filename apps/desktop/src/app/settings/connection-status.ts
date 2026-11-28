import type { ConnectionState } from '@hermes/shared'

import type { DesktopAgentRoster, DesktopRegistryConnection } from '@/global'

/** A saved connection's state as the app already knows it (18563:12574 OFFLINE / CONNECTED). */
export type ConnectionStatus = 'offline' | 'online' | 'pending'

interface ConnectionFacts {
  /** The connection this window is on: its own socket says how it is. */
  activeConnectionId: null | string
  gatewayState: ConnectionState
  /** The fleet probe: every other saved connection's reachability and the agents it serves. */
  roster: DesktopAgentRoster | null
}

const ACTIVE_STATUS: Record<ConnectionState, ConnectionStatus> = {
  open: 'online',
  connecting: 'pending',
  idle: 'pending',
  closed: 'offline',
  error: 'offline'
}

/**
 * The window's own connection reads its socket; any other reads the fleet probe. A connection the
 * probe has not reached yet, or only dials when asked (`connect-on-demand`), has no known state, so
 * it gets none rather than a guessed OFFLINE.
 */
export function connectionStatus(connectionId: string, facts: ConnectionFacts): ConnectionStatus | null {
  if (connectionId === facts.activeConnectionId) {
    return ACTIVE_STATUS[facts.gatewayState]
  }

  const source = facts.roster?.sources.find(entry => entry.connectionId === connectionId)

  if (!source || (!source.reachable && source.error === 'connect-on-demand')) {
    return null
  }

  return source.reachable ? 'online' : 'offline'
}

/** How many agents the fleet probe found on a reachable connection; null when it does not know. */
export function connectionAgentCount(connectionId: string, roster: DesktopAgentRoster | null): null | number {
  const source = roster?.sources.find(entry => entry.connectionId === connectionId)

  if (!roster || !source?.reachable) {
    return null
  }

  return roster.agents.filter(agent => agent.connectionId === connectionId).length
}

/** Where a saved connection lives, short: the URL's host (and any path it is served under), or
 *  `user@host:port` over SSH. */
export function connectionAddress(conn: DesktopRegistryConnection): string {
  if (conn.kind === 'ssh') {
    return conn.host ? `${conn.user ? `${conn.user}@` : ''}${conn.host}${conn.port ? `:${conn.port}` : ''}` : ''
  }

  if (conn.url && URL.canParse(conn.url)) {
    const url = new URL(conn.url)
    const path = url.pathname.replace(/\/+$/, '')

    return `${url.host}${path}`
  }

  return conn.url ?? ''
}
