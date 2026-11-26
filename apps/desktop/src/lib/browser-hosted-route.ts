import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import type { BrowserSavedConnections } from '@/lib/browser-saved-connections'

/**
 * Where a request goes on a host that runs no agent (`browserHostRunsNoAgent`). The host has no
 * `local` gateway: that name, and a request that names no connection, mean the agent this window
 * is on. REST, sockets, terminals, pop-outs and profile routes all resolve through this one route,
 * so they always name the same agent.
 */
export interface HostedRoute {
  /** The account's agents, each reached through `/api/relay/<id>` on this origin. */
  relays: BrowserSavedConnections
  /** The agent a request naming `connectionId` reaches: that agent, or with none (or `local`) the window's. */
  agentFor: (connectionId?: null | string) => Promise<string>
  /** `agentFor` without a read: undefined before this page has read the list. */
  knownAgentFor: (connectionId?: null | string) => string | undefined
  /**
   * The agent the window's own socket opened on, kept for the window's life. The store dials and
   * reuses that socket for every profile named with no agent (`/profile`, a bare new chat), so the
   * descriptor and socket for such a profile name this agent too, wherever the window has moved
   * since, as a served Webapp's name the serving gateway.
   */
  home: () => Promise<string>
}

interface HostedRouteOptions {
  relays: BrowserSavedConnections
  /** The agent boot or the last switch published for this window, if any. */
  windowAgent: () => null | string | undefined
  /** The agent this window was opened for, if any (a popped-out chat's address names it). */
  launchAgent: () => null | string | undefined
}

const named = (connectionId?: null | string) =>
  connectionId && connectionId !== LOCAL_CONNECTION_ID ? connectionId : undefined

export function createHostedRoute({ launchAgent, relays, windowAgent }: HostedRouteOptions): HostedRoute {
  let pinned: string | undefined

  // A candidate counts only while the account still lists it: an agent that left the list is
  // refused by the relay, so the next rung answers instead.
  const listed = (connectionId?: null | string) => {
    const id = named(connectionId)

    return id && relays.gateway(id) ? id : undefined
  }

  // The window's own socket: the agent it opened on while listed, else the one the window was
  // opened for, else the list's primary (or first). The next dial opens on that, so it becomes the pin.
  const knownHome = () => listed(pinned) ?? listed(launchAgent()) ?? relays.knownWindowConnection()

  const home = async () => {
    const primary = await relays.windowConnection()
    pinned = listed(pinned) ?? listed(launchAgent()) ?? primary

    return pinned
  }

  // No connection (or `local`): the agent boot or the last switch published while listed, else the
  // window's own socket's.
  const knownAgentFor = (connectionId?: null | string) => named(connectionId) ?? listed(windowAgent()) ?? knownHome()

  return {
    agentFor: async connectionId => {
      const id = named(connectionId)

      if (id) {
        return id
      }

      // Read the list once, so every rung is checked against it.
      await relays.windowConnection()

      return listed(windowAgent()) ?? home()
    },
    home,
    knownAgentFor,
    relays
  }
}
