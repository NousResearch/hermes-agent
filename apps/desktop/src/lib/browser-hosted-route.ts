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
}

const named = (connectionId?: null | string) =>
  connectionId && connectionId !== LOCAL_CONNECTION_ID ? connectionId : undefined

export function createHostedRoute({ relays, windowAgent }: HostedRouteOptions): HostedRoute {
  let pinned: null | Promise<string> = null

  const knownAgentFor = (connectionId?: null | string) =>
    named(connectionId) ?? named(windowAgent()) ?? relays.knownWindowConnection()

  // The first bare dial pins the agent it opened on; a failed read pins nothing.
  const home = () =>
    (pinned ??= relays.windowConnection().catch(error => {
      pinned = null
      throw error
    }))

  return {
    agentFor: async connectionId => knownAgentFor(connectionId) ?? (await relays.windowConnection()),
    home,
    knownAgentFor,
    relays
  }
}
