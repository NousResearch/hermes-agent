import { LOCAL_CONNECTION_ID } from '@hermes/shared'

import { translateNow } from '@/i18n/runtime'
import type { BrowserSavedConnections } from '@/lib/browser-saved-connections'

/**
 * The agent a window was opened for is not in the account's list. The window does not open on
 * another one in its place: a chat or a setting meant for that agent would land on another.
 */
export class HostedAgentUnavailableError extends Error {
  constructor(readonly agentId: string) {
    super(translateNow('boot.errors.hostedAgentUnavailable', agentId))
    this.name = 'HostedAgentUnavailableError'
  }
}

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
   * since, as a served Webapp's name the serving gateway. Rejects with `HostedAgentUnavailableError`
   * once that agent, or the one the address named at open, is not in the account's list.
   */
  home: () => Promise<string>
}

interface HostedRouteOptions {
  relays: BrowserSavedConnections
  /** The agent boot or the last switch published for this window, if any. */
  windowAgent: () => null | string | undefined
  /** The agent this window was opened for, if any (a popped-out chat's address names it). */
  launchAgent: () => null | string | undefined
  /** The agent the window was opened for is not listed: the window fails rather than open another. */
  onUnavailable?: (error: HostedAgentUnavailableError) => void
}

const named = (connectionId?: null | string) =>
  connectionId && connectionId !== LOCAL_CONNECTION_ID ? connectionId : undefined

export function createHostedRoute({
  launchAgent,
  onUnavailable,
  relays,
  windowAgent
}: HostedRouteOptions): HostedRoute {
  let pinned: string | undefined
  // Once the window could not open on its agent it opens on none: not on another one later either.
  let unavailable: HostedAgentUnavailableError | undefined

  // A candidate counts only while the account still lists it: an agent that left the list is
  // refused by the relay. The agent in view falls back to the window's own; that one never falls
  // back to another (`refuse`).
  const listed = (connectionId?: null | string) => {
    const id = named(connectionId)

    return id && relays.gateway(id) ? id : undefined
  }

  // The window's own socket: the agent it opened on, else the one the window was opened for, else
  // the list's primary (or first). The next dial opens on that, so it becomes the pin.
  const knownHome = () => {
    if (unavailable) {
      return undefined
    }

    const opened = pinned ?? named(launchAgent())

    return opened ? listed(opened) : relays.knownWindowConnection()
  }

  // An agent the window is for that the account no longer lists fails the window, naming the
  // agent, instead of opening another in its place: an address naming one (a bookmark, a tab
  // reopened after it left), or the one the window's socket opened on, which the socket cannot leave.
  const refuse = (agentId: string) => {
    if (!unavailable) {
      unavailable = new HostedAgentUnavailableError(agentId)
      onUnavailable?.(unavailable)
    }

    return unavailable
  }

  const home = async () => {
    const primary = await relays.windowConnection()

    if (unavailable) {
      throw unavailable
    }

    const opened = pinned ?? named(launchAgent())

    if (opened && !listed(opened)) {
      throw refuse(opened)
    }

    pinned = opened ?? primary

    return pinned
  }

  // Said when the list read that dropped it answers, not at the next request that names no agent.
  relays.connections.onChanged?.(({ connectionId, reason }) => {
    if (reason === 'removed' && connectionId === pinned) {
      refuse(connectionId)
    }
  })

  // No connection (or `local`): the agent boot or the last switch published while listed, else the
  // window's own socket's.
  const knownAgentFor = (connectionId?: null | string) => named(connectionId) ?? listed(windowAgent()) ?? knownHome()

  return {
    agentFor: async connectionId => {
      const id = named(connectionId)

      // Read the list once, so every rung is checked against it, and a request naming an agent finds
      // that agent's route (an early one would otherwise look in a list not read yet).
      await relays.windowConnection()

      if (id) {
        return id
      }

      return listed(windowAgent()) ?? home()
    },
    home,
    knownAgentFor,
    relays
  }
}
