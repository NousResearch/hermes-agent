import { atom } from 'nanostores'

import type { DesktopAgentRoster } from '@/global'

/** The roster plus the moment it was confirmed. The work dots read
 *  `fetchedAt` to tell a live claim from one nobody has re-checked. */
export type FleetRosterSnapshot = DesktopAgentRoster & { fetchedAt: number }

// The union agent roster — every profile on every registered gateway — as
// Electron enumerates it over REST/SSH (`hermes:agents:roster`). Bot Mode and
// the Capabilities scope selector already read it; the Sessions profile rail
// is its third consumer. Refreshed on demand (mount / focus / registry change)
// and by one slow timer while the rail is mounted: worker heartbeats are
// invisible to the gateway sockets, so a poll is the only truthful source for
// "is a bot working?" — but it reuses pooled backends and the SSH inventory
// cache, so it never spawns a gateway.
export const $fleetRoster = atom<FleetRosterSnapshot | null>(null)

const FLEET_ROSTER_STALE_MS = 60_000
// Focus/mount may retry a partial result sooner, but never on every event.
const FLEET_ROSTER_RETRY_MS = 5_000

let fetchedAt = 0
let inflight: null | Promise<void> = null
let forcedRefresh: null | Promise<void> = null

export async function refreshFleetRoster(options: { force?: boolean } = {}): Promise<void> {
  const bridge = window.hermesDesktop?.getAgentRoster

  if (!bridge) {
    return
  }

  const current = $fleetRoster.get()
  const incomplete = current?.sources.some(source => !source.reachable && source.error !== 'connect-on-demand')
  const staleMs = incomplete ? FLEET_ROSTER_RETRY_MS : FLEET_ROSTER_STALE_MS

  if (!options.force && current && Date.now() - fetchedAt < staleMs) {
    return
  }

  if (inflight) {
    if (options.force) {
      forcedRefresh ??= inflight.then(() => {
        forcedRefresh = null

        return refreshFleetRoster({ force: true })
      })

      return forcedRefresh
    }

    return inflight
  }

  inflight = bridge()
    .then(roster => {
      $fleetRoster.set({ ...roster, fetchedAt: Date.now() })
      fetchedAt = Date.now()
    })
    .catch((error: unknown) => {
      // A failed enumeration keeps the last roster: the rail should not lose
      // a machine's squares because one refresh hit a sleeping box.
      console.warn('[fleet-roster] enumeration failed; keeping the previous roster', error)
    })
    .finally(() => {
      inflight = null
    })

  return inflight
}

/** @internal */
export function _resetFleetRosterForTests(): void {
  $fleetRoster.set(null)
  fetchedAt = 0
  inflight = null
  forcedRefresh = null
}
