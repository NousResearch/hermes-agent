import { atom } from 'nanostores'

import type { DesktopAgentRoster } from '@/global'

// The union agent roster — every profile on every registered gateway — as
// Electron enumerates it over REST/SSH (`hermes:agents:roster`). Bot Mode and
// the Capabilities scope selector already read it; the Sessions profile rail
// is its third consumer. Fetched on demand only (mount / focus / registry
// change), never on a timer: the multi-connection docs rule out periodic
// fleet polling from the sidebar.
export const $fleetRoster = atom<DesktopAgentRoster | null>(null)

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
      $fleetRoster.set(roster)
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
