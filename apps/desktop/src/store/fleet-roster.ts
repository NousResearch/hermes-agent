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

let fetchedAt = 0
let inflight: null | Promise<void> = null

/**
 * True when every registered source answered with profiles (or is deliberately
 * connect-on-demand). An incomplete snapshot — oauth remotes timing out on
 * cold start (#101339) — must not sit in the 60s stale window and block focus
 * retries after Settings → Test warms the session.
 */
export function fleetRosterIsComplete(roster: DesktopAgentRoster | null | undefined): boolean {
  const sources = roster?.sources

  if (!Array.isArray(sources) || sources.length === 0) {
    return true
  }

  return sources.every(source => source?.reachable !== false || source?.error === 'connect-on-demand')
}

export async function refreshFleetRoster(options: { force?: boolean } = {}): Promise<void> {
  const bridge = window.hermesDesktop?.getAgentRoster

  if (!bridge) {
    return
  }

  const current = $fleetRoster.get()

  if (
    !options.force &&
    current &&
    Date.now() - fetchedAt < FLEET_ROSTER_STALE_MS &&
    fleetRosterIsComplete(current)
  ) {
    return
  }

  if (inflight) {
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
}
