import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { $gateway, activeGatewayProfileKey, gatewayRouteState, isActivePrimary } from '@/store/gateway'
import { $primaryGatewayRoute } from '@/store/gateway-primary-route'
import { $liveWorkRoster, rosterSourceHasWork, setLiveWorkRosterSource } from '@/store/live-work-roster'
import { $gatewayState, ownerLookupSessionRows } from '@/store/session'
import { ambientRequestFor } from '@/store/session-gone-latch'
import { isSessionOwnerRoute, requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

const POLL_MS = 10_000
// A closed socket whose last read showed work is re-dialed this often, not every tick.
const COLD_BUSY_MS = 60_000
// Window focus re-reads every source at most this often (alt-tab bursts).
const FOCUS_MIN_GAP_MS = 5_000
const MAX_BACKOFF_MS = 5 * 60_000

interface RosterResult {
  sessions?: unknown[]
}

interface RosterSource {
  open: boolean
  owner: SessionOwnerScope
}

const routeOf = (owner: SessionOwnerScope) =>
  isSessionOwnerRoute(owner)
    ? gatewayRouteState(owner.connectionId, owner.profile)
    : gatewayRouteState(null, String(owner ?? ''))

/** Every backend the sidebar shows rows for (recents, cron and messaging rows
 *  alike), keyed by the socket a read would ride, so one backend is asked once
 *  however many spellings its owner has (the launch profile used to be asked
 *  three times a tick). The ambient socket answers for the active profile; a row owned by another profile (or connection) is asked on
 *  the route its own session RPCs take, so each backend only reports
 *  conversations stored in that profile. */
function rosterSources(): Map<string, RosterSource> {
  // The ambient read carries no profile, so it shares a key only when it IS the
  // window primary answering for its own profile.
  const ambientKey = isActivePrimary() ? gatewayRouteState(null, activeGatewayProfileKey()).key : 'ambient'
  const sources = new Map<string, RosterSource>([[ambientKey, { open: true, owner: undefined }]])

  // One row list per tick, each owner resolved against its own row: resolving
  // through the whole list per row made a tick O(rows²).
  for (const row of ownerLookupSessionRows()) {
    const owner = knownOwnerForSession(row.id, [row])

    if (!owner) {
      continue
    }

    const route = routeOf(owner)

    if (!sources.has(route.key)) {
      sources.set(route.key, { open: route.open, owner })
    }
  }

  return sources
}

/** App-level hydration of `$liveWorkRoster` (sidebar rows not on screen, and
 *  every row after a reload / socket reconnect): one `subagent.roster` read per
 *  backend on connect and window focus, then every 10 s while the window is
 *  visible. Each source reads and publishes on its own, with a one-tick request
 *  timeout, so a hung backend never freezes the other sources' rows. Interval
 *  ticks are passive: after a source's first read on this connection they read
 *  only sockets already open, plus a closed one whose last read showed work once
 *  a minute (so a finished child clears without re-dialing every tick). A backend
 *  without the RPC stops for this connection; any other failure backs off (focus
 *  included) and keeps retrying, so a recovered socket comes back on its own. */
export function useLiveWorkRoster() {
  const gatewayState = useStore($gatewayState)
  // Route keys (`primary:<profile>`, `ambient`) name a socket, not a backend:
  // a soft re-home of the primary must start over, even while the foreground
  // socket (a secondary) stays open, or old buckets and stops outlive it.
  const { connectionId: primaryConnection, mode: primaryMode } = useStore($primaryGatewayRoute)

  useEffect(() => {
    // Counts from a dropped socket or a previous primary are unverified; the
    // (re)connect re-reads them.
    $liveWorkRoster.set({})

    if (gatewayState !== 'open') {
      return
    }

    let cancelled = false
    let lastHydrate = -Infinity
    let current = new Set<string>()
    const inFlight = new Set<string>()
    const unsupported = new Set<string>()
    // When each source was last asked on this connection: rows that load after
    // the connect (the session list, a newly shown profile) get their first read.
    const sentAt = new Map<string, number>()
    const backoff = new Map<string, { failures: number; until: number }>()

    const readSource = async (key: string, owner: SessionOwnerScope, ambient: ReturnType<typeof ambientRequestFor>) => {
      const at = Date.now()
      sentAt.set(key, at)
      inFlight.add(key)

      try {
        const result = await requestForSessionProfile<RosterResult>(owner, ambient, 'subagent.roster', {}, POLL_MS)
        backoff.delete(key)

        // A source whose rows left the sidebar mid-read stays pruned.
        if (!cancelled && current.has(key)) {
          setLiveWorkRosterSource(key, Array.isArray(result?.sessions) ? result.sessions : [], at)
        }
      } catch (error) {
        if (isMissingRpcMethod(error)) {
          unsupported.add(key)
        } else {
          const failures = (backoff.get(key)?.failures ?? 0) + 1
          backoff.set(key, { failures, until: Date.now() + Math.min(POLL_MS * 2 ** failures, MAX_BACKOFF_MS) })
        }

        if (!cancelled) {
          setLiveWorkRosterSource(key, null)
        }
      } finally {
        inFlight.delete(key)
      }
    }

    const refresh = (hydrate: boolean) => {
      const gateway = $gateway.get()

      if (cancelled || !gateway) {
        return
      }

      const ambient = ambientRequestFor(gateway)
      const sources = rosterSources()
      const roster = $liveWorkRoster.get()
      const now = Date.now()
      current = new Set(sources.keys())

      // Owners whose rows left the sidebar stop contributing.
      for (const key of Object.keys(roster)) {
        if (!current.has(key)) {
          setLiveWorkRosterSource(key, null)
        }
      }

      for (const [key, { open, owner }] of sources) {
        const last = sentAt.get(key)
        const due = !inFlight.has(key) && !unsupported.has(key) && now >= (backoff.get(key)?.until ?? 0)

        const warm =
          hydrate || open || last === undefined || (rosterSourceHasWork(roster, key) && now - last >= COLD_BUSY_MS)

        if (due && warm) {
          void readSource(key, owner, ambient)
        }
      }
    }

    const hydrate = () => {
      if (Date.now() - lastHydrate >= FOCUS_MIN_GAP_MS) {
        lastHydrate = Date.now()
        refresh(true)
      }
    }

    hydrate()

    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        refresh(false)
      }
    }, POLL_MS)

    window.addEventListener('focus', hydrate)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      window.removeEventListener('focus', hydrate)
    }
  }, [gatewayState, primaryConnection, primaryMode])
}
