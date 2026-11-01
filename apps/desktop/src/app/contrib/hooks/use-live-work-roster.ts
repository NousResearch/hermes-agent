import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { $gateway, activeGatewayProfileKey, gatewayRouteState, isActivePrimary } from '@/store/gateway'
import { $liveWorkRoster, rosterSourceHasWork, setLiveWorkRosterSource } from '@/store/live-work-roster'
import { $gatewayState, ownerLookupSessionRows } from '@/store/session'
import { ambientRequestFor } from '@/store/session-gone-latch'
import { isSessionOwnerRoute, requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

const POLL_MS = 10_000
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

  for (const row of ownerLookupSessionRows()) {
    const owner = knownOwnerForSession(row.id)

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
 *  visible. Interval ticks are passive: after a source's first read on this
 *  connection they read only sockets already open (plus sources whose last read
 *  showed live work, so a finished child clears), never re-dialing an idle
 *  historical owner every tick just to learn nothing runs there. A
 *  backend without the RPC stops for this connection; any other failure backs
 *  off and keeps retrying, so a recovered socket comes back on its own. */
export function useLiveWorkRoster() {
  const gatewayState = useStore($gatewayState)

  useEffect(() => {
    if (gatewayState !== 'open') {
      // Counts from a dropped socket are unverified; the reconnect re-reads them.
      $liveWorkRoster.set({})

      return
    }

    let cancelled = false
    let pending = false
    const unsupported = new Set<string>()
    // Sources read at least once on this connection: rows that load after the
    // connect (the session list, a newly shown profile) get their first read.
    const read = new Set<string>()
    const backoff = new Map<string, { failures: number; until: number }>()

    const readSource = async (key: string, owner: SessionOwnerScope, ambient: ReturnType<typeof ambientRequestFor>) => {
      const sentAt = Date.now()
      read.add(key)

      try {
        const result = await requestForSessionProfile<RosterResult>(owner, ambient, 'subagent.roster', {})

        if (!cancelled) {
          setLiveWorkRosterSource(key, Array.isArray(result?.sessions) ? result.sessions : [], sentAt)
        }

        backoff.delete(key)
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
      }
    }

    const refresh = async (hydrate: boolean) => {
      const gateway = $gateway.get()

      if (cancelled || pending || !gateway) {
        return
      }

      pending = true
      const ambient = ambientRequestFor(gateway)
      const sources = rosterSources()
      const roster = $liveWorkRoster.get()
      const now = Date.now()

      try {
        await Promise.all(
          [...sources].map(([key, { open, owner }]) => {
            const due = hydrate || now >= (backoff.get(key)?.until ?? 0)
            const warm = hydrate || open || !read.has(key) || rosterSourceHasWork(roster, key)

            return unsupported.has(key) || !due || !warm ? undefined : readSource(key, owner, ambient)
          })
        )

        if (cancelled) {
          return
        }

        // Owners whose rows left the sidebar stop contributing.
        for (const key of Object.keys($liveWorkRoster.get())) {
          if (!sources.has(key)) {
            setLiveWorkRosterSource(key, null)
          }
        }
      } finally {
        pending = false
      }
    }

    void refresh(true)

    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void refresh(false)
      }
    }, POLL_MS)

    const onFocus = () => void refresh(true)
    window.addEventListener('focus', onFocus)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      window.removeEventListener('focus', onFocus)
    }
  }, [gatewayState])
}
