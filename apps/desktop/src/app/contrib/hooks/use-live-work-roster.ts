import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { $gateway } from '@/store/gateway'
import { $liveWorkRoster, setLiveWorkRosterSource } from '@/store/live-work-roster'
import { $gatewayState, $sessions } from '@/store/session'
import { ambientRequestFor } from '@/store/session-gone-latch'
import { requestForSessionProfile, type SessionOwnerScope } from '@/store/session-request-router'
import { knownOwnerForSession } from '@/store/session-states'

const POLL_MS = 10_000
const MAX_FAILURES = 3
const AMBIENT = 'ambient'

interface RosterResult {
  sessions?: unknown[]
}

/** Every backend owner the sidebar shows rows for. The ambient socket answers
 *  for the launch profile; a row owned by another profile (or connection) is
 *  asked on the same route its own session RPCs take, so each backend only
 *  reports conversations stored in that profile. */
function rosterSources(): Map<string, SessionOwnerScope> {
  const sources = new Map<string, SessionOwnerScope>([[AMBIENT, undefined]])

  for (const row of $sessions.get()) {
    const owner = knownOwnerForSession(row.id)

    if (owner) {
      sources.set(JSON.stringify(owner), owner)
    }
  }

  return sources
}

/** App-level hydration of `$liveWorkRoster` (sidebar rows not on screen, and
 *  every row after a reload / socket reconnect): one `subagent.roster` read
 *  per backend owner on connect and every 10 s while the window is visible.
 *  A source that keeps failing (an older backend without the RPC) stops until
 *  the next reconnect or window focus, so it never hot-loops. */
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
    const failures = new Map<string, number>()

    const refresh = async () => {
      const gateway = $gateway.get()

      if (cancelled || pending || !gateway) {
        return
      }

      pending = true
      const ambient = ambientRequestFor(gateway)
      const sources = rosterSources()

      try {
        await Promise.all(
          [...sources].map(async ([key, owner]) => {
            if ((failures.get(key) ?? 0) >= MAX_FAILURES) {
              return
            }

            try {
              const result = await requestForSessionProfile<RosterResult>(owner, ambient, 'subagent.roster', {})

              if (!cancelled) {
                setLiveWorkRosterSource(key, Array.isArray(result?.sessions) ? result.sessions : [])
              }

              failures.delete(key)
            } catch {
              failures.set(key, (failures.get(key) ?? 0) + 1)

              if (!cancelled) {
                setLiveWorkRosterSource(key, null)
              }
            }
          })
        )

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

    void refresh()

    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void refresh()
      }
    }, POLL_MS)

    const retry = () => {
      failures.clear()
      void refresh()
    }

    window.addEventListener('focus', retry)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      window.removeEventListener('focus', retry)
    }
  }, [gatewayState])
}
