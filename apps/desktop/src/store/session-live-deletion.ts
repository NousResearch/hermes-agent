import type { GatewayEvent } from '@hermes/shared'

import {
  $cronSessions,
  $messagingSessions,
  $sessions,
  setCronSessions,
  setMessagingSessions,
  setSessions
} from './session'
import { tombstoneSessions } from './session-removal'
import { $sessionStates, $workingSessionIds } from './session-states'

/** Remove exact sidebar IDs without refreshing a transcript or touching a live
 * runtime. Existing tombstone generations fence older in-flight list pages. */
export function notifySessionsDeleted(payload: GatewayEvent<'sessions.deleted'>['payload'] | undefined): void {
  if (!payload) {
    return
  }

  const ids = new Set(payload.session_ids)
  const live = new Set($workingSessionIds.get())

  for (const state of Object.values($sessionStates.get())) {
    if (state.busy || state.awaitingResponse || state.turnLive || state.needsInput) {
      if (state.storedSessionId) {
        live.add(state.storedSessionId)
      }
    }
  }

  // Fence even IDs not loaded yet: their first page may already be in flight.
  // Existing tombstones are ID-keyed, so a known foreign-profile collision must
  // stay out of that overlay rather than hiding the other profile's row.
  const foreign = new Set<string>()

  for (const rows of [$sessions.get(), $cronSessions.get(), $messagingSessions.get()]) {
    for (const row of rows) {
      if ((row.profile ?? 'default') !== payload.profile) {
        foreign.add(row.id)
      }
    }
  }

  const removed = new Set([...ids].filter(id => !live.has(id) && !foreign.has(id)))

  if (!removed.size) {
    return
  }

  tombstoneSessions([...removed])

  const keep = (rows: typeof $sessions.value) => {
    const next = rows.filter(row => !removed.has(row.id) || (row.profile ?? 'default') !== payload.profile)

    return next.length === rows.length ? rows : next
  }

  setSessions(keep)
  setCronSessions(keep)
  setMessagingSessions(keep)
}
