/**
 * FOREIGN-LIVE — DB-derived liveness for sessions this renderer has no
 * runtime for (cli one-shots, cron runs, other profiles, TUI sessions).
 *
 * The backend already stamps every list row with `is_active`
 * (ended_at IS NULL AND now - last_active < 300, web_routers/sessions.py).
 * Claim a stored id ONLY when the row says active AND no runtime exists in
 * $sessionStates (event-owned sessions are the authoritative path and must
 * win). Refreshes ride the existing sessions.changed list refresh; a real
 * gateway event for the session removes it from this set by construction
 * (the runtime then exists, so the predicate stops claiming the id).
 */
import { computed } from 'nanostores'

import { $sessions, lineageAliases } from './session'
import { $sessionStates } from './session-states'

export const $foreignLiveSessionIds = computed([$sessions, $sessionStates], (sessions, states) => {
  const owned = new Set<string>()

  for (const state of Object.values(states)) {
    if (state?.storedSessionId) {
      owned.add(state.storedSessionId)
    }
  }

  const foreign = new Set<string>()

  for (const session of sessions) {
    if (!session.is_active || owned.has(session.id)) {
      continue
    }

    for (const alias of lineageAliases(session.id, sessions)) {
      foreign.add(alias)
    }
  }

  return [...foreign]
})
