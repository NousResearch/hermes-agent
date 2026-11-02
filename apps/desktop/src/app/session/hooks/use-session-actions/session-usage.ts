import { $sessions, sessionMatchesStoredId, setCurrentBranch, setCurrentUsage, setSessionStartedAt } from '@/store/session'
import type { UsageStats } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

// The primary statusbar reads one global usage atom; each session keeps its own copy in its runtime
// state. A switch publishes the switched-to session's copy wholesale: a merge kept the previously
// viewed session's cache-hit / t/s wherever the new one had no reading yet.
export function publishSessionUsage(usage: null | UsageStats | undefined) {
  setCurrentUsage(usage ?? { calls: 0, input: 0, output: 0, total: 0 })
}

// No reading yet (agentless since a gateway restart): the stored row's token totals, the same ones a
// cold open paints, rather than zeros.
function storedTotals(storedSessionId: null | string | undefined): null | UsageStats {
  const row = storedSessionId
    ? $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
    : undefined

  if (!row) {
    return null
  }

  const input = row.input_tokens || 0
  const output = row.output_tokens || 0

  return { calls: 0, input, output, total: input + output }
}

/** Warm switch: the cached runtime is this conversation's own truth, so its branch, clock and last
 *  usage reading paint at once instead of after its next turn. */
export function publishWarmSessionRuntime(state: ClientSessionState) {
  setCurrentBranch(state.branch)
  setSessionStartedAt(state.runtimeStartedAt)
  publishSessionUsage(state.usage ?? storedTotals(state.storedSessionId))
}
