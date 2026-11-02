import { setCurrentBranch, setCurrentUsage, setSessionStartedAt } from '@/store/session'
import type { UsageStats } from '@/types/hermes'

import type { ClientSessionState } from '../../../types'

// The primary statusbar reads one global usage atom; each session keeps its own copy in its runtime
// state. A switch publishes the switched-to session's copy wholesale: a merge kept the previously
// viewed session's cache-hit / t/s wherever the new one had no reading yet.
export function publishSessionUsage(usage: null | UsageStats | undefined) {
  setCurrentUsage(usage ?? { calls: 0, input: 0, output: 0, total: 0 })
}

/** Warm switch: the cached runtime is this conversation's own truth, so its branch, clock and last
 *  usage reading paint at once instead of after its next turn. */
export function publishWarmSessionRuntime(state: ClientSessionState) {
  setCurrentBranch(state.branch)
  setSessionStartedAt(state.runtimeStartedAt)
  publishSessionUsage(state.usage)
}
