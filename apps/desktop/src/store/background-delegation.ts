import { computed } from 'nanostores'

import { $subagentsBySession, activeSubagentCount } from './subagents'

export interface BackgroundResume {
  /** Running/queued background children belonging to this runtime session. */
  count: number
}

/** Background work outlives its spawning turn. The owning view gates this
 * signal on its own busy state; global focus and raw stream text are unrelated. */
export function sessionBackgroundResume(sessionId: null | string) {
  const $count = computed($subagentsBySession, bySession =>
    sessionId ? activeSubagentCount(bySession[sessionId] ?? []) : 0
  )

  // Stream frames must not repaint a count-only notice.
  return computed($count, (count): BackgroundResume | null => (count ? { count } : null))
}
