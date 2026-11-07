import { computed } from 'nanostores'

import { $subagentsBySession, activeSubagentCount, type SubagentProgress } from './subagents'

export interface BackgroundResume {
  /** Running/queued background children belonging to this runtime session. */
  count: number
  /** Epoch ms the oldest of those children started — the wait's timer origin. */
  since: number
}

const isActive = (item: SubagentProgress) => item.status === 'queued' || item.status === 'running'

/** Background work outlives its spawning turn. The owning view gates this
 * signal on its own busy state; global focus and raw stream text are unrelated. */
export function sessionBackgroundResume(sessionId: null | string) {
  const $items = computed($subagentsBySession, bySession => (sessionId ? (bySession[sessionId] ?? []) : []))
  const $count = computed($items, activeSubagentCount)

  const $since = computed($items, items =>
    items.filter(isActive).reduce((oldest, item) => Math.min(oldest, item.startedAt || Infinity), Infinity)
  )

  // Stream frames must not repaint a count-only notice: both inputs are
  // primitives, so a child's activity line changes neither.
  return computed([$count, $since], (count, since): BackgroundResume | null =>
    count ? { count, since: Number.isFinite(since) ? since : Date.now() } : null
  )
}
