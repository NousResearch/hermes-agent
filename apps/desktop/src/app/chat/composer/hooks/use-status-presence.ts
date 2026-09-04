import { useSyncExternalStore } from 'react'

import { $composerActionsBySession } from '@/store/composer-actions'
import { $statusItemsBySession } from '@/store/composer-status'
import { $previewStatusBySession } from '@/store/preview-status'
import { $todoHistoryBySession } from '@/store/todos'

/** Structural view of the per-session feeds — they hold different item
 *  types, and all this hook needs from each is "does this key have rows". */
interface PresenceFeed {
  get(): Record<string, undefined | unknown[]>
  listen(listener: () => void): () => void
}

const FEEDS: PresenceFeed[] = [
  $statusItemsBySession,
  $composerActionsBySession,
  $previewStatusBySession,
  $todoHistoryBySession
]

const subscribe = (onChange: () => void) => {
  const offs = FEEDS.map(feed => feed.listen(onChange))

  return () => {
    for (const off of offs) {
      off()
    }
  }
}

/**
 * Whether a session has any status items, micro actions, or previews, as a
 * coarse *edge*: the boolean only flips when the stack appears/disappears.
 * ChatBar uses it to toggle a styling data-attr — subscribing to the whole
 * `$statusItemsBySession` (a `computed` that rebuilds the entire map) /
 * `$previewStatusBySession` maps re-rendered the ~1.4k ChatBar on every
 * per-item mutation (a subagent tick, a 5s background poll) and on churn in
 * OTHER sessions. The boolean snapshot bails out of all of that, re-rendering
 * only on the actual show/hide transition.
 */
export function useSessionStatusPresence(sessionId: string | null): boolean {
  return useSyncExternalStore(subscribe, () => {
    if (!sessionId) {
      return false
    }

    return FEEDS.some(feed => (feed.get()[sessionId]?.length ?? 0) > 0)
  })
}
