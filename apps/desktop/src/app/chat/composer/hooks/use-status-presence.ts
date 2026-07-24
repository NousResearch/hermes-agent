import { useSyncExternalStore } from 'react'

import { $statusItemsBySession } from '@/store/composer-status'
import { $previewStatusBySession } from '@/store/preview-status'

const subscribe = (onChange: () => void) => {
  const offItems = $statusItemsBySession.listen(onChange)
  const offPreviews = $previewStatusBySession.listen(onChange)

  return () => {
    offItems()
    offPreviews()
  }
}

/**
 * Whether a session has any status items or previews, as a coarse *edge*: the
 * boolean only flips when the stack appears/disappears. ChatBar uses it to
 * toggle a styling data-attr — subscribing to the whole `$statusItemsBySession`
 * (a `computed` that rebuilds the entire map) / `$previewStatusBySession` maps
 * re-rendered the ~1.4k ChatBar on every per-item mutation (a subagent tick, a
 * 5s background poll) and on churn in OTHER sessions. The boolean snapshot bails
 * out of all of that, re-rendering only on the actual show/hide transition.
 */
export function useSessionStatusPresence(sessionId: string | null): boolean {
  return useSyncExternalStore(subscribe, () => {
    if (!sessionId) {
      return false
    }

    return (
      ($statusItemsBySession.get()[sessionId]?.length ?? 0) > 0 ||
      ($previewStatusBySession.get()[sessionId]?.length ?? 0) > 0
    )
  })
}
