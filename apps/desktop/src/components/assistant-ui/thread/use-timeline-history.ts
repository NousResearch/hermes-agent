import { useCallback, useEffect, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, getSessionOwnerHint } from '@/store/session'
import { transcriptTailState } from '@/store/transcript-tail'

import { cachedTimelineIndex, fetchTimelineIndex, type TimelineIndex, timelineIndexKey } from './timeline-index'

/** A bounded metadata page after paint; more titles load only on explicit demand. */
export function useTimelineHistory() {
  const view = useSessionView()
  const storedId = useStoreSelector(view.$storedId, id => id)
  const runtimeId = useStoreSelector(view.$runtimeId, id => id)

  const connectionId = useStoreSelector(
    $connection,
    connection => connection?.connectionId || (connection?.mode === 'local' ? 'local' : '')
  )

  const activeProfile = useStoreSelector($activeGatewayProfile, profile => profile)

  const owner = storedId
    ? getSessionOwnerHint(storedId, connectionId ? { connectionId, profile: activeProfile } : undefined)
    : undefined

  const scope = owner
    ? { connectionId: owner.connectionId, profile: owner.targetProfile || owner.profile }
    : (transcriptTailState(storedId)?.profile ?? { connectionId, profile: activeProfile })

  const key = timelineIndexKey(storedId ?? '', scope)
  const [index, setIndex] = useState<{ key: string; value: TimelineIndex } | null>(null)
  const [failed, setFailed] = useState<string | null>(null)

  const loadMore = useCallback(async () => {
    if (!storedId) {
      return
    }

    try {
      const value = await fetchTimelineIndex(storedId, scope)

      if (view.$storedId.get() === storedId && view.$runtimeId.get() === runtimeId) {
        setIndex({ key, value })
        setFailed(null)
      }
    } catch {
      // Old backends keep the loaded rail and their explicit Show earlier path.
      setFailed(key)
    }
    // Owner is represented by key; do not restart on object identity alone.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, storedId, runtimeId, view])

  useEffect(() => {
    if (!storedId) {
      return
    }

    const timer = window.setTimeout(() => {
      void loadMore()
    }, 200)

    return () => window.clearTimeout(timer)
  }, [loadMore, storedId])

  const value = index?.key === key ? index.value : cachedTimelineIndex(key)

  return { ...value, failed: failed === key, loadMore }
}
