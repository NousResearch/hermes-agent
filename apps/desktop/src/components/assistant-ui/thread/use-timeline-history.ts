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

  const loadMore = useCallback(
    async (beyondRowId?: number) => {
      if (!storedId) {
        return
      }

      try {
        const value = await fetchTimelineIndex(storedId, scope, beyondRowId)

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
    },
    [key, storedId, runtimeId, view]
  )

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

  // The newest persisted prompt in the live tail — a scalar, so streaming
  // deltas never re-run this. Each new turn moves it once.
  const newestPromptRowId = useStoreSelector(view.$messages, messages => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i]!

      if (message.role === 'user' && message.rowId !== undefined) {
        return message.rowId
      }
    }

    return undefined
  })

  // A complete index is a snapshot of the turns that existed when it was
  // read. When the live tail grows past its last mark (entries are
  // chronological), page it forward once per new prompt — no timer; an
  // incomplete index still pages on demand.
  const last = value?.entries.at(-1)?.rowId
  const stale =
    value?.complete === true && newestPromptRowId !== undefined && (last === undefined || last < newestPromptRowId)

  useEffect(() => {
    if (!stale || failed === key) {
      return
    }

    void loadMore(newestPromptRowId)
  }, [stale, failed, key, loadMore, newestPromptRowId])

  return { ...value, failed: failed === key, loadMore }
}
