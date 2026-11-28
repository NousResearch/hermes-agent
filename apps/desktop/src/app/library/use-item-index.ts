import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo } from 'react'

import type { ArtifactRecord } from '@/app/artifacts/artifact-utils'
import type { IndexedChat } from '@/app/artifacts/item-index-snapshot'
import { $activeConnectionId } from '@/store/connections'
import { $itemIndex, ownerIndex, ownerItems, ownerKeyOf, peekOwnerIndex, refreshItemIndex } from '@/store/item-index'

/** An index older than this is checked again when its page opens or the window regains focus. */
const STALE_AFTER_MS = 30_000

export interface ItemIndexView {
  /** Every indexed item, newest first. */
  items: ArtifactRecord[]
  chats: Record<string, IndexedChat>
  /** When the list was last checked; null before the first check finished. */
  checkedAt: null | number
  refreshing: boolean
  error: unknown
  refresh: () => Promise<void>
}

/** The active owner's item index, painted from what is kept and checked again on open and on focus
 *  when it is older than {@link STALE_AFTER_MS}. */
export function useItemIndex({ autoRefresh = true }: { autoRefresh?: boolean } = {}): ItemIndexView {
  const owner = ownerKeyOf(useStore($activeConnectionId))
  const index = useStore($itemIndex)[owner] ?? peekOwnerIndex(owner)

  useEffect(() => {
    ownerIndex(owner)

    if (!autoRefresh) {
      return
    }

    const refreshIfStale = () => {
      const checkedAt = ownerIndex(owner).snapshot.checkedAt

      if (checkedAt === null || Date.now() - checkedAt > STALE_AFTER_MS) {
        void refreshItemIndex()
      }
    }

    refreshIfStale()
    window.addEventListener('focus', refreshIfStale)

    return () => window.removeEventListener('focus', refreshIfStale)
  }, [autoRefresh, owner])

  const items = useMemo(() => ownerItems(index.snapshot), [index.snapshot])
  const refresh = useCallback(() => refreshItemIndex(), [])

  return {
    items,
    chats: index.snapshot.chats,
    checkedAt: index.snapshot.checkedAt,
    refreshing: index.refreshing,
    error: index.error,
    refresh
  }
}
