import { useCallback, useSyncExternalStore } from 'react'

import { registry } from '../registry'
import type { Contribution } from '../types'

/** Subscribe to the resolved contributions for an area. The subscription is
 *  scoped to `area`, so a slot re-renders only when ITS area mutates — a
 *  statusbar registration never re-renders a titlebar (or panes) slot. */
export function useContributions(area: string): readonly Contribution[] {
  const subscribe = useCallback((onChange: () => void) => registry.subscribeArea(area, onChange), [area])
  const getSnapshot = useCallback(() => registry.getArea(area), [area])

  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
}
