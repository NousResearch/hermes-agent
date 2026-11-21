import { atom } from 'nanostores'

/** Add/remove `item` in a readonly set, returning a fresh set — or null when
 *  membership already matches `present` (so callers can early-out on a no-op). */
export function toggledSet<T>(set: ReadonlySet<T>, item: T, present: boolean): Set<T> | null {
  if (set.has(item) === present) {
    return null
  }

  const next = new Set(set)

  if (present) {
    next.add(item)
  } else {
    next.delete(item)
  }

  return next
}

/**
 * Panes the current ROUTE takes off screen without unmounting them: the redesign shows no chat
 * sidebar beside a page, but leaving and returning must not rebuild it. Their zone collapses
 * (display:none, so no style, layout, focus or accessibility tree) exactly as a hidden one does,
 * while the pane stays shown in its group and so stays mounted. Never persisted and never the
 * user's hide (`$hiddenTreePanes`), which still unmounts.
 */
export const $routeCollapsedPanes = atom<ReadonlySet<string>>(new Set())

export function setPaneRouteCollapsed(paneId: string, collapsed: boolean) {
  const next = toggledSet($routeCollapsedPanes.get(), paneId, collapsed)

  if (next) {
    $routeCollapsedPanes.set(next)
  }
}
