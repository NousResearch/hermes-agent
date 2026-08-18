export type PaneLifecycle = 'visible' | 'hot-hidden' | 'parked'

export const DEFAULT_HOT_HIDDEN_PANE_CAP = 2

interface PaneLifecycleEntry {
  lifecycle: PaneLifecycle
  lastVisible: number
}

export interface PaneLifecycleState {
  clock: number
  entries: Record<string, PaneLifecycleEntry>
}

export const emptyPaneLifecycleState = (): PaneLifecycleState => ({ clock: 0, entries: {} })

interface ReconcilePaneLifecycleOptions {
  activeId: string
  hotHiddenCap?: number
  keepAlive?: (id: string) => boolean
  paneIds: readonly string[]
}

/**
 * Reconcile one zone's mounted pane cache.
 *
 * The foreground pane is visible, the most recently visible inactive panes stay
 * hot up to a small cap, and the rest park (unmount). Explicit keep-alive panes
 * such as the terminal remain hot outside that cap so hiding UI never kills the
 * stateful resource they host.
 */
export function reconcilePaneLifecycle(
  previous: PaneLifecycleState,
  {
    activeId,
    hotHiddenCap = DEFAULT_HOT_HIDDEN_PANE_CAP,
    keepAlive = () => false,
    paneIds
  }: ReconcilePaneLifecycleOptions
): PaneLifecycleState {
  const present = new Set(paneIds)
  const entries: Record<string, PaneLifecycleEntry> = {}
  let clock = previous.clock

  for (const id of paneIds) {
    const prior = previous.entries[id]

    if (prior) {
      entries[id] = { ...prior, lifecycle: 'parked' }
    }
  }

  if (present.has(activeId)) {
    const prior = previous.entries[activeId]

    if (!prior || prior.lifecycle !== 'visible') {
      clock += 1
    }

    entries[activeId] = { lifecycle: 'visible', lastVisible: clock }
  }

  const inactive = paneIds
    .filter(id => id !== activeId && entries[id])
    .sort((a, b) => entries[b].lastVisible - entries[a].lastVisible)

  for (const id of inactive.filter(keepAlive)) {
    entries[id] = { ...entries[id], lifecycle: 'hot-hidden' }
  }

  for (const id of inactive.filter(id => !keepAlive(id)).slice(0, Math.max(0, hotHiddenCap))) {
    entries[id] = { ...entries[id], lifecycle: 'hot-hidden' }
  }

  return { clock, entries }
}
