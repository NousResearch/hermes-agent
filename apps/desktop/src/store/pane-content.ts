import { atom, computed } from 'nanostores'

import { visiblePaneIds } from '@/components/pane-shell/tree/model'
import { $dismissedPanes, $hiddenTreePanes, $layoutTree } from '@/components/pane-shell/tree/store'

export type PaneContent =
  | { kind: 'chat'; storedSessionId: string | null }
  | { kind: 'page'; page: string }

export const $paneContentById = atom<Record<string, PaneContent>>({})

export function setPaneContent(paneId: string, content: PaneContent) {
  const current = $paneContentById.get()

  if (current[paneId] === content) {
    return
  }

  $paneContentById.set({ ...current, [paneId]: content })
}

export function clearPaneContent(paneId?: string) {
  if (!paneId) {
    $paneContentById.set({})
    return
  }

  const current = $paneContentById.get()

  if (!(paneId in current)) {
    return
  }

  const { [paneId]: _removed, ...rest } = current
  $paneContentById.set(rest)
}

/** Stored session ids in panes actually visible on screen. Hidden tabs,
 * minimized groups, chrome-hidden panes, dismissed panes, and page panes do
 * not contribute. */
export const $visibleSessionIds = computed(
  [$layoutTree, $hiddenTreePanes, $dismissedPanes, $paneContentById],
  (tree, hidden, dismissed, contentById) => {
    if (!tree) {
      return []
    }

    const unavailable = new Set([...hidden, ...dismissed])

    return visiblePaneIds(tree, unavailable).flatMap(paneId => {
      const content = contentById[paneId]
      return content?.kind === 'chat' && content.storedSessionId ? [content.storedSessionId] : []
    })
  }
)
