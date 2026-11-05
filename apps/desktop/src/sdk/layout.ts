/**
 * Session tiling surface for plugins: a READ-ONLY projection of the pane tree
 * (`host.state.sessionLayout`), the sidebar's pinned ids
 * (`host.state.pinnedSessionIds`), and a few layout verbs (`host.layout`).
 * Placing a NEW tile is `host.openSession(id, { intent: 'tab', dock })`.
 *
 * The verbs are thin doors over the writers the app's own sash drags, tab
 * strip and ⌘W use, so a plugin action and a hand gesture can never disagree.
 * Pane verbs accept only chat panes (`workspace`, `session-tile:<storedId>`):
 * a plugin may arrange conversations, not core chrome or another plugin's pane.
 */
import { computed, type ReadableAtom } from 'nanostores'

import type { LayoutNode, Orientation } from '@/components/pane-shell/tree/model'
import {
  $activeTreeGroup,
  $layoutTree,
  closeTabPane,
  commitTreeSplitWeights,
  setTreeSplitOrientation,
  swapTreePanes,
  treeSplitArity
} from '@/components/pane-shell/tree/store'
import { $pinnedSessionIds } from '@/store/layout'
import { $selectedStoredSessionId } from '@/store/session'
import { TILE_PANE_PREFIX } from '@/store/session-focus'

/** Public shape of the pane tree. Internal attributes (tab-strip prefs, editor
 *  state) are deliberately not part of the contract. */
export type PluginLayoutNode =
  | { type: 'split'; id: string; orientation: Orientation; weights: number[]; children: PluginLayoutNode[] }
  | { type: 'group'; id: string; panes: string[]; active: string; minimized: boolean }

export interface PluginSessionLayout {
  root: null | PluginLayoutNode
  /** The zone the user last interacted with. */
  focusedGroupId: null | string
  /** Stored id of the chat in the `workspace` (main) pane; null on a blank draft. */
  mainStoredSessionId: null | string
}

export const isChatPaneId = (id: string): boolean => id === 'workspace' || id.startsWith(TILE_PANE_PREFIX)

export function projectLayout(node: LayoutNode): PluginLayoutNode {
  return node.type === 'group'
    ? { type: 'group', id: node.id, panes: [...node.panes], active: node.active, minimized: Boolean(node.minimized) }
    : {
        type: 'split',
        id: node.id,
        orientation: node.orientation,
        weights: [...node.weights],
        children: node.children.map(projectLayout)
      }
}

export const $pluginSessionLayout: ReadableAtom<PluginSessionLayout> = computed(
  [$layoutTree, $activeTreeGroup, $selectedStoredSessionId],
  (tree, focusedGroupId, main) => ({
    root: tree ? projectLayout(tree) : null,
    focusedGroupId,
    mainStoredSessionId: main || null
  })
)

export const $pluginPinnedSessionIds: ReadableAtom<readonly string[]> = computed($pinnedSessionIds, ids => [...ids])

const validWeights = (weights: unknown): weights is number[] =>
  Array.isArray(weights) && weights.every(w => typeof w === 'number' && Number.isFinite(w) && w > 0)

export const layoutHost = {
  /** Resize a split: one positive weight per child. Persisted. */
  setSplitWeights: (splitId: string, weights: number[]): boolean => {
    const arity = splitId ? treeSplitArity(splitId) : null

    if (!arity || !validWeights(weights) || weights.length !== arity) {
      return false
    }

    commitTreeSplitWeights(splitId, [...weights])

    return true
  },

  /** Re-orient a split (`row` = side by side, `column` = stacked). Persisted. */
  setSplitOrientation: (splitId: string, orientation: Orientation): boolean => {
    if (!splitId || !treeSplitArity(splitId) || (orientation !== 'row' && orientation !== 'column')) {
      return false
    }

    setTreeSplitOrientation(splitId, orientation)

    return true
  },

  /** Exchange two chat panes' positions. */
  swapPanes: (a: string, b: string): boolean => {
    if (!isChatPaneId(a) || !isChatPaneId(b) || a === b) {
      return false
    }

    swapTreePanes(a, b)

    return true
  },

  /** Close a chat pane exactly like its tab's ✕ (main promotes a neighbour
   *  tile); the parent split collapses so the sibling takes the space. */
  closePane: (paneId: string): boolean => {
    if (!isChatPaneId(paneId)) {
      return false
    }

    closeTabPane(paneId)

    return true
  }
}
