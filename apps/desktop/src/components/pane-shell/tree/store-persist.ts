/**
 * What the layout tree store writes to disk — the persist writer and the
 * ephemeral-pane gatekeepers it shares with the load-time codec heal
 * (see `store.ts`).
 */

import { LAYOUT_KEYS } from '@/lib/layout-persistence'
import { modeLayout } from '@/store/interface-mode'
import { PREVIEW_TILE_PREFIX } from '@/store/preview-tile-id'
import { isBrowserWindow, isSecondaryWindow } from '@/store/windows'

import { allPaneIds, type LayoutNode, removePane } from './model'

/**
 * Panes that must never reach the persisted tree. Preview tiles are transient
 * by design: their pane set is re-derived from `$previewTabs` on every boot
 * (artifact and other non-restorable tabs are dropped outright at decode), so
 * a tree saved with `preview-tile:` panes bakes splits/weights around tabs the
 * next session may not have — and the user's hand-built arrangement re-assembles
 * differently on every restart. The LIVE tree keeps them for the session; only
 * the STORED copy is scrubbed, and boot adoption re-docks surviving preview
 * tabs at their remembered share (see `rememberPaneShare`).
 *
 * The prefix lives in the import-free `store/preview-tile-id` leaf, the same
 * definition the minting side (`preview-tile.tsx`) uses. It must stay off any
 * cycle with the tree store: `$layoutTree` decodes (and scrubs) while that
 * module is still evaluating.
 */
export const isEphemeralPane = (paneId: string): boolean => paneId.startsWith(`${PREVIEW_TILE_PREFIX}:`)

/** Depth-first, allocation-free: true the moment any pane id is ephemeral.
 *  `persist` runs on every commit, and most commits carry no preview tile —
 *  the no-op path must not materialize the `allPaneIds` array. */
function containsEphemeralPane(node: LayoutNode): boolean {
  return node.type === 'group' ? node.panes.some(isEphemeralPane) : node.children.some(containsEphemeralPane)
}

/** Remove every ephemeral tile pane from a tree COPY (`removePane` rebuilds;
 *  the input is never mutated). Returns the same reference when nothing matches
 *  — `persist` runs on every commit and must not churn allocations — and null
 *  only for a tree that was nothing but ephemeral panes. */
export function scrubEphemeralPanes(tree: LayoutNode | null): LayoutNode | null {
  if (!tree) {
    return null
  }

  if (!containsEphemeralPane(tree)) {
    return tree
  }

  let next: LayoutNode | null = tree

  for (const paneId of allPaneIds(tree)) {
    if (isEphemeralPane(paneId)) {
      next = next === null ? null : removePane(next, paneId)
    }
  }

  return next
}

/** The stored form of a tree for the mode-switch capture, which writes every
 *  scoped atom through its codec rather than through `persist()`. Same scrub;
 *  a tree that was nothing but ephemeral panes stores as nothing, which loads
 *  back as the mode's default tree (what the decode heal would give it). */
export function encodeStoredTree(tree: LayoutNode | null): null | string {
  const stored = scrubEphemeralPanes(tree)

  return stored ? JSON.stringify(stored) : null
}

export function persist(tree: LayoutNode | null) {
  // A secondary window (single-chat pop-out) shares the origin's localStorage;
  // writing its stripped-down DEFAULT tree back would wipe the primary's layout.
  // A popped-out Browser is the same class of window.
  if (isSecondaryWindow() || isBrowserWindow()) {
    return
  }

  if (tree) {
    const stored = scrubEphemeralPanes(tree)

    // A tree that was nothing but ephemeral panes has no layout worth keeping;
    // skip the write so the last good persisted tree survives.
    if (!stored) {
      return
    }

    modeLayout.write(LAYOUT_KEYS.tree, JSON.stringify(stored))

    return
  }

  modeLayout.write(LAYOUT_KEYS.tree, null)
}
