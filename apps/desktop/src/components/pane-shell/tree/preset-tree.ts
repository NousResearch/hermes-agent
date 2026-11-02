/**
 * A named layout preset is GEOMETRY (splits, rails, weights, which zones are
 * open) — not a snapshot of the conversations that happened to be on screen
 * when it was saved.
 *
 * `saveCurrentLayoutAs` clones the LIVE tree, which carries
 * `session-tile:<storedSessionId>`, `preview-tile:*` and `route-tile:*` panes.
 * Those ids are not portable: the layout tree and its presets are GLOBAL
 * (`hermes.desktop.layoutTree.v2` / `hermes.desktop.layoutPresets.v2`), while a
 * session tile belongs to the profile that owns the stored session. Re-applying
 * such a preset remounts the baked-in tiles, which fires `session.resume` /
 * agent init across profiles; the tile WebSocket then drops, the gateway
 * `ws_orphan_reap`s the runtime, and the UI keeps RPCing the dead id (#94260).
 *
 * This is deliberately NOT the same path as persisting the live tree across
 * restarts (preview tiles, #92818 / #93179) — those PRs never touch
 * `saveLayoutPresetTree` / `applyLayoutPreset`, and they do not strip
 * `session-tile:` at all. `applyTree` already ADOPTS the panes that are open
 * right now into the new geometry, so stripping here does not lose tabs: it
 * only stops them arriving from a stored snapshot.
 */

import { allPaneIds, type LayoutNode, normalize } from './model'

/** Panes that must never ride along in a named layout preset. */
export const PRESET_EXCLUDED_PANE_PREFIXES = ['session-tile:', 'preview-tile:', 'route-tile:'] as const

/** Is `paneId` a live/ephemeral tile that a preset must not carry? */
export const isPresetExcludedPaneId = (paneId: string): boolean =>
  PRESET_EXCLUDED_PANE_PREFIXES.some(prefix => paneId.startsWith(prefix))

/**
 * A copy of `tree` without live/ephemeral tile panes. Zones left empty collapse
 * through `normalize`, which also unwraps the single-child splits they leave
 * behind. Unchanged subtrees keep their reference, so an already-clean preset
 * comes back as the SAME object.
 *
 * Returns `null` when nothing structural remains — a preset that was only a
 * snapshot of open tabs carries no layout and must not be applied (applying it
 * would hand `applyTree` an empty tree).
 */
export function stripPresetLivePanes(tree: LayoutNode): LayoutNode | null {
  const walk = (node: LayoutNode): LayoutNode => {
    if (node.type === 'group') {
      const panes = node.panes.filter(paneId => !isPresetExcludedPaneId(paneId))

      if (panes.length === node.panes.length) {
        return node
      }

      // The active tab may be one of the stripped panes; fall to the first
      // survivor (an emptied zone keeps `''` and is dropped by normalize).
      return { ...node, panes, active: panes.includes(node.active) ? node.active : (panes[0] ?? '') }
    }

    const children = node.children.map(walk)

    // Reference-preserving: a split whose children all came back untouched is
    // returned as-is, so the clean-preset no-op stays allocation-free.
    if (children.every((child, i) => child === node.children[i])) {
      return node
    }

    return { ...node, children }
  }

  return normalize(walk(tree))
}

/** The `resting` entries that still name a pane the stripped geometry places.
 *  A resting record for a live tile names nothing on apply (the pane is gone),
 *  and leaving it behind would keep writing tile ids back into the preset. */
export function stripPresetResting(resting: readonly string[] | undefined, tree: LayoutNode): string[] {
  if (!resting) {
    return []
  }

  const placed = new Set(allPaneIds(tree))

  return resting.filter(paneId => placed.has(paneId))
}
