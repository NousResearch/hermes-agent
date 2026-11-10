import { findGroupOfPane, type LayoutNode } from '@/components/pane-shell/tree/model'

import { TILE_PANE_PREFIX } from './session-focus'

/** Reorder tiles to match layout-tree encounter order (stored ids in the order
 *  their `session-tile:` panes are walked). Restore replays the array through
 *  sequential adoption (each center tile APPENDS after the ones before it), so
 *  array order IS strip order — no `before` stamping needed; a stale `before`
 *  naming an absent pane falls back to append anyway (see insertAtGroup). Tiles
 *  not yet adopted sort after placed ones, stably. Returns `null` when nothing
 *  moves so callers can skip a needless persist. */
export function orderTilesByTree<T extends { storedSessionId: string }>(
  tree: LayoutNode | null,
  tiles: readonly T[]
): null | T[] {
  if (!tree || tiles.length < 2) {
    return null
  }

  const order: string[] = []

  const walk = (node: LayoutNode) => {
    if (node.type === 'group') {
      for (const id of node.panes) {
        if (id.startsWith(TILE_PANE_PREFIX)) {
          order.push(id.slice(TILE_PANE_PREFIX.length))
        }
      }

      return
    }

    node.children.forEach(walk)
  }

  walk(tree)

  const rank = new Map(order.map((id, i) => [id, i]))

  const next = [...tiles].sort(
    (a, b) => (rank.get(a.storedSessionId) ?? Infinity) - (rank.get(b.storedSessionId) ?? Infinity)
  )

  return next.some((t, i) => t !== tiles[i]) ? next : null
}

/** ⌘W promotes the next chat stacked with workspace, then a chat in another
 *  zone. Walk from workspace's next strip slot and wrap to preceding slots. */
export function nextWorkspaceSession(
  tree: LayoutNode | null,
  tiles: readonly { storedSessionId: string }[]
): null | string {
  const group = tree ? findGroupOfPane(tree, 'workspace') : null

  if (!group) {
    return null
  }

  const idx = group.panes.indexOf('workspace')
  const ordered = [...group.panes.slice(idx + 1), ...group.panes.slice(0, idx).reverse()]

  for (const paneId of ordered) {
    if (paneId.startsWith(TILE_PANE_PREFIX)) {
      const storedSessionId = paneId.slice(TILE_PANE_PREFIX.length)

      if (tiles.some(t => t.storedSessionId === storedSessionId)) {
        return storedSessionId
      }
    }
  }

  // Another zone can still shift into main; otherwise closing main would
  // incorrectly create an empty draft while another chat remains open.
  for (const tile of tiles) {
    if (tree && findGroupOfPane(tree, `${TILE_PANE_PREFIX}${tile.storedSessionId}`)) {
      return tile.storedSessionId
    }
  }

  return null
}
