import { findGroupOfPane, type LayoutNode } from '@/components/pane-shell/tree/model'

import { TILE_PANE_PREFIX } from './session-focus'

/** A tile's stored dock hint, in the shape pane adoption replays it. */
export interface TilePlacement {
  anchor: string
  before: null | string
  dir: 'center'
}

/** Where `paneId` sits in `tree`, as a dock hint that puts it back in the same
 *  zone and slot: stacked (`center`) into its zone, before its right-hand
 *  neighbour (`null` = last).
 *
 *  The anchor only has to name the zone, so it is the member least likely to
 *  leave the tree when tile buckets swap: a pane that is not a session tab
 *  (the workspace, a plugin page), else the left-hand neighbour, else the
 *  right-hand one. Adoption docks anchors first, so a neighbour that comes
 *  back in the same pass still works.
 *
 *  `null` when the pane is not in the tree, or is alone in its zone: an edge
 *  split is already described by the tile's stored `dir`/`anchor`, and a
 *  center hint cannot recreate a zone. */
export function tilePlacementInTree(tree: LayoutNode | null, paneId: string): null | TilePlacement {
  const group = tree ? findGroupOfPane(tree, paneId) : null

  if (!group || group.panes.length < 2) {
    return null
  }

  const index = group.panes.indexOf(paneId)

  const anchor =
    group.panes.find(id => id !== paneId && !id.startsWith(TILE_PANE_PREFIX)) ??
    group.panes[index - 1] ??
    group.panes[index + 1]!

  return { anchor, before: group.panes[index + 1] ?? null, dir: 'center' }
}

interface PlacedTile {
  anchor?: string
  before?: null | string
  dir?: string
  storedSessionId: string
}

/** Pin each tile's stored dock hint to the slot its pane holds in `tree`.
 *  Returns the updated list, or `null` when nothing changed. Tiles whose pane
 *  isn't in the tree (or sits alone in its zone) keep their hint.
 *
 *  The layout tree is the record of where each tab is. A tile's stored
 *  `anchor`/`before`/`dir` is only the hint adoption replays when the pane
 *  re-enters the tree, and it is written when the tab is opened or dragged
 *  onto a zone, so it goes stale. Panes do re-enter: tiles are stored per
 *  gateway-profile bucket while the tree is shared, boot starts on the
 *  placeholder `default` bucket so the tile mirror's first sync prunes every
 *  persisted tile pane that bucket doesn't list, and a gateway-profile swap
 *  prunes the outgoing bucket's tabs. Replaying a stale hint moved tabs
 *  between zones on restart with nobody touching them. */
export function pinTilePlacements<T extends PlacedTile>(tree: LayoutNode | null, tiles: readonly T[]): null | T[] {
  let changed = false

  const next = tiles.map(tile => {
    const placement = tilePlacementInTree(tree, `${TILE_PANE_PREFIX}${tile.storedSessionId}`)

    if (
      !placement ||
      (placement.anchor === tile.anchor && placement.before === (tile.before ?? null) && placement.dir === tile.dir)
    ) {
      return tile
    }

    changed = true

    return { ...tile, ...placement }
  })

  return changed ? next : null
}
