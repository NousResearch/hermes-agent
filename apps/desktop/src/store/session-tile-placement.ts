import { findGroupOfPane, findParentSplit, type GroupNode, type LayoutNode } from '@/components/pane-shell/tree/model'

import { TILE_PANE_PREFIX } from './session-focus'

/** A tile's stored dock hint, in the shape pane adoption replays it. */
export interface TilePlacement {
  anchor: string
  before: null | string
  dir: 'bottom' | 'center' | 'left' | 'right' | 'top'
}

/** The zone member least likely to leave the tree when tile buckets swap: a
 *  pane that is not a session tab (the workspace, a plugin page), else `fallback`. */
const stableMember = (group: GroupNode, fallback: string | undefined) =>
  group.panes.find(id => !id.startsWith(TILE_PANE_PREFIX)) ?? fallback

/** The zone of `node` nearest the seam on its `last` (or first) side. */
const seamGroup = (node: LayoutNode | undefined, last: boolean): GroupNode | undefined => {
  if (!node || node.type === 'group') {
    return node
  }

  return seamGroup(last ? node.children.at(-1) : node.children[0], last)
}

/** Where `paneId` sits in `tree`, as a dock hint that puts it back in the same
 *  zone and slot.
 *
 *  A pane sharing its zone is stacked (`center`) into it, before its
 *  right-hand neighbour (`null` = last). The anchor only has to name the
 *  zone: a stable member, else the left-hand neighbour, else the right-hand
 *  one. Adoption docks anchors first, so a neighbour that comes back in the
 *  same pass still works.
 *
 *  A pane ALONE in its zone gets an edge hint: anchored on the sibling zone
 *  before it in its parent split (below / right of it), else the one after
 *  it (above / left of it). An edge split beside a zone whose parent runs the
 *  same way flattens back into that parent, so this recreates the zone in its
 *  slot; adoption replays the share the pane held against that seam. A
 *  sibling that is itself split only offers its zone nearest the seam, which
 *  lands the pane beside that zone rather than beside the whole split.
 *
 *  `null` when the pane is not in the tree, or its zone is the whole tree. */
export function tilePlacementInTree(tree: LayoutNode | null, paneId: string): null | TilePlacement {
  const group = tree ? findGroupOfPane(tree, paneId) : null

  if (!group) {
    return null
  }

  const index = group.panes.indexOf(paneId)

  if (group.panes.length > 1) {
    const anchor = stableMember(group, group.panes[index - 1] ?? group.panes[index + 1])!

    return { anchor, before: group.panes[index + 1] ?? null, dir: 'center' }
  }

  const parent = findParentSplit(tree!, group.id)

  if (!parent) {
    return null
  }

  const at = parent.children.findIndex(child => child.id === group.id)
  const row = parent.orientation === 'row'
  const prev = parent.children[at - 1]
  const next = parent.children[at + 1]

  const [zone, after] =
    prev?.type === 'group'
      ? [prev, true]
      : next?.type === 'group'
        ? [next, false]
        : prev
          ? [seamGroup(prev, true), true]
          : [seamGroup(next, false), false]

  const anchor = zone && stableMember(zone, zone.panes[0])

  if (!anchor) {
    return null
  }

  return { anchor, before: null, dir: after ? (row ? 'right' : 'bottom') : row ? 'left' : 'top' }
}

interface PlacedTile {
  anchor?: string
  before?: null | string
  dir?: string
  storedSessionId: string
}

/** Pin each tile's stored dock hint to the slot its pane holds in `tree`.
 *  Returns the updated list, or `null` when nothing changed. Every tile whose
 *  pane is in the tree is re-pinned, so a hint whose anchor has left the tree
 *  never survives; a tile whose pane isn't in the tree keeps its hint.
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
