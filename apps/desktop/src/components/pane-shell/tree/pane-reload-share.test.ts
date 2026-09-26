import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { LayoutNode, SplitNode } from '@/components/pane-shell/tree/model'

// #108679: a hard reload boots with `registered` empty, so the pane-mirror
// prunes every persisted session-tile pane from the tree — and each removal
// records its share (0.5 for an evenly-split row) as if it were a user
// resize. When the tiles re-dock in anchor order, `recalledEdgeWeights`
// replays those stale shares against a DIFFERENTLY shaped row, so tiles later
// in the re-dock chain land at half weight and the persisted tree drifts
// [1,1,1,1] → [1,1,1,1,0.5,0.5,1]. Two guards close the class:
//
// 1. Share recording is SUSPENDED while the store is hydrating (a boot
//    prune is not remembered geometry).
// 2. A recorded share only replays against the seam partner it was recorded
//    with — a differently shaped row falls back to the even default.

describe('tile split-share memory across a hard reload', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const tree = await import('@/components/pane-shell/tree/store')
    const model = await import('@/components/pane-shell/tree/model')
    const { registry } = await import('@/contrib/registry')

    registry.register({
      id: 'workspace',
      area: 'panes',
      title: 'chat',
      data: { placement: 'main', uncloseable: true },
      render: () => null
    })

    const registerTile = (id: string, anchor: string) =>
      registry.register({
        id,
        area: 'panes',
        title: id,
        data: { placement: 'main', dock: { pane: anchor, pos: 'right' } },
        render: () => null
      })

    tree.declareDefaultTree(model.group(['workspace'], { id: 'grp-main' }))
    tree.watchContributedPanes()

    return { model, registerTile, registry, tree }
  }

  /** The root row's weights, normalized to shares of their sum. */
  function rowShares(root: LayoutNode) {
    if (root.type !== 'split') {
      throw new Error('expected a split root')
    }

    const total = root.weights.reduce((a, b) => a + b, 0)

    return root.weights.map(w => w / total)
  }

  /** All split nodes in the tree (row and column alike). */
  function splits(node: LayoutNode, acc: SplitNode[] = []): SplitNode[] {
    if (node.type === 'group') {
      return acc
    }

    acc.push(node)

    for (const child of node.children) {
      splits(child, acc)
    }

    return acc
  }

  it('a reload-cycle prune does not record stale half-shares (hydration suspend)', async () => {
    const { registerTile, tree } = await setup()

    // The reload shape: hydration begins (the mirror's registered map is
    // empty), every persisted tile pane is pruned, then the tiles re-register
    // and re-dock in anchor order. The prune pass must not remember shares.
    tree.beginLayoutHydration()
    for (const id of ['session-tile:a', 'session-tile:b', 'session-tile:c', 'session-tile:d']) {
      tree.removeTreePane(id)
    }
    tree.endLayoutHydration()

    // Persisted shares stay empty — nothing was recorded by the prune.
    expect(tree.$paneShareRecords.get()).toEqual({})

    // Re-dock: the tiles come back in anchor order.
    registerTile('session-tile:a', 'workspace')
    registerTile('session-tile:b', 'session-tile:a')
    registerTile('session-tile:c', 'session-tile:b')
    registerTile('session-tile:d', 'session-tile:c')

    // A reload cycle must land on the same WEIGHTS a fresh dock of the same
    // tiles produces — no half-weight drift for tiles later in the re-dock
    // chain. (Group ids are regenerated per insert, so compare the weight
    // vector only, against a fresh-boot reference of the same tile set.)
    const fresh = await setup()
    for (const [id, anchor] of [
      ['session-tile:a', 'workspace'],
      ['session-tile:b', 'session-tile:a'],
      ['session-tile:c', 'session-tile:b'],
      ['session-tile:d', 'session-tile:c']
    ] as const) {
      fresh.registerTile(id, anchor)
    }

    const afterTree = tree.$layoutTree.get()!
    const freshTree = fresh.tree.$layoutTree.get()!

    if (afterTree.type !== 'split' || freshTree.type !== 'split') {
      throw new Error('expected split roots')
    }

    expect(afterTree.weights).toEqual(freshTree.weights)
  })

  it('a stale recorded share does not replay against a different seam partner', async () => {
    const { registerTile, tree } = await setup()

    // The user really did leave tile A at a quarter of the pair — recorded
    // against the WORKSPACE seam.
    const disposeA = registerTile('session-tile:a', 'workspace')
    const root = tree.$layoutTree.get()!

    if (root.type !== 'split') {
      throw new Error('expected a split root')
    }

    tree.setTreeSplitWeights(root.id, [3, 1])
    disposeA()
    tree.removeTreePane('session-tile:a')

    // The share was recorded against the workspace as seam partner.
    expect(tree.$paneShareRecords.get()['session-tile:a']).toBeCloseTo(0.25)

    // But the tree changes shape before the tile returns: it now re-docks
    // beside ANOTHER tile, not the workspace. The remembered share belongs
    // to a seam that no longer exists — replay must fall back to even.
    registerTile('session-tile:b', 'workspace')
    registerTile('session-tile:a', 'session-tile:b')

    // The seam holding tile A must be even: the share was recorded against
    // the workspace, not against tile B. (The tree may flatten the nested
    // seam into the root row — assert on the WEIGHTS of tile A's slot vs its
    // row, normalized: an even seam means A's slot equals its neighbor's.)
    const root2b = tree.$layoutTree.get()!

    if (root2b.type !== 'split') {
      throw new Error('expected a split root')
    }

    const seam = splits(root2b).find(s =>
      s.children.some(c => c.type === 'group' && c.panes.includes('session-tile:a'))
    )

    expect(seam).toBeDefined()

    // Tile A's slot must take the same share as its seam neighbor: an even
    // split, not the 3:1 recorded against the workspace.
    const at = seam!.children.findIndex(c => c.type === 'group' && c.panes.includes('session-tile:a'))
    const total = seam!.weights.reduce((x, y) => x + y, 0)
    const aShare = seam!.weights[at]! / total
    const neighborShare = seam!.weights[at === 0 ? 1 : at - 1]! / total

    expect(aShare).toBeCloseTo(neighborShare)
  })

  it('a genuine user resize still records and replays against the same partner', async () => {
    const { registerTile, tree } = await setup()

    const disposeA = registerTile('session-tile:a', 'workspace')
    const root = tree.$layoutTree.get()!

    if (root.type !== 'split') {
      throw new Error('expected a split root')
    }

    tree.setTreeSplitWeights(root.id, [3, 1])
    disposeA()
    tree.removeTreePane('session-tile:a')

    // Same seam partner back: the quarter share replays.
    registerTile('session-tile:a', 'workspace')

    const shares = rowShares(tree.$layoutTree.get()!)

    expect(shares[0]).toBeCloseTo(0.75)
    expect(shares[1]).toBeCloseTo(0.25)
  })
})
