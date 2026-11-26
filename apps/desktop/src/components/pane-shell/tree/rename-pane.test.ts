import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { LayoutNode } from '@/components/pane-shell/tree/model'

// A session tile's conversation rotates to a new stored id on auto-compression
// (#98622), so its pane id changes while the surface stays the same. Removing
// the old pane and adopting the new one re-split the anchor zone [1, 1] and
// normalize flattened the user's grid. Re-keying in place keeps every weight.

const OLD = 'session-tile:tip-old'
const NEW = 'session-tile:tip-new'
const OTHER = 'session-tile:other'

/** A resized grid: main | (old over other), with non-default weights. */
async function grid() {
  const { group, split } = await import('@/components/pane-shell/tree/model')

  return split(
    'row',
    [
      group(['workspace'], { id: 'grp-main' }),
      split('column', [group([OLD], { id: 'grp-old' }), group([OTHER], { id: 'grp-other' })], [3, 1], 'col')
    ],
    [2, 5],
    'row'
  )
}

const renamed = (tree: LayoutNode) => JSON.parse(JSON.stringify(tree).replaceAll(OLD, NEW)) as LayoutNode

describe('renameTreePane', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    vi.resetModules()
  })

  async function setup() {
    const tree = await import('@/components/pane-shell/tree/store')
    const { registry } = await import('@/contrib/registry')
    const initial = await grid()

    registry.register({
      id: 'workspace',
      area: 'panes',
      title: 'chat',
      data: { placement: 'main', uncloseable: true },
      render: () => null
    })

    const registerTile = (id: string, dock: { pane: string; pos: 'bottom' | 'right' }) =>
      registry.register({ id, area: 'panes', title: id, data: { placement: 'main', dock }, render: () => null })

    const disposeOld = registerTile(OLD, { pane: 'workspace', pos: 'right' })

    registerTile(OTHER, { pane: OLD, pos: 'bottom' })
    tree.declareDefaultTree(initial)
    tree.watchContributedPanes()

    return { disposeOld, initial, registerTile, tree }
  }

  it('re-keys the pane in place: same zone, same structure, same weights', async () => {
    const { initial, tree } = await setup()

    expect(tree.$layoutTree.get()).toEqual(initial)
    expect(tree.renameTreePane(OLD, NEW)).toBe(true)
    expect(tree.$layoutTree.get()).toEqual(renamed(initial))
  })

  // The pane mirror's order: register the new contribution (adoption runs),
  // then dispose the old one (adoption runs again), unpark it, remove its pane.
  async function swap(setupResult: Awaited<ReturnType<typeof setup>>) {
    const { setTreePaneParked } = await import('@/components/pane-shell/tree/parked-panes')

    setupResult.registerTile(NEW, { pane: 'workspace', pos: 'right' })
    setupResult.disposeOld()
    setTreePaneParked(OLD, false)
    setupResult.tree.removeTreePane(OLD)
  }

  it("survives the owner's contribution swap (the pane mirror's adopt + remove)", async () => {
    const ctx = await setup()

    ctx.tree.renameTreePane(OLD, NEW)
    await swap(ctx)

    expect(ctx.tree.$layoutTree.get()).toEqual(renamed(ctx.initial))
  })

  it('without it, the same swap re-splits the anchor and loses the grid (the bug)', async () => {
    const ctx = await setup()

    await swap(ctx)

    expect(ctx.tree.$layoutTree.get()).not.toEqual(renamed(ctx.initial))
  })

  it('is a no-op when the old id is absent or the new id is already placed', async () => {
    const { tree } = await setup()
    const before = tree.$layoutTree.get()

    expect(tree.renameTreePane('session-tile:missing', NEW)).toBe(false)
    expect(tree.renameTreePane(OLD, OTHER)).toBe(false)
    expect(tree.renameTreePane(OLD, OLD)).toBe(false)
    expect(tree.$layoutTree.get()).toBe(before)
  })

  it('carries the per-pane size records to the new id', async () => {
    const { tree } = await setup()
    const { $paneStates } = await import('@/store/panes')

    tree.$paneShareRecords.set({ [OLD]: 0.3, [OTHER]: 0.6 })
    $paneStates.set({ [OLD]: { open: true, widthOverride: 420 } })

    tree.renameTreePane(OLD, NEW)

    expect(tree.$paneShareRecords.get()).toEqual({ [NEW]: 0.3, [OTHER]: 0.6 })
    expect($paneStates.get()).toEqual({ [NEW]: { open: true, widthOverride: 420 } })
  })
})
