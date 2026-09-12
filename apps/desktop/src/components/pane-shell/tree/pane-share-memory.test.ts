import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { LayoutNode } from '@/components/pane-shell/tree/model'

// Closing and re-opening a docked tile (the in-app browser) must respect the
// size the user left it at. Adoption's edge insert used to split the anchor
// zone [1, 1] every time, so each agent-triggered browser open re-took half
// the chat — "it keeps squishing my convo". The share the pane held against
// its seam neighbor is remembered on removal and re-applied on re-insert.

describe('tile split-share memory across close/reopen', () => {
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

    const registerBrowser = () =>
      registry.register({
        id: 'preview-tile:url:browser',
        area: 'panes',
        title: 'Browser',
        data: { placement: 'main', dock: { pane: 'workspace', pos: 'right' } },
        render: () => null
      })

    tree.declareDefaultTree(model.group(['workspace'], { id: 'grp-main' }))
    tree.watchContributedPanes()

    return { model, registerBrowser, registry, tree }
  }

  const sessionTile = (id: string, anchor: string) => ({
    id: `session-tile:${id}`,
    area: 'panes',
    title: id,
    data: { placement: 'main', dock: { pane: anchor, pos: 'right' as const } },
    render: () => null
  })

  /** The root row's weights, normalized to shares of their sum. */
  function rowShares(root: LayoutNode) {
    if (root.type !== 'split') {
      throw new Error('expected a split root')
    }

    const total = root.weights.reduce((a, b) => a + b, 0)

    return root.weights.map(w => w / total)
  }

  it('first open splits the anchor evenly', async () => {
    const { registerBrowser, tree } = await setup()

    registerBrowser()

    expect(rowShares(tree.$layoutTree.get()!)).toEqual([0.5, 0.5])
  })

  it('reopening after reload restores the share when the seam partner matches', async () => {
    const { registerBrowser, tree } = await setup()

    const dispose = registerBrowser()

    // The user drags the seam: browser down to a quarter of the pair.
    const root = tree.$layoutTree.get()!

    if (root.type !== 'split') {
      throw new Error('expected a split root')
    }

    tree.setTreeSplitWeights(root.id, [3, 1])

    // Close (the mirror disposes the contribution, then removes the pane)…
    dispose()
    tree.removeTreePane('preview-tile:url:browser')
    expect(tree.$layoutTree.get()!.type).toBe('group')

    // Simulate a hard reload: both the layout and split-share memory must be
    // read back from storage before the browser contribution returns.
    vi.resetModules()
    const reloaded = await setup()

    // Re-open against the same workspace seam partner.
    reloaded.registerBrowser()

    const shares = rowShares(reloaded.tree.$layoutTree.get()!)

    expect(shares[0]).toBeCloseTo(0.75)
    expect(shares[1]).toBeCloseTo(0.25)
  })

  it('rejects stale shares when chained session tiles re-adopt beside different partners', async () => {
    const { registry, tree } = await setup()
    const disposeA = registry.register(sessionTile('a', 'workspace'))
    const disposeB = registry.register(sessionTile('b', 'session-tile:a'))
    const root = tree.$layoutTree.get()!

    if (root.type !== 'split') {
      throw new Error('expected a split root')
    }

    // Give each departing tile a distinctive share against its current seam:
    // b owns 1/4 beside a, then a owns 3/4 beside workspace.
    tree.setTreeSplitWeights(root.id, [1, 3, 1])
    disposeB()
    tree.removeTreePane('session-tile:b')
    disposeA()
    tree.removeTreePane('session-tile:a')

    // A reload/profile reconciliation can re-adopt the same tiles in a new
    // chain. Neither old share belongs to the new seam, so both insertions use
    // the existing even default instead of replaying 1/4 and 3/4 stale shares.
    registry.register(sessionTile('b', 'workspace'))
    registry.register(sessionTile('a', 'session-tile:b'))

    expect(rowShares(tree.$layoutTree.get()!)).toEqual([0.5, 0.25, 0.25])
  })

  it('a stacked tab records no share (its removal changes no geometry)', async () => {
    const { model, tree } = await setup()
    const { registry } = await import('@/contrib/registry')

    // Stacks INTO the workspace zone instead of splitting beside it.
    const dispose = registry.register({
      id: 'preview-tile:file:notes',
      area: 'panes',
      title: 'notes',
      data: { placement: 'main', dock: { pane: 'workspace', pos: 'center' } },
      render: () => null
    })

    expect(tree.$layoutTree.get()!.type).toBe('group')

    dispose()
    tree.removeTreePane('preview-tile:file:notes')

    // Re-register docking to an EDGE: no remembered share exists, so the
    // split falls back to the even default.
    registry.register({
      id: 'preview-tile:file:notes',
      area: 'panes',
      title: 'notes',
      data: { placement: 'main', dock: { pane: 'workspace', pos: 'right' } },
      render: () => null
    })

    expect(model.allPaneIds(tree.$layoutTree.get()!)).toContain('preview-tile:file:notes')
    expect(rowShares(tree.$layoutTree.get()!)).toEqual([0.5, 0.5])
  })
})
