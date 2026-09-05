import { beforeEach, describe, expect, it, vi } from 'vitest'

import { findGroupOfPane, split, group } from '@/components/pane-shell/tree/model'
import type { LayoutNode } from '@/components/pane-shell/tree/model'

// The right-side toggle must be POSITIONAL: it acts on whatever column is
// physically rightmost in the root row — including a preview-tile column
// (the Browser), whose panes register with `placement: 'main'` and are
// therefore invisible to the semantic side derivation. Ground truth for the
// "⌘J / titlebar toggle does nothing to the browser pane" bug: the old
// pane-bound toggle pressed the files pane, which the user had dragged into
// the left stack.

describe('positional right-side toggle', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  async function setup() {
    const tree = await import('@/components/pane-shell/tree/store')
    const layout = await import('@/store/layout')
    const { registry } = await import('@/contrib/registry')

    // Mirror controller.tsx registrations: sessions left, files right, and a
    // preview tile (Browser) that docks beside main with placement 'main'.
    registry.register({ id: 'sessions', area: 'panes', title: 'sessions', data: { placement: 'left' }, render: () => null })
    registry.register({ id: 'files', area: 'panes', title: 'files', data: { placement: 'right' }, render: () => null })
    registry.register({ id: 'preview-tile:url:browser', area: 'panes', title: 'Browser', data: { placement: 'main' }, render: () => null })
    registry.register({ id: 'workspace', area: 'panes', title: 'workspace', data: { placement: 'main' }, render: () => null })

    // User's arrangement: left stack holds sessions+files (dragged), browser
    // column is its own zone on the right of the root row.
    tree.declareDefaultTree(
      split('row', [
        group(['sessions', 'files']),
        group(['workspace']),
        group(['preview-tile:url:browser'])
      ])
    )

    return { tree, layout }
  }

  it('folds the rightmost side column (the browser) to a minimized rail', async () => {
    const { tree, layout } = await setup()

    layout.toggleRightSide()

    const browserGroup = findGroupOfPane(tree.$layoutTree.get() as LayoutNode, 'preview-tile:url:browser')
    expect(browserGroup?.minimized).toBe(true)
  })

  it('round-trips: a second press restores the zone', async () => {
    const { tree, layout } = await setup()

    layout.toggleRightSide()
    layout.toggleRightSide()

    const browserGroup = findGroupOfPane(tree.$layoutTree.get() as LayoutNode, 'preview-tile:url:browser')
    expect(browserGroup?.minimized).toBe(false)
  })

  it('leaves the left stack (sessions+files) untouched', async () => {
    const { tree, layout } = await setup()

    layout.toggleRightSide()

    const filesGroup = findGroupOfPane(tree.$layoutTree.get() as LayoutNode, 'files')
    expect(Boolean(filesGroup?.minimized)).toBe(false)
  })

  it('$rightSideOpen tracks the folded column truthfully', async () => {
    const { layout } = await setup()

    expect(layout.$rightSideOpen.get()).toBe(true)

    layout.toggleRightSide()
    expect(layout.$rightSideOpen.get()).toBe(false)

    layout.toggleRightSide()
    expect(layout.$rightSideOpen.get()).toBe(true)
  })
})
