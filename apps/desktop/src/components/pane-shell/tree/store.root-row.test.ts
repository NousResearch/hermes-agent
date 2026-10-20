import { beforeEach, describe, expect, it, vi } from 'vitest'

// rootRow() has its own hasMain() closure, separate from paneRootSide()'s own
// mainIndices check (a sibling of the fix in aa05e5c0f4 "resolve side ownership
// before workspace registration"). In a column-root layout (Terminal deck, Quad),
// rootRow() runs FIRST inside paneRootSide() -- if its hasMain() fails to
// recognize the workspace pane before the registry has registered its
// data.placement === 'main' metadata, rootRow() returns null and every caller
// (paneRootSide, layoutHasRootSide, restoreMinimizedTreeSide, treeSideOfPane)
// short-circuits before aa05e5c0f4's own fix (inside paneRootSide's mainIndices)
// ever runs.
describe('rootRow() column-root workspace recognition', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('paneRootSide finds the side even when the workspace pane has not registered its placement yet', async () => {
    const { split, group } = await import('./model')
    const tree = await import('./store')
    const { registry } = await import('@/contrib/registry')

    const sidebar = group(['sessions'], { id: 'sidebar' })
    const main = group(['workspace'], { id: 'main' })
    const files = group(['files'], { id: 'files-zone' })
    // Column root: a row of sidebar/main/files nested under a column split --
    // the shape rootRow() must unwrap to find the side-eligible row.
    tree.declareDefaultTree(split('column', [split('row', [sidebar, main, files])]))

    // Deliberately do NOT register 'workspace' in the panes area -- rootRow()'s
    // hasMain() must still recognize it by id, the same way paneRootSide()'s own
    // mainIndices check already does.
    const disposers = [
      registry.register({ area: 'panes', id: 'sessions', data: { placement: 'left' } }),
      registry.register({ area: 'panes', id: 'files', data: { placement: 'right' } })
    ]

    try {
      expect(tree.paneRootSide('sessions')).toBe('left')
      expect(tree.paneRootSide('files')).toBe('right')
      expect(tree.layoutHasRootSide('left')).toBe(true)
      expect(tree.layoutHasRootSide('right')).toBe(true)
    } finally {
      disposers.forEach(dispose => dispose())
    }
  })

  it('restoreMinimizedTreeSide recovers a minimized side in a column-root layout without a registered workspace pane', async () => {
    const { split, group, findGroup } = await import('./model')
    const tree = await import('./store')
    const { registry } = await import('@/contrib/registry')

    const sidebar = group(['sessions'], { id: 'sidebar' })
    const main = group(['workspace'], { id: 'main' })
    const files = group(['files'], { id: 'files-zone' })
    tree.declareDefaultTree(split('column', [split('row', [sidebar, main, files])]))

    const disposers = [
      registry.register({ area: 'panes', id: 'sessions', data: { placement: 'left' } }),
      registry.register({ area: 'panes', id: 'files', data: { placement: 'right' } })
    ]

    try {
      tree.setTreeGroupMinimized('sidebar', true)
      expect(findGroup(tree.$layoutTree.get()!, 'sidebar')?.minimized).toBe(true)

      const restored = tree.restoreMinimizedTreeSide('left')

      expect(restored).toBe(true)
      expect(findGroup(tree.$layoutTree.get()!, 'sidebar')?.minimized).toBe(false)
    } finally {
      disposers.forEach(dispose => dispose())
    }
  })
})
