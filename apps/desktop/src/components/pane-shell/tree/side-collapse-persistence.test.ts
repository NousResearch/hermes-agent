import { beforeEach, describe, expect, it, vi } from 'vitest'

import { LAYOUT_KEYS } from '@/lib/layout-persistence'

/** Re-import real stores over the same storage, as a page refresh does. The
 * workspace is available; Files keeps its own toggle, independent of its side. */
async function boot() {
  vi.resetModules()
  const tree = await import('./store')
  const model = await import('./model')
  const layout = await import('@/store/layout')
  const terminal = await import('@/app/right-sidebar/store')
  const { registry } = await import('@/contrib/registry')
  const { DEFAULT_TREE } = await import('@/app/contrib/layout-presets')

  for (const [id, placement] of [
    ['sessions', 'left'],
    ['workspace', 'main'],
    ['files', 'right'],
    ['review', 'right'],
    ['terminal', 'bottom']
  ]) {
    registry.register({ id, area: 'panes', title: id, data: { placement }, render: () => null })
  }

  tree.declareDefaultTree(DEFAULT_TREE)
  tree.bindTreeSideVisibility('left', layout.$sidebarOpen, layout.setSidebarOpen)
  tree.bindTreeSideVisibility('right', layout.$fileBrowserOpen, layout.setFileBrowserOpen)
  tree.bindPaneVisibility(
    'files',
    layout.$fileBrowserOpen,
    () => layout.setFileBrowserOpen(false),
    () => layout.setFileBrowserOpen(true)
  )
  tree.bindToolPaneCollapse(
    'terminal',
    terminal.$terminalTakeover,
    () => terminal.setTerminalTakeover(false),
    () => terminal.setTerminalTakeover(true)
  )

  return { tree, model, layout, terminal }
}

describe('side visibility survives a page refresh independently of Files', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.history.replaceState(null, '', '/')
    vi.resetModules()
  })

  it('restores the last presentation without revealing hidden, minimized or inactive panes', async () => {
    let state = await boot()
    // An old installation without a side record still uses its chrome default.
    expect(state.tree.$collapsedTreeSides.get().has('right')).toBe(true)
    state.tree.togglePaneVisible('terminal')

    async function refresh(expectSameTree = true) {
      const before = structuredClone(state.tree.$layoutTree.get())
      state.tree.persistTree()
      state = await boot()

      if (expectSameTree) {
        expect(state.tree.$layoutTree.get()).toEqual(before)
      }

      expect(state.layout.$fileBrowserOpen.get()).toBe(false)
      expect(state.tree.$hiddenTreePanes.get().has('files')).toBe(true)
    }

    expect(state.terminal.$terminalTakeover.get()).toBe(true)
    expect(state.tree.$collapsedTreeSides.get().has('right')).toBe(false)
    await refresh()
    // Regression: boot used Files=false to hide the whole right column, even
    // though both the open terminal flag and its complete tree survived.
    expect(state.tree.$collapsedTreeSides.get().has('right')).toBe(false)
    expect(state.tree.isPaneVisible('terminal')).toBe(true)

    state.tree.setTreeGroupMinimized('grp-terminal', true)
    await refresh()
    expect(state.terminal.$terminalTakeover.get()).toBe(true)
    expect(state.tree.isPaneVisible('terminal')).toBe(false)
    expect(state.tree.$collapsedTreeSides.get().has('right')).toBe(false)

    state.tree.restoreTreePane('terminal')
    state.tree.moveTreePane('review', { groupId: 'grp-terminal', pos: 'center' })
    state.tree.activateTreePane('grp-terminal', 'review')
    await refresh()
    expect(state.terminal.$terminalTakeover.get()).toBe(true)
    expect(state.tree.isPaneVisible('review')).toBe(true)
    expect(state.tree.isPaneVisible('terminal')).toBe(false)

    // Explicit side hide wins even while an open terminal owns the active tab.
    state.tree.restoreTreePane('terminal')
    state.layout.setFileBrowserOpen(false)
    await refresh()
    expect(state.tree.$collapsedTreeSides.get().has('right')).toBe(true)
    expect(state.terminal.$terminalTakeover.get()).toBe(true)
    expect(state.tree.isPaneVisible('terminal')).toBe(false)

    state.tree.togglePaneVisible('terminal')
    expect(state.tree.isPaneVisible('terminal')).toBe(true)
    state.tree.closeToolPane('terminal')
    // Default declaration can re-adopt the id, but its dismissal must remain.
    await refresh(false)
    expect(state.tree.$dismissedPanes.get().has('terminal')).toBe(true)
    expect(state.terminal.$terminalTakeover.get()).toBe(false)
    expect(state.tree.isPaneVisible('terminal')).toBe(false)

    // Pop-outs share localStorage, but must not inherit or overwrite the
    // primary layout's sides. Keep the Files default opposite the saved side.
    state.tree.setTreeSideCollapsed('right', false)
    state.tree.setTreeSideCollapsed('left', true)
    const sidesKey = LAYOUT_KEYS.collapsed
    const savedSides = window.localStorage.getItem(sidesKey)

    for (const win of ['secondary', 'browser']) {
      window.history.replaceState(null, '', `/?win=${win}`)
      const auxiliary = await boot()
      expect(auxiliary.tree.$collapsedTreeSides.get().has('right')).toBe(true)
      expect(auxiliary.tree.$collapsedTreeSides.get().has('left')).toBe(false)
      auxiliary.tree.setTreeSideCollapsed('right', true)
      auxiliary.tree.setTreeSideCollapsed('left', false)
      expect(window.localStorage.getItem(sidesKey)).toBe(savedSides)
    }

    window.history.replaceState(null, '', '/')
    state = await boot()
    expect(state.tree.$collapsedTreeSides.get().has('right')).toBe(false)
    expect(state.tree.$collapsedTreeSides.get().has('left')).toBe(true)

    // Invalid storage falls back to this mode's chrome defaults.
    const malformed = JSON.stringify(['left', 'invalid-side'])
    window.localStorage.setItem(sidesKey, malformed)
    state = await boot()
    expect(state.tree.$collapsedTreeSides.get().has('left')).toBe(false)
    expect(state.tree.$collapsedTreeSides.get().has('right')).toBe(true)
  })

  it('toggles an ancestor-hidden pane on the first press and keeps unrelated zones visible', async () => {
    const { tree, model, layout, terminal } = await boot()
    tree.togglePaneVisible('terminal')
    const $visible = tree.$paneVisible('terminal')
    const changes: boolean[] = []
    const unlisten = $visible.subscribe(value => changes.push(value))

    try {
      expect($visible.get()).toBe(true)
      layout.setFileBrowserOpen(false)
      expect(terminal.$terminalTakeover.get()).toBe(true)
      expect(tree.isPaneVisible('terminal')).toBe(false)
      expect($visible.get()).toBe(false)
      expect(changes).toEqual([true, false])
      expect(tree.isPaneVisible('workspace')).toBe(true)
      expect(tree.isPaneVisible('sessions')).toBe(true)

      tree.togglePaneVisible('terminal')
      expect(tree.isPaneVisible('terminal')).toBe(true)
      expect($visible.get()).toBe(true)
      expect(layout.$fileBrowserOpen.get()).toBe(false)
      tree.togglePaneVisible('terminal')
      expect(tree.isPaneVisible('terminal')).toBe(false)
      expect(terminal.$terminalTakeover.get()).toBe(false)

      // The same predicate follows physical position after a layout flip.
      tree.restoreTreePane('terminal')
      tree.mirrorLayoutTree()
      layout.setSidebarOpen(false)
      expect(tree.treeSideOfPane('terminal')).toBe('left')
      expect(tree.isPaneVisible('terminal')).toBe(false)
      tree.togglePaneVisible('terminal')
      expect(tree.isPaneVisible('terminal')).toBe(true)
      expect(layout.$sidebarOpen.get()).toBe(false)

      // A terminal below the root row is not owned by either sidebar.
      tree.applyTree(
        model.split('column', [
          model.split('row', [model.group(['sessions']), model.group(['workspace']), model.group(['files', 'review'])]),
          model.group(['terminal'])
        ]),
        'terminal-deck'
      )
      layout.setFileBrowserOpen(false)
      layout.setSidebarOpen(false)
      expect(tree.treeSideOfPane('terminal')).toBeNull()
      expect(tree.isPaneVisible('terminal')).toBe(true)
      expect(tree.isPaneVisible('workspace')).toBe(true)
    } finally {
      unlisten()
    }
  })
})
